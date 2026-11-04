# Capacity score, Langevin step with the dispatch following, replica batch and weather upload on
# the device, against the CPU reference (docs/system/testing.md, device order): (1) the device
# code on Julia arrays against the CPU, with the same noise; (2) the same code compiled by
# Reactant (CPU backend by default, REACTANT_BACKEND) against (1).
#
# CPU reference: `energy_cost(…; grad=true)` + `log_measure_grad` for the scores (through the
# prototype's `lgv_scores`); the prototype's `lgv_knew`/`lgv_apply!` (scripts/replicas/replicas.jl)
# for the step and the dispatch following, `compensate!` for the weather upload. To be switched to
# the src/ functions of the replica sampler once they are merged.
#   julia --project=device -t 4 device/test/test_langevin_dev.jl
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice
include(joinpath(@__DIR__, "mockrng.jl"))
Random.randn(r::MockRNG, n::Integer) = [randn(r) for _ in 1:n]
include(joinpath(@__DIR__, "setup_all_terms.jl"))       # 40 nodes × 6 h, all model terms, physical measure
module Proto                                             # the prototype (reference until it is in src/)
include(joinpath(@__DIR__, "..", "..", "scripts", "langevin", "langevin.jl"))
include(joinpath(@__DIR__, "..", "..", "scripts", "replicas", "replicas.jl"))
end
LMc = ML.LocalModel(P, L)
runits = D.ren_units(P)
KP = D.KParams(M)
ph = Val(P.E.measure == :physical)

# chains: perturbed copies of the heuristic start (generic residuals); replicas: fresh prior weather, compensated
rng = MersenneTwister(3)
bases = ML.State[]
for c in 1:2
    s = copy(sL)
    s.v .+= 0.02 .* c .* randn(rng, length(s.v)); s.k .*= 1 .+ 0.05 .* randn(rng, length(s.k))
    push!(bases, ML.evaluate!(P, s))
end
nch, Mrep = 2, 2
C = nch * Mrep
reps = ML.State[]
for ch in 1:nch, j in 1:Mrep
    b = bases[ch]
    mm, z0, ws = ML.sample_weather(rng, P.W, L)
    push!(reps, Proto.compensate!(P, LMc, copy(ML.LocalCost(P, LMc, b).r), ML.make_state(P, L, mm, z0, ws, copy(b.k), copy(b.v))))
end
Sd = D.to_device(P, M, ix, reps)
st0 = D.flat_state(M, Sd)
RP, RS0 = D.pack_ren(P, M, reps)
dims = Val((C, M.T, M.N, length(ix.gidx), length(ix.sidx), length(M.kcap), Mrep))
rel(a, b) = maximum(abs.(a .- b) ./ max.(1.0, abs.(b)))
A_(x) = x isa Union{Number,AbstractArray} ? Array(x) : x

# ---------------------------------------------------------------- 1. score
"Largest relative error of the device scores and forces against the CPU's, per unit class."
function cmp_score(sc, ss)
    worst = Dict{Symbol,Float64}()
    for (c, s) in enumerate(ss)
        sk, fk = Proto.lgv_scores(P, s)
        for (blk, fblk, idx, kfof) in ((:sg, :fg, ix.gidx, g -> P.E.gens[g].kf), (:ss, :fs, ix.sidx, g -> P.E.stors[g].kf),
                                       (:sr, :fr, runits, g -> P.E.gens[g].kf))
            for (q, (g, i)) in enumerate(idx)
                kf = kfof(g); kf > 0 || continue
                a, b = Array(getfield(sc, blk))[c, q], sk[kf][i]
                fa, fb = Array(getfield(sc, fblk))[c, q], fk[kf][i]
                worst[blk] = max(get(worst, blk, 0.0), abs(a - b) / max(abs(b), 1e-3 * fb * P.βv / P.βk))
                worst[fblk] = max(get(worst, fblk, 0.0), abs(fa - fb) / abs(fb))
            end
        end
    end
    return worst
end
r0 = D.rebuild_r(M, Sd)
sc = D.cap_score(KP, RP, Sd.ω, Sd.uc, Sd.e, Sd.kg, Sd.ks, RS0.kr, RS0.A, r0, ph; βv=P.βv, βk=P.βk)
@testset "capacity score (arrays) = energy_cost gradient + log_measure_grad" begin
    w = cmp_score(sc, reps)
    println("  score, relative: ", w)
    @test all(v < 1e-10 for v in values(w))
end

# ---------------------------------------------------------------- 2. Langevin step + dispatch following
ηw = fill(5e-4, 6); κsv = [500.0, 1000.0, 200.0, 300.0, 100.0, 100.0]
W = fill(1 / Mrep, C)
rngξ = MersenneTwister(21)
ξ = [[randn(rngξ, length(P.E.kspecs[f].lo)) for f in eachindex(P.E.kspecs)] for _ in 1:nch]

"The prototype: per chain the weighted replica scores, `lgv_knew` with the given normals, `lgv_apply!` per replica."
function lgv_reference(reps, W, ξ; ηw, κs, clip=0.02, box=true, κmin=1.0)
    units = Proto.lgv_units(P, LMc)
    out = ML.State[]
    for ch in 1:nch
        js = (ch - 1) * Mrep .+ (1:Mrep)
        scs = [Proto.lgv_scores(P, reps[c]) for c in js]
        sk̄ = [sum(W[c] .* scs[a][1][f] for (a, c) in enumerate(js)) for f in eachindex(P.E.kspecs)]
        Kraw = ML.knodes(P, L, reps[js[1]].k)
        z = reduce(vcat, [ξ[ch][kf] for kf in unique(u[4] for u in units) if ηw[kf] != 0])
        knew, _ = Proto.lgv_knew(MockRNG(z, Float64[]), P, units, Kraw, sk̄, scs[1][2]; ηw, κs, κmin, clip, box)
        append!(out, [Proto.lgv_apply!(P, LMc, copy(reps[c]), units, Kraw, knew; κmin) for c in js])
    end
    return out
end
"The same normals per chain and device unit (C/M, U)."
function ξ_dev(ξ)
    gk = [P.E.gens[g].kf for (g, i) in ix.gidx]; sk = [P.E.stors[s].kf for (s, i) in ix.sidx]; rk = [P.E.gens[g].kf for (g, i) in runits]
    mk(kf, idx) = [kf[q] > 0 ? ξ[ch][kf[q]][idx[q][2]] : 0.0 for ch in 1:nch, q in eachindex(idx)]
    return mk(gk, ix.gidx), mk(sk, ix.sidx), mk(rk, runits)
end
"Errors of a device step (`lgv_pass` outputs) against the CPU states."
function cmp_step(out, cpu)
    st2, kg, ks, cp, RS, dnet, cren, capren, diag = map(A_, out)
    Sh = deepcopy(Sd); D.unflat!(Sh, st2); r2 = reshape(Array(st2.r), C, M.T, M.N)
    Sc = D.to_device(P, M, ix, cpu); _, RSc = D.pack_ren(P, M, cpu)
    return (; kg=rel(Array(kg), Sc.kg), ks=rel(Array(ks), Sc.ks), kr=rel(Array(RS.kr), RSc.kr),
            ω=maximum(abs.(Sh.ω .- Sc.ω)), uc=maximum(abs.(Sh.uc .- Sc.uc)), e=maximum(abs.(Sh.e .- Sc.e)),
            r=maximum(abs.(r2 .- D.rebuild_r(M, Sc))), dnet=maximum(abs.(Array(dnet) .- Sc.dnet)),
            cren=rel(Array(cren), Sc.cren), capren=rel(Array(capren), Sc.capren))
end
okstep(e) = e.kg < 1e-12 && e.ks < 1e-12 && e.kr < 1e-12 && e.ω < 1e-10 && e.uc < 1e-12 && e.e == 0 &&
            e.r < 1e-6 && e.dnet < 1e-6 && e.cren < 1e-12 && e.capren < 1e-12
cpu_out = lgv_reference(reps, W, ξ; ηw, κs=κsv)
LP = D.lgv_params(P, M, ix; κs=κsv, ηw)
rg, sgp = D.lgv_groups(P, ix)
wsh = D.load_shares(P, reps)
ξg, ξs, ξr = ξ_dev(ξ)
acc0 = D.ScoreAcc(C, length(ix.gidx), length(ix.sidx), length(runits))
kw = (; rr=P.βk / P.βv, βk=P.βk, clip=0.02, κmin=1.0, box=true)
function dev_step(st, kg, ks, RS, dnet, cren, capren, KP, RP, LP, acc, W, ξg, ξs, ξr, wsh, rg, sgp)
    acc = D.score_acc(st, kg, ks, RS, KP, RP, acc, 1.0, dims, ph; βv=P.βv, βk=P.βk)
    return D.lgv_pass(st, kg, ks, RS, dnet, cren, capren, KP, RP, LP, acc, W, ξg, ξs, ξr, wsh, rg, sgp, dims; kw...)
end
step_args() = (st0, copy(Sd.kg), copy(Sd.ks), deepcopy(RS0), copy(Sd.dnet), copy(Sd.cren), copy(Sd.capren), KP, RP, LP, acc0, W,
               ξg, ξs, ξr, wsh, rg, sgp)
out_a = dev_step(step_args()...)
@testset "Langevin step + dispatch following (arrays) = prototype" begin
    e = cmp_step(out_a, cpu_out)
    println("  step: ", e, "  clipped drifts per chain ", out_a[9].nclip)
    @test okstep(e)
    Sh = deepcopy(Sd); D.unflat!(Sh, out_a[1])
    @test count(Sh.ω .!= Sd.ω) > 0                          # the compensators moved
    kg2 = Array(out_a[2])
    @test all(kg2[(ch-1)*Mrep+1, :] == kg2[(ch-1)*Mrep+j, :] for ch in 1:nch, j in 1:Mrep)   # k shared within a chain
end

# ---------------------------------------------------------------- 3. weather upload (replicas 1 and 3)
chg = Float64[isodd(c) for c in 1:C]
rngw = MersenneTwister(31)
new_states = ML.State[]; cpu_w = ML.State[]
for (c, s) in enumerate(reps)
    if chg[c] > 0
        mm, z0, ws = ML.sample_weather(rngw, P.W, L)
        s1 = ML.make_state(P, L, mm, z0, ws, copy(s.k), copy(s.v))
        push!(new_states, s1)
        push!(cpu_w, Proto.compensate!(P, LMc, copy(ML.LocalCost(P, LMc, s).r), copy(s1)))
    else
        push!(new_states, s); push!(cpu_w, copy(s))
    end
end
D0, cren0, Aw = D.weather_arrays(P, M, new_states)
wsh1 = D.load_shares(P, new_states)
wu(st, kg, ks, RS, dnet, cren, KP, RP, D0, cren0, A, chg, wsh) =
    D.weather_update(st, kg, ks, RS, dnet, cren, KP, RP, D0, cren0, A, chg, wsh, dims, Val(true))
wu_args() = (st0, copy(Sd.kg), copy(Sd.ks), deepcopy(RS0), copy(Sd.dnet), copy(Sd.cren), KP, RP, D0, cren0, Aw, chg, wsh1)
function cmp_weather(wout)
    st2, RS2, dn2, cr2, cp2 = map(A_, wout)
    Sh = deepcopy(Sd); D.unflat!(Sh, st2); r2 = reshape(Array(st2.r), C, M.T, M.N)
    Sc = D.to_device(P, M, ix, cpu_w); _, RSc = D.pack_ren(P, M, cpu_w)
    cpc = D.chain_params(M, Sc; cls=D.proxy_class(Sc.dnet, wsh1))
    return (; ω=maximum(abs.(Sh.ω .- Sc.ω)), r=maximum(abs.(r2 .- D.rebuild_r(M, Sc))), dnet=maximum(abs.(Array(dn2) .- Sc.dnet)),
            cren=rel(Array(cr2), Sc.cren), A=maximum(abs.(Array(RS2.A) .- RSc.A)), cls=count(Array(cp2.cls) .!= cpc.cls),
            κg=maximum(abs.(Array(cp2.κg) .- cpc.κg)))
end
okw(e) = e.ω < 1e-12 && e.r < 1e-6 && e.dnet < 1e-6 && e.cren < 1e-12 && e.A == 0 && e.cls == 0 && e.κg == 0
wout_a = wu(wu_args()...)
@testset "weather upload with compensation (arrays) = prototype compensate!" begin
    e = cmp_weather(wout_a)
    println("  weather: ", e)
    @test okw(e)
end

# ---------------------------------------------------------------- compiled by Reactant = arrays
Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
R = Reactant.to_rarray
"Largest difference of two output tuples (arrays and structs of arrays), elementwise."
function maxdiff(a, b)
    a isa Union{Number,AbstractArray} && return maximum(abs.(Array(a) .- Array(b)); init=0.0)
    a isa NamedTuple && return maximum(maxdiff(x, y) for (x, y) in zip(values(a), values(b)))
    a isa Tuple && return maximum(maxdiff(x, y) for (x, y) in zip(a, b))
    return maximum(maxdiff(getfield(a, f), getfield(b, f)) for f in fieldnames(typeof(a)))
end
@testset "compiled by Reactant = arrays" begin
    args = map(R, step_args())
    tc = @elapsed f = @compile dev_step(args...)
    out_c = f(args...)
    d = maxdiff(out_c, out_a)
    @printf("  score + Langevin step: compile %.1f s, max |compiled − arrays| %.2e\n", tc, d)
    @test okstep(cmp_step(out_c, cpu_out))
    @test d < 1e-6
    args = map(R, wu_args())
    tc = @elapsed f = @compile wu(args...)
    wout_c = f(args...)
    d = maxdiff(wout_c, wout_a)
    @printf("  weather upload: compile %.1f s, max |compiled − arrays| %.2e\n", tc, d)
    @test okw(cmp_weather(wout_c))
    @test d < 1e-6
    # the noise from Reactant's RNG: same shapes, finite, and the step with it runs
    rngd = Reactant.ReactantRNG(R(UInt64[7, 0x9e3779b9]))
    g(st, kg, ks, RS, dnet, cren, capren, KP, RP, LP, acc, W, rng, wsh, rg, sgp) =
        D.lgv_pass_rng(st, kg, ks, RS, dnet, cren, capren, KP, RP, LP, acc, W, rng, wsh, rg, sgp, dims; kw...)
    a2 = map(R, (st0, copy(Sd.kg), copy(Sd.ks), deepcopy(RS0), copy(Sd.dnet), copy(Sd.cren), copy(Sd.capren), KP, RP, LP, acc0, W))
    a3 = (R(wsh), map(R, rg), map(R, sgp))
    fr = @compile g(a2..., rngd, a3...)
    o = fr(a2..., rngd, a3...)
    k1 = Array(o[2])
    o = fr(a2..., rngd, a3...)                               # the RNG state advances between calls
    @test Array(o[2]) != k1
    @test all(isfinite, Array(o[2])) && all(isfinite, Array(o[1].r))
end
