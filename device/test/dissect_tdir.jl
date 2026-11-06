# Dissect test_comp cases F/G (dense angle directions): per family and round, the compiled round
# (Reactant CPU backend) against the array round FROM THE SAME INPUT STATE, per site.
# Run from the repository root (a worktree), CASE=F or G:
#   CASE=G julia --project=device -t 4 device/test/dissect_tdir.jl   (LOOP=1: the run_rounds program per family and for both;
#   SKIP1=1 skips the first family alone; CHAIN=1: the compiled side continues from its own state; about 10 GB)
#   on the cluster: hpc/jobs/devtest.sbatch with the test spec dissect_tdir.jl:CASE=G,LOOP=1,SKIP1=1
cd(get(ENV, "REPO", pwd()))
include(joinpath(pwd(), "device/test/setup40.jl"))
include(joinpath(pwd(), "device/test/helpers.jl"))
using Random, Printf
using Reactant: to_rarray
Reactant.set_default_backend("cpu")
say(x...) = (println(x...); flush(stdout))
mem() = @sprintf("%.1f GB", Sys.maxrss() / 2^30)

const NSITE = parse(Int, get(ENV, "NSITE", "600"))
weyl(n, off) = [mod(0.5 + (off + k) * 0.6180339887498949, 1.0) for k in 1:n]
case = get(ENV, "CASE", "G")
tracking_model = case == "G"
fams = case == "G" ? (:tdirL, :tdir0) : (:tdir, :tdir0)
C = 2; nround = 2

P, M, ix, S, ss = setup40(; C=C, losses=tracking_model, link_losses=tracking_model)
E = P.E; β = P.βv; ℓ = ss[1].ℓ
LM = ML.LocalModel(P, ℓ)
LCs = [ML.LocalCost(P, LM, s) for s in ss]
AM = ML.AdaptiveModes(P, ℓ; K=2, lengths=[2, 4, 8])
rng = MersenneTwister(4)
for n in eachindex(AM.β); AM.β[n] .= 0.1 .+ 0.8 .* rand(rng, size(AM.β[n])...); end
cls = zeros(C, M.T * M.N)
for c in 1:C; cls[c, :] .= vec(permutedims(Float64.(S.dnet[c, :, :] .< 0))); end
D.TDIR_CHAINS[] = C
D.TDIR_SOURCE[] = (fam, τ, rng, c) -> begin
    sys = E.sys
    rand(rng) < 0.3 && return D.tline_dir(sys, LM, rand(rng, eachindex(sys.line_s)))
    d = randn(rng, sys.N); d .-= sum(d) / sys.N; d ./= sqrt(sum(abs2, d))
    D.tdir_g(sys, d)
end
say("setup done ", mem())
pl = D.plan(M, S, β; families=fams, cpumap=true, cls=cls, bmax=Dict(f => NSITE for f in fams))
say("plan done ", mem(), "; form ", [(f, D.jumpform(pl, f), D.jumpcap(pl, f)) for f in pl.fams])
qof = Dict(ix.gidx[q] => q for q in eachindex(ix.gidx))
for n in eachindex(AM.runits), (j, (f, i)) in enumerate(AM.runits[n])
    g = findfirst(gg -> gg.vf == f, E.gens)
    q = qof[(g, i)]
    pl.sp.β1[q] = AM.β[n][j, 1]; pl.sp.β2[q] = AM.β[n][j, 2]
end
st0 = cpu_r_state(pl, LCs)
rounds = NamedTuple{pl.fams}(Tuple(pl.sched[f][1:min(nround, end)] for f in pl.fams))
tabs = NamedTuple{pl.fams}(Tuple(D.device_table(pl, f, D.host_table(f, pl.tp, rounds[f])) for f in pl.fams))
off = Ref(0)
U = NamedTuple{pl.fams}(Tuple(begin
    B, R = size(tabs[f].on)
    x = reshape(weyl(C * B * 3 * R, off[]), C, B, 3, R); off[] += length(x); x
end for f in pl.fams))
say("tables done ", mem(), "; sites ", [(f, size(tabs[f].on)) for f in pl.fams])

const CHAIN = get(ENV, "CHAIN", "0") == "1"      # 1: the compiled side continues from its own state
st = st0
stc = st0
hostst(o) = D.FlatState(map(Array, (o.θ, o.ω, o.uc, o.e, o.u, o.r))...)
for f in (get(ENV, "LOOP", "0") == "1" ? () : pl.fams)
    global st, stc
    tab = tabs[f]
    step(st, cp, sp, k, tab, r, u) = D.do_round(Val(f), st, cp, sp, k, tab, r, u)
    R = size(tab.on, 2)
    rr = Reactant.ConcreteRNumber(1)
    args = (to_rarray(st), to_rarray(pl.cp), to_rarray(pl.sp), to_rarray(pl.k), to_rarray(tab), rr, to_rarray(U[f][:, :, :, 1]))
    tc = @elapsed fc = @compile step(args...)
    say(@sprintf("%s: compiled one round in %.0f s, %s", f, tc, mem()))
    for r in 1:R
        u = U[f][:, :, :, r]
        stA, tA, gA = D.do_round(Val(f), st, pl.cp, pl.sp, pl.k, tab, r, u)
        o = fc(to_rarray(CHAIN ? stc : st), to_rarray(pl.cp), to_rarray(pl.sp), to_rarray(pl.k), to_rarray(tab), Reactant.ConcreteRNumber(r), to_rarray(u))
        tR = Array(o[2]); gR = map(x -> x === nothing ? nothing : Array(x), o[3])
        d = abs.(tR .- tA) ./ max.(1.0, abs.(tA))
        bad = findall(>(1e-9), d)
        say(@sprintf("  %s r %d: max rel |Δt| %.2e, %d of %d sites > 1e-9; root guards arrays %d compiled %d; cap guards %s",
                     f, r, maximum(d), length(bad), length(d), sum(gA[1]), sum(gR[1]),
                     gA[2] === nothing ? "none" : string(sum(gA[2]), " / ", sum(gR[2]))))
        for I in first(bad, 5)
            c, b = I[1], I[2]
            say(@sprintf("    site (c %d, b %d): t arrays %.6e compiled %.6e; root guard %s/%s%s", c, b, tA[I], tR[I],
                         gA[1][c, b, 1], gR[1][c, b, 1], gA[2] === nothing ? "" : string("; cap guard ", gA[2][c, b, 1], "/", gR[2][c, b, 1])))
        end
        st = stA                                     # continue from the array state
        if CHAIN
            stc = hostst(o[1])
            for fld in (:θ, :ω, :uc, :e, :u, :r)
                a, b = getfield(stc, fld), getfield(stA, fld)
                dd = maximum(abs.(a .- b)); sc = max(maximum(abs.(b)), 1.0)
                say(@sprintf("    state %-2s max |compiled − arrays| %.2e (rel %.1e)", fld, dd, dd / sc))
            end
        end
    end
end
say("done ", mem())

# LOOP=1: the test's program (`run_rounds`, traced loops over the rounds) for each family alone and
# for all families, against the array chain of the same families
if get(ENV, "LOOP", "0") == "1"
    function arrays_chain(fs)
        s = st0
        for f in fs, r in 1:size(tabs[f].on, 2)
            s, _ = D.do_round(Val(f), s, pl.cp, pl.sp, pl.k, tabs[f], r, U[f][:, :, :, r])
        end
        return s
    end
    for fs in (get(ENV, "SKIP1", "0") == "1" ? (map(f -> (f,), pl.fams[2:end])..., pl.fams) : (map(f -> (f,), pl.fams)..., pl.fams))
        tb = NamedTuple{fs}(Tuple(tabs[f] for f in fs)); Uf = NamedTuple{fs}(Tuple(U[f] for f in fs))
        fv = Val(fs)
        runU(st, cp, sp, k, tabs, U) = D.run_rounds(st, cp, sp, k, tabs, 1, nothing, U, fv)
        args = (to_rarray(st0), to_rarray(pl.cp), to_rarray(pl.sp), to_rarray(pl.k), to_rarray(tb), to_rarray(Uf))
        tc = @elapsed fr = @compile runU(args...)
        o = hostst(fr(args...))
        sA = arrays_chain(fs)
        say(@sprintf("run_rounds %s: compiled in %.0f s, %s", fs, tc, mem()))
        for fld in (:θ, :ω, :uc, :e, :u, :r)
            a, b = getfield(o, fld), getfield(sA, fld)
            dd = maximum(abs.(a .- b)); sc = max(maximum(abs.(b)), 1.0)
            say(@sprintf("    state %-2s max |compiled − arrays| %.2e (rel %.1e; bitwise %.1f %%)", fld, dd, dd / sc, 100 * count(a .== b) / length(a)))
        end
    end
end
