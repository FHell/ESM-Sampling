# Phase 3, part 1: compensated moves against the CPU with mock uniforms (no statistics).
#   model A: 40 nodes × 6 h, all model terms on (line losses, lossy links: tracking kinks on the CPU)
#            families :spair (UNIT), :tsocL, :tpotL (LINEAR; learned β random per unit, proxy class
#            from the chain's net load)
#   model B: the same without line losses and lossy links (soft limits, vintages, UC still on)
#            families :tsoc, :tpot (T2: static kinks + the Jacobian independence step)
#   model A, families :cell, :cellpair (cell templates on every partition level, level-bucketed)
#   model A, :sblkL2, :sblkL4 (storage blocks, LINEAR); model B, :sblk2, :sblk4 (T2); plateau and shapes
# For each: the first rounds of every family (a prefix of NSITE sites), the same explicit sites on
# the CPU (prep_node!, prorata!, prep_tpot!(…; n), compensate_learned! via `learn`), the same (u1,
# u2, u3) per site (MockRNG for sample_line!, u3 for the Jacobian step); both sides commit the
# device's t; then states, r and cost; then Reactant (CPU backend) against the Julia-array run.
# Run: cd device && nice -n 19 julia --project=. -t 4 test/test_comp.jl   (CASE=A|B|C|D|E, SKIP=fam1,fam2 leaves families out)
include("setup40.jl")
include("helpers.jl")
using Test, Random
using Reactant: to_rarray
Reactant.set_default_backend(get(ENV, "DEVICE_BACKEND", "cpu"))

const NSITE = parse(Int, get(ENV, "NSITE", "600"))
const DEV_RE = get(ENV, "DEVICE_REACTANT", "1") == "1"
weyl(n, off) = [mod(0.5 + (off + k) * 0.6180339887498949, 1.0) for k in 1:n]

function comp_case(name; tracking_model::Bool, fams, C=2, nround=2, cells=false)
    for L in D.BLOCK_LENGTHS                     # block profiles: the plateau and two shapes with zero ends
        D.set_profiles!(L, hcat([vcat(0.0, rand(MersenneTwister(L + j), L - 1) .- 0.3, 0.0) for j in 1:2]...))
    end
    P, M, ix, S, ss = setup40(; C=C, losses=tracking_model, link_losses=tracking_model)
    E = P.E
    β = P.βv
    ℓ = ss[1].ℓ
    LM = ML.LocalModel(P, ℓ)
    LCs = [ML.LocalCost(P, LM, s) for s in ss]
    # learned tables: random β per (unit, class); proxy classes from each chain's FlowFrames
    AM = ML.AdaptiveModes(P, ℓ; K=2, lengths=[2, 4, 8])
    rng = MersenneTwister(4)
    for n in eachindex(AM.β); AM.β[n] .= 0.1 .+ 0.8 .* rand(rng, size(AM.β[n])...); end
    # proxy class: surplus where the net load (load − renewables) is negative (state-independent; the
    # CPU's compensate_learned! reads only FF.surplus, so a stand-in carries it; FlowFrames needs the
    # flow basis, this setup is graph-Haar)
    FFs = [(surplus=BitMatrix(permutedims(S.dnet[c, :, :] .< 0)),) for c in 1:C]
    cls = zeros(C, M.T * M.N)
    for c in 1:C; cls[c, :] .= vec(permutedims(Float64.(FFs[c].surplus))); end
    ci = cells ? D.cell_info(P, LM, ix) : nothing
    fams == :cells && (fams = D.cell_families(ci))
    fams = Tuple(f for f in fams if !(string(f) in split(get(ENV, "SKIP", ""), ",")))   # SKIP=cellpairu1: leave out
    # dense angle directions (tdir families): random node directions per chain, a third binding-line
    # normals of random lines (the CPU reference compensates the same g, `cpu_move`)
    D.TDIR_CHAINS[] = C
    D.TDIR_SOURCE[] = (fam, τ, rng, c) -> begin
        sys = E.sys
        rand(rng) < 0.3 && return D.tline_dir(sys, LM, rand(rng, eachindex(sys.line_s)))
        d = randn(rng, sys.N); d .-= sum(d) / sys.N; d ./= sqrt(sum(abs2, d))      # unit norm, as the modes
        D.tdir_g(sys, d)
    end
    # sparse angle directions: Haar atoms of random modes (as `localize` of the proxy and learned modes)
    Hθ = ML.haar_theta(E)
    Dm = randn(MersenneTwister(5), E.sys.N, 4); Dm .-= sum(Dm; dims=1) ./ E.sys.N
    D.SPARSE_SOURCE[] = D.sparse_source(P, LM, ix; atoms=() -> Any[(Hθ, ML.localize(Dm, Hθ))])
    pl = D.plan(M, S, β; families=fams, cpumap=true, cls=cls, bmax=Dict(f => NSITE for f in fams), cells=ci)
    qof = Dict(ix.gidx[q] => q for q in eachindex(ix.gidx))
    for n in eachindex(AM.runits), (j, (f, i)) in enumerate(AM.runits[n])
        g = findfirst(gg -> gg.vf == f, E.gens)
        q = qof[(g, i)]
        pl.sp.β1[q] = AM.β[n][j, 1]; pl.sp.β2[q] = AM.β[n][j, 2]
    end
    st0 = cpu_r_state(pl, LCs)
    rounds = NamedTuple{pl.fams}(Tuple(pl.sched[f][1:min(nround, end)] for f in pl.fams))
    tabs = NamedTuple{pl.fams}(Tuple(D.device_table(f, D.host_table(f, pl.tp, rounds[f])) for f in pl.fams))
    off = Ref(0)
    U = NamedTuple{pl.fams}(Tuple(begin
        B, R = size(tabs[f].on)
        x = reshape(weyl(C * B * 3 * R, off[]), C, B, 3, R); off[] += length(x); x
    end for f in pl.fams))
    learnof(f, c) = (f in (:tsocL, :tpotL, :tdirL, :tcutL, :tcutLw) || startswith(string(f), "sblkL")) ? (AM=AM, FF=FFs[c]) : nothing
    cpumove(c, LC, f, site) = (jac = NTuple{5,Float64}[];
                              bm = cpu_move(P, LM, LC, ix, f, site; jac=jac, learn=learnof(f, c), chain=c); (bm, jac))
    ok_conf = all(D.check_conflicts(f, D.host_table(f, pl.tp, pl.sched[f]))[1] for f in pl.fams)
    # exactness of the dense form with kinked and LINEAR entries: ≤ 2 sign changes per secondary quantity
    maxroots = Dict(f => maximum(maximum(D.root_count(D.build(Val(f), st0, pl.cp, pl.sp, tabs[f], r, pl.k)[1], pl.k))
                                 for r in 1:size(tabs[f].on, 2)) for f in pl.fams)
    t0 = time()
    stJ, stats, scpu = mock_compare(pl, st0, LCs, ss, pl.fams, rounds, tabs, U, cpumove; P=P, β=β)
    @printf("%s: families %s, %d draws in %.0f s\n", name, pl.fams, stats[:n], time() - t0)
    @testset "$name: device = CPU (mock uniforms)" begin
        @test ok_conf
        println("  max sign changes of a secondary quantity: ", maxroots, "; sites beyond the kept roots (device stays, exact guard): ",
                stats[:guard], " of ", stats[:n])
        @test stats[:guard] <= 0.01 * stats[:n]
        @printf("  (a) device own points vs CPU: %d agree (max %.1e), %d differ (CPU ray splitting)\n", stats[:own_ok], stats[:worst_own], stats[:own_bad])
        @printf("  (c) device sampler on the CPU points vs CPU: max %.1e, %d beyond 1e-9; CPU ray overshoot excluded: %d; CPU walk truncated (a deeper region beyond its 40-nat stop) excluded: %d; improper mismatch %d\n",
                stats[:worst_cpu], stats[:cpupts_bad], stats[:overshoot], stats[:truncated], stats[:improper_mismatch])
        @printf("  Jacobian step: CPU rejections %d, decision mismatches on the same t %d\n", stats[:jac_rej], stats[:jac_mismatch])
        @test stats[:cpupts_bad] == 0
        @test stats[:improper_mismatch] == 0
        @test stats[:jac_mismatch] == 0
        @test stats[:overshoot] <= 1e-3 * stats[:n]
        @test stats[:truncated] <= 1e-2 * stats[:n]
        stC = cpu_flat(P, ix, LCs)
        for fld in (:θ, :ω, :uc, :e, :u, :r)
            a, b = getfield(stJ, fld), getfield(stC, fld)
            d = maximum(abs.(a .- b)); sc = max(maximum(abs.(b)), 1.0)
            @printf("  %-2s max |device − CPU| = %.2e (relative %.1e; bitwise equal %.1f %%)\n", fld, d, d / sc, 100 * count(a .== b) / length(a))
            @test d <= 1e-10 * sc
        end
        S1 = deepcopy(S); rc = D.unflat!(S1, stJ)
        rb, parts = D.cost_parts(M, S1)
        @test maximum(abs.(rb .- rc)) < 1e-8
        for c in 1:C
            rel = abs(parts.op[c] / scpu[c].parts.op - 1)
            @printf("  chain %d: op device %.10e, CPU energy_cost %.10e (rel %.1e)\n", c, parts.op[c], scpu[c].parts.op, rel)
            @test rel < 1e-11
        end
    end
    if DEV_RE
        @testset "$name: Reactant = Julia arrays" begin
            fv = Val(pl.fams)
            runU(st, cp, sp, k, tabs, U) = D.run_rounds(st, cp, sp, k, tabs, 1, nothing, U, fv)
            args = (to_rarray(st0), to_rarray(pl.cp), to_rarray(pl.sp), to_rarray(pl.k), to_rarray(tabs), to_rarray(U))
            tc = @elapsed f = @compile runU(args...)
            o = f(args...)
            oR = D.FlatState(map(Array, (o.θ, o.ω, o.uc, o.e, o.u, o.r))...)
            @printf("  compile %.0f s\n", tc)
            for fld in (:θ, :ω, :uc, :e, :u, :r)
                a, b = getfield(oR, fld), getfield(stJ, fld)
                d = maximum(abs.(a .- b)); sc = max(maximum(abs.(b)), 1.0)
                @printf("  %-2s max |reactant − julia| = %.2e (relative %.1e; bitwise %.1f %%)\n", fld, d, d / sc, 100 * count(a .== b) / length(a))
                @test d <= 1e-10 * sc
            end
        end
    end
    return pl, stats
end

if get(ENV, "CASE", "ABC") in ("A", "ABC")
    comp_case("A (all terms; UNIT, LINEAR)"; tracking_model=true, fams=(:spair, :tsocL, :tpotL))
end
if get(ENV, "CASE", "ABC") in ("B", "ABC")
    comp_case("B (no losses; T2)"; tracking_model=false, fams=(:tsoc, :tpot))
end
if get(ENV, "CASE", "ABC") in ("D", "ABC")
    comp_case("D (all terms; storage blocks, LINEAR)"; tracking_model=true, fams=(:sblkL2, :sblkL4))
end
if get(ENV, "CASE", "ABC") in ("E", "ABC")
    comp_case("E (no losses; storage blocks, T2)"; tracking_model=false, fams=(:sblk2, :sblk4))
end
if get(ENV, "CASE", "ABC") in ("F",)
    comp_case("F (no losses; dense angle directions, T2 and none)"; tracking_model=false, fams=(:tdir, :tdir0))
end
if get(ENV, "CASE", "ABC") in ("G",)
    comp_case("G (all terms; dense angle directions, LINEAR and none)"; tracking_model=true, fams=(:tdirL, :tdir0))
end
if get(ENV, "CASE", "ABC") in ("H",)
    comp_case("H (no losses; sparse angle directions, T2 and none)"; tracking_model=false, fams=(:tcut, :tcutw, :xloc, :xlocc, :xpair))
end
if get(ENV, "CASE", "ABC") in ("I",)
    comp_case("I (all terms; sparse angle directions, LINEAR and none)"; tracking_model=true, fams=(:tcutL, :tcutLw, :xloc, :xpair))
end
if get(ENV, "CASE", "ABC") in ("C", "ABC")
    # cell templates: every partition level, one round per level (cell_power = 1) and time class
    comp_case("C (all terms; cell templates)"; tracking_model=true, fams=:cells, cells=true)
end
