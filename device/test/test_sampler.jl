# Phase-2 acceptance (docs/reactant_design.md §8, adapted), part 1: the term batch and the dense
# line sampler on the 40-node model with all model terms on (as test_cost.jl).
#   (i)   ΔC(t) of the device term batch = CPU eval_move! (BlockMove over τ−1, τ) per family
#   (ii)  [only with KS=1] KS of device line draws against quadrature and against the CPU sample_line! draws
#         (stiff c_shed case, SoC two-step case, and one site of every other family), on Julia
#         arrays with given uniforms and traced by Reactant with the device RNG
#   (iv)  check_conflicts on every round of the schedule (and a negative control)
#   roots every secondary quantity has ≤ 2 sign changes on every site (exactness of the dense form)
# Run: cd device && nice -n 19 julia --project=. -t 4 test/test_sampler.jl   (≈ 10 min)
include("setup40.jl")
include("helpers.jl")
using Test, Statistics

const NKS = parse(Int, get(ENV, "NKS", "40000"))
# statistical tests (KS) are off by default (test strategy: deterministic comparisons with mock
# uniforms, test_mock.jl); KS=1 runs them
const RUN_KS = get(ENV, "KS", "0") == "1"
const DEV_KS = get(ENV, "DEVICE_REACTANT", "1") == "1"

P, M, ix, S, ss = setup40()
pl = D.plan(M, S, P.βv; bmax=Dict(:theta => 4096))
E = P.E
LM = ML.LocalModel(P, ss[1].ℓ)
LCs = [ML.LocalCost(P, LM, s) for s in ss]
stc = cpu_r_state(pl, LCs)                 # loop state with the CPU residual (ΔC comparisons at round-off)
C = length(ss)
β = P.βv
SC = Dict(:gen => 0.3, :uc => 0.3, :pair => 300.0, :soc => 0.2, :link => 0.2, :theta => 300.0)

@testset "(i) ΔC(t) = CPU eval_move!" begin
    rng = MersenneTwister(5)
    for fam in pl.fams
        worst_rel = 0.0
        worst_abs = 0.0
        nbad = 0
        for rep in 1:150
            site = rand(rng, rand(rng, pl.sched[fam]))
            tb, _ = dev_tb(pl, fam, [site], stc)
            t = SC[fam] .* randn(rng, C, 1, 1)
            dv = eval_terms(tb, pl.k, t)[1] ./ β
            sc = balscale(pl, tb)
            for c in 1:C
                bm = cpu_move(P, LM, LCs[c], ix, fam, site)
                dc = bm === nothing ? 0.0 : ML.mv_eval!(bm, t[c])
                bm === nothing && (dv[c] = 0.0)          # κ_min mask of the pair family (t = 0 on the device)
                err = abs(dv[c] - dc)
                worst_abs = max(worst_abs, err)
                abs(dc) > 1e-6 && (worst_rel = max(worst_rel, err / abs(dc)))
                ok = err <= 1e-9 * abs(dc) + sc[c]
                ok || (nbad += 1; nbad <= 3 && println("  $fam site $site chain $c t $(t[c]): device $(dv[c]) CPU $dc"))
            end
        end
        @printf("  %-6s max |ΔC_dev − ΔC_cpu| = %.2e €, max relative %.1e where |ΔC| > 1e-6 € (450 draws; bound 1e-9 |ΔC| + 1e-12 Δt c_shed max|r0|)\n",
                fam, worst_abs, worst_rel)
        @test nbad == 0
    end
end

@testset "(iv) check_conflicts on every round" begin
    for fam in pl.fams
        ok, n = D.check_conflicts(fam, pl.host[fam])
        @printf("  %-6s %3d rounds × %6d sites: conflicts %d\n", fam, size(pl.host[fam].on, 2), size(pl.host[fam].on, 1), n)
        @test ok
    end
    # negative controls: two neighbouring θ sites, two units of one node, two SoC steps τ, τ+1
    n1 = 1; n2 = pl.tp.nbrs[1][1]
    @test !D.check_conflicts(:theta, D.host_table(:theta, pl.tp, [[(n1, 5), (n2, 5)]]))[1]
    g = first(n for n in 1:M.N if length(pl.tp.nodegens[n]) >= 2)
    @test !D.check_conflicts(:gen, D.host_table(:gen, pl.tp, [[(pl.tp.nodegens[g][1], 5), (pl.tp.nodegens[g][2], 5)]]))[1]
    @test !D.check_conflicts(:soc, D.host_table(:soc, pl.tp, [[(1, 5), (1, 6)]]))[1]
    @test D.check_conflicts(:soc, D.host_table(:soc, pl.tp, [[(1, 5), (1, 7)]]))[1]
end

@testset "dense form: ≤ 2 sign changes per secondary quantity" begin
    for fam in pl.fams
        m = 0.0
        for r in 1:size(pl.dev[fam].on, 2)
            tb, _ = D.build(Val(fam), pl.st, pl.cp, pl.sp, pl.dev[fam], r, pl.k)
            m = max(m, maximum(D.root_count(tb, pl.k)))
        end
        @printf("  %-6s max sign changes of one secondary quantity over all sites: %d\n", fam, m)
        @test m <= 2
    end
end

# ---------------------------------------------------------------------------
# (ii) line draws
# ---------------------------------------------------------------------------

"Stiff case: a generator site whose node sheds (r > 0) with the shed kink inside the unit's box."
function stiff_site()
    T = M.T
    for τ in 1:T, n in 1:M.N, q in pl.tp.nodegens[n]
        c = 1
        r0 = LCs[c].r[n, τ]
        y0 = pl.st.ω[c, τ+(q-1)*T]
        κ = pl.cp.κg[c, q]
        (r0 > 0 && 0.05 < y0 < 0.95 && κ * (1 - y0) > 1.5r0 && r0 > 1.0) && return (q, τ)
    end
    error("no stiff site")
end
"SoC two-step case: interior state of charge, unit with charging and discharging both possible."
function soc_site()
    T = M.T
    for τ in 2:T-1, s in eachindex(pl.tp.snode)
        e0 = pl.st.e[1, τ+(s-1)*T]
        (0.2 < e0 < 0.8 && pl.sp.sηin[s] > 0 && pl.cp.ks[1, s] > 100) && return (s, τ)
    end
    error("no SoC site")
end

"CDF of ∝ exp(−βΔC) on a grid (trapezoid, CPU evaluation) and the KS distance of samples to it."
function ks_quad(bm, ts)
    lo, hi = extrema(ts)
    w = hi - lo
    grid = collect(range(lo - 0.3w, hi + 0.3w; length=200_001))
    pot = [β * ML.mv_eval!(bm, x) for x in grid]
    m = minimum(pot)
    dens = exp.(-(pot .- m))
    cdf = cumsum(vcat(0.0, 0.5 .* (dens[1:end-1] .+ dens[2:end]) .* diff(grid)))
    cdf ./= cdf[end]
    st = sort(ts)
    n = length(st)
    return maximum(abs(searchsortedlast(st, x) / n - cdf[i]) for (i, x) in enumerate(grid))
end
"Two-sample KS distance."
function ks2(a, b)
    a, b = sort(a), sort(b)
    xs = sort(vcat(a, b))
    return maximum(abs(searchsortedlast(a, x) / length(a) - searchsortedlast(b, x) / length(b)) for x in xs)
end

"Device draws for B copies of `site` (Julia arrays, given uniforms): (C, B)."
function dev_draws(fam, site, B; rng=MersenneTwister(17))
    tb, _ = dev_tb(pl, fam, fill(site, B))
    u = rand(rng, C, B, 2)
    return D.sample_dense(tb, pl.k, u[:, :, 1:1], u[:, :, 2:2])[:, :, 1]
end

"Device draws traced by Reactant with the device RNG (on the GPU)."
function dev_draws_reactant(fam, site, B)
    ht = D.host_table(fam, pl.tp, [fill(site, B)]); dt = D.device_table(fam, ht)
    function draw(st, cp, sp, k, tab, rng)
        tb, _ = D.build(Val(fam), st, cp, sp, tab, 1, k)
        u = rand(rng, Float64, (C, B, 2))
        return D.sample_dense(tb, k, u[:, :, 1:1], u[:, :, 2:2])
    end
    args = (Reactant.to_rarray(pl.st), Reactant.to_rarray(pl.cp), Reactant.to_rarray(pl.sp), Reactant.to_rarray(pl.k),
            Reactant.to_rarray(dt), Reactant.ReactantRNG(Reactant.to_rarray(UInt64[23, 5])))
    f = @compile draw(args...)
    return Array(f(args...))[:, :, 1]
end

"CPU draws with sample_line! (n per chain)."
cpu_draws(bm, n; rng=MersenneTwister(29)) = [ML.sample_line!(rng, bm, β, Float64[]) for _ in 1:n]

RUN_KS && @testset "(ii) KS of the line draws" begin
    cases = Any[(:gen, stiff_site(), "stiff (c_shed kink)"), (:soc, soc_site(), "SoC two-step")]
    for fam in (:uc, :pair, :link, :theta)
        rr = pl.sched[fam][1]
        push!(cases, (fam, rr[length(rr)÷2+1], "site of the family"))
    end
    crit = 1.95 / sqrt(NKS)                       # one-sample KS, α ≈ 0.001
    crit2 = 1.95 * sqrt(2 / NKS)
    for (fam, site, what) in cases
        tdev = dev_draws(fam, site, NKS)
        tre = DEV_KS && fam in (:gen, :soc) ? dev_draws_reactant(fam, site, NKS) : nothing
        for c in 1:C
            bm = cpu_move(P, LM, LCs[c], ix, fam, site)
            bm === nothing && (@test all(==(0.0), tdev[c, :]); continue)
            tc = cpu_draws(bm, NKS)
            ksq = ks_quad(bm, tdev[c, :])
            ksc = ks2(tdev[c, :], tc)
            ksr = tre === nothing ? NaN : ks_quad(bm, tre[c, :])
            @printf("  %-6s %-20s site %-14s chain %d: KS(dev, quad) %.4f  KS(dev, CPU) %.4f  KS(reactant RNG, quad) %.4f  (crit %.4f / %.4f)\n",
                    fam, what, string(site), c, ksq, ksc, ksr, crit, crit2)
            @test ksq < crit
            @test ksc < crit2
            isnan(ksr) || @test ksr < crit
        end
    end
end
