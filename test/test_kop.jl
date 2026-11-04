# Capacity moves with their operation (src/kop.jl, docs/system/moves/capacity.md): the line
# density against the global log target at α = 0 and α = 1, the bias change against the
# difference of the bias, the local and bias caches after commits and after a sweep, and the
# curve: the reverse move returns to the start, the line densities of the two directions are
# antisymmetric (the Jacobians are inverse), and the pro-rata coordinates (σ, e) of the
# compensating generators are unchanged.
using Test, Random, LinearAlgebra, Statistics
using ESMSampling
const MK = ESMSampling

"Pro-rata coordinates (shares, excess) of the dispatchable generators at node n and step τ."
function prorata_coords(P, LM, LC, n, τ; κmin=1.0)
    ps, Qs = Float64[], 0.0
    for (f, i) in LM.genunits[n]
        P.E.gens[LM.role[f][2]].wf > 0 && continue
        κ = MK.κunit(P, LM, LC, f, i, τ)
        abs(κ) >= κmin || continue
        push!(ps, κ * LC.Y[f][i, τ]); Qs += κ * clamp(LC.Y[f][i, τ], 0.0, 1.0)
    end
    (isempty(ps) || abs(sum(ps)) < 1e-6) && return Float64[], NaN
    return ps ./ sum(ps), sum(ps) - max(LC.r[n, τ] + Qs, 0.0)
end

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    cfg = MK.load_config(joinpath(@__DIR__, "..", "configs", "haar_poc.toml")); cfg.verbose = false
    cfg.use_bias = true; cfg.bias_warmstart = 20.0; cfg.bias_monotone = true; cfg.measure = "physical"
    P1, sL, info = MK.setup(cfg; rng=MersenneTwister(cfg.seed))
    L = sL.ℓ
    MK.lift_bias!(P1, L; rng=MersenneTwister(5), nsteps=2048)
    P1.bias.θ .*= 3.0                                 # a bias of realistic size (as in test_bias.jl)
    MK.rebuild_tables!(P1.bias)
    LM = MK.LocalModel(P1, L)

    @testset "kop: capacity moves with their operation (α = $(Int(P.use_bias)))" for P in (MK.with_bias(P1, false), P1)
        rng = MersenneTwister(41)
        s = MK.evaluate!(P, copy(sL))
        groups = MK.kop_units(P, LM)
        @test length(groups) == count(g -> g.kf > 0 && g.vf == 0, P.E.gens) + count(st -> st.kf > 0, P.E.stors)
        ncheck, nprorata = 0, 0
        for g in groups, grp in ([g[rand(rng, 1:length(g))]], g), u in (0.3, -0.5, 1.2)
            s1 = deepcopy(s)
            LC = MK.LocalCost(P, LM, s1); BC = MK.BiasCache(P, s1)
            mv = MK.kop_prepare(P, LM, LC, s1, grp)
            (mv === nothing || u < mv.lo) && continue
            lf, lj, Δb = MK.kop_logdensity(P, LM, LC, mv, u; BC=BC)
            L0, b0 = s1.L, MK.bias_term(P, s1)
            # pro-rata coordinates of the first unit's node before the move
            n = mv.terms[1].n
            pc0 = [prorata_coords(P, LM, LC, n, τ) for τ in 1:LM.T]
            MK.kop_commit!(P, LM, LC, s1, mv, u; BC=BC)
            MK.evaluate!(P, s1)
            N = length(mv.terms)
            # incremental against global: the log target changes by the line density without the Jacobians
            @test isapprox(s1.L - L0, lf - lj - N * u; atol=1e-12 * abs(L0) + 1e-8 * (1 + abs(lf)))
            # the bias change against the difference of the bias (0 at α = 0)
            @test isapprox(Δb, MK.bias_term(P, s1) - b0; atol=1e-9 * (1 + abs(b0)))
            # caches equal a rebuild
            LC2 = MK.LocalCost(P, LM, s1)
            @test maximum(abs.(LC.S .- LC2.S)) < 1e-6 && maximum(abs.(LC.r .- LC2.r)) < 1e-6
            @test maximum(maximum(abs.(LC.Y[f] .- LC2.Y[f]); init=0.0) for f in eachindex(LC.Y)) < 1e-9
            if P.use_bias
                BC2 = MK.BiasCache(P, s1)
                @test abs(MK.bias_value(BC, P.bias) - MK.bias_term(P, s1)) <= 1e-9 * (1 + abs(b0))
                @test maximum(abs.(BC.U1x .- BC2.U1x)) < 1e-12 && maximum(abs.(BC.U1s .- BC2.U1s)) < 1e-12
                @test maximum(abs.(BC.zc .- BC2.zc)) < 1e-6 && abs(BC.firm - BC2.firm) < 1e-6
                @test maximum(maximum(abs.(BC.Kw[f] .- BC2.Kw[f]); init=0.0) for f in eachindex(BC.Kw)) < 1e-6
            end
            # the pro-rata coordinates (σ, e) are constant along the curve, also across its kinks
            eσ, ee = 0.0, 0.0
            for τ in 1:LM.T
                σ0, e0 = pc0[τ]
                isempty(σ0) && continue
                σ1, e1 = prorata_coords(P, LM, LC, n, τ)
                eσ = max(eσ, maximum(abs.(σ1 .- σ0)))
                ee = max(ee, abs(e1 - e0) / max(1.0, abs(e0)))
                nprorata += 1
            end
            @test eσ < 1e-9 && ee < 1e-6
            # the curve from the moved state: the move with −u returns to the start, with the opposite density
            mv2 = MK.kop_prepare(P, LM, LC, s1, grp)
            lf2, _, _ = MK.kop_logdensity(P, LM, LC, mv2, -u; BC=BC)
            @test isapprox(lf2, -lf; atol=1e-8 * (1 + abs(lf)))
            MK.kop_commit!(P, LM, LC, s1, mv2, -u; BC=BC)
            @test maximum(abs.(s1.k .- s.k)) < 1e-6 && maximum(abs.(s1.v .- s.v)) < 1e-9
            ncheck += 1
        end
        @test ncheck >= 25 && nprorata > 0
        # a sweep keeps both caches consistent with a fresh evaluation
        s1 = deepcopy(s)
        LC = MK.LocalCost(P, LM, s1); BC = MK.BiasCache(P, s1)
        s1, nm = MK.kop_sweep!(MersenneTwister(3), P, LM, s1; LC=LC, BC=BC)
        @test nm > 0
        @test MK.evaluate!(P, deepcopy(s1)).L ≈ s1.L
        @test isapprox(MK.cache_cost(P, LM, LC), s1.parts.op; rtol=1e-9)
        LC2 = MK.LocalCost(P, LM, s1)
        @test maximum(abs.(LC.S .- LC2.S)) < 1e-6
        if P.use_bias
            @test abs(MK.bias_value(BC, P.bias) - MK.bias_term(P, s1)) <= 1e-9 * (1 + abs(MK.bias_term(P, s1)))
            BC2 = MK.BiasCache(P, s1)
            @test maximum(abs.(BC.U1x .- BC2.U1x)) < 1e-12 && maximum(abs.(BC.zc .- BC2.zc)) < 1e-6
        end
    end
end
