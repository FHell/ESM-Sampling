# kscale and kpeak at the capacity box (src/local.jl, src/adaptive.jl,
# docs/system/moves/capacity.md): the moves scale the utilizations against the clamped
# capacity, so a raw capacity above its box can return to it. Checks, at α = 0 and α = 1,
# on a state with raw capacities above their box: the line density against the global log
# target per move (moves inside the box, above it and across its edge, for dispatchable and
# storage units), the curve (the reverse move returns to the start with the opposite
# density), the local and bias caches after the moves and after a sweep, and that one
# kscale and one kpeak sweep bring the raw capacities above the box toward it.
# Run after test_kop.jl (reuses its problem P1, LM and start sL) or on its own.
using Test, Random, LinearAlgebra, Statistics
using ESMSampling
const MQ = ESMSampling
include(joinpath(@__DIR__, "..", "device", "test", "mockrng.jl"))

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    if !@isdefined(P1)
        cfgq = MQ.load_config(joinpath(@__DIR__, "..", "configs", "haar_poc.toml")); cfgq.verbose = false
        cfgq.use_bias = true; cfgq.bias_warmstart = 20.0; cfgq.bias_monotone = true; cfgq.measure = "physical"
        P1, sL, _ = MQ.setup(cfgq; rng=MersenneTwister(cfgq.seed))
        MQ.lift_bias!(P1, sL.ℓ; rng=MersenneTwister(5), nsteps=2048)
        P1.bias.θ .*= 3.0
        MQ.rebuild_tables!(P1.bias)
        LM = MQ.LocalModel(P1, sL.ℓ)
    end
    LMq = MQ.LocalModel(P1, sL.ℓ)

    "Set the raw node capacity of unit i of k field kf to knew (coefficients through the field basis)."
    function set_raw!(P, s, kf, i, knew)
        lk = P.lk[s.ℓ+1]
        Ut = P.E.kspecs[kf].B.Ut
        Δ = knew - MQ.knodes(P, s.ℓ, s.k)[kf][i]
        for q in MQ.nzrange(Ut, i)
            s.k[lk.off[kf]+MQ.rowvals(Ut)[q]] += Δ * MQ.nonzeros(Ut)[q]
        end
        return s
    end
    "Total raw capacity above the box of k field kf."
    excess(P, s, kf) = (sp = P.E.kspecs[kf]; sum(max.(MQ.knodes(P, s.ℓ, s.k)[kf] .- sp.hi, 0.0)))

    function cache_checks(P, LMq, LC, BC, s1)
        LC2 = MQ.LocalCost(P, LMq, s1)
        @test isapprox(MQ.cache_cost(P, LMq, LC), s1.parts.op; rtol=1e-9)
        @test maximum(abs.(LC.S .- LC2.S)) < 1e-6 && maximum(abs.(LC.r .- LC2.r)) < 1e-6
        @test maximum(maximum(abs.(LC.Y[f] .- LC2.Y[f]); init=0.0) for f in eachindex(LC.Y)) < 1e-9
        @test maximum(maximum(abs.(LC.kgen[g] .- LC2.kgen[g]); init=0.0) for g in eachindex(LC.kgen)) < 1e-6
        @test maximum(maximum(abs.(LC.kst[g] .- LC2.kst[g]); init=0.0) for g in eachindex(LC.kst)) < 1e-6
        if P.use_bias
            BC2 = MQ.BiasCache(P, s1)
            @test abs(MQ.bias_value(BC, P.bias) - MQ.bias_term(P, s1)) <= 1e-9 * (1 + abs(MQ.bias_term(P, s1)))
            @test maximum(abs.(BC.U1x .- BC2.U1x)) < 1e-12 && maximum(abs.(BC.zc .- BC2.zc)) < 1e-6
            @test abs(BC.firm - BC2.firm) < 1e-6
        end
    end

    @testset "kscale and kpeak at the capacity box (α = $(Int(P.use_bias)))" for P in (MQ.with_bias(P1, false), P1)
        E = P.E
        T = LMq.T
        # start: the dispatchable and storage capacities of every second node 1.6 × above their box
        s = MQ.evaluate!(P, deepcopy(sL))
        dunits = [(1, gi, i) for (gi, g) in enumerate(E.gens) if g.kf > 0 && g.vf > 0 for i in eachindex(g.bus)
                  if isfinite(E.kspecs[g.kf].hi[i]) && E.kspecs[g.kf].hi[i] > 100]
        sunits = [(2, si, i) for (si, st) in enumerate(E.stors) if st.kf > 0 for i in eachindex(st.bus)
                  if isfinite(E.kspecs[st.kf].hi[i]) && E.kspecs[st.kf].hi[i] > 100]
        @test length(dunits) >= 4 && length(sunits) >= 2
        kfof(kind, gi) = kind == 1 ? E.gens[gi].kf : E.stors[gi].kf
        above = vcat(dunits[1:2:end], sunits[1:2:end])
        for (kind, gi, i) in above
            kf = kfof(kind, gi)
            set_raw!(P, s, kf, i, 1.6 * E.kspecs[kf].hi[i])
        end
        MQ.evaluate!(P, s)
        kfs = unique(kfof(u[1], u[2]) for u in above)
        ex0 = sum(excess(P, s, kf) for kf in kfs)
        @test ex0 > 0
        # single moves: (unit, u) with u inside the box, above it and across its edge
        cases = Tuple{NTuple{3,Int},Float64}[]
        for (kind, gi, i) in vcat(dunits[1:4], sunits[1:2])
            kf = kfof(kind, gi)
            k0, hi = MQ.knodes(P, s.ℓ, s.k)[kf][i], E.kspecs[kf].hi[i]
            if k0 > hi
                append!(cases, [((kind, gi, i), -0.2), ((kind, gi, i), log(0.8hi / k0)), ((kind, gi, i), 0.1)])
            else
                append!(cases, [((kind, gi, i), 0.3), ((kind, gi, i), -0.4)])
                k0 * exp(0.3) < hi && push!(cases, ((kind, gi, i), log(1.3hi / k0)))
            end
        end
        nk, np = 0, 0
        for ((kind, gi, i), u) in cases, move in (:kscale, :kpeak)
            (move == :kpeak && kind == 2) && continue
            s1 = deepcopy(s)
            LC = MQ.LocalCost(P, LMq, s1); BC = P.use_bias ? MQ.BiasCache(P, s1) : nothing
            sel(kd, g_, j) = (kd, g_, j) == (kind, gi, i)
            kf = kfof(kind, gi)
            k0 = MQ.knodes(P, s.ℓ, s.k)[kf][i]
            nu = 2000                                       # more than the units of any sweep
            rec = []
            # Metropolis form with one proposal: z = u/σ (kscale) or δ (kpeak, σk = 1), uniform 0 (accepted)
            domove(s_, z, rec_) = move == :kscale ?
                MQ.kscale_sweep!(MockRNG(fill(z / 0.3, nu), fill(0.0, nu)), P, LMq, s_; LC=LC, BC=BC, mh=true,
                                 σ=0.3, order=:fixed, select=sel, rec=rec_)[1] :
                MQ.kpeak_sweep!(MockRNG(fill(z, nu), fill(0.0, nu)), P, LMq, s_; LC=LC, BC=BC, mh=true,
                                σk=ones(nu), order=:fixed, select=(_, g_, j) -> sel(1, g_, j), rec=rec_)[1]
            L0 = s1.L
            s1 = domove(s1, move == :kscale ? u : k0 * expm1(u), rec)
            @test length(rec) == 1
            r1 = rec[1]
            # incremental against global: the log target changes by the line density without its Jacobian
            @test isapprox(s1.L - L0, r1.lf - r1.logjac; atol=1e-12 * abs(L0) + 1e-8 * (1 + abs(r1.lf)))
            cache_checks(P, LMq, LC, BC, s1)
            # the curve: the reverse move from the moved state returns to the start, with the opposite density
            rec2 = []
            k1 = MQ.knodes(P, s1.ℓ, s1.k)[kf][i]
            s1 = domove(s1, move == :kscale ? -u : k0 - k1, rec2)
            @test isapprox(rec2[1].lf, -r1.lf; atol=1e-8 * (1 + abs(r1.lf)))
            @test isapprox(rec2[1].logjac, -r1.logjac; atol=1e-9 * (1 + abs(r1.logjac)))
            @test maximum(abs.(s1.k .- s.k)) < 1e-6 && maximum(abs.(s1.v .- s.v)) < 1e-9
            move == :kscale ? (nk += 1) : (np += 1)
        end
        @test nk >= 12 && np >= 8
        # a slice sweep of each keeps the caches consistent and brings the raw capacities toward the box
        s1 = deepcopy(s)
        LC = MQ.LocalCost(P, LMq, s1); BC = P.use_bias ? MQ.BiasCache(P, s1) : nothing
        s1, n1 = MQ.kscale_sweep!(MersenneTwister(7), P, LMq, s1; LC=LC, BC=BC)
        ex1 = sum(excess(P, s1, kf) for kf in kfs)
        s1, n2 = MQ.kpeak_sweep!(MersenneTwister(8), P, LMq, s1; LC=LC, BC=BC)
        ex2 = sum(excess(P, s1, kf) for kf in kfs)
        @test n1 > 0 && n2 > 0
        @test MQ.evaluate!(P, deepcopy(s1)).L ≈ s1.L
        cache_checks(P, LMq, LC, BC, s1)
        println("  excess above the box: start $(round(ex0 / 1e3; digits=1)) GW, after kscale $(round(ex1 / 1e3; digits=1)), after kpeak $(round(ex2 / 1e3; digits=1))")
        @test ex1 < 0.8 * ex0 && ex2 < 0.5 * ex0
    end
end
