# LP baselines (src/lp_baseline.jl; docs/system/validation.md §2.7): historical weather, the
# multi-year LP and the MGA sweep, on a 5-node sub-network at a 5-day step (73 steps).
using Test, Random, JuMP
using ESMSampling
const MLB = ESMSampling

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    @testset "LP baselines on historical years" begin
        sys = MLB.subsystem(MLB.load_system(joinpath(@__DIR__, "..", "data", "processed", "model.h5")), [6, 10, 13, 17, 36])
        E = MLB.build_energy_model(sys)
        levels = [(2, 120), (5, 120)]
        lk, lv = MLB.Layout(E.kspecs, levels[2]...), MLB.Layout(E.vspecs, levels[2]...)
        yr = 2013
        y = findfirst(==(yr), sys.years)

        # the historical year at the level is the weather model's training data there
        W = MLB.fit_weather_model(MersenneTwister(1), E, levels; K=2)
        X = MLB.historical_weather(E, lv, yr)
        Xw = MLB.weather_nodes(W, 1, W.data[2][y])
        @test all(isapprox(X[f], Xw[f]; rtol=1e-10, atol=1e-9 * maximum(abs, X[f])) for f in eachindex(X))
        @test_throws ErrorException MLB.historical_weather(E, lv, 1990)
        @test_throws ErrorException MLB.historical_weather(E, MLB.Layout(E.vspecs, levels[1]...), yr)

        # the LP's cost is the sampler's cost at its solution, up to the curtailment that the
        # sampler prices (dump at c_dump plus the renewable's marginal cost) and the LP does not
        lp = MLB.solve_lp(E, lk, lv, MLB.datamapped(E, X))
        r1, lpm1 = MLB.solve_lp_years(E, lv, [yr])
        bias = MLB.init_bias(E, W, MLB.Layout(E.kspecs, levels[1]...), 1e-3; rng=MersenneTwister(2), nsamples=4)
        P = MLB.build_problem(E, W, levels, 4e-8, 4e-5, bias; use_bias=false)
        zk = MLB.to_modes(lk, E.kspecs, [reshape(k, :, 1) for k in lp.k])
        zv = MLB.to_modes(lv, E.vspecs, lp.v)
        xs = [W.data[1][y], W.data[2][y]]
        s = MLB.State(1, ones(Int, W.lays[1].T), copy(xs[1]), [zeros(W.trs[1].n)], xs, zk, zv, -Inf,
                      MLB.CostParts(NaN, NaN, NaN, NaN, NaN, NaN, NaN))
        MLB.evaluate!(P, s)
        @test s.parts.capex ≈ lp.capex rtol = 1e-10
        @test s.parts.shed < 1e-6 * sum(X[1]) * lv.h
        @test s.parts.capex + s.parts.op ≈ lp.objective + r1.stats[1]["curtail_cost_eur"] rtol = 1e-5
        @test r1.stats[1]["curtail_cost_eur"] > 0

        # the multi-year LP with one year is the single-year LP; with the year twice, the same optimum
        # (the same LP; equal to round-off, HiGHS's parallel factorization need not be bitwise reproducible)
        @test r1.objective ≈ lp.objective rtol = 1e-9
        @test vcat(r1.k...) ≈ vcat(lp.k...) rtol = 1e-6
        @test r1.sols[1].dF_dX[1] ≈ lp.dF_dX[1] rtol = 1e-6
        @test r1.capex + r1.opex[1] ≈ r1.objective rtol = 1e-12
        r2, _ = MLB.solve_lp_years(E, lv, [yr, yr])
        @test r2.objective ≈ lp.objective rtol = 1e-6
        @test r2.opex[1] ≈ r2.opex[2] rtol = 1e-5
        @test r2.sols[1].dF_dX[1] ≈ lp.dF_dX[1] rtol = 1e-3       # duals per unit weight: those of the year
        @test all(sum(r2.k[f]) ≈ sum(lp.k[f]) for f in eachindex(lp.k) if sum(lp.k[f]) > 1e-3 * sum(sum, lp.k) / length(lp.k))
        r3, _ = MLB.solve_lp_years(E, lv, [2010, yr])
        @test r3.objective ≈ r3.capex + 0.5 * (r3.opex[1] + r3.opex[2]) rtol = 1e-9
        @test r3.k != lp.k

        # MGA: every extreme satisfies the cost bound, also when its dispatch is re-optimized
        # at its capacities; min ≤ optimum ≤ max per carrier
        ε = 0.02
        res = Dict{Tuple{Int,Symbol},Any}()
        fields = [1, 4]                                  # onwind, gas
        MLB.mga_sweep!(lpm1, E, r1.objective, [ε], fields; on_result=(e, f, sense, r) -> (res[(f, sense)] = r))
        @test length(res) == 4
        C = r1.objective
        Wd = MLB.datamapped(E, X)
        for f in fields
            mn, mx = res[(f, :min)], res[(f, :max)]
            @test !startswith(mn.status, "OTHER") && !startswith(mx.status, "OTHER")
            @test mn.cost <= (1 + ε) * C * (1 + 1e-6) && mx.cost <= (1 + ε) * C * (1 + 1e-6)
            opt = sum(r1.k[f])
            @test mn.total <= opt * (1 + 1e-6) + 1e-6 && opt <= mx.total * (1 + 1e-6) + 1e-6
            @test mx.total > mn.total * 1.01
            for r in (mn, mx)
                d = MLB.solve_lp(E, lk, lv, Wd; kfixed=r.k)      # dispatch LP at the extreme's capacities
                @test d.objective <= (1 + ε) * C * (1 + 1e-6)
            end
        end
        # the model is restored: the optimum again
        optimize!(lpm1.model)
        @test objective_value(lpm1.model) ≈ C rtol = 1e-6
    end
end
