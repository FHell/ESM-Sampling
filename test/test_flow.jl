# Tests of the flow-based hierarchy (docs/flow_hierarchy.md): batched congestion
# projection, transport modes and angle basis, storage shapes, active-set frames,
# and the transport and storage moves.
using Test, Random, LinearAlgebra, Statistics, SparseArrays, JuMP, HiGHS
using ESMSampling
const MF = ESMSampling

const FLOWCFG = joinpath(@__DIR__, "..", "configs", "flow.toml")
if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    sys = MF.load_system(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))

    @testset "batched congestion projection (ADMM) vs QP" begin
        khat = MF.proxy_capacities(sys)
        p, _, _ = MF.proxy_injections(sys, khat; h=24, years=1:1)
        Pm = p[:, 1:60, 1]
        Q = MF.CongestionADMM(sys)
        θ, load, mask = MF.project_congestion(Q, Pm)
        @test maximum(abs.(load)) <= 1 + 1e-2
        agree = Float64[]
        for τ in 1:6:60
            m = Model(HiGHS.Optimizer); set_silent(m)
            @variable(m, th[1:sys.N]); nk = length(sys.link_cap)
            @variable(m, -sys.link_cap[l] <= u[l in 1:nk] <= sys.link_cap[l])
            out = [AffExpr(0.0) for _ in 1:sys.N]
            fl = Vector{AffExpr}(undef, length(sys.line_s))
            for e in eachindex(sys.line_s)
                a, b = sys.line_bus0[e], sys.line_bus1[e]
                fl[e] = @expression(m, sys.line_b[e] * (th[a] - th[b]))
                @constraint(m, -sys.line_s[e] <= fl[e] <= sys.line_s[e])
                add_to_expression!(out[a], fl[e]); add_to_expression!(out[b], -1.0, fl[e])
            end
            for l in 1:nk
                add_to_expression!(out[sys.link_bus0[l]], u[l]); add_to_expression!(out[sys.link_bus1[l]], -1.0, u[l])
            end
            @objective(m, Min, sum((out[n] - Pm[n, τ])^2 for n in 1:sys.N))
            optimize!(m)
            bq = [abs(value(fl[e])) >= sys.line_s[e] * (1 - 1e-3) for e in eachindex(sys.line_s)]
            push!(agree, mean(bq .== mask[:, τ]))
        end
        @test mean(agree) > 0.97
    end

    @testset "flow design on the PyPSA-Eur model" begin
        cfg = MF.load_config(FLOWCFG); cfg.verbose = false
        P, sL, info = MF.setup(cfg; rng=MersenneTwister(cfg.seed))
        E = P.E
        FB = E.flow
        N = E.sys.N
        Zc = MF.component_indicators(N, E.sys.line_bus0, E.sys.line_bus1)
        @test FB.K >= 1
        @test FB.modes' * FB.modes ≈ I
        @test norm(Zc' * FB.modes) < 1e-8
        Bθ = E.vspecs[E.fθ].B
        @test Matrix(Bθ.U' * Bθ.U) ≈ I
        @test Bθ.rank_at[1] == FB.K && Bθ.rank_at[end] == size(Bθ.U, 2)
        @test norm(Zc' * Matrix(Bθ.U)) < 1e-8
        Φ = MF.storage_shapes(FB, E.sys, 24, 8, FB.cell[1])
        @test size(Φ, 1) == 9 && all(Φ[1, :] .== 0) && all(Φ[end, :] .== 0)
        @test all(maximum(abs.(Φ); dims=1) .≈ 1)

        rng = MersenneTwister(21)
        ℓ = 1
        s = MF.restrict_state(P, sL, ℓ)
        LM = MF.LocalModel(P, ℓ)
        FF = MF.FlowFrames(P, ℓ, s)
        # projected modes do not change the loading of predicted binding lines
        W = zeros(length(E.sys.line_s), N)
        for e in eachindex(E.sys.line_s)
            w = E.sys.line_b[e] / E.sys.line_s[e]
            W[e, E.sys.line_bus0[e]] += w; W[e, E.sys.line_bus1[e]] -= w
        end
        τb = findfirst(!isempty, FF.binding)
        if τb !== nothing
            # relative to the unprojected modes (the projection carries a tiny ridge for dependent rows)
            @test maximum(abs.(W[FF.binding[τb], :] * FF.D[:, :, τb])) < 1e-4 * maximum(abs.(W[FF.binding[τb], :] * FB.modes))
        end
        LC = MF.LocalCost(P, LM, s)
        ws = MF.MoveWS(P); jac = NTuple{5,Float64}[]
        for trial in 1:150
            τ = rand(rng, 1:LM.T)
            MF.clear!(ws)
            ok = isodd(trial) ? MF.prep_tmode!(rng, ws, P, LM, LC, FF, τ, jac) : MF.prep_tline!(rng, ws, P, LM, LC, FF, τ, jac)
            ok || continue
            MF.finalize!(ws, P, LM, LC, τ)
            t = 1e-5 * randn(rng)
            Δ = MF.eval_move!(ws, P, LM, LC, τ, t)
            op0 = s.parts.op
            MF.commit!(ws, P, LM, LC, s.v, τ, t); MF.evaluate!(P, s)
            @test isapprox(s.parts.op - op0, Δ; atol=1e-12 * abs(op0) + 1e-9 * abs(Δ))
        end
        # exact T2 compensation: (σ, e = P − Q₊) invariant along the line, also across the kink Q = 0
        LC = MF.LocalCost(P, LM, s)
        τk = something(findfirst(!isempty, FF.binding), 1)
        jac = NTuple{5,Float64}[]
        MF.clear!(ws)
        if MF.prep_tline!(rng, ws, P, LM, LC, FF, τk, jac) && !isempty(jac)
            MF.finalize!(ws, P, LM, LC, τk)
            node = E.sys.line_bus0[FF.binding[τk][1]]
            units = [(f, i) for (f, i) in LM.units[node] if LM.role[f][1] == MF.ROLE_GEN && LM.complete[f]]
            κ = [MF.κunit(P, LM, LC, f, i, τk) for (f, i) in units]
            inv(L_) = (ps = κ .* [L_.Y[f][i, τk] for (f, i) in units];
                       Qn = L_.r[node, τk] + sum(κ .* clamp.([L_.Y[f][i, τk] for (f, i) in units], 0, 1));
                       (ps ./ sum(ps), sum(ps) - max(Qn, 0), Qn))
            σ0, e0, Q0 = inv(LC)
            for t in (-2abs(Q0) - 10, 3.0, 2abs(Q0) + 10)
                sx = deepcopy(s); LCx = deepcopy(LC)
                MF.commit!(ws, P, LM, LCx, sx.v, τk, t)
                σ1, e1, Q1 = inv(LCx)
                @test maximum(abs.(σ1 .- σ0)) < 1e-9
                @test abs(e1 - e0) < 1e-6 * max(1, abs(e0))
            end
        end
        LC = MF.LocalCost(P, LM, s)
        s, st = MF.storage_ms_sweep!(rng, P, LM, s; LC=LC, lengths=[2, 8, 32])
        @test sum(st[:, 1]) > 0
        @test isapprox(MF.cache_cost(P, LM, LC), s.parts.op; rtol=1e-10)
        LC2 = MF.LocalCost(P, LM, s)
        @test maximum(abs.(LC.S .- LC2.S)) < 1e-6
        s, st1, st2 = MF.flow_sweep!(rng, P, LM, FF, s)
        @test isfinite(s.L)
        # adaptive modes: geometric update schedule, learned moves keep the caches consistent
        AM = MF.AdaptiveModes(P, ℓ; K=4, lengths=[2, 4, 8])
        for it in 1:5
            s, _, st2 = MF.adaptive_sweep!(rng, P, LM, FF, AM, s)
        end
        @test AM.updates == [4]
        @test size(AM.Dθ, 2) == 4
        @test norm(Zc' * AM.Dθ) < 1e-8
        @test all(Φ -> all(Φ[1, :] .== 0) && all(Φ[end, :] .== 0), [AM.shapes[k] for k in keys(AM.shapes) if k[1] == :soc])
        LC = MF.LocalCost(P, LM, s)
        @test isapprox(MF.cache_cost(P, LM, LC), s.parts.op; rtol=1e-10)
        # transported peak-set capacity moves keep the caches consistent
        LC = MF.LocalCost(P, LM, s)
        s, nm = MF.kpeak_sweep!(rng, P, LM, s; LC=LC, AM=AM)
        @test nm > 0
        @test isapprox(MF.cache_cost(P, LM, LC), s.parts.op; rtol=1e-10)
        LC2 = MF.LocalCost(P, LM, s)
        @test maximum(abs.(LC.S .- LC2.S)) < 1e-6
        LC = LC2
        # learned linear compensation: coefficients fitted, moves consistent with the full cost
        @test any(n -> !isempty(AM.runits[n]) && AM.rcnt[n, 1] > 0, eachindex(AM.runits))
        ws = MF.MoveWS(P); jac = NTuple{5,Float64}[]
        for trial in 1:40
            τ = rand(rng, 1:LM.T); MF.clear!(ws)
            MF.prep_tmode!(rng, ws, P, LM, LC, FF, τ, jac; learn=(AM=AM, FF=FF)) || continue
            MF.finalize!(ws, P, LM, LC, τ)
            t = 1e-4 * randn(rng); Δ = MF.eval_move!(ws, P, LM, LC, τ, t); op0 = s.parts.op
            MF.commit!(ws, P, LM, LC, s.v, τ, t); MF.evaluate!(P, s)
            @test isapprox(s.parts.op - op0, Δ; atol=1e-12 * abs(op0) + 1e-9 * abs(Δ))
        end
    end
end
