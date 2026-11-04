# Joint low-frequency block moves of the level-0 weather (src/weather_blocks.jl,
# docs/system/moves/weather.md): the directions, the VAR quadratic against the log prior of
# the weather model, the line density against the global log target at α = 0 and α = 1, the
# bias change against the difference of the bias, the local and bias caches and the anomalies
# against rebuilds, the reverse move with the opposite line density, the sweeps, and the
# chain of repeated moves of one block against quadrature of its line density with the
# Metropolis bias correction.
# The density, reverse-move and cache checks of single block moves are also made by
# test_moves.jl (wblock, B ≤ 4); this file keeps the directions, the VAR quadratic and the KS test.
using Test, Random, LinearAlgebra, Statistics
using ESMSampling
const MW = ESMSampling

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    cfg = MW.load_config(joinpath(@__DIR__, "..", "configs", "haar_poc.toml")); cfg.verbose = false
    cfg.use_bias = true; cfg.bias_warmstart = 20.0; cfg.bias_monotone = true; cfg.measure = "physical"
    P1, sL, info = MW.setup(cfg; rng=MersenneTwister(cfg.seed))
    L = sL.ℓ
    MW.lift_bias!(P1, L; rng=MersenneTwister(5), nsteps=2048)
    P1.bias.θ .*= 3.0                                 # a bias of realistic size (as in test_bias.jl)
    MW.rebuild_tables!(P1.bias)
    W = P1.W
    LM = MW.LocalModel(P1, L)
    T0 = W.lays[1].T
    V = MW.z0_block_directions(W)
    newblock(P, LC, B) = MW.BlockMove(MW.BlockPool(P, MW.z0_block_span(W, L, B)).wss, 0, Int[], P, LM, LC, Float64[], Float64[])

    @testset "z0 block moves: directions and VAR quadratic" begin
        @test size(V) == (MW.n0(W), 6)
        @test V' * V ≈ I
        # lengths whose window at the finest level (6 h here) fits the cap
        @test MW.z0_block_span(W, L, 365) == ((365 + 2) * W.trs[1].r + 2) * W.trs[2].r
        @test MW.z0_block_lengths(W, L, (16, 64, 365), 600) == (16, 64)
        @test MW.z0_block_lengths(W, L, (16, 64, 365), 10^6) == (16, 64, 365)
        @test isempty(MW.z0_block_lengths(W, L, (16, 64, 365), 10))
        rng = MersenneTwister(51)
        s = MW.evaluate!(P1, copy(sL))
        for trial in 1:20
            B = rand(rng, (1, 7, 30, 365))
            a = rand(rng, 1:T0); b = min(a + B - 1, T0)
            v = V[:, rand(rng, 1:size(V, 2))]
            κ, β = MW.z0_block_quad(W, s.m, s.z0, a, b, v)
            dz = MW.z0_block_delta(W, a, b, v)
            lp0 = MW.logp_z0(W, s.m, s.z0)[1]
            for t in (0.3, -1.1)
                @test isapprox(MW.logp_z0(W, s.m, s.z0 .+ t .* dz)[1] - lp0, -(0.5 * κ * t^2 + β * t); atol=1e-8 * (1 + abs(lp0)))
            end
            @test κ > 0
        end
    end

    @testset "z0 block moves (α = $(Int(P.use_bias)))" for P in (MW.with_bias(P1, false), P1)
        rng = MersenneTwister(53)
        s = MW.evaluate!(P, copy(sL))
        ncheck = 0
        for B in (1, 4, 16), comp in (true, false), trial in 1:2
            s1 = deepcopy(s)
            LC = MW.LocalCost(P, LM, s1); BC = MW.BiasCache(P, s1)
            a = rand(rng, 1:T0 - B + 1); b = a + B - 1
            v = V[:, rand(rng, 1:size(V, 2))]
            bm = newblock(P, LC, B)
            jac = NTuple{5,Float64}[]
            pr = MW.z0_block_prepare!(bm, P, LM, LC, s1, jac, a, b, v; comp=comp)
            pr === nothing && continue
            ch, dz, (κ, β) = pr
            t = 0.05 * randn(rng)
            ΔC = MW.mv_eval!(bm, t)
            Δb = BC === nothing ? 0.0 : MW.bias_delta_weather(BC, P.bias, P.E, bm, t)
            lj = isempty(jac) ? 0.0 : MW.logjac(jac, t, bm)
            L0, b0 = s1.L, MW.bias_term(P, s1)
            MW.z0_block_commit!(P, s1, bm, ch, dz, t; BC=BC)
            MW.evaluate!(P, s1)
            BC === nothing || MW.bias_resum!(BC, P.bias)
            # incremental against global (the log target excludes the weather prior)
            @test isapprox(s1.L - L0, -P.βv * ΔC - Δb; atol=1e-12 * abs(L0) + 1e-8 * (1 + abs(P.βv * ΔC)))
            @test isapprox(Δb, MW.bias_term(P, s1) - b0; atol=1e-9 * (1 + abs(b0)))
            # anomalies of every level against a fresh evaluation, caches against rebuilds
            @test maximum(abs.(MW.weather_at_level(W, L, s1.m, s1.z0, s1.ws) .- s1.xs[end])) < 1e-10
            LC2 = MW.LocalCost(P, LM, s1)
            @test maximum(abs.(LC.S .- LC2.S)) < 1e-6 && maximum(abs.(LC.r .- LC2.r)) < 1e-6
            @test maximum(maximum(abs.(LC.X[f] .- LC2.X[f])) for f in eachindex(LC.X)) < 1e-10
            @test maximum(maximum(abs.(LC.Y[f] .- LC2.Y[f]); init=0.0) for f in eachindex(LC.Y)) < 1e-9
            if BC !== nothing
                BC2 = MW.BiasCache(P, s1)
                @test abs(MW.bias_value(BC, P.bias) - MW.bias_term(P, s1)) <= 1e-9 * (1 + abs(b0))
                @test maximum(abs.(BC.U1x .- BC2.U1x)) < 1e-12 && maximum(abs.(BC.Dx .- BC2.Dx)) < 1e-6
                @test maximum(maximum(abs.(BC.At[g] .- BC2.At[g]); init=0.0) for g in eachindex(BC.At)) < 1e-12
            end
            # the reverse move from the moved state returns to the start with the opposite line density
            LC = MW.LocalCost(P, LM, s1); BCr = MW.BiasCache(P, s1)
            jac2 = NTuple{5,Float64}[]
            ch2, dz2, (κ2, β2) = MW.z0_block_prepare!(bm, P, LM, LC, s1, jac2, a, b, v; comp=comp)
            ΔC2 = MW.mv_eval!(bm, -t)
            lj2 = isempty(jac2) ? 0.0 : MW.logjac(jac2, -t, bm)
            ℓf = -P.βv * ΔC - (0.5 * κ * t^2 + β * t) + lj
            ℓr = -P.βv * ΔC2 - (0.5 * κ2 * t^2 - β2 * t) + lj2
            @test isapprox(ℓf, -ℓr; atol=1e-8 * (1 + abs(ℓf)))
            MW.z0_block_commit!(P, s1, bm, ch2, dz2, -t; BC=BCr)
            @test maximum(abs.(s1.z0 .- s.z0)) < 1e-12 && maximum(abs.(s1.v .- s.v)) < 1e-9
            @test maximum(abs.(s1.xs[end] .- s.xs[end])) < 1e-10
            ncheck += 1
        end
        @test ncheck >= 10
        # sweeps keep the caches consistent with a fresh evaluation
        s1 = deepcopy(s)
        LC = MW.LocalCost(P, LM, s1); BC = MW.BiasCache(P, s1)
        s1, (nm, nmoved) = MW.z0_block_sweep!(MersenneTwister(7), P, LM, s1, 6; LC=LC, BC=BC, lengths=(4, 16))
        @test nm == 6 && nmoved > 0
        @test MW.evaluate!(P, deepcopy(s1)).L ≈ s1.L
        @test maximum(abs.(LC.S .- MW.LocalCost(P, LM, s1).S)) < 1e-6
        BC === nothing || @test abs(MW.bias_value(BC, P.bias) - MW.bias_term(P, s1)) <= 1e-9 * (1 + abs(MW.bias_term(P, s1)))
        s2, st = MW.fine_weather_sweep!(MersenneTwister(8), P, LM, deepcopy(s); frac=0.005, frac0=0.005, zblocks=2.0)
        @test MW.evaluate!(P, deepcopy(s2)).L ≈ s2.L
    end

    @testset "z0 block move: repeated draws against quadrature (α = 1, Metropolis bias correction)" begin
        # Repeated moves of one block in one direction without compensation (a straight line in
        # (z0, v)): the chain must sample p(t) ∝ exp(−β_v ΔC(t) − prior(t) − Δb̂(t)) on the line
        # through the start, computed by quadrature from the start state's move.
        P = P1
        b = P.bias
        rng = MersenneTwister(57)
        s = MW.evaluate!(P, copy(sL))
        best = nothing
        for trial in 1:200                           # a block with a wide line and a bias change
            a = rand(rng, 1:T0 - 3); bb = a + 3
            v = V[:, rand(rng, 1:size(V, 2))]
            s1 = copy(s); LC = MW.LocalCost(P, LM, s1); BC = MW.BiasCache(P, s1)
            bm = newblock(P, LC, 4)
            pr = MW.z0_block_prepare!(bm, P, LM, LC, s1, NTuple{5,Float64}[], a, bb, v; comp=false)
            pr === nothing && continue
            _, _, (κ, β) = pr
            grid = range(-4.0, 4.0; length=4001)
            lp = [-P.βv * MW.mv_eval!(bm, x) - 0.5 * κ * x^2 - β * x for x in grid]
            w = exp.(lp .- maximum(lp)); cdf = cumsum(w) ./ sum(w)
            lo, hi = grid[searchsortedfirst(cdf, 0.01)], grid[searchsortedfirst(cdf, 0.99)]
            hi - lo > 0.2 || continue
            Δb = abs(MW.bias_delta_weather(BC, b, P.E, bm, hi) - MW.bias_delta_weather(BC, b, P.E, bm, lo))
            Δb > 1e-3 || continue
            best = (a, bb, v, Δb, lo, hi)
            break
        end
        @test best !== nothing
        a, bb, v, Δb, lo, hi = best
        θs = copy(b.θ)
        b.θ .*= 2.0 / Δb                              # a bias change of about 2 nats across the line
        MW.rebuild_tables!(b)
        # start the chain inside the bulk of its line: move the state to the middle of [lo, hi]
        s1 = MW.evaluate!(P, copy(s)); LC = MW.LocalCost(P, LM, s1)
        bmc = newblock(P, LC, 4)
        ch, dzc, _ = MW.z0_block_prepare!(bmc, P, LM, LC, s1, NTuple{5,Float64}[], a, bb, v; comp=false)
        MW.z0_block_commit!(P, s1, bmc, ch, dzc, 0.5 * (lo + hi))
        MW.evaluate!(P, s1)
        lo, hi = lo - 0.5 * (lo + hi), hi - 0.5 * (lo + hi)
        LC = MW.LocalCost(P, LM, s1); BC = MW.BiasCache(P, s1)
        bm0 = newblock(P, LC, 4)
        _, _, (κ, β) = MW.z0_block_prepare!(bm0, P, LM, LC, s1, NTuple{5,Float64}[], a, bb, v; comp=false)
        grid = range(lo - 3 * (hi - lo), hi + 3 * (hi - lo); length=8001)
        lp = [-P.βv * MW.mv_eval!(bm0, x) - 0.5 * κ * x^2 - β * x - MW.bias_delta_weather(BC, b, P.E, bm0, x) for x in grid]
        lp0 = [-P.βv * MW.mv_eval!(bm0, x) - 0.5 * κ * x^2 - β * x for x in grid]
        cdfof(lp) = (w = exp.(lp .- maximum(lp)); cumsum(w) ./ sum(w))
        cdf, cdf0 = cdfof(lp), cdfof(lp0)
        # the chain of moves (the caches follow the accepted moves)
        LC = MW.LocalCost(P, LM, s1); BC = MW.BiasCache(P, s1)
        bm = newblock(P, LC, 4)
        jac, sp = NTuple{5,Float64}[], Float64[]
        z00 = copy(s1.z0)
        dz = MW.z0_block_delta(W, a, bb, v)
        n = 20000
        ts = zeros(n)
        nacc = 0
        for it in 1:n
            t = MW.z0_block_move!(rng, P, LM, LC, s1, bm, jac, sp, a, bb, v; comp=false, BC=BC)
            nacc += t != 0.0
            ts[it] = dot(s1.z0 .- z00, dz) / dot(dz, dz)
        end
        Fq(x, c) = c[clamp(searchsortedlast(grid, x), 1, length(grid))]
        ks = maximum(abs(mean(ts .<= x) - Fq(x, cdf)) for x in quantile(ts, 0.02:0.02:0.98))
        ks0 = maximum(abs(mean(ts .<= x) - Fq(x, cdf0)) for x in quantile(ts, 0.02:0.02:0.98))
        println("  z0 block MH with the bias: acceptance $(round(nacc / n; digits=3)), KS $(round(ks; digits=4)) ",
                "(against the bias-free conditional $(round(ks0; digits=4)))")
        @test ks < 0.03
        @test ks0 > 2 * ks
        b.θ .= θs
        MW.rebuild_tables!(b)
    end
end
