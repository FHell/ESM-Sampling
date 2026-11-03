# Model completion (docs/model_completion.md): losses, soft limits, lossy links,
# vintages, CO₂ price, fleet unit commitment (item 17) and fine-level weather moves
# (item 18). Checks: incremental move costs equal the global evaluation, caches stay
# consistent, gradients match finite differences, exact line draws match quadrature
# (requirement-tracking kinks, commitment, Gaussian weather prior).
using Test, Random, Statistics
import ESMSampling as ML

function model_ks(rng, P, bm, nsamp; quad=(0.0, 0.0))
    ts = [ML.sample_line!(rng, bm, P.βv, Float64[]; quad=quad) for _ in 1:nsamp]
    lo, hi = quantile(ts, 0.0005), quantile(ts, 0.9995)
    hi - lo < 1e-12 && return NaN
    grid = range(lo - (hi - lo), hi + (hi - lo); length=40001)
    lp = [-P.βv * ML.mv_eval!(bm, x) - 0.5 * quad[1] * x^2 - quad[2] * x for x in grid]
    w = exp.(lp .- maximum(lp)); cdf = cumsum(w) ./ sum(w)
    Fq(x) = cdf[clamp(searchsortedlast(grid, x), 1, length(grid))]
    return maximum(abs(mean(ts .<= x) - Fq(x)) for x in quantile(ts, 0.01:0.01:0.99))
end

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    @testset "model completion: all terms on (40 nodes)" begin
        cfg = ML.load_config(joinpath(@__DIR__, "..", "configs", "haar_poc.toml")); cfg.verbose = false
        cfg.storage = "soc"; cfg.renewables = "load"; cfg.measure = "physical"; cfg.init = "heuristic"
        cfg.losses = true; cfg.thermal = "soft"; cfg.link_losses = true; cfg.uc = ["gas", "nuclear"]
        cfg.co2_price = 120.0
        P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
        rng = MersenneTwister(13)
        L = sL.ℓ
        s = copy(sL)
        LM = ML.LocalModel(P, L)
        LC = ML.LocalCost(P, LM, s)
        @test isapprox(ML.cache_cost(P, LM, LC), s.parts.op; rtol=1e-10)
        @test s.parts.co2 > 0
        # random local and transport moves: incremental = global
        ws, wa = ML.MoveWS(P), ML.MoveWS(P); wa.follow = false
        jac = NTuple{5,Float64}[]
        nbad = 0
        for trial in 1:150
            τ = rand(rng, 2:LM.T); τm = τ - 1
            ML.clear!(ws); ML.clear!(wa)
            fam = rand(rng, [:atom, :unit, :pair, :xpair, :potential, :cell, :tpot, :tsoc])
            if fam == :atom
                f = rand(rng, eachindex(P.E.vspecs))
                ML.prep_atom!(ws, P, LM, f, rand(rng, 1:P.lv[L+1].m[f]), τ)
            elseif fam == :tpot
                ML.prep_tpot!(rng, ws, P, LM, LC, τ, jac) || continue
            elseif fam == :tsoc
                ML.prep_tsoc!(rng, wa, ws, P, LM, LC, τm, τ, jac) || continue
            else
                ML.prep_random!(rng, ws, P, LM, LC, fam, τ) || continue
            end
            ML.finalize!(ws, P, LM, LC, τ); ML.finalize!(wa, P, LM, LC, τm); ML.link_soc!(wa, ws, LM)
            isempty(ws.ent) && continue
            bm = ML.BlockMove([wa, ws], 2, [τm, τ], P, LM, LC, Float64[], Float64[])
            t = 0.05 * randn(rng)
            op0 = s.parts.op
            Δ = ML.mv_eval!(bm, t)
            ML.commit_block!(bm, s.v, t)
            ML.evaluate!(P, s)
            nbad += abs(s.parts.op - op0 - Δ) > 1e-12 * abs(op0) + 1e-9 * abs(Δ) + 1e-6
        end
        @test nbad == 0
        LC2 = ML.LocalCost(P, LM, s)
        @test maximum(abs.(LC.S .- LC2.S)) < 1e-6
        # exact line draws with requirement-tracking kinks and with commitment
        ks = Float64[]
        for rep in 1:4
            for trial in 1:50
                τ = rand(rng, 2:LM.T)
                ML.clear!(ws); ML.clear!(wa)
                ok = rep <= 2 ? ML.prep_tpot!(rng, ws, P, LM, LC, τ, jac) :
                     (g = P.E.gens[findfirst(g -> g.ucf > 0, P.E.gens)]; ML.prep_node!(ws, P, LM, g.vf, rand(rng, eachindex(g.bus)), 1.0, τ); true)
                ok || continue
                ML.finalize!(ws, P, LM, LC, τ); ML.finalize!(wa, P, LM, LC, τ - 1); ML.link_soc!(wa, ws, LM)
                isempty(ws.ent) && continue
                bm = ML.BlockMove([wa, ws], 2, [τ - 1, τ], P, LM, LC, Float64[], Float64[])
                bm.bps = vcat(wa.bps, ws.bps)
                x = model_ks(rng, P, bm, 20000)
                isnan(x) && continue
                push!(ks, x)
                break
            end
        end
        @test !isempty(ks) && maximum(ks) < 0.015
        # gradients at level 0 (finite differences)
        s0 = ML.restrict_state(P, sL, 0)
        Xn = ML.weather_nodes(P.W, 0, s0.xs[end])
        zv = s0.v .+ 0.01 .* randn(rng, length(s0.v))
        p, (gkc, gko, gv, gX) = ML.energy_cost(P.E, P.lk[1], P.lv[1], s0.k, zv, Xn; grad=true)
        nok = 0
        for i in rand(rng, 1:length(zv), 20)
            h = 1e-4 * max(1, abs(zv[i]))
            zp, zm = copy(zv), copy(zv); zp[i] += h; zm[i] -= h
            fd = (ML.energy_cost(P.E, P.lk[1], P.lv[1], s0.k, zp, Xn).op - ML.energy_cost(P.E, P.lk[1], P.lv[1], s0.k, zm, Xn).op) / 2h
            nok += isapprox(fd, gv[i]; rtol=0.05, atol=1e-6 * abs(p.op) / length(zv))
        end
        @test nok >= 19
        # fine-level weather moves: incremental = global, caches, representation, line draws
        W = P.W
        span = 1; for ℓ in 1:L; span = (span + 2) * W.trs[ℓ].r; end
        pool = ML.BlockPool(P, span)
        bmw = ML.BlockMove(pool.wss, 0, Int[], P, LM, LC, Float64[], Float64[])
        nbad = 0; nmv = 0
        for trial in 1:60
            ℓ = rand(rng, 1:L)
            q = rand(rng, 1:W.trs[ℓ].n)
            f, part, i, b, j = ML.innovation_coord(W, ℓ, q)
            ch = ML.wdelta_chain(W, s.m, ML.wdelta_innovation(W, s.m, ℓ, f, part, i, b, j), L)
            ML.prep_weather_move!(bmw, P, LM, LC, ch, jac) === nothing && continue
            t = 0.3 * randn(rng)
            op0 = s.parts.op
            Δ = ML.mv_eval!(bmw, t)
            ML.commit_block!(bmw, s.v, t)
            s.ws[ℓ][q] += t; ML.apply_wdelta!(s, W, ch, t)
            ML.evaluate!(P, s)
            nmv += 1
            nbad += abs(s.parts.op - op0 - Δ) > 1e-9 * abs(op0) + 1e-6 * abs(Δ) + 1e-3
        end
        @test nmv > 20 && nbad == 0
        LC2 = ML.LocalCost(P, LM, s)
        @test maximum(maximum(abs.(LC.X[f] .- LC2.X[f])) for f in eachindex(LC.X)) < 1e-8
        @test maximum(abs.(LC.r .- LC2.r)) < 1e-6
        s2 = ML.make_state(P, L, s.m, s.z0, s.ws, s.k, s.v)
        @test maximum(maximum(abs.(s2.xs[l] .- s.xs[l])) for l in eachindex(s.xs)) < 1e-9
        s3 = copy(s); ML.fine_regime_step!(rng, P, s3)
        s4 = ML.make_state(P, L, s3.m, s3.z0, s3.ws, s3.k, s3.v)
        @test maximum(maximum(abs.(s4.xs[l] .- s.xs[l])) for l in eachindex(s.xs)) < 1e-9
        ksw = Float64[]
        for rep in 1:3
            ℓ = rand(rng, 1:L); q = rand(rng, 1:W.trs[ℓ].n)
            f, part, i, b, j = ML.innovation_coord(W, ℓ, q)
            ch = ML.wdelta_chain(W, s.m, ML.wdelta_innovation(W, s.m, ℓ, f, part, i, b, j), L)
            ML.prep_weather_move!(bmw, P, LM, LC, ch, jac) === nothing && continue
            x = model_ks(rng, P, bmw, 20000; quad=(1.0, s.ws[ℓ][q]))
            isnan(x) || push!(ksw, x)
        end
        @test !isempty(ksw) && maximum(ksw) < 0.015
        # z0 prior along a coordinate: the local quadratic equals the global log density change
        o, τ0 = 3, 17
        κ, bq = ML.z0_quad(W, s.m, s.z0, o, τ0)
        lay0 = W.lays[1]
        fz, rows = 1, 0
        while rows + lay0.m[fz] < o; rows += lay0.m[fz]; fz += 1; end
        iz = lay0.off[fz] + (τ0 - 1) * lay0.m[fz] + (o - rows)
        z1 = copy(s.z0); z1[iz] += 0.7
        @test isapprox(-(ML.logp_z0(W, s.m, z1)[1] - ML.logp_z0(W, s.m, s.z0)[1]), 0.5κ * 0.49 + bq * 0.7; rtol=1e-8, atol=1e-8)
    end
end
