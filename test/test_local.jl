# Tests of the sparse graph-Haar bases and the local move machinery (src/local.jl).
using Test, Random, LinearAlgebra, Statistics, SparseArrays
using ESMSampling
const ML = ESMSampling

@testset "sparse graph Haar field bases" begin
    rng = MersenneTwister(11)
    N = 30
    country = rand(rng, 1:5, N)
    parts = [ones(Int, N), country, collect(1:N)]
    edges = [(i, i + 1, 1.0) for i in 1:N-1]
    Ug, gcounts, _ = ML.haar_basis(parts; edges=edges)
    @test Matrix(Ug' * Ug) ≈ I
    @test gcounts == [1, length(unique(country)), N]
    sup = sort(randperm(rng, N)[1:17])
    B = ML.haar_field_basis(parts, gcounts, sup; edges=edges)
    @test Matrix(B.U' * B.U) ≈ I
    @test nnz(B.U) <= length(sup) * (ceil(Int, log2(length(sup))) + 3)
    for M_ in (1, gcounts[2], N)
        m = ML.modes_at(B, M_)
        Um = Matrix(B.U[:, 1:m])
        r = randn(rng, length(sup))
        out = zeros(length(sup))
        ML.project!(out, B, m, r, zeros(length(sup)))
        @test out ≈ Um * (Um' * r)
        # the level space is the cell-constant fields of the partition
        q = findlast(<=(M_), gcounts)
        for c in unique(zip([p[sup] for p in parts[1:q]]...))
            v = Float64.([Tuple(p[n] for p in parts[1:q]) == c for n in sup])
            @test Um * (Um' * v) ≈ v
        end
    end
    # potentials: no component constants
    comp = [i <= 20 ? 1 : 2 for i in 1:N]
    Bθ = ML.haar_field_basis([comp, country, collect(1:N)], gcounts, collect(1:N); edges=edges, drop_top=true)
    @test ML.nmodes(Bθ) == N - 2
    @test norm(Bθ.U' * Float64.(comp .== 1)) < 1e-10
    @test norm(Bθ.U' * Float64.(comp .== 2)) < 1e-10
end

@testset "piecewise-exponential segment helpers" begin
    rng = MersenneTwister(12)
    for (g, L) in ((3.0, 0.7), (-2.0, 1.3), (1e-12, 2.0), (-40.0, 0.5))
        # log mass
        xs = range(0, L; length=200001)
        num = log(sum(exp.(-g .* xs)) * step(xs))
        @test isapprox(ML.logseg(L, g * L), num; atol=1e-4)
        # inversion sampler: mean
        s = [ML.trexp_offset(rng, g, L) for _ in 1:200000]
        w = exp.(-g .* xs); μ = sum(xs .* w) / sum(w)
        @test all(0 .<= s .<= L)
        @test abs(mean(s) - μ) < 0.01 * L
    end
end

const CFG = joinpath(@__DIR__, "..", "configs", "haar_poc.toml")
if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    @testset "local moves on the PyPSA-Eur model" begin
        cfg = ML.load_config(CFG); cfg.verbose = false
        P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
        rng = MersenneTwister(13)
        for ℓ in (0, 1)
            s = ML.restrict_state(P, sL, ℓ)
            LM = ML.LocalModel(P, ℓ)
            LC = ML.LocalCost(P, LM, s)
            @test isapprox(ML.cache_cost(P, LM, LC), s.parts.op; rtol=1e-12)
            # localized re-routing templates: balance kept except at n, m and the listed leak nodes
            sys = P.E.sys
            errx = 0.0
            for LMx in (LM, ML.LocalModel(P, ℓ; xr_ball=1)), n in 1:sys.N, (j, m) in enumerate(LMx.xpart[n])
                d = zeros(sys.N); d[LMx.xnod[n][j]] .= LMx.xval[n][j]
                dr = zeros(sys.N)
                for e in eachindex(sys.line_s)
                    f = sys.line_b[e] * (d[sys.line_bus0[e]] - d[sys.line_bus1[e]])
                    dr[sys.line_bus0[e]] += f; dr[sys.line_bus1[e]] -= f
                end
                dr[n] -= 1.0; dr[m] += 1.0
                dr[LMx.xlkn[n][j]] .-= LMx.xlkg[n][j]
                errx = max(errx, maximum(abs, dr))
            end
            @test errx < 1e-9
            # cell-move levels: fixed probabilities ∝ number of cells
            @test LM.qcum[end] ≈ 1 && issorted(LM.qcum)
            ws = ML.MoveWS(P)
            wa = ML.MoveWS(P)
            jac = NTuple{5,Float64}[]
            for trial in 1:300
                τ = rand(rng, 1:LM.T)
                ML.clear!(ws)
                fam = rand(rng, [:atom, :unit, :pair, :xpair, :xloc, :xlocc, :potential, :cellpair, :cell, :spair])
                τm = mod(τ - 2, LM.T) + 1
                ML.clear!(wa)
                if fam == :atom
                    f = rand(rng, eachindex(P.E.vspecs))
                    ML.prep_atom!(ws, P, LM, f, rand(rng, 1:P.lv[ℓ+1].m[f]), τ)
                elseif fam == :spair
                    ℓ == 1 || continue
                    ML.prep_spair!(rng, wa, ws, P, LM, LC, τm, τ) || continue
                elseif fam == :xlocc
                    ML.prep_xloc!(rng, ws, P, LM, LC, τ, jac) || continue
                else
                    ML.prep_random!(rng, ws, P, LM, LC, fam, τ) || continue
                end
                ML.finalize!(ws, P, LM, LC, τ)
                isempty(ws.ent) && continue
                t = 0.05 * randn(rng)
                op0 = s.parts.op
                # linked two-step move (τ−1, τ)
                ML.finalize!(wa, P, LM, LC, τm)
                ML.link_soc!(wa, ws, LM)
                bm = ML.BlockMove([wa, ws], 2, [τm, τ], P, LM, LC, Float64[], Float64[])
                # the cached walk evaluation (record at t = 0, then evaluate) is the plain one, bitwise
                Δc = (ML.mv_eval!(bm, 0.0; mode=1); ML.mv_eval!(bm, t; mode=2)); secc = copy(ML.mv_sec(bm))
                Δ = ML.mv_eval!(bm, t)
                @test Δc == Δ && secc == ML.mv_sec(bm)
                ML.commit_block!(bm, s.v, t)
                ML.evaluate!(P, s)
                @test isapprox(s.parts.op - op0, Δ; atol=1e-12 * abs(op0) + 1e-9 * abs(Δ))
            end
            # caches stay consistent with the coefficients
            LC2 = ML.LocalCost(P, LM, s)
            @test maximum(abs.(LC.S .- LC2.S)) < 1e-6
            @test maximum(abs.(LC.F .- LC2.F)) < 1e-6
            begin                                 # single-step sampler (a generator move, no storage term)
                # exact line sampler: distribution of t for a fixed move against quadrature
                τ = 7
                ML.clear!(ws)
                usable(x) = LM.complete[x[1]] || ℓ == 0
                nu = first(n for n in 1:P.E.sys.N if any(usable, LM.units[n]))
                u = first(x for x in LM.units[nu] if usable(x))
                if ℓ == 1
                    ML.prep_node!(ws, P, LM, u[1], u[2], 1.0, τ)
                else
                    ML.prep_atom!(ws, P, LM, u[1], 1, τ)
                end
                ML.finalize!(ws, P, LM, LC, τ)
                ts = [ML.sample_line!(rng, ws, P, LM, LC, τ, P.βv) for _ in 1:40000]
                lo, hi = quantile(ts, 0.0005), quantile(ts, 0.9995)
                grid = range(lo - (hi - lo), hi + (hi - lo); length=20001)
                lp = [-P.βv * ML.eval_move!(ws, P, LM, LC, τ, x) for x in grid]
                w = exp.(lp .- maximum(lp))
                cdf = cumsum(w) ./ sum(w)
                Femp(x) = mean(ts .<= x)
                Fq(x) = cdf[clamp(searchsortedlast(grid, x), 1, length(grid))]
                ks = maximum(abs(Femp(x) - Fq(x)) for x in quantile(ts, 0.01:0.01:0.99))
                @test ks < 0.015
            end
            if ℓ == 1                                # the same for the linked two-step sampler (cached walk evaluations)
                τ = 7; τm = 6
                ML.clear!(ws); ML.clear!(wa)
                nd = first(n for n in 1:P.E.sys.N if !isempty(LM.nodestor[n]) && any(x -> LM.complete[x[1]], LM.units[n]))
                u = first(x for x in LM.units[nd] if LM.complete[x[1]])
                ML.prep_node!(ws, P, LM, u[1], u[2], 1.0, τ)
                si, i = LM.nodestor[nd][1]
                ML.prep_node!(ws, P, LM, P.E.stors[si].fe, i, 0.3, τ)      # also moves the storage term of τm
                ML.finalize!(ws, P, LM, LC, τ); ML.finalize!(wa, P, LM, LC, τm); ML.link_soc!(wa, ws, LM)
                bm = ML.BlockMove([wa, ws], 2, [τm, τ], P, LM, LC, Float64[], Float64[])
                append!(bm.bps, wa.bps); append!(bm.bps, ws.bps)
                ts = [ML.sample_line!(rng, bm, P.βv, ws.secprev) for _ in 1:40000]
                lo, hi = quantile(ts, 0.0005), quantile(ts, 0.9995)
                grid = range(lo - (hi - lo), hi + (hi - lo); length=20001)
                lp = [-P.βv * ML.mv_eval!(bm, x) for x in grid]
                w = exp.(lp .- maximum(lp))
                cdf = cumsum(w) ./ sum(w)
                Femp2(x) = mean(ts .<= x)
                Fq2(x) = cdf[clamp(searchsortedlast(grid, x), 1, length(grid))]
                ks = maximum(abs(Femp2(x) - Fq2(x)) for x in quantile(ts, 0.01:0.01:0.99))
                @test ks < 0.015
            end
            if ℓ == 1
                # the walk's points of lines with storage terms: the potential is linear between
                # consecutive points (a storage injection kinks at the roots of its own secondaries,
                # which bend the cell balances inside a primary segment; walk! splits there)
                worst = 0.0
                nlines = 0
                jacs = NTuple{5,Float64}[]
                for trial in 1:240
                    τ = rand(rng, 2:LM.T); τm = τ - 1
                    ML.clear!(ws); ML.clear!(wa)
                    kind = trial % 3
                    if kind == 0
                        ML.prep_spair!(rng, wa, ws, P, LM, LC, τm, τ) || continue
                    elseif kind == 1
                        empty!(jacs)
                        ML.prep_tsoc!(rng, wa, ws, P, LM, LC, τm, τ, jacs) || continue
                    else
                        nd = rand(rng, [n for n in 1:P.E.sys.N if !isempty(LM.nodestor[n])])
                        si, i = rand(rng, LM.nodestor[nd])
                        ML.prep_node!(ws, P, LM, P.E.stors[si].fe, i, 1.0, τ)
                        us = [x for x in LM.units[nd] if LM.complete[x[1]] && x[1] != P.E.stors[si].fe]
                        isempty(us) || (u = rand(rng, us); ML.prep_node!(ws, P, LM, u[1], u[2], randn(rng), τ))
                    end
                    ML.finalize!(ws, P, LM, LC, τ)
                    isempty(ws.ent) && continue
                    ML.finalize!(wa, P, LM, LC, τm); ML.link_soc!(wa, ws, LM)
                    bm = ML.BlockMove([wa, ws], 2, [τm, τ], P, LM, LC, Float64[], Float64[])
                    append!(bm.bps, wa.bps); append!(bm.bps, ws.bps)
                    ML.sample_line!(rng, bm, P.βv, ws.secprev)
                    lb = ML.linebuf(bm)
                    pts, cs = copy(lb.pts), copy(lb.cs)
                    for j in 1:length(pts)-1
                        pts[j+1] > pts[j] || continue
                        cm = P.βv * ML.mv_eval!(bm, 0.5 * (pts[j] + pts[j+1]))
                        worst = max(worst, abs(cm - 0.5 * (cs[j] + cs[j+1])))
                    end
                    nlines += 1
                end
                @test nlines > 100
                @test worst < 1e-6                       # nats (before the split: up to 2.6e2)
            end
            # investment scaling moves keep the caches consistent (complete level only)
            if ℓ == 1 && all(ML.iscomplete(P.lk[2], P.E.kspecs, f) for f in eachindex(P.E.kspecs))
                LC = ML.LocalCost(P, LM, s)
                s, nm = ML.kscale_sweep!(rng, P, LM, s; LC=LC)
                @test nm > 0
                @test isapprox(ML.cache_cost(P, LM, LC), s.parts.op; rtol=1e-9)
                LC2 = ML.LocalCost(P, LM, s)
                @test maximum(abs.(LC.S .- LC2.S)) < 1e-6
            end
        end
    end
end
