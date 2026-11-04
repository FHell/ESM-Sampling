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

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    @testset "long-range storage moves (slong, slevel)" begin
        cfg = ML.load_config(CFG); cfg.verbose = false
        P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
        rng = MersenneTwister(17)
        ℓ = 1
        s = ML.restrict_state(P, sL, ℓ)
        LM = ML.LocalModel(P, ℓ)
        E = P.E
        T = LM.T
        units = [(si, i) for (si, st) in enumerate(E.stors) for i in eachindex(st.bus)]
        pool = ML.BlockPool(P, 3)
        newmove(LC) = ML.SLong(pool.wss[1], pool.wss[2], pool.wss[3], P, LM, LC)
        jac = NTuple{5,Float64}[]
        drawL() = rand(rng) < 0.25 ? T : rand(rng, [1, 2, 3, rand(rng, 4:T-1)])
        # incremental = global; recorded = plain (bitwise); lazy interior = full evaluation (round-off)
        LC = ML.LocalCost(P, LM, s)
        m = newmove(LC)
        nmv, nin, nout = 0, 0, 0
        for trial in 1:120
            si, i = rand(rng, units)
            ML.prep_slong!(m, si, i, rand(rng, 1:T), drawL(), jac) || continue
            inI = m.ilo <= m.ihi && rand(rng) < 0.5
            t = inI ? m.ilo + rand(rng) * (m.ihi - m.ilo) : 0.02 * randn(rng)
            Δ = ML.mv_eval!(m, t); sec = copy(ML.mv_sec(m))
            Δc = (ML.mv_fill!(m); ML.mv_evalc!(m, t))
            @test Δc == Δ && ML.mv_sec(m) == sec
            lo, hi, qlo, qhi = m.ilo, m.ihi, copy(m.qlo), copy(m.qhi)
            m.ilo, m.ihi = Inf, -Inf; fill!(m.qlo, Inf); fill!(m.qhi, -Inf)
            Δf = ML.mv_eval!(m, t)
            @test isapprox(Δf, Δ; atol=1e-9 * abs(Δ) + 1e-6)          # € (round-off of the full sum over the steps)
            @test maximum(abs, ML.mv_sec(m) .- sec; init=0.0) < 1e-6 * (1 + maximum(abs, sec; init=0.0))
            m.ilo, m.ihi = lo, hi; m.qlo .= qlo; m.qhi .= qhi
            (lo <= t <= hi) ? (nin += 1) : (nout += 1)
            op0 = s.parts.op
            ML.commit_slong!(m, s.v, t)
            ML.evaluate!(P, s)
            @test isapprox(s.parts.op - op0, Δ; atol=1e-12 * abs(op0) + 1e-9 * abs(Δ))
            nmv += 1
        end
        @info "slong moves" nmv nin nout
        @test nmv > 60 && nin > 5 && nout > 5
        LC2 = ML.LocalCost(P, LM, s)
        @test isapprox(ML.cache_cost(P, LM, LC), s.parts.op; rtol=1e-10)
        @test maximum(abs.(LC.S .- LC2.S)) < 1e-6 && maximum(abs.(LC.r .- LC2.r)) < 1e-6
        @test maximum(maximum(abs.(a .- b)) for (a, b) in zip(LC.Y, LC2.Y)) < 1e-9
        # the sweep keeps the caches consistent
        LC = ML.LocalCost(P, LM, s)
        s, st = ML.storage_long_sweep!(rng, P, LM, s; LC=LC, nmoves=16, hmin=0.0)   # threaded: cells in parallel
        @test st[1, 1] > 0 && st[2, 1] > 0
        @test isapprox(ML.cache_cost(P, LM, LC), s.parts.op; rtol=1e-10)
        LC2 = ML.LocalCost(P, LM, s)
        @test maximum(abs.(LC.S .- LC2.S)) < 1e-6 && maximum(abs.(LC.r .- LC2.r)) < 1e-6
        # invariance of the curve: the move prepared at s(t) with the same selection, at −t,
        # returns to s (shares and excess of the compensation kept, also across its kink), and
        # the Jacobian of the way back is the inverse
        LC = ML.LocalCost(P, LM, s)
        m = newmove(LC)
        jac2 = NTuple{5,Float64}[]
        worst, worstj, nk = 0.0, 0.0, 0
        for trial in 1:40
            si, i = rand(rng, units)
            a, Lb = rand(rng, 1:T), rand(rng, 1:T-1)
            ML.prep_slong!(m, si, i, a, Lb, jac) || continue
            isempty(jac) && continue
            Q, g = jac[1][3], jac[1][4]
            t = isodd(trial) ? 0.3 * randn(rng) * abs(Q / g) : -2.0 * Q / g      # beyond the kink of step a
            v0 = copy(s.v)
            lj = ML.logjac(jac, t, m)
            s1 = deepcopy(s); LC1 = ML.LocalCost(P, LM, s1)
            m.LC = LC1; ML.prep_slong!(m, si, i, a, Lb, jac); ML.commit_slong!(m, s1.v, t)
            LC1 = ML.LocalCost(P, LM, ML.evaluate!(P, s1))
            m.LC = LC1; ML.prep_slong!(m, si, i, a, Lb, jac2)
            lj2 = ML.logjac(jac2, -t, m)
            ML.commit_slong!(m, s1.v, -t)
            worst = max(worst, maximum(abs, s1.v .- v0) / (1 + maximum(abs, v0)))
            worstj = max(worstj, abs(lj + lj2))
            nk += abs(t) > abs(Q / g)
            m.LC = LC
        end
        @test nk > 5
        @test worst < 1e-9
        @test worstj < 1e-8
        # the same for the storage multiscale plateaus (one unit, a cell aggregate) and tsoc: the
        # requirement of their compensation follows the nominal storage injection (soc_qadj)
        bpool = ML.BlockPool(P, 20)
        worst, worstj = 0.0, 0.0
        cellunits = Dict{Tuple{Int,Int},Vector{NTuple{2,Int}}}()
        for (si, i) in units
            push!(get!(cellunits, (E.parts[2][E.stors[si].bus[i]], si), NTuple{2,Int}[]), (si, i))
        end
        for trial in 1:40
            kind = trial % 3
            us = kind == 2 ? rand(rng, collect(values(cellunits))) : [rand(rng, units)]
            Lb, b0 = rand(rng, [1, 2, 8]), rand(rng, 2:T-10)
            function prep(LCx, jx)
                bm = ML.blockmove(bpool.wss, bpool.wss[1], P, LM, LCx)
                for q in 1:12; ML.clear!(bpool.wss[q]); end
                if kind == 0
                    wa, ws = bpool.wss[1], bpool.wss[2]
                    ok = ML.prep_tsoc!(MersenneTwister(trial), wa, ws, P, LM, LCx, b0 - 1, b0, jx)
                    ML.finalize!(ws, P, LM, LCx, b0); ML.finalize!(wa, P, LM, LCx, b0 - 1); ML.link_soc!(wa, ws, LM)
                    bm.n = 2; append!(bm.τs, [b0 - 1, b0]); append!(bm.bps, wa.bps); append!(bm.bps, ws.bps)
                    return ok ? bm : nothing
                end
                return ML.prep_storage_ms!(bm, P, LM, LCx, us, b0, Lb, nothing, jx) ? bm : nothing
            end
            LC = ML.LocalCost(P, LM, s)
            bm = prep(LC, jac)
            (bm === nothing || isempty(jac)) && continue
            t = 0.5 * randn(rng) * abs(jac[1][3] / jac[1][4])
            lj = ML.logjac(jac, t, bm)
            s1 = deepcopy(s); ML.commit_block!(bm, s1.v, t); ML.evaluate!(P, s1)
            bm = prep(ML.LocalCost(P, LM, s1), jac2)
            lj2 = ML.logjac(jac2, -t, bm)
            ML.commit_block!(bm, s1.v, -t)
            worst = max(worst, maximum(abs, s1.v .- s.v) / (1 + maximum(abs, s.v)))
            worstj = max(worstj, abs(lj + lj2))
        end
        for q in 1:12; ML.clear!(bpool.wss[q]); end
        @test worst < 1e-9
        @test worstj < 1e-8
        # the walk's points: the potential is linear between consecutive points
        worst, nl = 0.0, 0
        for trial in 1:60
            si, i = rand(rng, units)
            ML.prep_slong!(m, si, i, rand(rng, 1:T), drawL(), jac) || continue
            ML.sample_line!(rng, m, P.βv, pool.wss[1].secprev)
            lb = ML.linebuf(m)
            pts, cs = copy(lb.pts), copy(lb.cs)
            for j in 1:length(pts)-1
                pts[j+1] > pts[j] || continue
                cm = P.βv * ML.mv_eval!(m, 0.5 * (pts[j] + pts[j+1]))
                worst = max(worst, abs(cm - 0.5 * (cs[j] + cs[j+1])))
            end
            nl += 1
        end
        @test nl > 40
        @test worst < 1e-6
        # line draws against quadrature: a long plateau and a level move of the largest-energy unit
        si, i = argmax(u -> LC.kst[u[1]][u[2]] * E.stors[u[1]].hours[u[2]], units)
        for Lb in (T ÷ 2, T)
            ML.prep_slong!(m, si, i, 5, Lb, jac)
            ts = [ML.sample_line!(rng, m, P.βv, pool.wss[1].secprev) for _ in 1:40000]
            lo, hi = quantile(ts, 0.0005), quantile(ts, 0.9995)
            grid = range(lo - (hi - lo), hi + (hi - lo); length=20001)
            lp = [-P.βv * ML.mv_eval!(m, x) for x in grid]
            w = exp.(lp .- maximum(lp)); cdf = cumsum(w) ./ sum(w)
            Fq(x) = cdf[clamp(searchsortedlast(grid, x), 1, length(grid))]
            @test maximum(abs(mean(ts .<= x) - Fq(x)) for x in quantile(ts, 0.01:0.01:0.99)) < 0.015
        end
    end
end


if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    @testset "cross-node storage pairs (sxpair, sxlong)" begin
        cfg = ML.load_config(CFG); cfg.verbose = false
        P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
        rng = MersenneTwister(19)
        ℓ = 1
        s = ML.restrict_state(P, sL, ℓ)
        E = P.E
        # every unit pairs (sx_hmin = 0); default templates (partners 1 hop, ball 1 hop) and partners
        # within 2 hops with the flow re-routed on the pair only (large leaks)
        LMs = (ML.LocalModel(P, ℓ; sx_hmin=0.0), ML.LocalModel(P, ℓ; sx_hmin=0.0, sx_ball=0, sx_partner=2))
        LM = LMs[1]
        T = LM.T
        @test !isempty(LM.sxunits) && all(p[1] == n for (n, p) in enumerate(LM.sxpart))
        # footprints: the cells of every node a template touches
        okf = true
        for LMx in LMs, n in 1:E.sys.N, j in eachindex(LMx.sxpart[n])
            nd = vcat([n, LMx.sxpart[n][j]], LMx.sxlkn[n][j], LMx.sxnod[n][j])
            okf &= issubset(unique(LMx.cell[nd]), LMx.sxfoot[n][j])
        end
        @test okf
        # the family interface (one entry of a move-consistency table): prepare with seeded discrete
        # choices, the line, the commit, the Jacobian
        pool = ML.BlockPool(P, 3)
        ws, wa = ML.MoveWS(P), ML.MoveWS(P)
        jac, jac2 = NTuple{5,Float64}[], NTuple{5,Float64}[]
        function prepare(kind, LMx, LCx, seed, jx)
            r = MersenneTwister(seed)
            if kind == :sxpair
                return ML.sxpair_move!(r, wa, ws, P, LMx, LCx, rand(r, 1:T), jx)
            end
            m = ML.SXLong(pool.wss[1], pool.wss[2], pool.wss[3], P, LMx, LCx)
            return ML.prep_sxlong_random!(r, m, jx) ? m : nothing
        end
        line(mv) = ML.sample_line!(rng, mv, P.βv, mv isa ML.SXLong ? mv.wa.secprev : ws.secprev)
        commit!(mv, v, t) = mv isa ML.SXLong ? ML.commit_sxlong!(mv, v, t) : ML.commit_block!(mv, v, t)
        kinds = (:sxpair, :sxlong)
        # incremental = global; recorded = plain (bitwise); for sxlong the lazy interior = full evaluation
        nmv = Dict(k => 0 for k in kinds)
        nin, nout, nkink = 0, 0, 0
        for LMx in LMs
            LC = ML.LocalCost(P, LMx, s)
            for trial in 1:200
                kind = kinds[mod1(trial, 2)]
                mv = prepare(kind, LMx, LC, rand(rng, UInt), jac)
                mv === nothing && continue
                if kind == :sxlong && mv.ilo <= mv.ihi && rand(rng) < 0.5
                    t = mv.ilo + rand(rng) * (mv.ihi - mv.ilo)
                else
                    t = rand(rng) < 0.5 ? line(mv) : 0.02 * randn(rng)
                end
                Δ = ML.mv_eval!(mv, t); sec = copy(ML.mv_sec(mv))
                Δc = (ML.mv_fill!(mv); ML.mv_evalc!(mv, t))
                @test Δc == Δ && ML.mv_sec(mv) == sec
                if kind == :sxlong
                    lo, hi, qlo, qhi = mv.ilo, mv.ihi, copy(mv.qlo), copy(mv.qhi)
                    mv.ilo, mv.ihi = Inf, -Inf; fill!(mv.qlo, Inf); fill!(mv.qhi, -Inf)
                    Δf = ML.mv_eval!(mv, t)
                    @test isapprox(Δf, Δ; atol=1e-9 * abs(Δ) + 1e-6)          # € (round-off of the full sum over the steps)
                    @test maximum(abs, ML.mv_sec(mv) .- sec; init=0.0) < 1e-6 * (1 + maximum(abs, sec; init=0.0))
                    mv.ilo, mv.ihi = lo, hi; mv.qlo .= qlo; mv.qhi .= qhi
                    (lo <= t <= hi) ? (nin += 1) : (nout += 1)
                end
                op0 = s.parts.op
                commit!(mv, s.v, t)
                ML.evaluate!(P, s)
                @test isapprox(s.parts.op - op0, Δ; atol=1e-12 * abs(op0) + 1e-9 * abs(Δ))
                nmv[kind] += 1
                nkink += length(jac)
            end
            LC2 = ML.LocalCost(P, LMx, s)
            @test isapprox(ML.cache_cost(P, LMx, LC), s.parts.op; rtol=1e-10)
            @test maximum(abs.(LC.S .- LC2.S)) < 1e-6 && maximum(abs.(LC.r .- LC2.r)) < 1e-6
            @test maximum(abs.(LC.F .- LC2.F)) < 1e-6
            @test maximum(maximum(abs.(LC.Y[f] .- LC2.Y[f])) for f in eachindex(LC.Y) if f != E.fθ) < 1e-9   # (angles: up to component constants)
        end
        @info "storage pairs" nmv nin nout nkink
        @test nmv[:sxpair] > 80 && nmv[:sxlong] > 80 && nin > 5 && nout > 5 && nkink > 20
        # the threaded sweeps keep the caches consistent: sxpair in the local sweep (steps of one
        # parity in parallel), sxlong in the long-range storage sweep (rounds of disjoint footprints)
        LC = ML.LocalCost(P, LM, s)
        s, st = ML.local_sweep!(rng, P, LM, s; LC=LC, sxpair=1.0, atoms=false, extra=1.0)
        @test st[end-1, 1] > 0 && st[end-1, 2] > 0                   # the sxpair row (the last family)
        @test isapprox(ML.cache_cost(P, LM, LC), s.parts.op; rtol=1e-10)
        s, st = ML.storage_long_sweep!(rng, P, LM, s; LC=LC, nmoves=2, hmin=0.0, nxmoves=8)
        @test st[3, 1] > 0 && st[3, 2] > 0
        @test isapprox(ML.cache_cost(P, LM, LC), s.parts.op; rtol=1e-10)
        LC2 = ML.LocalCost(P, LM, s)
        @test maximum(abs.(LC.S .- LC2.S)) < 1e-6 && maximum(abs.(LC.r .- LC2.r)) < 1e-6 && maximum(abs.(LC.F .- LC2.F)) < 1e-6
        @test maximum(maximum(abs.(LC.Y[f] .- LC2.Y[f])) for f in eachindex(LC.Y) if f != E.fθ) < 1e-9
        # rounds: the footprints of the moves of one round are disjoint
        mvs = [x for x in (ML.sxlong_choice(rng, P, LM, LC, rand(rng, LM.sxunits)...) for _ in 1:200) if x !== nothing]
        okr = true
        for rd in ML.sx_rounds(P, LM, mvs)
            fp = reduce(vcat, [LM.sxfoot[E.stors[mvs[k][1]].bus[mvs[k][2]]][mvs[k][5]] for k in rd])
            okr &= allunique(fp)
        end
        @test okr
        # reversibility of the curve: prepared again at s(t) with the same seed, the move makes the
        # same discrete choices, its line density is the forward one shifted by t (cost and log
        # Jacobian), −t returns to s (shares and excess of every compensation kept, also across a
        # kink), and the Jacobian of the way back is the inverse
        worst, worstj, worstd, nk, nr = 0.0, 0.0, 0.0, 0, 0
        for trial in 1:120
            kind = kinds[mod1(trial, 2)]
            seed = rand(rng, UInt)
            LC = ML.LocalCost(P, LM, s)
            mv = prepare(kind, LM, LC, seed, jac)
            (mv === nothing || isempty(jac)) && continue
            k = argmin(x -> abs(jac[x][3] / jac[x][4]), eachindex(jac))           # the nearest kink
            Q, g = jac[k][3], jac[k][4]
            t = trial % 4 < 2 ? 0.3 * randn(rng) * abs(Q / g) : -2.0 * Q / g      # beyond it
            us = (-t, -0.5t, 0.7t, 0.01 * randn(rng))
            fwd = [(ML.mv_eval!(mv, t + u), ML.logjac(jac, t + u, mv)) for u in us]
            c0, lj = ML.mv_eval!(mv, t), ML.logjac(jac, t, mv)
            choice = kind == :sxpair ? copy(ws.ent) : mv.us
            s1 = deepcopy(s)
            mv = prepare(kind, LM, ML.LocalCost(P, LM, s1), seed, jac)
            commit!(mv, s1.v, t)
            ML.evaluate!(P, s1)
            mv = prepare(kind, LM, ML.LocalCost(P, LM, s1), seed, jac2)
            @test mv !== nothing && (kind == :sxpair ? Set(ws.ent) == Set(choice) : mv.us == choice)
            for (u, (cf, jf)) in zip(us, fwd)
                cb = ML.mv_eval!(mv, u)
                worstd = max(worstd, P.βv * abs(cb - (cf - c0)))                   # nats
                worstj = max(worstj, abs(ML.logjac(jac2, u, mv) - (jf - lj)))
            end
            lj2 = ML.logjac(jac2, -t, mv)
            commit!(mv, s1.v, -t)
            worst = max(worst, maximum(abs, s1.v .- s.v) / (1 + maximum(abs, s.v)))
            worstj = max(worstj, abs(lj + lj2))
            nk += abs(t) > abs(Q / g)
            nr += 1
        end
        @info "storage pair reversibility" nr nk worst worstj worstd
        @test nr > 60 && nk > 10
        @test worst < 1e-9
        @test worstj < 1e-8
        @test worstd < 1e-6
        # the walk's points: the potential is linear between consecutive points
        LC = ML.LocalCost(P, LM, s)
        worst, nl = 0.0, 0
        for trial in 1:120
            mv = prepare(kinds[mod1(trial, 2)], LM, LC, rand(rng, UInt), jac)
            mv === nothing && continue
            line(mv)
            lb = ML.linebuf(mv)
            pts, cs = copy(lb.pts), copy(lb.cs)
            for j in 1:length(pts)-1
                pts[j+1] > pts[j] || continue
                cm = P.βv * ML.mv_eval!(mv, 0.5 * (pts[j] + pts[j+1]))
                worst = max(worst, abs(cm - 0.5 * (cs[j] + cs[j+1])))
            end
            nl += 1
        end
        @test nl > 80
        @test worst < 1e-6                       # nats
    end
end
