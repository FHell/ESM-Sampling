# Per-move statistics of the local exact kernel (no changes to src/: the line
# sampler and the evaluator are wrapped at run time by method redefinition).
# Usage: nice -n 19 julia -t 1 --project=. external/reactant-bench/instrument.jl [data] [N] [sweeps] [kind]
using ESMSampling, Random, Statistics, Printf
const M = ESMSampling

const NEVAL = Ref(0)
const REC = Vector{NTuple{9,Int}}()   # (evals, primary bps, entries, lines, nodes, cells, nsec, steps)
const TLINE = Ref(0.0)

M.mv_eval!(m::M.StepMove, t) = (NEVAL[] += 1; M.eval_move!(m.ws, m.P, m.LM, m.LC, m.τ, t))
function M.mv_eval!(m::M.BlockMove, t)
    NEVAL[] += 1
    Δ = 0.0
    empty!(m.sec)
    for q in 1:m.n
        prev = q > 1 ? m.wss[q-1] : nothing
        next = q < m.n ? m.wss[q+1] : nothing
        Δ += M.eval_move!(m.wss[q], m.P, m.LM, m.LC, m.τs[q], t; prev=prev, next=next)
        append!(m.sec, m.wss[q].sec)
    end
    return Δ
end

wss(m::M.StepMove) = (m.ws,)
wss(m::M.BlockMove) = Tuple(m.wss[1:m.n])

function M.sample_line!(rng, mv::Union{M.StepMove,M.BlockMove}, β::Float64, secprev::Vector{Float64}; maxnats=40.0)
    n0 = NEVAL[]
    t0 = time_ns()
    t = invoke_line(rng, mv, β, secprev, maxnats)
    dt = time_ns() - t0
    TLINE[] += dt * 1e-9
    W = wss(mv)
    nb = count(isfinite, M.mv_bps(mv))
    push!(REC, (NEVAL[] - n0, nb, sum(length(w.ent) for w in W), sum(length(w.lines) for w in W),
                sum(length(w.nodes) for w in W), sum(length(w.cells) for w in W), length(M.mv_sec(mv)), length(W), Int(dt)))
    return t
end

# copy of the original sampler body (src/local.jl, sample_line!), called by the wrapper
function invoke_line(rng, mv, β, secprev, maxnats)
    bp = filter(isfinite, M.mv_bps(mv))
    right = sort!(filter(>(0.0), bp))
    left = sort!(filter(<(0.0), bp); rev=true)
    pr, cr = Float64[], Float64[]
    pl, cl = Float64[], Float64[]
    M.walk!(pr, cr, mv, β, right, 1.0, maxnats, secprev) || return 0.0
    M.walk!(pl, cl, mv, β, left, -1.0, maxnats, secprev) || return 0.0
    pts = vcat(reverse(pl), 0.0, pr)
    cs = vcat(reverse(cl), 0.0, cr)
    p = sortperm(pts)
    pts, cs = pts[p], cs[p]
    nseg = length(pts) - 1
    PTS[] += nseg
    lm = Vector{Float64}(undef, nseg)
    for j in 1:nseg
        L = pts[j+1] - pts[j]
        lm[j] = L > 0 ? -cs[j] + M.logseg(L, cs[j+1] - cs[j]) : -Inf
    end
    mx = maximum(lm)
    u = rand(rng) * sum(x -> exp(x - mx), lm)
    acc = 0.0
    j = nseg
    for q in 1:nseg
        acc += exp(lm[q] - mx)
        if u <= acc
            j = q
            break
        end
    end
    L = pts[j+1] - pts[j]
    g = (cs[j+1] - cs[j]) / L
    return pts[j] + M.trexp_offset(rng, g, L)
end
const PTS = Ref(0)

function report(tag, rec, secs, nsw)
    n = length(rec)
    col(i) = [r[i] for r in rec]
    q(x, p) = quantile(x, p)
    println(@sprintf("\n[%s] %d line moves in %d sweeps, %.2f s/sweep, %.2f µs/move (line sampler %.0f %% of time)",
                     tag, n, nsw, secs / nsw, 1e6 * secs / n, 100 * TLINE[] / secs))
    for (i, name) in enumerate(("evaluations", "primary bps", "entries", "lines", "nodes", "cells", "secondary", "steps", "ns/line"))
        x = col(i)
        println(@sprintf("  %-12s mean %7.2f  median %5.0f  p90 %6.0f  p99 %6.0f  max %6d", name, mean(x), q(x, 0.5), q(x, 0.9), q(x, 0.99), maximum(x)))
    end
    println(@sprintf("  segments per move (mean) %.2f", PTS[] / n))
end

function main()
    data = length(ARGS) > 0 ? ARGS[1] : "data/processed/model.h5"
    N = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 40
    nsw = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 3
    kind = length(ARGS) > 3 ? ARGS[4] : "local"
    cfg = M.load_config("configs/flow.toml"); cfg.verbose = false
    cfg.data = data; cfg.levels = [(10, 24), (N, 24)]; cfg.init = "heuristic"; cfg.flow_theta = "haar"
    cfg.out = "results/flow.h5"
    t = @elapsed (P, s, info) = M.setup(cfg; rng=MersenneTwister(1))
    E = P.E
    ℓ = 1
    lv = P.lv[ℓ+1]
    println(@sprintf("setup %.0f s: N = %d, lines %d, T = %d, coordinates %d, per step %d", t, E.sys.N, length(E.sys.line_s), lv.T, length(s.v), length(s.v) ÷ lv.T))
    LM = M.LocalModel(P, ℓ)
    wsets = [M.MoveWS(P) for _ in 1:2Threads.maxthreadid()]
    rng = MersenneTwister(3)
    FF = kind == "flow" ? M.FlowFrames(P, ℓ, s) : nothing
    sweep(s) = kind == "flow" ? M.local_sweep!(rng, P, LM, s; wsets=wsets, families=M.flow_families(E)[1], extra=4.0 * M.flow_families(E)[2], frames=FF)[1] :
                                M.local_sweep!(rng, P, LM, s; wsets=wsets)[1]
    for _ in 1:2                       # warm up (compilation, first relaxation)
        s = sweep(s)
    end
    empty!(REC); TLINE[] = 0.0; PTS[] = 0; NEVAL[] = 0
    secs = @elapsed for _ in 1:nsw
        s = sweep(s)
    end
    report("$kind, $(E.sys.N) nodes daily", REC, secs, nsw)
    # move-size histogram in entries
    ent = [r[3] for r in REC]
    for (a, b) in ((1, 2), (3, 8), (9, 32), (33, 128), (129, 10^9))
        sel = [a <= r[3] <= b for r in REC]
        tt = sum(r[9] for r in REC); ww = sum(r[1] * (r[3] + r[4] + r[5] + r[6] + r[7]) for r in REC)
        println(@sprintf("  entries %4d–%-10d: %5.1f %% of moves, %5.1f %% of line-sampler time, %5.1f %% of eval work (evals × touched items), mean evals %5.1f, mean bps %6.1f",
                         a, b, 100 * count(sel) / length(ent), 100 * sum(r[9] for r in REC[sel]; init=0) / tt,
                         100 * sum(r[1] * (r[3] + r[4] + r[5] + r[6] + r[7]) for r in REC[sel]; init=0) / ww,
                         mean(r[1] for r in REC[sel]), mean(r[2] for r in REC[sel])))
    end
end

main()
