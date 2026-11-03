# Per-family and per-sweep timings of the local exact kernel (todo items 5–7).
# Usage (from the package directory):
#   nice -n 19 julia -t 1 --project=. scripts/bench_perf.jl fam   [data] [N] [extra]
#   nice -n 19 julia -t 4 --project=. scripts/bench_perf.jl sweep [data] [N] [nsweeps]
#   nice -n 19 julia -t 4 --project=. scripts/bench_perf.jl regress [data] [N] [outfile]
# `fam`: µs per sampled move of each family (local_sweep! with one family, atoms off,
# minus the final evaluate!), plus the move statistics (entries, lines) per family.
# `sweep`: s/sweep of the default local sweep, the flow sweep, the storage multiscale
# sweep and the block sweep. `regress`: fixed-seed sweeps, writes s.v (bitwise checks).
using ESMSampling, Random, Statistics, Printf, Serialization
const M = ESMSampling

function setup_bench(data, N)
    cfg = M.load_config("configs/flow.toml"); cfg.verbose = false
    cfg.data = data; cfg.levels = [(10, 24), (N, 24)]; cfg.init = "heuristic"; cfg.flow_theta = "haar"
    cfg.out = "results/bench.h5"
    t = @elapsed (P, s, info) = M.setup(cfg; rng=MersenneTwister(1))
    E = P.E
    lv = P.lv[2]
    println(@sprintf("setup %.0f s: N = %d, lines %d, T = %d, coordinates %d", t, E.sys.N, length(E.sys.line_s), lv.T, length(s.v)))
    LM = M.LocalModel(P, 1)
    FF = M.FlowFrames(P, 1, s)
    return P, s, LM, FF
end


function bench_fam(P, s, LM, FF, extra)
    E = P.E
    wsets = [M.MoveWS(P) for _ in 1:2Threads.maxthreadid()]
    rng = MersenneTwister(3)
    # relax a little first (heuristic start), compile
    for _ in 1:2
        s, _ = M.local_sweep!(rng, P, LM, s; wsets=wsets, families=M.flow_families(E)[1], extra=1.0, frames=FF)
    end
    LC = M.LocalCost(P, LM, s)
    M.evaluate!(P, s)
    tev = minimum(@elapsed(M.evaluate!(P, s)) for _ in 1:3)
    bev = @allocated M.evaluate!(P, s)
    fams = Symbol[]
    for f in vcat(first.(M.flow_families(E)[1]), [:xpair, :xloc, :xlocc])
        f in fams || push!(fams, f)
    end
    println(@sprintf("%-10s %10s %10s %12s %10s", "family", "moves", "µs/move", "alloc B/mv", "accept"))
    res = Dict{Symbol,Float64}()
    for fam in vcat([:atom], fams)
        fl = fam == :atom ? [(:unit, 1.0)] : [(fam, 1.0)]
        ex = fam == :atom ? 0.0 : extra
        at = fam == :atom
        try
            M.local_sweep!(rng, P, LM, s; wsets=wsets, families=fl, extra=ex * 0.05, atoms=at, frames=FF, LC=LC)   # compile
        catch err
            println(@sprintf("%-10s  (not available: %s)", fam, sprint(showerror, err)[1:min(end, 60)]))
            continue
        end
        GC.gc()
        local st
        b = @allocated (t = @elapsed ((_, st) = M.local_sweep!(rng, P, LM, s; wsets=wsets, families=fl, extra=ex, atoms=at, frames=FF, LC=LC)))
        n = sum(st[:, 1])
        acc = sum(st[:, 2]) / max(n, 1)
        n == 0 && (println(@sprintf("%-10s %10d", fam, 0)); continue)
        us = 1e6 * (t - tev) / n
        res[fam] = us
        println(@sprintf("%-10s %10d %10.2f %12.0f %10.3f", fam, n, us, (b - bev) / n, acc))
    end
    return res
end

function bench_sweep(P, s, LM, FF, nsw)
    E = P.E
    wsets = [M.MoveWS(P) for _ in 1:2Threads.maxthreadid()]
    pools = [M.BlockPool(P, 140) for _ in 1:Threads.maxthreadid()]
    rng = MersenneTwister(4)
    fams, mult = M.flow_families(E)
    s, _ = M.local_sweep!(rng, P, LM, s; wsets=wsets)                       # compile + relax
    s, _ = M.flow_sweep!(rng, P, LM, FF, s; wsets=wsets, pools=pools)
    s, _ = M.storage_ms_sweep!(rng, P, LM, s; pools=pools)
    s, _ = M.block_sweep!(rng, P, LM, s; pools=pools)
    GC.gc()
    for (name, f) in (("local_sweep! (default families)", x -> M.local_sweep!(rng, P, LM, x; wsets=wsets)[1]),
                      ("local_sweep! (flow families)", x -> M.local_sweep!(rng, P, LM, x; wsets=wsets, extra=4.0 * mult, families=fams, frames=FF)[1]),
                      ("storage_ms_sweep!", x -> M.storage_ms_sweep!(rng, P, LM, x; pools=pools)[1]),
                      ("block_sweep!", x -> M.block_sweep!(rng, P, LM, x; pools=pools)[1]))
        ts = Float64[]
        local bytes = 0
        for _ in 1:nsw
            bytes = @allocated (t = @elapsed (s = f(s)))
            push!(ts, t)
        end
        println(@sprintf("%-34s  %8.3f s/sweep (min %8.3f)  %8.1f MB alloc/sweep", name, mean(ts), minimum(ts), bytes / 2^20))
    end
end

function regress(P, s, LM, FF, out)
    E = P.E
    wsets = [M.MoveWS(P) for _ in 1:2Threads.maxthreadid()]
    pools = [M.BlockPool(P, 140) for _ in 1:Threads.maxthreadid()]
    plain = [(:unit, 0.3), (:pair, 0.25), (:xpair, 0.15), (:potential, 0.1), (:spair, 0.3)]
    rng = MersenneTwister(5)
    # the flow families of the main copy, without those whose arithmetic order changed (tcut: node order of
    # prep_tsparse!; cell, cellpair: level probabilities)
    noTsparse = [(:unit, 0.3), (:pair, 0.25), (:xpair, 0.15), (:potential, 0.1), (:spair, 0.3),
                 (:tpot, 0.1), (:tmode, 0.05), (:tline, 0.05), (:tmode0, 0.05), (:tline0, 0.05), (:tsoc, 0.1)]
    s, _ = M.local_sweep!(rng, P, LM, s; wsets=wsets, families=plain, extra=2.0)
    v1 = copy(s.v)
    s, _ = M.local_sweep!(rng, P, LM, s; wsets=wsets, families=noTsparse, extra=2.0, frames=FF)
    v2 = copy(s.v)
    s, _ = M.storage_ms_sweep!(rng, P, LM, s; pools=pools)
    v3 = copy(s.v)
    serialize(out, (v1, v2, v3))
    println("wrote ", out, "  op = ", s.parts.op)
end

function main()
    mode = ARGS[1]
    data = length(ARGS) > 1 ? ARGS[2] : "data/processed/model.h5"
    N = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 40
    P, s, LM, FF = setup_bench(data, N)
    if mode == "fam"
        bench_fam(P, s, LM, FF, length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 1.0)
    elseif mode == "sweep"
        bench_sweep(P, s, LM, FF, length(ARGS) > 3 ? parse(Int, ARGS[4]) : 3)
    else
        regress(P, s, LM, FF, ARGS[4])
    end
end

main()
