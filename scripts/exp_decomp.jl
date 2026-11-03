# Recursive decomposed LP start vs the full reference LP (docs/flow_hierarchy.md §14).
# 40 nodes, SoC + renewables as load. For each finest step and region cap: LP time and the
# total cost (capex + operation, evaluated in the sampler's model) of the start, against
# the full LP start (same weather), plus the cost after a few local sweeps.
# Usage: nice -n 19 julia -t 4 --project=. scripts/exp_decomp.jl
using ESMSampling, Random, Statistics, Printf
const M = ESMSampling

function run_case(levels, init; cap=50)
    cfg = M.load_config("configs/haar_poc.toml"); cfg.verbose = false
    cfg.storage = "soc"; cfg.renewables = "load"; cfg.levels = levels
    cfg.init = init; cfg.decomp_cap = cap
    t = @elapsed P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    L = M.nlevels(P) - 1
    LM = M.LocalModel(P, L)
    rng = MersenneTwister(5)
    wsets = [M.MoveWS(P) for _ in 1:Threads.maxthreadid()]
    s = sL
    vals = [s.parts.capex + s.parts.op]
    for it in 1:10
        s, _ = M.local_sweep!(rng, P, LM, s; wsets=wsets)
        it in (1, 10) && push!(vals, s.parts.op + s.parts.capex)
    end
    caps = [sum(x) for x in M.knodes(P, L, sL.k)]
    return t, vals, info, caps, P
end

for levels in ([(10, 24), (40, 24)], [(10, 24), (40, 24), (40, 6)])
    h = levels[end][2]
    tl, vl, il, cl, P = run_case(levels, "lp")
    println(@sprintf("\nfinest %d h — full LP start: setup %.0f s, LP objective %.4e; total cost of the start %.4e, after 1 / 10 sweeps %.4e / %.4e",
                     h, tl, il.lp.objective, vl...))
    println("   capacities [GW]: ", join([@sprintf("%s %.1f", String(P.E.kspecs[f].name)[3:end], cl[f] / 1e3) for f in eachindex(cl)], " "))
    for cap in (50, 4)
        td, vd, id, cd, _ = run_case(levels, "decomposed"; cap=cap)
        println(@sprintf("   decomposed (cap %2d): setup %.0f s; total cost of the start %.4e (%+.1f %% vs LP objective), after 1 / 10 sweeps %.4e / %.4e",
                         cap, td, vd[1], 100 * (vd[1] / il.lp.objective - 1), vd[2], vd[3]))
        println("   capacities [GW]: ", join([@sprintf("%s %.1f", String(P.E.kspecs[f].name)[3:end], cd[f] / 1e3) for f in eachindex(cd)], " "))
        flush(stdout)
    end
end
