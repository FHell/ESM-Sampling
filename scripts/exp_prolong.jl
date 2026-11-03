# Prolongation rules from an equilibrated daily state to 6 h (docs/flow_hierarchy.md §12):
# block-constant, T2 (shares and excess kept), T2 clamped to the unit boxes. Excess over the
# 6 h LP (fixed k) in units of the equipartition prediction, after 0, 1, 5, 10, 25 sweeps.
using ESMSampling, Random, Statistics, Printf
const M = ESMSampling
cfg = M.load_config("configs/haar_poc.toml"); cfg.verbose = false
cfg.storage = "soc"; cfg.renewables = "load"; cfg.levels = [(10, 24), (40, 24), (40, 6)]
P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
E = P.E
rng = MersenneTwister(3)
wsets = [M.MoveWS(P) for _ in 1:Threads.maxthreadid()]
lv, lk = P.lv[3], P.lk[3]
Xn = M.weather_nodes(P.W, 2, sL.xs[end])
Wd = [map(v -> M.datamap(E.wspecs[f], v), Xn[f]) for f in eachindex(Xn)]
lp = M.solve_lp(E, lk, lv, Wd; kfixed=info.lp.k)
Cop, npin = lp.objective - lp.capex, lp.n_v_total - lp.n_v_free
ex(x) = (x - Cop) / (npin / P.βv)
sc = M.restrict_state(P, sL, 1)
LMc = M.LocalModel(P, 1)
for _ in 1:30
    global sc, _ = M.local_sweep!(rng, P, LMc, sc; wsets=wsets)
end
LM = M.LocalModel(P, 2)
for (name, rule) in (("block-constant", M.prolong_constant), ("T2", M.prolong_t2), ("T2 clamped", M.prolong_t2c))
    x = rule(P, sc, sL)
    vals = [ex(x.parts.op)]
    parts0 = x.parts
    for it in 1:25
        x, _ = M.local_sweep!(rng, P, LM, x; wsets=wsets)
        it in (1, 5, 10, 25) && push!(vals, ex(x.parts.op))
    end
    println(@sprintf("%-14s excess/pred at 0, 1, 5, 10, 25 sweeps: %s   (start: shed %.2e MWh, dump %.2e MWh, penalty %.2e €)",
                     name, join([@sprintf("%.1f", v) for v in vals], "  "), parts0.shed, parts0.dump, parts0.penalty))
    flush(stdout)
end
