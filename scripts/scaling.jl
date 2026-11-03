# Scaling check: how the LP and the sampler scale with the time resolution of
# the finest level (40 nodes). The sampler needs the LP only at the first
# spatially complete level (here daily); finer levels are reached by MLDA.
#
# Usage: julia -t 8 --project=. scripts/scaling.jl [lp|sampler|all]
using ESMSampling, Random, Statistics, Printf, HDF5, JuMP
const M = ESMSampling

part = length(ARGS) > 0 ? ARGS[1] : "all"
mkpath("results")

if part in ("lp", "all")
    println("== LP at 40 nodes (perfect foresight, one weather year) ==")
    sys = M.load_system("data/processed/model.h5")
    E = M.build_energy_model(sys)
    rng = MersenneTwister(1)
    for h in (24, 12, 6, 3)
        levels = [(4, 24), (40, h)]
        W = M.fit_weather_model(rng, E, levels; K=4)
        m, z0, ws = M.sample_weather(rng, W, 1)
        x = M.weather_at_level(W, 1, m, z0, ws)
        Xn = M.weather_nodes(W, 1, x)
        Wd = [map(v -> M.datamap(E.wspecs[f], v), Xn[f]) for f in eachindex(Xn)]
        lk, lv = M.Layout(E.kspecs, 40, h), M.Layout(E.vspecs, 40, h)
        GC.gc()
        t = @elapsed ok = try
            M.solve_lp(E, lk, lv, Wd; time_limit=2400)
            true
        catch err
            println("   ", sprint(showerror, err)[1:min(end, 120)])
            false
        end
        println(@sprintf("  h = %2d h: %7d steps, ~%.2e LP variables, %s in %.0f s, max RSS %.1f GB", h, lv.T,
                         lv.T * (sum(M.nsupport(s.B) for s in E.vspecs) + 2 * sys.N), ok ? "solved" : "NOT solved",
                         t, Sys.maxrss() / 1e9))
        flush(stdout)
        ok || break
    end
end

if part in ("sampler", "all")
    println("== sampler with finest level at h_L (LP only at the daily, spatially complete level) ==")
    for hL in (6, 3, 1)
        cfg = M.Config(; out="results/scaling_h$(hL).h5", levels=[(4, 24), (12, 24), (40, 24), (40, hL)],
                       warmup0=40, mass_windows=[20], bias_start=10, bias_every=5, n0=20, J=[3, 2, 2],
                       mlda_warmup=10, nsamples=30, alpha0=false, verbose=false)
        t0 = time()
        P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
        tsetup = time() - t0
        tev = minimum(@elapsed(M.evaluate!(P, sL)) for _ in 1:3)
        t1 = time()
        res = M.run(cfg)
        trun = time() - t1
        acc = [st.nacc / max(st.nprop, 1) for st in res.S.stats]
        inf = [st.ninfeasible / max(st.nprop, 1) for st in res.S.stats]
        println(@sprintf("  h_L = %d h: finest dim %.2e, setup %.0f s, finest evaluation %.3f s, run %.0f s, MLDA acceptance per level %s (infeasible %s), max RSS %.1f GB",
                         hL, length(sL.k) + length(sL.v), tsetup, tev, trun, string(round.(acc; digits=3)),
                         string(round.(inf; digits=3)), Sys.maxrss() / 1e9))
        flush(stdout)
    end
end
