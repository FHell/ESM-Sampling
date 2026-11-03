# R4 of scripts/exp_relax.jl on its own: operational LP at fixed capacities
# (40 nodes × daily), cold dual simplex, then warm-started dual simplex after the
# same +3σ network-mean load change for days 10–23.
using ESMSampling, Random, Printf, JuMP
const M = ESMSampling

function main()
    cfg = M.load_config("configs/haar_poc.toml"); cfg.verbose = false
    P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    E, W, lp = P.E, P.W, info.lp
    s1 = M.restrict_state(P, sL, 1)
    lw1 = W.lays[2]
    x0 = s1.xs[end]
    x1 = copy(x0)
    x1[[lw1.off[1] + (τ - 1) * lw1.m[1] + 1 for τ in 10:23]] .+= 3.0
    Wd(x) = (Xn = M.weather_nodes(W, 1, x); [map(v -> M.datamap(E.wspecs[f], v), Xn[f]) for f in eachindex(Xn)])
    Wd0, Wd1 = Wd(x0), Wd(x1)
    tcold = @elapsed (sol0, model, balc) = M.solve_lp(E, P.lk[2], P.lv[2], Wd0; kfixed=lp.k, return_model=true, solver="simplex")
    obj0 = objective_value(model)
    for n in axes(balc, 1), t in axes(balc, 2)
        Wd1[1][n, t] != Wd0[1][n, t] && set_normalized_rhs(balc[n, t], -Wd1[1][n, t])
    end
    twarm = @elapsed optimize!(model)
    obj1 = objective_value(model)
    println(@sprintf("cold dual simplex %.0f s (opex %.4e); warm re-solve after the load change %.1f s, status %s; LP Δopex %.4e",
                     tcold, obj0 - sol0.capex, twarm, termination_status(model), obj1 - obj0))
end
main()
