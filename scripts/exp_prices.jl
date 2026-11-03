# Sampled nodal prices vs LP duals (40 nodes × 24 h, capacities fixed at the LP, one weather
# draw). The chain starts from the level's own LP solution, runs `nsw` local sweeps and
# accumulates the sampled price E[c_shed 1(r>0) − c_dump 1(r<0)] over the last 3/4.
# Usage: julia -t 8 --project=. scripts/exp_prices.jl [sweeps]
using ESMSampling, Random, Statistics, Printf
const M = ESMSampling

function main()
    nsw = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 200
    cfg = M.load_config("configs/haar_poc.toml"); cfg.verbose = false
    P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    E = P.E
    ℓ = 1
    lv, lk = P.lv[2], P.lk[2]
    s0 = M.restrict_state(P, sL, ℓ)
    Xn = M.weather_nodes(P.W, ℓ, s0.xs[end])
    Wd = [map(v -> M.datamap(E.wspecs[f], v), Xn[f]) for f in eachindex(Xn)]
    lp = M.solve_lp(E, lk, lv, Wd; kfixed=info.lp.k)
    center = M.interior_center(E, lv)
    s = copy(s0); s.v = M.to_modes(lv, E.vspecs, [0.98 .* lp.v[f] .+ 0.02 .* center[f] for f in eachindex(center)]); M.evaluate!(P, s)
    λlp = lp.dF_dX[1] ./ lv.h                         # €/MWh per node and step
    LM = M.LocalModel(P, ℓ)
    wsets = [M.MoveWS(P) for _ in 1:Threads.maxthreadid()]
    acc = M.PriceAcc(E.sys.N, lv.T)
    rng = MersenneTwister(3)
    for it in 1:nsw
        LC = M.LocalCost(P, LM, s)
        s, _ = M.local_sweep!(rng, P, LM, s; wsets=wsets, LC=LC)
        it > nsw ÷ 4 && M.accumulate_prices!(acc, P, LM, LC)
    end
    λ = M.prices(acc)
    println(@sprintf("mean price: sampled %.1f €/MWh, LP %.1f €/MWh; corr(node×step) %.3f; corr(node means) %.3f; corr(daily system mean) %.3f",
                     mean(λ), mean(λlp), cor(vec(λ), vec(λlp)), cor(vec(mean(λ; dims=2)), vec(mean(λlp; dims=2))),
                     cor(vec(mean(λ; dims=1)), vec(mean(λlp; dims=1)))))
    q = [0.1, 0.5, 0.9, 0.99]
    println("quantiles sampled ", round.(quantile(vec(λ), q); digits=1), "  LP ", round.(quantile(vec(λlp), q); digits=1))
    # load-weighted average (what consumers pay)
    D = Wd[1]
    println(@sprintf("load-weighted mean: sampled %.1f, LP %.1f €/MWh", sum(λ .* D) / sum(D), sum(λlp .* D) / sum(D)))
end
main()
