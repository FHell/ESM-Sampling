# Time-axis experiments (docs/flow_hierarchy.md §12), 40 nodes, SoC + renewables as load,
# capacities from the daily LP, one weather draw, local kernel:
#  E1  autocorrelation times of coarse (block-mean) and intra-block observables in
#      equilibrated chains at 24, 6, 3 h (own-LP start, burn-in, recorded sweeps);
#  E2  prolongation from the equilibrated daily state to 6 h and 3 h: block-constant vs
#      T2 (shares and excess kept, dispatch from the fine requirement): excess after n sweeps;
#  E3  principal components of the intra-block details (gas dispatch, state of charge,
#      angles) in the equilibrated fine chains.
# Usage: nice -n 19 julia -t 4 --project=. scripts/exp_time.jl [burn] [record]
using ESMSampling, Random, Statistics, Printf
const M = ESMSampling

function main()
    burn = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 30
    nrec = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 60
    cfg = M.load_config("configs/haar_poc.toml"); cfg.verbose = false
    cfg.storage = "soc"; cfg.renewables = "load"
    cfg.levels = [(10, 24), (40, 24), (40, 6), (40, 3)]
    P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    E = P.E
    rng = MersenneTwister(3)
    wsets = [M.MoveWS(P) for _ in 1:Threads.maxthreadid()]
    gasf = E.gens[findfirst(g -> g.carrier == "gas", E.gens)].vf
    socf = [s.fe for s in E.stors]
    ref = Dict{Int,Tuple{Float64,Int}}()          # h → (C*_op, n_pinned)
    eq = Dict{Int,M.State}()
    for h in (24, 6, 3)
        ℓ = findfirst(l -> l == (E.sys.N, h), P.levels) - 1
        lv, lk = P.lv[ℓ+1], P.lk[ℓ+1]
        s0 = M.restrict_state(P, sL, ℓ)
        Xn = M.weather_nodes(P.W, ℓ, s0.xs[end])
        Wd = [map(v -> M.datamap(E.wspecs[f], v), Xn[f]) for f in eachindex(Xn)]
        tlp = @elapsed lp = M.solve_lp(E, lk, lv, Wd; kfixed=info.lp.k)
        ref[h] = (lp.objective - lp.capex, lp.n_v_total - lp.n_v_free)
        center = M.interior_center(E, lv)
        s = copy(s0); s.v = M.to_modes(lv, E.vspecs, [0.98 .* lp.v[f] .+ 0.02 .* center[f] for f in eachindex(center)])
        M.evaluate!(P, s)
        LM = M.LocalModel(P, ℓ)
        ex(x) = (x - ref[h][1]) / (ref[h][2] / P.βv)
        t = @elapsed for _ in 1:burn
            s, _ = M.local_sweep!(rng, P, LM, s; wsets=wsets)
        end
        r = 24 ÷ h
        # observables: op; network gas dispatch mean over day d (coarse) and its intra-day contrast
        # (first vs second half; detail); network SoC at a day boundary vs mid-day deviation
        days = 10:30:340
        obs = Dict(k => Float64[] for k in ("op", "gas_mean", "gas_detail", "soc_boundary", "soc_mid"))
        states = M.State[]
        t2 = @elapsed for it in 1:nrec
            s, _ = M.local_sweep!(rng, P, LM, s; wsets=wsets)
            push!(obs["op"], s.parts.op)
            Y = Matrix(M.nodevals(s.v, lv, E.vspecs, gasf))
            gm = [mean(Y[:, (d-1)*r+1:d*r]) for d in days]
            gd = r > 1 ? [mean(Y[:, (d-1)*r+1:(d-1)*r+r÷2]) - mean(Y[:, (d-1)*r+r÷2+1:d*r]) for d in days] : zeros(length(days))
            Ye = sum(vec(sum(Matrix(M.nodevals(s.v, lv, E.vspecs, f)); dims=1)) for f in socf)   # network total per instant
            push!(obs["gas_mean"], mean(gm)); push!(obs["gas_detail"], mean(gd))
            push!(obs["soc_boundary"], mean(Ye[(d-1)*r+1] for d in days))
            push!(obs["soc_mid"], r > 1 ? mean(Ye[(d-1)*r+r÷2+1] - 0.5 * (Ye[(d-1)*r+1] + Ye[mod(d*r, lv.T)+1]) for d in days) : 0.0)
            it % 5 == 0 && push!(states, copy(s))
        end
        τint(x) = length(x) / max(M.ess(x), 1e-9)
        println(@sprintf("\n[E1] h = %2d h (r = %d per day, T = %d): LP %.0f s, burn %d sweeps %.0f s, record %d sweeps %.1f s/sweep; excess/pred at end %.2f",
                         h, r, lv.T, tlp, burn, t, nrec, t2 / nrec, ex(s.parts.op)))
        for k in ("op", "gas_mean", "gas_detail", "soc_boundary", "soc_mid")
            r == 1 && k in ("gas_detail", "soc_mid") && continue
            println(@sprintf("     τ_int(%-12s) = %6.1f sweeps", k, τint(obs[k])))
        end
        if r > 1
            Xg = reduce(hcat, [M.block_details(P, x, gasf, r) for x in states])
            Xe = reduce(hcat, [M.block_details(P, x, f, r) for x in states for f in socf])
            Xθ = reduce(hcat, [M.block_details(P, x, E.fθ, r) for x in states])
            println("[E3] explained by 1..4 intra-block components: gas ", round.(M.explained(Xg); digits=3),
                    "  SoC ", round.(M.explained(Xe); digits=3), "  angles ", round.(M.explained(Xθ); digits=3))
        end
        eq[h] = s
        flush(stdout)
    end
    # E2: prolongation from the equilibrated daily state
    sc = eq[24]
    for h in (6, 3)
        ℓ = findfirst(l -> l == (E.sys.N, h), P.levels) - 1
        LM = M.LocalModel(P, ℓ)
        ex(x) = (x - ref[h][1]) / (ref[h][2] / P.βv)
        # from 24 h to h: chain through the intermediate level for 3 h
        for (name, rule) in (("block-constant", M.prolong_constant), ("T2", M.prolong_t2))
            x = sc
            for l in 2:ℓ
                x = rule(P, x, M.restrict_state(P, sL, l))
            end
            vals = [ex(x.parts.op)]
            for it in 1:25
                x, _ = M.local_sweep!(rng, P, LM, x; wsets=wsets)
                it in (1, 5, 10, 25) && push!(vals, ex(x.parts.op))
            end
            println(@sprintf("[E2] 24 h → %d h, %-14s excess/pred at 0, 1, 5, 10, 25 sweeps: %s", h, name, join([@sprintf("%.2f", v) for v in vals], "  ")))
            flush(stdout)
        end
    end
end
main()
