# Relaxation from a far start (daily LP solution of the setup, restricted to 40 nodes × 24 h —
# the "prolonged" start of scripts/exp_local.jl) with (a) local sweeps only and (b) local
# sweeps + temporal block sweeps, for equal wall time. Prints the cost parts against the
# level's own operational LP (fixed k) to show where the excess sits.
# Usage: julia -t 8 --project=. scripts/exp_farstart.jl [seconds] [hours]
using ESMSampling, Random, Statistics, Printf
const M = ESMSampling

function main()
    secs = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 120.0
    h = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 24
    cfg = M.load_config(get(ENV, "CONFIG", "configs/haar_poc.toml")); cfg.verbose = false
    haskey(ENV, "CONFIG") || (cfg.levels = [(10, 24), (40, 24), (40, 12), (40, 6)])
    haskey(ENV, "STORAGE") && (cfg.storage = ENV["STORAGE"])
    haskey(ENV, "RENEW") && (cfg.renewables = ENV["RENEW"])
    haskey(ENV, "MEASURE") && (cfg.measure = ENV["MEASURE"])
    haskey(ENV, "BASIS") && (cfg.basis = ENV["BASIS"])
    haskey(ENV, "INIT") && (cfg.init = ENV["INIT"])
    variants = Symbol.(split(get(ENV, "VARIANTS", "local,local_block"), ","))
    println("storage form: $(cfg.storage), renewables: $(cfg.renewables), measure: $(cfg.measure)")
    P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    E = P.E
    ℓ = findfirst(l -> l == (E.sys.N, h), P.levels) - 1
    lv, lk = P.lv[ℓ+1], P.lk[ℓ+1]
    s_pro = M.restrict_state(P, sL, ℓ)
    Xn = M.weather_nodes(P.W, ℓ, s_pro.xs[end])
    Wd = [map(v -> M.datamap(E.wspecs[f], v), Xn[f]) for f in eachindex(Xn)]
    if cfg.init == "heuristic"          # large models: no reference LP; absolute cost trajectories only
        Cop, npin = NaN, 1
        s_lp = s_pro
    else
        lp = M.solve_lp(E, lk, lv, Wd; kfixed=info.lp.k)
        s_lp = copy(s_pro); s_lp.v = M.to_modes(lv, E.vspecs, lp.v); M.evaluate!(P, s_lp)
        npin = lp.n_v_total - lp.n_v_free
        Cop = lp.objective - lp.capex
    end
    fmt(p) = @sprintf("op %.4e gen %.4e stor %.3e shed %.3e MWh dump %.3e MWh pen %.3e", p.op, p.gen, p.storage, p.shed, p.dump, p.penalty)
    caps(s) = join([@sprintf("%s %.0f", String(E.kspecs[f].name)[3:end], sum(x)) for (f, x) in enumerate(M.knodes(P, ℓ, s.k))], " ")
    println("LP (fixed k) at $(h) h: ", fmt(s_lp.parts), "  (objective op $(@sprintf("%.4e", Cop)), pinned $npin)")
    println("far start:           ", fmt(s_pro.parts))
    println("capacities: setup state vs LP k (sum, MW): ", round(sum(sum.(M.knodes(P, ℓ, s_pro.k)))), " vs ", round(sum(sum.(info.lp.k))))
    rng = MersenneTwister(4)
    LM = M.LocalModel(P, ℓ)
    wsets = [M.MoveWS(P) for _ in 1:Threads.maxthreadid()]
    pools = [M.BlockPool(P, 140) for _ in 1:Threads.maxthreadid()]
    FF = E.flow === nothing ? nothing : M.FlowFrames(P, ℓ, s_pro)
    AM = M.AdaptiveModes(P, ℓ)
    AMn = M.AdaptiveModes(P, ℓ)
    for variant in variants
        s = copy(s_pro)
        t0 = time(); it = 0; nextp = 0.0
        while time() - t0 < secs
            it += 1
            fams = variant == :local_tpot ? vcat([(:unit, 0.3), (:pair, 0.25), (:xpair, 0.15), (:potential, 0.1), (:cellpair, 0.1), (:cell, 0.1)],
                                                 M.issoc(E) ? [(:spair, 0.3), (:tsoc, 0.3)] : Tuple{Symbol,Float64}[], [(:tpot, 0.3)]) : nothing
            variant in (:flow, :adaptive, :adaptive_nl) || ((s, _) = M.local_sweep!(rng, P, LM, s; wsets=wsets, families=fams))
            variant == :adaptive_nl && ((s, _, _) = M.adaptive_sweep!(rng, P, LM, FF, AMn, s; wsets=wsets, pools=pools, learned=false))
            variant == :adaptive && ((s, _, _) = M.adaptive_sweep!(rng, P, LM, FF, AM, s; wsets=wsets, pools=pools))
            variant == :local_block && ((s, _) = M.block_sweep!(rng, P, LM, s; pools=pools))
            variant == :local_k && ((s, _) = M.kscale_sweep!(rng, P, LM, s))
            variant == :flow && ((s, _, _) = M.flow_sweep!(rng, P, LM, FF, s; wsets=wsets, pools=pools))
            if time() - t0 >= nextp
                println(@sprintf("  %-12s %4d its %6.0f s  excess/pred %8.2f  ", variant, it, time() - t0, (s.parts.op - Cop) / (npin / P.βv)), fmt(s.parts))
                flush(stdout)
                nextp += secs / 8
            end
        end
        println(@sprintf("  %-12s final %4d its: excess/pred %.2f", variant, it, (s.parts.op - Cop) / (npin / P.βv)), "  capacities [MW]: ", caps(s))
    end
end
main()
