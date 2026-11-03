# Several finest-level chains from dispersed starts and their convergence diagnostics
# (docs/validation_plan.md Part A). Used by `run` when `nchains > 0`.
#
# * Starts: capacities scaled (low ×0.5, high ×2, clamped to the boxes) or as set up
#   (heuristic or LP); with fine weather moves every chain also starts from its own
#   exact weather draw (without them the weather is the same in all chains, since each
#   chain would otherwise sample a different conditional).
# * Every chain has its own sampler (kernels, adaptive modes, RNG); the learned bias is
#   shared, refitted from the pooled scores during the warmup and frozen after it
#   (diminishing adaptation: the measurement window has a fixed target).
# * Diagnostics per monitored quantity: rank-normalized split-R̂, bulk and tail ESS,
#   MCSE/sd, on the window after the warmup (the configured warmup, or the detected one
#   if later).

"Names of the monitored quantities (`monitor_quantities`)."
function monitor_names(P::Problem)
    E = P.E
    cs = unique(E.sys.country)
    names = copy(quantity_names(P))
    for f in E.kspecs, c in cs
        push!(names, "$(f.name)_$c")
    end
    append!(names, ["line_loading_mean", "H2_soc_mean", "price_loadweighted", "price_mean"])
    return names
end

"Monitored quantities of a state: `quantities` plus per-country capacities, line loading, H₂ SoC, prices."
function monitor_quantities(P::Problem, s::State; LM=nothing)
    E = P.E
    sys = E.sys
    q = quantities(P, s)
    lk = P.lk[s.ℓ+1]
    cs = unique(sys.country)
    for (f, sp) in enumerate(E.kspecs)
        kn = vec(nodevals(s.k, lk, E.kspecs, f))
        bus = sp.B.support
        for c in cs
            push!(q, sum((kn[i] for i in eachindex(bus) if sys.country[bus[i]] == c); init=0.0))
        end
    end
    LM = LM === nothing ? LocalModel(P, s.ℓ) : LM
    LC = LocalCost(P, LM, s)
    push!(q, mean(abs.(LC.F) ./ sys.line_s))
    h2 = findfirst(st -> st.carrier == "H2", E.stors)
    push!(q, h2 === nothing ? NaN : mean(clamp.(LC.Y[E.stors[h2].fe], 0.0, 1.0)))
    Xn = weather_nodes(P.W, s.ℓ, s.xs[end])
    pr = [x > 0 ? E.c_shed : (x < 0 ? -E.c_dump : 0.0) for x in LC.S]   # cells = nodes at the complete level
    pc = pr[LM.cell, :]
    d = max.(Xn[1], 0.0)
    push!(q, sum(pc .* d) / max(sum(d), 1e-9))
    push!(q, mean(pc))
    return q
end

"Capacities of `s` scaled by `a` at node level and clamped to the boxes (complete capacity fields)."
function scale_capacities(P::Problem, s::State, a::Float64)
    E = P.E
    s2 = copy(s)
    lk = P.lk[s.ℓ+1]
    for (f, sp) in enumerate(E.kspecs)
        kn = vec(nodevals(s.k, lk, E.kspecs, f))
        kn = clamp.(a .* kn, sp.lo, sp.hi)
        fieldview(s2.k, lk, E.kspecs, f) .= cols(sp.B, lk.m[f])' * reshape(kn, :, 1)
    end
    return evaluate!(P, s2)
end

"Starting state of chain c from the set-up state `sL`."
function chain_start(rng, P::Problem, sL::State, kind::String, newweather::Bool)
    s = kind == "low" ? scale_capacities(P, sL, 0.5) :
        kind == "high" ? scale_capacities(P, sL, 2.0) : copy(sL)
    if newweather
        m, z0, ws = sample_weather(rng, P.W, s.ℓ)
        s = make_state(P, s.ℓ, m, z0, ws, s.k, s.v)
    end
    return s
end

"""
    run_chains(cfg) -> results

The multi-chain mode of the driver (`nchains > 0`): level-0 warmup for the bias (α = 1),
then `nchains` finest-level chains for `mlda_warmup + nsamples` sweeps each, diagnostics,
HDF5 output (`chains/<k>/Q`, `convergence/…`).
"""
function run_chains(cfg::Config)
    rng = MersenneTwister(cfg.seed)
    P, sL, info = setup(cfg; rng=rng)
    L = nlevels(P) - 1
    t0 = time()
    if P.use_bias                                  # level-0 warmup and bias learning, as in `run`
        S0 = new_sampler(P, sL, cfg)
        s0 = predescend(P, restrict_state(P, sL, 0), cfg)
        logmsg(cfg, "level-0 warmup and bias learning ($(cfg.warmup0) its)")
        s0, _, _, _ = run_level0!(rng, P, s0, S0, cfg.warmup0, cfg; learn=true, label="warmup")
        P.bias.active = false
    end
    K = cfg.nchains
    kinds = [cfg.chain_starts[mod1(c, length(cfg.chain_starts))] for c in 1:K]
    newweather = cfg.fine_weather > 0
    newweather || logmsg(cfg, "fine_weather = 0: all chains share the weather of the set-up state")
    chains = [chain_start(MersenneTwister(cfg.seed + 7 * c), P, sL, kinds[c], newweather) for c in 1:K]
    samplers = [new_sampler(P, chains[c], cfg) for c in 1:K]
    foreach(S -> (S.p_fine = 1.0), samplers)       # finest-level kernel only (MLDA shelved)
    rngs = [MersenneTwister(cfg.seed + 1000 * c) for c in 1:K]
    nwarm, nprod = cfg.mlda_warmup, cfg.nsamples
    online = P.use_bias && cfg.bias_online
    OB = OnlineBias(; every=cfg.bias_online_every, freeze=K * nwarm)
    PB = (P.use_bias && cfg.bias_post) ? PostBias() : nothing
    names = monitor_names(P)
    LM = LocalModel(P, L)
    traces = [Vector{Vector{Float64}}() for _ in 1:K]
    logmsg(cfg, "$K chains (starts: $(join(kinds, ", "))), $nwarm warmup + $nprod sweeps each")
    for it in 1:nwarm+nprod
        for c in 1:K
            chains[c] = level_step(rngs[c], P, chains[c], samplers[c], L)
            if online && it <= nwarm
                chains[c], upd = online_bias_step!(P, chains[c], OB; monotone=bias_monotone_args(P, cfg))
                upd && (chains = [evaluate!(P, x) for x in chains])   # a refit changes the target of every chain
            end
            q = monitor_quantities(P, chains[c]; LM=LM)
            push!(traces[c], q)
            (PB !== nothing && it > nwarm) && post_record!(PB, P, chains[c], q)
        end
        if it % max(1, (nwarm + nprod) ÷ 10) == 0
            tot = [tr[end][length(P.E.kspecs)+3] for tr in traces]
            logmsg(cfg, @sprintf("  sweep %4d/%d  total per chain %s  [%.0fs]", it, nwarm + nprod,
                                 join((@sprintf("%.4e", x) for x in tot), " "), time() - t0))
        end
    end
    return chains_report(cfg, P, names, traces, kinds, PB, nwarm, nprod, t0)
end

"""
    chains_report(cfg, P, names, traces, kinds, PB, nwarm, nprod, t0)

Convergence diagnostics of per-chain traces (`traces[c][sweep]` = monitored quantities),
the bias post-correction, and the HDF5 output of the multi-chain modes (CPU and device).
"""
function chains_report(cfg::Config, P::Problem, names, traces, kinds, PB, nwarm, nprod, t0)
    K = length(traces)
    # ---- diagnostics
    nq = length(names)
    qtr = [[[tr[t][q] for t in eachindex(tr)] for tr in traces] for q in 1:nq]
    good = [all(isfinite, reduce(vcat, qtr[q])) for q in 1:nq]
    wdet = warmup_end(qtr[good]; threshold=cfg.rhat_screen, every=max(5, (nwarm + nprod) ÷ 40))
    w0 = max(nwarm, wdet)
    w0 >= nwarm + nprod - 4 && (w0 = nwarm)        # no usable window: report the configured one
    summ = [good[q] ? convergence_summary([c[w0+1:end] for c in qtr[q]]; rhat_max=cfg.rhat_max,
                                          ess_min=cfg.ess_min, mcse_max=cfg.mcse_max) : nothing for q in 1:nq]
    passed = all(x -> x === nothing || x.pass, summ)
    worst = argmax([x === nothing ? -Inf : x.rhat for x in summ])
    logmsg(cfg, @sprintf("convergence: warmup detected %s (window from %d), max R̂ %.3f (%s), min bulk ESS %.0f, min tail ESS %.0f → %s",
                         wdet > 0 ? "at sweep $wdet" : "never", w0, summ[worst].rhat, names[worst],
                         minimum(x.ess_bulk for x in summ if x !== nothing), minimum(x.ess_tail for x in summ if x !== nothing),
                         passed ? "converged" : "not converged"))
    pc = (PB !== nothing && !isempty(PB.bused)) ? post_correct(P, PB; monotone=bias_monotone_args(P, cfg)) : nothing
    pc === nothing || logmsg(cfg, @sprintf("bias post-correction: weight ESS %.1f of %d samples", pc.ess, length(pc.w)))
    mkpath(dirname(cfg.out))
    h5open(cfg.out, "w") do f
        f["quantity_names"] = names
        for c in 1:K
            f["chains/$c/Q"] = reduce(hcat, traces[c])
            f["chains/$c/start"] = kinds[c]
        end
        sel = [x === nothing ? NaN : getfield(x, k) for x in summ, k in (:rhat, :ess_bulk, :ess_tail, :mcse_rel, :mean, :sd)]
        f["convergence/rhat"] = sel[:, 1]
        f["convergence/ess_bulk"] = sel[:, 2]
        f["convergence/ess_tail"] = sel[:, 3]
        f["convergence/mcse_rel"] = sel[:, 4]
        f["convergence/mean"] = sel[:, 5]
        f["convergence/sd"] = sel[:, 6]
        f["convergence/pass"] = [x === nothing ? -1 : Int(x.pass) for x in summ]
        f["convergence/warmup_detected"] = wdet
        f["convergence/window_start"] = w0
        f["convergence/converged"] = Int(passed)
        f["convergence/thresholds"] = [cfg.rhat_max, cfg.ess_min, cfg.mcse_max, cfg.rhat_screen]
        if pc !== nothing
            f["post/weights"] = pc.w; f["post/ess"] = pc.ess; f["post/Q_weighted"] = pc.Qw
        end
    end
    logmsg(cfg, @sprintf("wrote %s  [total %.0fs]", cfg.out, time() - t0))
    return (; names, traces, summ, wdet, w0, passed)
end
