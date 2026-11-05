# The CPU driver (`run`): several finest-level chains from dispersed starts and their
# convergence diagnostics (docs/system/sampler.md §9, docs/system/validation.md).
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
    run_chains(cfg; hook=nothing) -> results

The CPU driver (`run`): level-0 warmup for the bias (α = 1) and its lift to the
finest level, then `nchains` finest-level chains for `mlda_warmup + nsamples` sweeps each. During the warmup the
online bias is refitted from the pooled scores of all chains and, with `co2_cap > 0`,
the CO₂ price is tuned to the cap; both, and the learned quantities of the adaptive
fine kernel, are frozen for production. Production
records the monitored quantities, the post-correction samples and the sampled
prices. Output: diagnostics and traces (`chains_report`) plus prices, the learned
quantities and the calibration (`write_run_outputs`).

**Snapshots and resume** (`checkpoint` ≠ "", checkpoint.jl; docs/system/hpc.md §7): at
sweep boundaries the chain states are written every `checkpoint_minutes`, once after the
level-0 warmup, and when `stop_file` exists (then the run stops: `results.exitcode = 75`,
`results.states` the saved states). If the snapshot file exists at the start, the run
resumes from it: the phase and the sweep counter continue, the chains restart from the
saved states (`results.resume_states`), the random streams are re-seeded from the seed and
the restart count, and the kernels are rebuilt (fresh adaptation in the rest of the
warmup, none in production). Not restored: the bias learned in the level-0 warmup (a
resume after it lifts the set-up fit), the online-bias observations, the learned modes, and
the post-correction and price accumulators (they cover the production sweeps since the last
start). `hook(phase, it)` is called at every sweep boundary when snapshots are on (tests).
"""
function run_chains(cfg::Config; hook=nothing)
    C = Checkpointer(cfg; hook=hook)
    clock = StageClock(dirname(abspath(cfg.out)))      # stages.csv: wall and CPU time per stage (hpc/cost_table.jl)
    phase = Ref("setup")
    rng = MersenneTwister(cfg.seed)
    P, sL, info = setup(cfg; rng=rng)
    stage!(clock, "setup"); phase[] = "level-0 warmup"
    L = nlevels(P) - 1
    t0 = time()
    snap = read_snapshot(cfg.checkpoint, cfg, P)
    r = snap === nothing ? 0 : snap.restarts + 1          # resumes of this run so far
    C.restarts = r
    resume_states = nothing
    if snap !== nothing
        logmsg(cfg, "resuming from snapshot $(cfg.checkpoint) (restart $r, written ",
               Libc.strftime("%Y-%m-%d %H:%M:%S", snap.written), "): $(snap.where_)")
        snap.co2 === P.E.co2[] || set_co2_price!(P.E, snap.co2)
        copy!(rng, MersenneTwister(restart_seed(cfg, r)))
    end
    C.last = time()
    try
        if P.use_bias                                  # level-0 warmup and bias learning
            if snap === nothing || snap.phase == :level0
                S0 = new_sampler(P, sL, cfg)
                if snap === nothing
                    s0, it0 = predescend(P, restrict_state(P, sL, 0), cfg), 0
                else
                    s0, it0 = snap.states[1], snap.it
                    s0.ℓ == 0 || error("snapshot $(cfg.checkpoint): the level-0 state is at level $(s0.ℓ)")
                    resume_states = [copy(s0)]
                    snap.sigma_scale === nothing || (S0.σscale .= snap.sigma_scale)
                    if snap.gibbs !== nothing
                        S0.gibbs.wv .= snap.gibbs.wv; S0.gibbs.wz .= snap.gibbs.wz; S0.gibbs.wk .= snap.gibbs.wk
                    end
                end
                logmsg(cfg, "level-0 warmup and bias learning ($(cfg.warmup0) its", it0 > 0 ? ", from $it0)" : ")")
                level0_data = s -> (; states=[s], sigma_scale=S0.σscale, gibbs=(wv=S0.gibbs.wv, wz=S0.gibbs.wz, wk=S0.gibbs.wk))
                s0, _, _, _ = run_level0!(rng, P, s0, S0, cfg.warmup0, cfg; learn=true, label="warmup", start=it0,
                                          at_sweep=(it, s) -> boundary!(C, cfg, P, :level0, it, () -> level0_data(s)))
            else
                @warn "resumed after the level-0 warmup: the bias is the set-up fit, not the one learned in the warmup"
            end
            P.bias.active = false
            lift_bias!(P, cfg)                         # the fine chains evaluate and refit the bias at their level
            sL = evaluate!(P, sL)
        elseif snap !== nothing && snap.phase == :level0
            error("snapshot $(cfg.checkpoint) is in the level-0 warmup, but use_bias = false")
        end
        K = cfg.nchains
        @assert K >= 1 "nchains must be at least 1"
        nwarm, nprod = cfg.mlda_warmup, cfg.nsamples
        names = monitor_names(P)
        if snap !== nothing && snap.phase == :fine
            length(snap.states) == K || error("snapshot $(cfg.checkpoint) has $(length(snap.states)) chains, nchains = $K")
            kinds = length(snap.kinds) == K ? snap.kinds : [cfg.chain_starts[mod1(c, length(cfg.chain_starts))] for c in 1:K]
            chains = [evaluate!(P, s) for s in snap.states]          # under the lifted bias
            resume_states = [copy(s) for s in chains]
            it0 = snap.it
            traces = [tr[1:min(length(tr), it0)] for tr in snap.traces]
            all(tr -> length(tr) == it0, traces) || @warn "snapshot traces cover $(length.(traces)) sweeps, the counter is $it0"
            μs = copy(snap.co2_trace)
        else
            kinds = [cfg.chain_starts[mod1(c, length(cfg.chain_starts))] for c in 1:K]
            newweather = cfg.fine_weather > 0
            newweather || logmsg(cfg, "fine_weather = 0: all chains share the weather of the set-up state")
            chains = [chain_start(MersenneTwister(restart_seed(cfg, r, 7 * c)), P, sL, kinds[c], newweather) for c in 1:K]
            it0 = 0
            traces = [Vector{Vector{Float64}}() for _ in 1:K]
            μs = Float64[]
        end
        samplers = [new_sampler(P, chains[c], cfg) for c in 1:K]
        rngs = [MersenneTwister(restart_seed(cfg, r, 1000 * c)) for c in 1:K]
        if it0 >= nwarm                                # resumed in production: no adaptation of the rebuilt kernels
            for S in samplers
                S.loc.mode == :adaptive && (S.loc.adapt[L] = AdaptiveModes(P, L; freeze=-1))
            end
        end
        online = P.use_bias && cfg.bias_online
        OB = OnlineBias(; every=cfg.bias_online_every, freeze=K * (nwarm - min(it0, nwarm)), nmin=cfg.bias_online_min,
                        trust=cfg.bias_trust)          # refits during the (rest of the) warmup only
        PB = (P.use_bias && cfg.bias_post) ? PostBias() : nothing
        LM = LocalModel(P, L)
        pacc = PriceAcc(P.E.sys.N, P.lv[L+1].T)
        raccs = Dict("country" => RegionPriceAcc(P.E.sys.country), "system" => RegionPriceAcc(fill("all", P.E.sys.N)))
        fine_data = () -> (; states=chains, kinds, traces, co2_trace=μs)
        logmsg(cfg, "$K chains (starts: $(join(kinds, ", "))), $nwarm warmup + $nprod sweeps each",
               it0 > 0 ? ", from sweep $it0" : "")
        it0 == 0 && boundary!(C, cfg, P, :fine, 0, fine_data; force=true)   # at least one snapshot after the level-0 warmup
        stage!(clock, it0 == 0 ? "level-0 warmup and fine set-up" : "resume")
        phase[] = it0 < nwarm ? "warmup" : "production"
        for it in it0+1:nwarm+nprod
            for c in 1:K
                chains[c] = fine_local_step(rngs[c], P, chains[c], samplers[c])
                if online && it <= nwarm
                    chains[c], upd = online_bias_step!(P, chains[c], OB; monotone=bias_monotone_args(P, cfg))
                    upd && (chains = [evaluate!(P, x) for x in chains])   # a refit changes the target of every chain
                end
                q = monitor_quantities(P, chains[c]; LM=LM)
                push!(traces[c], q)
                if it > nwarm
                    PB === nothing || post_record!(PB, P, chains[c], q)
                    LC = LocalCost(P, LM, chains[c])
                    accumulate_prices!(pacc, P, LM, LC)
                    foreach(a -> accumulate_region_prices!(a, P, LM, LC, chains[c]), values(raccs))
                end
            end
            if it == nwarm                              # adaptation of the fine kernels ends with the warmup
                for S in samplers, AM in values(S.loc.adapt)
                    AM.freeze = min(AM.freeze, AM.nobs)
                end
            end
            if cfg.co2_cap > 0 && it <= nwarm           # canonical price tuned to the cap (Robbins–Monro on the chain mean)
                em = mean(x.parts.co2 for x in chains)
                μ = P.E.co2[] * exp(0.5 / it^0.6 * clamp(em / cfg.co2_cap - 1, -1.0, 1.0))
                set_co2_price!(P.E, μ)
                chains = [evaluate!(P, x) for x in chains]
                push!(μs, μ)
                it == nwarm && logmsg(cfg, @sprintf("CO₂ price tuned to the cap: μ = %.1f €/t (mean emissions %.3g t)", μ, em))
            end
            if it % max(1, (nwarm + nprod) ÷ 10) == 0
                tot = [tr[end][length(P.E.kspecs)+3] for tr in traces]
                logmsg(cfg, @sprintf("  sweep %4d/%d  total per chain %s  [%.0fs]", it, nwarm + nprod,
                                     join((@sprintf("%.4e", x) for x in tot), " "), time() - t0))
            end
            it == nwarm && (stage!(clock, "warmup"); phase[] = "production")
            it < nwarm + nprod && boundary!(C, cfg, P, :fine, it, fine_data)
        end
        stage!(clock, "production")
        res = chains_report(cfg, P, names, traces, kinds, PB, nwarm, nprod, t0)
        write_run_outputs(cfg, P, info, samplers, pacc, raccs, online ? OB : nothing, μs)
        h5open(cfg.out, "r+") do f
            f["run/restarts"] = r
            f["run/resumed_at_sweep"] = (snap !== nothing && snap.phase == :fine) ? snap.it : -1
        end
        return (; res..., P, chains, samplers, info, status=:finished, exitcode=0, resume_states, restarts=r)
    catch e
        e isa CheckpointStop || rethrow()
        stage!(clock, phase[] * " (stopped)")
        logmsg(cfg, "stop file $(cfg.stop_file): stopped after the snapshot $(e.path) ($(e.where_)); exit code $EXIT_CHECKPOINTED")
        return (; status=:stopped, exitcode=EXIT_CHECKPOINTED, states=e.states, resume_states, restarts=r, P, info)
    end
end

"""
    write_run_outputs(cfg, P, info, samplers, pacc, raccs, OB, μs)

Outputs of the CPU driver beyond `chains_report` (appended to `cfg.out`): sampled
prices, the bias coefficients and refit times, the tuned CO₂ price, the learned
quantities of the adaptive kernel per chain, the weather model and the calibration.
"""
function write_run_outputs(cfg::Config, P::Problem, info, samplers, pacc, raccs, OB, μs)
    L = nlevels(P) - 1
    h5open(cfg.out, "r+") do f
        if pacc.n > 0
            f["prices/sampled"] = prices(pacc)            # €/MWh per node and finest step, all chains
            (info.ref == L && cfg.init == "lp") && (f["prices/lp"] = info.lp.dF_dX[1] ./ P.levels[end][2])   # LP duals at the same level
            for (rn, a) in raccs                           # per region and season (DJF, MAM, JJA, SON)
                for (key, val) in region_price_summary(a)
                    f["prices/regions/$rn/$key"] = val
                end
            end
        end
        f["bias/theta"] = P.bias.θ
        f["bias/a"] = P.bias.a
        f["bias/c"] = P.bias.c
        f["bias/level"] = P.bias.ℓb
        f["bias/Dbar"] = P.bias.Dbar
        f["bias/sample_steps"] = length(P.bias.sD)
        OB === nothing || (f["bias/online_updates"] = OB.updates)
        isempty(μs) || (f["co2/price_trace"] = μs; f["co2/price"] = P.E.co2[])
        for (c, S) in enumerate(samplers), (ℓ, AM) in S.loc.adapt   # learned quantities of the adaptive fine kernel
            g = "adaptive/chain$c/level$ℓ"
            f["$g/updates"] = AM.updates
            f["$g/transport_modes"] = AM.Dθ
            for ((kind, Lb, cl), Φ) in AM.shapes
                f["$g/shapes/$(kind)_L$(Lb)_season$cl"] = Φ
            end
            nz = [n for n in eachindex(AM.runits) if !isempty(AM.runits[n])]
            f["$g/response/nodes"] = nz
            f["$g/response/beta_deficit"] = reduce(vcat, [AM.β[n][:, 1] for n in nz]; init=Float64[])
            f["$g/response/beta_surplus"] = reduce(vcat, [AM.β[n][:, 2] for n in nz]; init=Float64[])
            f["$g/response/units_per_node"] = [length(AM.runits[n]) for n in nz]
        end
        f["weather/transition"] = P.W.Ptr
        f["weather/em_loglik"] = P.W.loglik
        f["calibration/beta"] = [P.βk, P.βv]
        f["calibration/lp_objective"] = info.lp.objective
        f["calibration/lp_capacity"] = [sum(k) for k in info.lp.k]
        f["config/levels"] = reduce(hcat, [[a, b] for (a, b) in cfg.levels])
    end
end

"""
    chains_report(cfg, P, names, traces, kinds, PB, nwarm, nprod, t0)

Convergence diagnostics of per-chain traces (`traces[c][sweep]` = monitored quantities),
the bias post-correction, and the HDF5 output of the drivers (CPU and device).
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
    pc = (PB !== nothing && !isempty(PB.bused)) ? post_correct(P, PB; monotone=bias_monotone_args(P, cfg)) : nothing
    pc === nothing || logmsg(cfg, @sprintf("bias post-correction: weight ESS %.1f of %d samples", pc.ess, length(pc.w)))
    # the post-corrected estimates are only as good as their weights (docs/system/validation.md §2):
    # a weight ESS below 30 % of the samples counts as not converged
    post_ok = pc === nothing || pc.ess >= 0.3 * length(pc.w)
    passed = all(x -> x === nothing || x.pass, summ) && post_ok
    worst = argmax([x === nothing ? -Inf : x.rhat for x in summ])
    logmsg(cfg, @sprintf("convergence: warmup detected %s (window from %d), max R̂ %.3f (%s), min bulk ESS %.0f, min tail ESS %.0f%s → %s",
                         wdet > 0 ? "at sweep $wdet" : "never", w0, summ[worst].rhat, names[worst],
                         minimum(x.ess_bulk for x in summ if x !== nothing), minimum(x.ess_tail for x in summ if x !== nothing),
                         post_ok ? "" : @sprintf(", post-correction weight ESS %.1f < 30 %% of %d", pc.ess, length(pc.w)),
                         passed ? "converged" : "not converged"))
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
        f["convergence/post_weight_ess_ok"] = Int(post_ok)
        f["convergence/thresholds"] = [cfg.rhat_max, cfg.ess_min, cfg.mcse_max, cfg.rhat_screen]
        if pc !== nothing
            f["post/weights"] = pc.w; f["post/ess"] = pc.ess; f["post/Q_weighted"] = pc.Qw
            f["post/bias_used"] = PB.bused; f["post/bias_refit"] = pc.btilde
        end
    end
    logmsg(cfg, @sprintf("wrote %s  [total %.0fs]", cfg.out, time() - t0))
    return (; names, traces, summ, wdet, w0, passed)
end
