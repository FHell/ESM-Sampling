# Host loop of the replica–Langevin sampler on the device (docs/replica_sampler.md,
# docs/system/device.md §5–§6): `nchains` chains × M weather replicas in one batch
# (C = nchains · M, the replicas of a chain contiguous), the capacities shared within a chain.
#
# One step:
#   1. weather, on the host, per replica from its own random stream (the prototype's prior-reversible
#      process: pCN bursts of angle φ every `burst_every` steps, staggered over the replicas, with a
#      regime window move; or continuous pCN of angle θ); the changed replicas' load and
#      availability are uploaded, and the device sets their net load, shifts the residual and
#      re-initialises the dispatch compensated (`weather_update`);
#   2. `inner` operation sweeps of every replica (`run_sweeps`, Reactant's RNG), the capacity scores
#      accumulated after each (`score_acc`);
#   3. the Langevin step of the shared capacities on the weighted replica mean of the scores, the
#      dispatch following in every replica (`lgv_pass_rng`); the weights are 1/M_active over the
#      replicas whose last burst is at least `gate` steps old, times the stress tilt exp(λΦ)
#      (λ = 0 by default: exactly 1/M_active).
# A host checkpoint every `device_sweeps` steps: download, states, monitored quantities, snapshot.
#
# The host weather process is the prototype's (scripts/replicas/replicas.jl, module `ReplicaProto`)
# until the replica sampler is in src/.

"The prototype of the replica sampler (host weather process, tilt weights; scripts/replicas/)."
module ReplicaProto
include(joinpath(@__DIR__, "..", "..", "scripts", "langevin", "langevin.jl"))
include(joinpath(@__DIR__, "..", "..", "scripts", "replicas", "replicas.jl"))
end

"Default offsets κ_s of w = log(k + κ_s) per carrier (MW; the prototype's E24 values), else 300 MW."
const REPLICA_KAPPA = Dict("onwind" => 500.0, "solar" => 1000.0, "offwind" => 200.0, "gas" => 300.0, "battery" => 100.0, "H2" => 100.0)

"Settings of the replica sampler on the device (config table `device_replica`; docs/system/config.md)."
Base.@kwdef mutable struct ReplicaCtl
    M::Int = 4                         # weather replicas per chain
    eta_w::Vector{Float64} = Float64[] # Langevin step in w per k field (one value: all fields; default 5e-4)
    kappa_s::Vector{Float64} = Float64[]   # offsets κ_s per k field (MW; default REPLICA_KAPPA)
    clip::Float64 = 0.02               # drift clip per step in w
    box::Bool = true                   # reflect at the upper capacity bound
    kappa_min::Float64 = 1.0           # reflection at k = κ_min; compensators need |A k| ≥ κ_min
    inner::Int = 1                     # operation sweeps per replica and step
    theta::Float64 = 0.0               # continuous pCN angle per step (burst_every = 0)
    nreg::Int = 0                      # continuous mode: regime window every nreg steps (0: never)
    burst_every::Int = 40              # burst mode: a pCN step of angle phi per replica every that many steps (staggered)
    phi::Float64 = 0.3                 # burst angle
    gate::Int = 25                     # steps after a burst in which a replica's scores get weight 0
    burst_regime::Float64 = 0.5        # probability of a regime window move with a burst
    reg_window::Int = 30               # regime window length (level-0 steps)
    comp::Bool = true                  # compensated initialisation of the dispatch after a weather step
    lambda::Float64 = 0.0              # stress tilt exp(λΦ) of the replica weights (0: off)
    phi_mean::Float64 = 0.0            # Φ: annual mean residual load at the set-up capacities (GW), standardised
    phi_sd::Float64 = 1.0
    init_sweeps::Int = 40              # operation sweeps at fixed k after the fresh weather draws of a new run
end

"The replica settings of `cfg` (`device_replica`), with the per-field defaults filled in."
function replica_ctl(cfg::ML.Config, P::ML.Problem)
    ctl = ReplicaCtl()
    for (k, v) in cfg.device_replica
        s = Symbol(k)
        hasfield(ReplicaCtl, s) || error("device_replica: unknown key $k")
        T = fieldtype(ReplicaCtl, s)
        setfield!(ctl, s, T <: Vector ? Float64.(v isa AbstractVector ? v : [v]) : convert(T, v))
    end
    names = [replace(String(sp.name), "k_" => "") for sp in P.E.kspecs]
    nk = length(names)
    ctl.eta_w = isempty(ctl.eta_w) ? fill(5e-4, nk) : length(ctl.eta_w) == 1 ? fill(ctl.eta_w[1], nk) : ctl.eta_w
    ctl.kappa_s = isempty(ctl.kappa_s) ? [get(REPLICA_KAPPA, n, 300.0) for n in names] :
                  length(ctl.kappa_s) == 1 ? fill(ctl.kappa_s[1], nk) : ctl.kappa_s
    (length(ctl.eta_w) == nk && length(ctl.kappa_s) == nk) || error("device_replica: eta_w and kappa_s need one value or one per k field ($names)")
    return ctl
end

"Replica weights (C,): per chain the tilt weights over its active replicas (`tilt_weights`; 1/M_active at λ = 0)."
function replica_weights(ctl::ReplicaCtl, Φs, active, nch)
    W = zeros(nch * ctl.M)
    for ch in 1:nch
        js = (ch - 1) * ctl.M .+ (1:ctl.M)
        W[js] .= ReplicaProto.tilt_weights(ctl.lambda, Φs[js], active[js])
    end
    return W
end

"""
    run_replicas_device(cfg; families) -> results

The replica–Langevin sampler on the device (module comment): `mlda_warmup + nsamples` host
checkpoints of `device_sweeps` steps each. Snapshots and resume as `run_device` (`checkpoint`,
`checkpoint_minutes`, `stop_file`; exit code 75 after a stop).
"""
function run_replicas_device(cfg::ML.Config; families=FAMILIES_FAST, budget=DENSE_BUDGET[])
    t0 = time()
    P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
    P.use_bias && error("device_sampler = \"replica\" needs use_bias = false (the learned bias is retired)")
    ctl = replica_ctl(cfg, P)
    L = ML.nlevels(P) - 1
    nch, Mr = cfg.nchains, ctl.M
    C = nch * Mr
    LM = ML.LocalModel(P, L)
    F = ReplicaProto.VarFactors(P.W)
    kinds = [cfg.chain_starts[mod1(ch, length(cfg.chain_starts))] for ch in 1:nch]
    rkinds = [kinds[(c - 1) ÷ Mr + 1] for c in 1:C]
    # ---- snapshot: resume or a fresh start
    Ck = ML.Checkpointer(cfg)
    snap = read_device_snapshot(cfg.checkpoint, cfg, P)
    rr = snap === nothing ? 0 : snap.restarts + 1
    Ck.restarts = rr
    wrngs = [MersenneTwister(ML.restart_seed(cfg, rr, c)) for c in 1:C]       # weather stream of every replica
    if snap !== nothing
        (get(snap.extra, "driver", "") == "replica" && snap.extra["replicas"] == Mr && snap.extra["nchains"] == nch) ||
            error("snapshot $(cfg.checkpoint) is not a replica run with $nch chains × $Mr replicas")
        reps = snap.states
        ck0 = snap.it
        since = Vector{Int}(snap.extra["since"])
        traces = snap.traces
        ML.logmsg(cfg, "resuming from snapshot $(cfg.checkpoint) (restart $rr, written ", snap.where_, ")")
    else
        reps = ML.State[]
        for ch in 1:nch
            s0 = ML.evaluate!(P, ML.chain_start(MersenneTwister(cfg.seed + 7 * ch), P, sL, kinds[ch], false))
            r0 = copy(ML.LocalCost(P, LM, s0).r)
            for j in 1:Mr                              # fresh prior weather per replica, compensated
                c = (ch - 1) * Mr + j
                mm, z0, ws = ML.sample_weather(wrngs[c], P.W, L)
                s = ML.make_state(P, L, mm, z0, ws, copy(s0.k), copy(s0.v))
                push!(reps, ctl.comp ? ReplicaProto.compensate!(P, LM, r0, s) : s)
            end
        end
        ck0 = 0
        since = fill(typemax(Int) ÷ 2, C)
        traces = [Vector{Vector{Float64}}() for _ in 1:C]
    end
    # ---- device model, plan, programs
    M, ix = pack(P, L)
    S = to_device(P, M, ix, reps)
    wsh = load_shares(P, reps)
    fams = families === :all ? device_families(M, cell_info(P, LM, ix)) : families
    pl = plan(M, S, P.βv; cls=proxy_class(S.dnet, wsh), families=fams, cells=cell_info(P, LM, ix), budget=budget)
    Ug, Us, Ul = length(ix.gidx), length(ix.sidx), length(M.kcap)
    nr = length(ren_units(P))
    dims = Val((C, M.T, M.N, Ug, Us, Ul, Mr))
    RP, RS = pack_ren(P, M, reps)
    LPh = lgv_params(P, M, ix; κs=ctl.kappa_s, ηw=ctl.eta_w)
    rg, sgp = lgv_groups(P, ix)
    Reactant.set_default_backend(cfg.device_backend)
    R = Reactant.to_rarray
    st, cp, sp, kk, tabs = R(pl.st), R(pl.cp), R(pl.sp), R(pl.k), R(pl.dev)
    kgd, ksd, RSd = R(S.kg), R(S.ks), R(RS)
    dn, crd, cap = R(S.dnet), R(S.cren), R(S.capren)
    KPd, RPd, LPd = R(KParams(M)), R(RP), R(LPh)
    rgd, sgd = map(R, rg), map(R, sgp)
    wshd = R(wsh)
    drng = Reactant.ReactantRNG(R(device_rng_seed(cfg, rr)))
    fv = Val(pl.fams)
    ph = Val(P.E.measure == :physical)
    βv, βk = P.βv, P.βk
    kw = (; rr=βk / βv, βk, clip=ctl.clip, κmin=ctl.kappa_min, box=ctl.box)
    comp = Val(ctl.comp)
    sw(st, cp, sp, k, tabs, ns, rng) = run_sweeps(st, cp, sp, k, tabs, ns, rng, fv)
    sa(st, kg, ks, RS, KP, RP, acc, wt) = score_acc(st, kg, ks, RS, KP, RP, acc, wt, dims, ph; βv, βk)
    lg(st, kg, ks, RS, dn, cr, cap, KP, RP, LP, acc, W, rng, wsh, rg, sg) =
        lgv_pass_rng(st, kg, ks, RS, dn, cr, cap, KP, RP, LP, acc, W, rng, wsh, rg, sg, dims; kw...)
    wu(st, kg, ks, RS, dn, cr, KP, RP, D0, c0, A, chg, wsh) = weather_update(st, kg, ks, RS, dn, cr, KP, RP, D0, c0, A, chg, wsh, dims, comp)
    acc0 = R(ScoreAcc(C, Ug, Us, nr))
    wt = Reactant.ConcreteRNumber(1.0 / ctl.inner)
    Wd = R(fill(1.0 / Mr, C))
    D0h, c0h, Ah = weather_arrays(P, M, reps)
    chg0 = R(zeros(C))
    tc = Dict{String,Float64}()
    tc["operation"] = @elapsed fsw = @compile sw(st, cp, sp, kk, tabs, Reactant.ConcreteRNumber(1), drng)
    tc["score"] = @elapsed fsa = @compile sa(st, kgd, ksd, RSd, KPd, RPd, acc0, wt)
    tc["langevin"] = @elapsed flg = @compile lg(st, kgd, ksd, RSd, dn, crd, cap, KPd, RPd, LPd, acc0, Wd, drng, wshd, rgd, sgd)
    tc["weather"] = @elapsed fwu = @compile wu(st, kgd, ksd, RSd, dn, crd, KPd, RPd, R(D0h), R(c0h), R(Ah), chg0, wshd)
    ML.logmsg(cfg, @sprintf("device (%s): replica sampler, %d chains × %d replicas; compiled: operation %.0f s, score %.0f s, Langevin %.0f s, weather %.0f s",
                            cfg.device_backend, nch, Mr, tc["operation"], tc["score"], tc["langevin"], tc["weather"]))
    if ck0 == 0 && ctl.init_sweeps > 0                 # the dispatch of the fresh weather at fixed k
        st = fsw(st, cp, sp, kk, tabs, Reactant.ConcreteRNumber(ctl.init_sweeps), drng)
    end
    # ---- loop
    nwarm, nprod, spc = cfg.mlda_warmup, cfg.nsamples, cfg.device_sweeps
    names = ML.monitor_names(P)
    kref = ML.knodes(P, L, sL.k)
    Φf(s) = ctl.lambda == 0 ? 0.0 : (ReplicaProto.wfeatures(P, s.xs[end], kref).resmean - ctl.phi_mean) / ctl.phi_sd
    Φs = [Φf(s) for s in reps]
    nb = ctl.burst_every
    Sh = deepcopy(S)
    tim = Dict(k => 0.0 for k in ("weather_host", "weather", "operation", "score", "langevin", "download", "host"))
    sync(x) = (Reactant.synchronize(x); x)
    ns1 = Reactant.ConcreteRNumber(1)
    nclip = zeros(nch)
    stopped = false
    for ck in ck0+1:nwarm+nprod
        for it in (ck - 1) * spc + 1:ck * spc
            # 1. weather on the host, upload of the changed replicas
            th = time()
            chg = zeros(C)
            for c in 1:C
                j = mod1(c, Mr)
                if nb == 0
                    θc, reg = ctl.theta, ctl.nreg > 0 && it % ctl.nreg == 0
                else
                    burst = mod(it + (j - 1) * nb ÷ Mr, nb) == 0
                    θc = burst ? ctl.phi : 0.0
                    reg = burst && rand(wrngs[c]) < ctl.burst_regime
                    since[c] = burst ? 0 : since[c] + 1
                end
                if θc > 0 || reg
                    reps[c] = ReplicaProto.weather_step(wrngs[c], P, F, reps[c], θc; regime=reg, regmode=:window, B=ctl.reg_window)
                    Φs[c] = Φf(reps[c])
                    chg[c] = 1.0
                end
            end
            tim["weather_host"] += time() - th
            if any(>(0), chg)
                tim["weather"] += @elapsed begin
                    D0h, c0h, Ah = weather_arrays(P, M, reps)
                    wshd = R(load_shares(P, reps))
                    st, RSd, dn, crd, cp = fwu(st, kgd, ksd, RSd, dn, crd, KPd, RPd, R(D0h), R(c0h), R(Ah), R(chg), wshd)
                    sync(st.r)
                end
            end
            # 2. operation sweeps and scores
            acc = acc0
            for _ in 1:ctl.inner
                tim["operation"] += @elapsed (st = fsw(st, cp, sp, kk, tabs, ns1, drng); sync(st.r))
                tim["score"] += @elapsed (acc = fsa(st, kgd, ksd, RSd, KPd, RPd, acc, wt); sync(acc.sg))
            end
            # 3. Langevin step of the shared capacities, the dispatch following
            active = [nb == 0 || since[c] >= ctl.gate for c in 1:C]
            Wd = R(replica_weights(ctl, Φs, active, nch))
            tim["langevin"] += @elapsed begin
                st, kgd, ksd, cp, RSd, dn, crd, cap, diag = flg(st, kgd, ksd, RSd, dn, crd, cap, KPd, RPd, LPd, acc, Wd, drng, wshd, rgd, sgd)
                sync(kgd)
            end
            nclip .+= Array(diag.nclip)
        end
        # ---- host checkpoint
        th0 = time()
        tim["download"] += @elapsed begin
            unflat!(Sh, st)
            Sh.kg .= Array(kgd); Sh.ks .= Array(ksd)
            krh = Array(RSd.kr)
        end
        for c in 1:C
            ren_to_host!(P, krh, c, reps[c])
            reps[c] = to_host!(P, M, ix, Sh, c, reps[c])
            push!(traces[c], ML.monitor_quantities(P, reps[c]; LM=LM))
        end
        tim["host"] += time() - th0
        if ck % max(1, (nwarm + nprod) ÷ 10) == 0 || ck == nwarm + nprod
            tot = [sum(k) / 1e3 for k in ML.knodes(P, L, reps[1].k)]
            ML.logmsg(cfg, @sprintf("  checkpoint %4d/%d (%d steps each)  chain 1 k [%s] GW  shed %.3g MWh  clipped %d  [%.0fs]", ck, nwarm + nprod,
                                    spc, join((@sprintf("%.1f", x) for x in tot), ", "), sum(s.parts.shed for s in reps) / C, sum(nclip), time() - t0))
        end
        data() = (; states=reps, kinds=rkinds, traces)
        extra() = Dict{String,Any}("driver" => "replica", "replicas" => Mr, "nchains" => nch, "since" => since, "step" => ck * spc,
                                   "eta_w" => ctl.eta_w)
        if device_boundary!(Ck, cfg, P, ck, data, extra; force=(ck == nwarm + nprod))
            stopped = true
            break
        end
    end
    stopped && return (; status=:stopped, exitcode=ML.EXIT_CHECKPOINTED, states=reps, restarts=rr, P, info)
    ncall = nwarm + nprod - ck0
    ML.logmsg(cfg, "device profile per checkpoint ($spc steps, $(ctl.inner) operation sweeps each, $C replicas): " *
                   join((@sprintf("%s %.3f s", k, tim[k] / max(ncall, 1)) for k in sort(collect(keys(tim)))), ", "))
    res = ML.chains_report(cfg, P, names, traces, rkinds, nothing, nwarm, nprod, t0)
    ML.HDF5.h5open(cfg.out, "r+") do f
        for (k, v) in tim
            f["device_profile/$k"] = v / max(ncall, 1)
        end
        for (k, v) in tc
            f["device_profile/compile_$k"] = v
        end
        f["replica/M"] = Mr; f["replica/nchains"] = nch
        f["replica/eta_w"] = ctl.eta_w; f["replica/kappa_s"] = ctl.kappa_s
        f["run/restarts"] = rr
        f["run/resumed_at_checkpoint"] = snap === nothing ? -1 : snap.it
    end
    return (; res..., P, states=reps, status=:finished, exitcode=0, restarts=rr)
end
