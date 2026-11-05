# The replica–Langevin sampler with the batch on the device (docs/system/sampler.md §11,
# docs/system/device.md §5.2, §6.2): the driver of src/replicas.jl (`run_replicas`) with the
# operation, the capacity scores, the Langevin step with the dispatch following and the weather
# upload of all `nchains` × M replicas as device programs. Device rows: per chain its M replicas
# and, with probes (`replica_probe_every > 0`), one probe slot, contiguous (C = nchains · M_p,
# M_p = M + 1; the capacities shared within a chain).
#
# One step, as `replica_step!` of the CPU driver:
#   1. weather, on the host, per replica from its own stream (`replica_weather!` of src/replicas.jl
#      without its compensation: the schedule, the window and FFBS, the gate reset at a burst); the
#      moved replicas' load and availability are uploaded (their rows only) and the device sets
#      their net load, shifts the residual and re-initialises the dispatch compensated
#      (`weather_rows`);
#   2. `replica_inner` operation sweeps of every row (`run_sweeps`), the scores accumulated
#      after each (`score_acc`) and downloaded;
#   3. on the host: the gate of every replica on its relaxation indicator (`update_gate!`), the
#      in-flight windows of the probe, the drift weights with the tilt (`drift_weights`);
#   4. the Langevin step of the shared capacities on the weighted replica scores and the dispatch
#      following in every replica (`lgv_pass_rng`; per-chain steps η_w); the probe slots are frozen;
#   5. every `replica_probe_every` steps, per chain, a lag probe on the device: the candidate
#      replica is copied into the chain's probe slot (`copy_rows`), which then runs the device's
#      operation sweeps at frozen (k, x) with weight 0 in the drift; after `replica_probe_len`
#      sweeps its equilibrated score gives the gap, and in the warmup the step adaptation
#      (`adapt_step!`).
# A host checkpoint every `device_sweeps` steps: the residual cache rebuilt (`rebuild_flat`), the
# download, the replicas' `State`s, the monitored quantities and the replica records, and the
# snapshot boundary (`write_replica_snapshot`, the CPU driver's layout). `compile = false` runs the
# same driver on plain Julia arrays.

"Columns of the device unit blocks for every node of every k field: (block 1 dispatchable | 2 storage | 3 renewable, column)."
function unit_map(P::ML.Problem, ix::DeviceIndex)
    E = P.E
    cols = [fill((0, 0), length(sp.lo)) for sp in E.kspecs]
    for (q, (g, i)) in enumerate(ix.gidx)
        E.gens[g].kf > 0 && (cols[E.gens[g].kf][i] = (1, q))
    end
    for (q, (s, i)) in enumerate(ix.sidx)
        E.stors[s].kf > 0 && (cols[E.stors[s].kf][i] = (2, q))
    end
    for (q, (g, i)) in enumerate(ren_units(P))
        cols[E.gens[g].kf][i] = (3, q)
    end
    return cols
end

"Per-k-field node vectors of row c of the device blocks (A1, A2, A3) (`capacity_scores` layout; 0 for units without a column)."
fields_of(um, A1, A2, A3, c) = [[b == 0 ? 0.0 : (b == 1 ? A1 : b == 2 ? A2 : A3)[c, q] for (b, q) in col] for col in um]

"Capex force β_k ∂K per k field at the raw node capacities `Kraw` (the `fk` of `capacity_scores`)."
host_forces(P::ML.Problem, Kraw) = [P.βk .* ML.f_capex(P.E, f) .* (1 .+ P.E.γk .* sign.(Kraw[f] .- clamp.(Kraw[f], sp.lo, sp.hi)))
                                    for (f, sp) in enumerate(P.E.kspecs)]

"""
    copy_rows(st, cp, kg, ks, RS, dnet, cren, capren, src) -> (st, cp, kg, ks, RS, dnet, cren, capren)

Every per-row device array of the replica batch with row b taken from row src[b] (src (C,), the
identity except at the rows written): one gather per array, exact. Starts a probe (the candidate
replica into the chain's probe slot).
"""
function copy_rows(st::FlatState, cp::ChainParams, kg, ks, RS, dnet, cren, capren, src)
    g2(X) = X[src, :]
    g3(X) = X[src, :, :]
    return (FlatState(g2(st.θ), g2(st.ω), g2(st.uc), g2(st.e), g2(st.u), g2(st.r)),
            ChainParams(g2(cp.κg), g2(cp.wg), g2(cp.plo), g2(cp.yk), g2(cp.ks), g2(cp.ws), g2(cp.cls)),
            g2(kg), g2(ks), RenState(g2(RS.kr), g3(RS.A)), g3(dnet), cren[src], capren[src])
end

"""
    weather_rows(st, kg, ks, RS, dnet, cren, KP, RP, D0, cren0, A, pos, chg, wsh, Val(dims), Val(comp))

`weather_update` with the weather of the moved replicas only: D0 (B, T, N), cren0 (B,), A (B, T, U_r)
of up to B uploaded rows, `pos` (C,) the uploaded row of every device row (any row where chg = 0),
chg (C,) 0/1. The rows are gathered on the device (bitwise the full upload of the same rows).
"""
weather_rows(st, kg, ks, RS, dnet, cren, KP, RP, D0, cren0, A, pos, chg, wsh, dv::Val, cv::Val) =
    weather_update(st, kg, ks, RS, dnet, cren, KP, RP, D0[pos, :, :], cren0[pos], A[pos, :, :], chg, wsh, dv, cv)

"""
Device programs of the replica driver: Reactant-compiled (`compile = true`, under `ftcompile`) or
the plain functions on Arrays. `args` holds the arguments of every program to compile (`pc`, the
row copy, only with probe slots). Returns (programs as a NamedTuple sw, sa, lg, wu, pc; compile
seconds).
"""
function replica_programs(fv::Val, dims::Val, ph::Val, comp::Val, kw, args; compile::Bool)
    sw(st, cp, sp, k, tabs, ns, rng) = run_sweeps(st, cp, sp, k, tabs, ns, rng, fv)
    sa(st, kg, ks, RS, KP, RP, acc, wt) = score_acc(st, kg, ks, RS, KP, RP, acc, wt, dims, ph; βv=kw.βv, βk=kw.βk)
    lg(st, kg, ks, RS, dn, cr, cap, KP, RP, LP, acc, W, rng, wsh, rg, sg) =
        lgv_pass_rng(st, kg, ks, RS, dn, cr, cap, KP, RP, LP, acc, W, rng, wsh, rg, sg, dims;
                     rr=kw.rr, βk=kw.βk, clip=kw.clip, κmin=kw.κmin, box=kw.box)
    wu(st, kg, ks, RS, dn, cr, KP, RP, D0, c0, A, pos, chg, wsh) = weather_rows(st, kg, ks, RS, dn, cr, KP, RP, D0, c0, A, pos, chg, wsh,
                                                                                dims, comp)
    pc(st, cp, kg, ks, RS, dn, cr, cap, src) = copy_rows(st, cp, kg, ks, RS, dn, cr, cap, src)
    fns = (; sw, sa, lg, wu, pc)
    tc = Dict{String,HostFloat}()
    compile || return fns, tc
    lab = (sw="operation", sa="score", lg="langevin", wu="weather", pc="copy")
    out = Dict{Symbol,Any}()
    for k in keys(fns)
        haskey(args, k) || continue
        tc[lab[k]] = @elapsed out[k] = ftcompile(() -> @compile fns[k](args[k]...))
    end
    return NamedTuple{keys(fns)}(Tuple(get(out, k, nothing) for k in keys(fns))), tc
end

"""
Weather rows uploaded per call of the weather program: the most replicas the schedule can move
in one step (bursts only: per chain the largest number of replicas bursting at the same step;
continuous mode or a continuous angle: all).
"""
function weather_chunk(ctl, nch::Int, Mr::Int)
    (ctl.θ > 0 || ctl.burst_every == 0) && return nch * Mr
    return nch * maximum(count(j -> ML.weather_schedule(ctl, it, j)[2], 1:Mr) for it in 1:ctl.burst_every)
end

"A running probe of one chain (its probe slot on the device); host values in `HostFloat` (Float64 also in the F32 build)."
mutable struct DeviceProbe
    j::Int                                   # the probed replica
    it::Int                                  # step of the copy
    hist::Vector{Vector{Vector{HostFloat}}}  # the replica's in-flight window at the copy (r ŝ_k per step)
    mask::Any                                # interior nodes at the frozen k
    fk::Vector{HostFloat}                    # β_k ∂K, interior means at the frozen k
    op::HostFloat; shed::HostFloat           # opex and shed of the replica at the copy
    sc::Vector{Vector{Vector{HostFloat}}}    # r ŝ_k of the probe slot after each step since the copy
end

"Steps of a device probe (`replica_probe_len` sweeps) and of its equilibrated tail (`replica_probe_tail`), in steps of `replica_inner` sweeps."
probe_steps(ctl) = (cld(ctl.probe_len, ctl.inner), cld(ctl.probe_tail, ctl.inner))

"State of a device replica run (host and device sides)."
mutable struct ReplicaDevice
    cfg::Any; P::Any; info::Any; ctl::Any; ctlw::Any; LM::Any; F::Any; kref::Any
    M::Any; ix::Any; um::Any; dims::Any
    nch::Int; Mr::Int; Mp::Int                # chains, replicas per chain, device rows per chain (replicas + probe slot)
    sets::Any; kinds::Any; restarts::Int; snap::Any
    dev::Dict{Symbol,Any}          # device arrays and programs
    Sh::Any                        # host DeviceState buffer of the downloads
    wsh::Matrix{Float64}           # load shares of every device row (host; uploaded when changed)
    wb::Int                        # rows per weather upload
    probes::Vector{Any}            # per chain: the running DeviceProbe or nothing
    due::Vector{Bool}              # per chain: a probe is due (started when the slot is free)
    fresh::Bool                    # the host States equal the device state (after a full download)
    sync::Bool                     # synchronize after every program (per-part timings)
    compile::Bool
    tim::Dict{String,HostFloat}
    tc::Dict{String,HostFloat}
end

"Device row of slot j of chain c (replicas 1 … M, the probe slot M + 1)."
slotrow(RD::ReplicaDevice, c::Int, j::Int) = (c - 1) * RD.Mp + j

"""
    replica_device_setup(cfg; families, budget, jump, compile=true, sync=true) -> ReplicaDevice

`setup`, the chains and their replicas (`init_replicas` without its operation sweeps: the
replicas' fresh weather draws and the compensated initialisation, as the CPU driver), or the
replicas of the snapshot `checkpoint`; the device model and plan (`jump`, `backend =
device_backend` as `device_setup`), the device arrays in the module's float type (`ftconv`) and
the programs (compiled, or plain on Arrays), then `replica_init_sweeps` operation sweeps of a new
run on the device. The probe slot of every chain starts as a copy of its replica 1 (idle until
its first probe). The random streams are those of `run_replicas` for the replicas' weather; the
operation sweeps (also those of the probes) and the Langevin noise use two device streams
(`device_rng_seed`). `sync = false`: no synchronization after every program (the host's work
overlaps the device's; the per-part timings then lump the device time into the parts that
download).
"""
function replica_device_setup(cfg::ML.Config; families=FAMILIES_FAST, budget=DENSE_BUDGET[], jump=:auto, compile::Bool=true,
                              sync::Bool=true)
    P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
    cfg.use_bias && ML.logmsg(cfg, "device_sampler = replicas: the learned bias is not used (use_bias ignored)")
    P = ML.with_bias(P, false)
    sL = ML.evaluate!(P, sL)
    L = ML.nlevels(P) - 1
    snap = ML.read_replica_snapshot(cfg.checkpoint, cfg, P)
    rr = snap === nothing ? 0 : snap.restarts + 1
    snap === nothing || snap.co2 === P.E.co2[] || ML.set_co2_price!(P.E, snap.co2)
    nch, Mr = cfg.nchains, cfg.replicas
    LM = ML.LocalModel(P, L; cell_power=cfg.cell_level_power, xr_partner=cfg.xloc_partner_hops, xr_ball=cfg.xloc_ball_hops)
    F = ML.VarFactors(P.W)
    kref = ML.knodes(P, sL.ℓ, sL.k)
    ctl = ML.ReplicaCtl(cfg, P)
    cfg.tilt_functional == "resload_mean" && (ctl.Φ = ML.TiltFunctional(P, kref; rng=MersenneTwister(cfg.seed + 5151), n=cfg.tilt_samples))
    ctlw = deepcopy(ctl); ctlw.comp = false                 # the host weather step; the device compensates
    kinds = [cfg.chain_starts[mod1(c, length(cfg.chain_starts))] for c in 1:nch]
    wrngs = [[MersenneTwister(ML.restart_seed(cfg, rr, 1000c + j)) for j in 1:Mr] for c in 1:nch]
    sets = ML.ReplicaSet[]
    if snap === nothing
        for c in 1:nch
            s0 = ML.chain_start(MersenneTwister(ML.restart_seed(cfg, rr, 7c)), P, sL, kinds[c], false)
            Ks = ML.replica_kernels(P, s0, cfg, Mr, LM)
            push!(sets, ML.init_replicas(nothing, P, Ks, LM, s0, wrngs[c], ctl, 0))
        end
    else
        length(snap.sets) == nch || error("snapshot $(cfg.checkpoint) has $(length(snap.sets)) chains, nchains = $nch")
        kinds = snap.kinds
        for c in 1:nch
            sv = snap.sets[c]
            length(sv.reps) == Mr || error("snapshot $(cfg.checkpoint): $(length(sv.reps)) replicas, replicas = $Mr")
            R = ML.ReplicaSet(sv.reps, wrngs[c], sv.ηw)
            R.since .= sv.since; R.window .= sv.window; R.active .= sv.active
            push!(sets, R)
        end
    end
    # device rows: per chain its replicas, then (with probes) its probe slot, a copy of replica 1 until the first probe
    np = ctl.probe_every > 0 ? 1 : 0
    Mp = Mr + np
    C = nch * Mp
    slots = [s for R in sets for s in (np > 0 ? vcat(R.reps, R.reps[1:1]) : R.reps)]
    # ---- device model, plan, arrays, programs
    M, ix = pack(P, L)
    S = to_device(P, M, ix, slots)
    wsh = load_shares(P, slots)
    fams = families === :all ? device_families(M, cell_info(P, LM, ix)) : families
    set_sources!(P, LM, ix)                                             # host sources of the drawn sites (sources.jl)
    pl = plan(M, S, P.βv; cls=proxy_class(S.dnet, wsh), families=fams, cells=cell_info(P, LM, ix), budget=budget, jump=jump,
              backend=cfg.device_backend, rng=MersenneTwister(ML.restart_seed(cfg, rr) + 303))
    Ug, Us, Ul = length(ix.gidx), length(ix.sidx), length(M.kcap)
    nr = length(ren_units(P))
    dims = Val((C, M.T, M.N, Ug, Us, Ul, Mp, np))
    RP, RS = pack_ren(P, M, slots)
    LPh = lgv_params(P, M, ix; κs=ctl.κs, ηw=[R.ηw for R in sets])
    rg, sgp = lgv_groups(P, ix)
    compile && Reactant.set_default_backend(cfg.device_backend)
    R = compile ? (x -> Reactant.to_rarray(ftconv(x))) : (x -> ftconv(deepcopy(x)))     # the module's float type (F32: Float32)
    rngd(seed) = compile ? Reactant.ReactantRNG(R(seed)) : MersenneTwister(collect(reinterpret(UInt32, seed)))
    dev = Dict{Symbol,Any}()
    dev[:st], dev[:cp], dev[:sp], dev[:kk], dev[:tabs] = R(pl.st), R(pl.cp), R(pl.sp), R(pl.k), R(pl.dev)
    dev[:kg], dev[:ks], dev[:RS] = R(S.kg), R(S.ks), R(RS)
    dev[:dn], dev[:cr], dev[:cap] = R(S.dnet), R(S.cren), R(S.capren)
    dev[:KP], dev[:RP], dev[:LP] = R(KParams(M)), R(RP), R(LPh)
    dev[:rg], dev[:sg] = map(R, rg), map(R, sgp)
    dev[:wsh] = R(wsh)
    dev[:pl], dev[:up], dev[:srng] = pl, R, MersenneTwister(ML.restart_seed(cfg, rr) + 202)   # site redraws (sources.jl)
    dev[:drng] = rngd(device_rng_seed(cfg, rr))                         # operation sweeps
    dev[:lrng] = rngd(device_rng_seed(cfg, rr) .+ UInt64[0, 1])        # Langevin noise
    dev[:R] = R
    num(x) = compile ? Reactant.ConcreteRNumber(ftconv(x)) : ftconv(x)
    dev[:num] = num
    dev[:one] = num(1)
    dev[:acc0] = R(ScoreAcc(C, Ug, Us, nr))     # zeros, uploaded once (score_acc does not change its input)
    dev[:wt] = num(1.0 / ctl.inner)
    dev[:nclip] = zeros(nch)
    wb = weather_chunk(ctl, nch, Mr)
    D0h, c0h, Ah = weather_arrays(P, M, slots[1:wb])
    kw = ftconv((; rr=P.βk / P.βv, P.βk, P.βv, clip=ctl.clip, κmin=ctl.κmin, box=ctl.box))    # scalars of the programs (F32: Float32)
    W0 = R(fill(1.0 / Mr, C))
    args = (; sw=(dev[:st], dev[:cp], dev[:sp], dev[:kk], dev[:tabs], dev[:one], dev[:drng]),
            sa=(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:KP], dev[:RP], dev[:acc0], dev[:wt]),
            lg=(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:cap], dev[:KP], dev[:RP], dev[:LP], dev[:acc0], W0,
                dev[:lrng], dev[:wsh], dev[:rg], dev[:sg]),
            wu=(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:KP], dev[:RP], R(D0h), R(c0h), R(Ah), R(ones(Int, C)),
                R(zeros(C)), dev[:wsh]))
    np > 0 && (args = merge(args, (; pc=(dev[:st], dev[:cp], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:cap], R(collect(1:C))))))
    progs, tc = replica_programs(Val(pl.fams), dims, Val(P.E.measure == :physical), Val(ctl.comp), kw, args; compile=compile)
    dev[:sw], dev[:sa], dev[:lg], dev[:wu], dev[:pc] = progs.sw, progs.sa, progs.lg, progs.wu, progs.pc
    dev[:rb], tcr = rebuild_program(M, rebuild_dims(C, M, ix), R, (dev[:st], dev[:kg], dev[:ks], dev[:dn], dev[:cr], dev[:cap]);
                                    compile=compile)
    compile && (tc["rebuild"] = tcr)
    ML.logmsg(cfg, @sprintf("device (%s): replica sampler, %d chains × %d replicas%s%s", compile ? cfg.device_backend : "arrays", nch, Mr,
                            np > 0 ? " + 1 probe slot" : "",
                            isempty(tc) ? "" : "; compiled: " * join((@sprintf("%s %.0f s", k, v) for (k, v) in tc), ", ")))
    RD = ReplicaDevice(cfg, P, info, ctl, ctlw, LM, F, kref, M, ix, unit_map(P, ix), dims, nch, Mr, Mp, sets, kinds, rr, snap,
                       dev, deepcopy(S), copy(wsh), wb, Any[nothing for _ in 1:nch], falses(nch), true, sync, compile,
                       Dict(k => 0.0 for k in ("weather_host", "weather", "operation", "score", "gate", "langevin", "probe", "download",
                                               "rebuild", "host")), tc)
    if snap === nothing && cfg.replica_init_sweeps > 0         # the dispatch of the fresh weather at fixed k
        dev[:st] = dev[:sw](dev[:st], dev[:cp], dev[:sp], dev[:kk], dev[:tabs], num(cfg.replica_init_sweeps), dev[:drng])
        RD.fresh = false
    end
    return RD
end

host(x) = x isa AbstractArray ? Array(x) : x
sync!(x) = (x isa Reactant.ConcreteRArray && Reactant.synchronize(x); x)

"Download the device state and write every replica's capacities and operation into its host `State` (the probe slots stay on the device)."
function replica_download!(RD::ReplicaDevice)
    dev, P = RD.dev, RD.P
    unflat!(RD.Sh, dev[:st])
    RD.Sh.kg .= host(dev[:kg]); RD.Sh.ks .= host(dev[:ks])
    krh = host(dev[:RS].kr)
    for c in 1:RD.nch, j in 1:RD.Mr
        b = slotrow(RD, c, j)
        R = RD.sets[c]
        ren_to_host!(P, krh, b, R.reps[j])
        R.reps[j] = to_host!(P, RD.M, RD.ix, RD.Sh, b, R.reps[j])
    end
    RD.fresh = true
    return RD
end

"""
Rebuild the residual cache on the device from the node arrays (`rebuild_flat`, at a host
checkpoint); returns the drift max |r_cache − r_rebuilt| of every device row before the rebuild.
"""
function replica_rebuild!(RD::ReplicaDevice)
    dev = RD.dev
    dev[:st], dr, _, _ = dev[:rb](dev[:st], dev[:kg], dev[:ks], dev[:dn], dev[:cr], dev[:cap])
    return HostFloat.(host(dr))
end

"""
    replica_device_step!(RD, it) -> per-chain step records

One step of every chain's replicas (module comment): `replica_device_weather!`,
`replica_device_operation!`, `replica_device_capacity!`. Returns, per chain, the step's record in
the form of `replica_step!` (`wts`, `Φs`, `moved`, `rs`, `fk`, `sks`, `Kraw`, `tw`, and `probe` when
a probe ended; `nclip` is filled at the host checkpoints, from `dev[:nclip]`).
"""
function replica_device_step!(RD::ReplicaDevice, it::Int)
    moved, tw = replica_device_weather!(RD, it)
    acc = replica_device_operation!(RD)
    return replica_device_capacity!(RD, it, acc, moved, tw)
end

"""
Step part 1: the replicas' weather steps on the host (`replica_weather!` without compensation)
and the upload of the moved replicas. The host `State`s then hold the new weather with the
operation before the compensation (which the device does): they are stale (`RD.fresh = false`)
until the next `replica_download!`. Returns (moved per chain, seconds).
"""
function replica_device_weather!(RD::ReplicaDevice, it::Int)
    P, tim = RD.P, RD.tim
    nch, Mr = RD.nch, RD.Mr
    th = time()
    moved = [falses(Mr) for _ in 1:nch]
    for c in 1:nch, j in 1:Mr
        moved[c][j] = ML.replica_weather!(RD.sets[c], j, P, RD.LM, RD.F, RD.ctlw, it)
    end
    tim["weather_host"] += time() - th
    tw = time() - th
    cj = [(c, j) for c in 1:nch for j in 1:Mr if moved[c][j]]
    if !isempty(cj)
        RD.fresh = false
        tim["weather"] += @elapsed for part in Iterators.partition(cj, RD.wb)
            replica_weather_upload!(RD, part)
        end
    end
    return moved, tw
end

"Upload the weather of the replicas `cj` ((chain, replica) pairs, at most `RD.wb`) and apply it on the device (`weather_rows`)."
function replica_weather_upload!(RD::ReplicaDevice, cj)
    P, dev = RD.P, RD.dev
    R = dev[:R]
    C, B, n = RD.nch * RD.Mp, RD.wb, length(cj)
    ss = [RD.sets[c].reps[j] for (c, j) in cj]
    D0, c0, A = weather_arrays(P, RD.M, ss)
    padr(X) = (Y = zeros(eltype(X), B, size(X)[2:end]...); selectdim(Y, 1, 1:n) .= X; Y)
    rows = [slotrow(RD, c, j) for (c, j) in cj]
    pos, chg = ones(Int, C), zeros(C)
    pos[rows] .= 1:n
    chg[rows] .= 1.0
    RD.wsh[rows, :] .= load_shares(P, ss)
    dev[:wsh] = R(RD.wsh)
    dev[:st], dev[:RS], dev[:dn], dev[:cr], dev[:cp] = dev[:wu](dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:KP], dev[:RP],
                                                               R(padr(D0)), R(padr(c0)), R(padr(A)), R(pos), R(chg), dev[:wsh])
    RD.sync && sync!(dev[:st].r)
    return nothing
end

"Step part 2: `replica_inner` operation sweeps of every row with the scores accumulated after each. Returns the score sums."
function replica_device_operation!(RD::ReplicaDevice)
    ctl, dev, tim = RD.ctl, RD.dev, RD.tim
    acc = dev[:acc0]
    for _ in 1:ctl.inner
        tim["operation"] += @elapsed (dev[:st] = dev[:sw](dev[:st], dev[:cp], dev[:sp], dev[:kk], dev[:tabs], dev[:one], dev[:drng]);
                                      RD.sync && sync!(dev[:st].r))
        tim["score"] += @elapsed (acc = dev[:sa](dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:KP], dev[:RP], acc, dev[:wt]);
                                  RD.sync && sync!(acc.sg))
    end
    RD.fresh = false
    return acc
end

"""
Step part 3: the gates, probe windows and drift weights on the host (src/replicas.jl), the
Langevin step with the dispatch following on the device (the probe slots frozen, weight 0), the
device probes and the step adaptation (`replica_device_probes!`).
"""
function replica_device_capacity!(RD::ReplicaDevice, it::Int, acc, moved, tw)
    P, ctl, dev, tim = RD.P, RD.ctl, RD.dev, RD.tim
    nch, Mr, C = RD.nch, RD.Mr, RD.nch * RD.Mp
    rr = P.βk / P.βv
    # gates, probe windows, weights
    th = time()
    A1, A2, A3 = host(acc.sg), host(acc.ss), host(acc.sr)
    kg, ks, kr = host(dev[:kg]), host(dev[:ks]), host(dev[:RS].kr)
    W = zeros(C)
    recs = Any[]
    for c in 1:nch
        R = RD.sets[c]
        b1 = slotrow(RD, c, 1)
        Kraw = fields_of(RD.um, kg, ks, kr, b1)
        mask = ML.interior_nodes(P, Kraw)
        fk = host_forces(P, Kraw)
        fkm = ML.masked_means(fk, mask)
        sks = [fields_of(RD.um, A1, A2, A3, b1 + j - 1) for j in 1:Mr]
        for j in 1:Mr
            ML.update_gate!(R, j, ML.masked_means([rr .* x for x in sks[j]], mask) ./ fkm, ctl)
            if R.active[j]
                push!(R.hist[j], [rr .* x for x in sks[j]])
                length(R.hist[j]) > ctl.probe_win && popfirst!(R.hist[j])
            end
        end
        pb = RD.probes[c]                        # the probe slot's scores of this step (frozen k and x)
        pb === nothing || push!(pb.sc, [rr .* x for x in fields_of(RD.um, A1, A2, A3, slotrow(RD, c, RD.Mp))])
        Φs = ctl.Φ === nothing ? zeros(Mr) : [ctl.Φ(P, s.xs[end]) for s in R.reps]
        wts = ML.drift_weights(ctl, Φs, R.active)
        W[b1:b1+Mr-1] .= wts                     # the probe slot: weight 0
        sk̄ = [sum(wts[j] .* sks[j][f] for j in 1:Mr) for f in eachindex(Kraw)]
        push!(recs, (; it, sks, wts, Φs, moved=moved[c], Kraw, rs=ML.masked_means([rr .* x for x in sk̄], mask), fk=ML.masked_means(fk, mask),
                     tw, nclip=0))
    end
    tim["gate"] += time() - th
    # Langevin step and the dispatch following
    tim["langevin"] += @elapsed begin
        out = dev[:lg](dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:cap], dev[:KP], dev[:RP], dev[:LP], acc, dev[:R](W),
                       dev[:lrng], dev[:wsh], dev[:rg], dev[:sg])
        dev[:st], dev[:kg], dev[:ks], dev[:cp], dev[:RS], dev[:dn], dev[:cr], dev[:cap], diag = out
        RD.sync && sync!(dev[:kg])
    end
    dev[:nclip] = diag.nclip                     # read at the host checkpoint (no wait for the step here)
    RD.fresh = false
    RD.Mp > RD.Mr && (tim["probe"] += @elapsed replica_device_probes!(RD, it, recs))
    return recs
end

"""
    replica_device_probes!(RD, it, recs)

The lag probes on the device, after the Langevin step of step `it` (as `lag_probe` of the CPU
driver, which probes after the step): every `replica_probe_every` steps a probe is due per chain;
when the chain's probe slot is free, its candidate (`probe_candidate`; none: the probe is
skipped, as on the CPU) is copied into the slot (`copy_rows`) with its in-flight window, its
interior nodes and β_k∂K at the frozen k and its cost. The slot then runs the device's operation
sweeps at frozen (k, x), weight 0 in the drift. After `probe_steps` steps the probe ends: the
equilibrated score is the mean of the slot's scores over the tail steps, gap = ⟨r ŝ_in − r ŝ_eq⟩/
⟨β_k∂K⟩ on the interior nodes (`lag_probe`'s formula), the record goes into `recs[c].probe` (step
of the copy, replica, adapted), and in the warmup (`it ≤ mlda_warmup` at the end) `adapt_step!`
adapts the chain's steps, which are uploaded.
"""
function replica_device_probes!(RD::ReplicaDevice, it::Int, recs)
    P, ctl, dev = RD.P, RD.ctl, RD.dev
    nch, C = RD.nch, RD.nch * RD.Mp
    n, nt = probe_steps(ctl)
    nt = min(nt, n)
    it % ctl.probe_every == 0 && (RD.due .= true)
    done = [c for c in 1:nch if RD.probes[c] !== nothing && length(RD.probes[c].sc) >= n]
    cand = Pair{Int,Int}[]
    for c in 1:nch
        (RD.due[c] && (RD.probes[c] === nothing || c in done)) || continue      # busy: stays due
        RD.due[c] = false
        j = ML.probe_candidate(RD.sets[c], ctl)
        j === nothing || push!(cand, c => j)
    end
    (isempty(done) && isempty(cand)) && return nothing
    _, _, opd, shd = dev[:rb](dev[:st], dev[:kg], dev[:ks], dev[:dn], dev[:cr], dev[:cap])     # costs only (the state is kept)
    op, shed = HostFloat.(host(opd)), HostFloat.(host(shd))
    adapted = false
    for c in done
        pb, R = RD.probes[c], RD.sets[c]
        p = slotrow(RD, c, RD.Mp)
        eq = sum(pb.sc[end-nt+1:end]) ./ nt
        inflight = sum(pb.hist) ./ length(pb.hist)
        rs_eq, rs_in = ML.masked_means(eq, pb.mask), ML.masked_means(inflight, pb.mask)
        pr = (; gap=(rs_in .- rs_eq) ./ pb.fk, rs_eq, rs_in, fk=pb.fk, op_in=pb.op, op_eq=op[p], shed_in=pb.shed, shed_eq=shed[p])
        η0 = copy(R.ηw)
        ad = ctl.adapt && it <= RD.cfg.mlda_warmup && ML.adapt_step!(R, pr.gap, ctl)
        adapted |= ad
        recs[c] = merge(recs[c], (; probe=ML.probe_record(pb.it, pb.j, pr, η0, copy(R.ηw), ad)))
        RD.probes[c] = nothing
    end
    adapted && (dev[:LP] = dev[:R](lgv_params(P, RD.M, RD.ix; κs=ctl.κs, ηw=[R.ηw for R in RD.sets])))
    isempty(cand) && return nothing
    kg, ks, kr = host(dev[:kg]), host(dev[:ks]), host(dev[:RS].kr)
    src = collect(1:C)
    for (c, j) in cand
        R = RD.sets[c]
        b, p = slotrow(RD, c, j), slotrow(RD, c, RD.Mp)
        src[p] = b
        Kraw = fields_of(RD.um, kg, ks, kr, b)
        mask = ML.interior_nodes(P, Kraw)
        RD.probes[c] = DeviceProbe(j, it, copy(R.hist[j]), mask, ML.masked_means(host_forces(P, Kraw), mask), op[b], shed[b],
                                   Vector{Vector{HostFloat}}[])
        RD.wsh[p, :] .= RD.wsh[b, :]
    end
    R = dev[:R]
    dev[:st], dev[:cp], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:cap] =
        dev[:pc](dev[:st], dev[:cp], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:cap], R(src))
    dev[:wsh] = R(RD.wsh)
    return nothing
end

"""
    run_replicas_device(cfg; families, budget, compile=true, sync=true) -> results

The replica–Langevin driver on the device (`device_sampler = "replicas"`): `replica_device_setup`,
then `mlda_warmup + nsamples` steps of `replica_device_step!` (both multiples of `device_sweeps`),
a host checkpoint every `device_sweeps` steps (the residual cache rebuilt on the device and its
drift recorded; monitored quantities: the replica mean, as `run_replicas`; the replica records;
the snapshot boundary). Outputs: `chains_report` over the chains, `write_replica_outputs`,
`write_replica_summary` and the device timings. Snapshots and resume in the CPU driver's layout
(`write_replica_snapshot`); exit code 75 after a stop.
"""
function run_replicas_device(cfg::ML.Config; families=FAMILIES_FAST, budget=DENSE_BUDGET[], compile::Bool=true, hook=nothing,
                             sync::Bool=true)
    t0 = time()
    nwarm, nprod, spc = cfg.mlda_warmup, cfg.nsamples, cfg.device_sweeps
    (nwarm % spc == 0 && nprod % spc == 0) || error("device replica run: mlda_warmup and nsamples must be multiples of device_sweeps ($spc)")
    Ck = ML.Checkpointer(cfg; hook=hook)
    RD = replica_device_setup(cfg; families, budget, compile, sync)
    P, sets, ctl = RD.P, RD.sets, RD.ctl
    Ck.restarts = RD.restarts
    snap = RD.snap
    snap === nothing || ML.logmsg(cfg, "resuming from snapshot $(cfg.checkpoint) (restart $(RD.restarts)): $(snap.where_)")
    it0 = snap === nothing ? 0 : snap.it
    it0 % spc == 0 || error("snapshot $(cfg.checkpoint) at step $it0: not a host checkpoint of device_sweeps = $spc")
    nch = RD.nch
    traces = snap === nothing ? [Vector{Vector{HostFloat}}() for _ in 1:nch] : snap.traces
    rtr = snap === nothing ? [Dict{String,Vector{Vector{HostFloat}}}() for _ in 1:nch] : snap.rtraces
    ptr = snap === nothing ? [Dict{String,Vector{Vector{HostFloat}}}() for _ in 1:nch] : snap.ptraces
    names = ML.monitor_names(P)
    # a snapshot holds the host States: downloaded first if the device moved since (never stale)
    snapdata = () -> (RD.fresh || replica_download!(RD);
                      (; states=[s for R in sets for s in R.reps], sets, kinds=RD.kinds, traces, rtraces=rtr, ptraces=ptr))
    try
        if it0 == 0
            replica_download!(RD)
            ML.boundary!(Ck, cfg, P, :replicas, 0, snapdata; force=true, writer=ML.write_replica_snapshot)
        end
        for it in it0+1:nwarm+nprod
            recs = replica_device_step!(RD, it)
            for c in 1:nch
                haskey(recs[c], :probe) || continue
                for (k, v) in recs[c].probe
                    push!(get!(ptr[c], k, Vector{Vector{HostFloat}}()), v)
                end
            end
            it % spc == 0 || continue
            # ---- host checkpoint
            th = time()
            RD.tim["rebuild"] += @elapsed (drift = replica_rebuild!(RD))
            RD.tim["download"] += @elapsed replica_download!(RD)
            RD.tim["host"] += @elapsed replica_redraw!(RD)         # new random sites, same shapes (sources.jl)
            ncl = host(RD.dev[:nclip])
            for c in 1:nch
                push!(traces[c], ML.replica_monitor(P, RD.LM, sets[c]))
                for (k, v) in ML.replica_record(P, sets[c], merge(recs[c], (; nclip=Int(ncl[c]))), RD.kref, time() - t0)
                    push!(get!(rtr[c], k, Vector{Vector{HostFloat}}()), v)
                end
                push!(get!(rtr[c], "residual_drift", Vector{Vector{HostFloat}}()), drift[slotrow(RD, c, 1):slotrow(RD, c, RD.Mr)])
            end
            RD.tim["host"] += time() - th
            if (it ÷ spc) % max(1, (nwarm + nprod) ÷ spc ÷ 10) == 0 || it == nwarm + nprod
                for c in 1:nch
                    R = sets[c]
                    ML.logmsg(cfg, @sprintf("  step %4d/%d chain %d: k [%s] GW, active %d/%d, op %.4e, shed %.3g MWh, ηw [%s], r drift %.2g MW  [%.0fs]",
                                            it, nwarm + nprod, c, join((@sprintf("%.1f", x / 1e3) for x in rtr[c]["ktot"][end]), ", "),
                                            count(R.active), RD.Mr, ML.mean(s.parts.op for s in R.reps),
                                            ML.mean(s.parts.shed for s in R.reps), join((@sprintf("%.1e", x) for x in R.ηw), ", "),
                                            maximum(rtr[c]["residual_drift"][end]), time() - t0))
                end
            end
            it < nwarm + nprod && ML.boundary!(Ck, cfg, P, :replicas, it, snapdata; writer=ML.write_replica_snapshot)
        end
        nst = nwarm + nprod - it0
        ML.logmsg(cfg, "device profile per step ($(ctl.inner) operation sweeps, $(nch * RD.Mr) replicas, $(RD.Mp - RD.Mr) probe slot(s) per chain): " *
                       join((@sprintf("%s %.3f s", k, RD.tim[k] / max(nst, 1)) for k in sort(collect(keys(RD.tim)))), ", "))
        res = ML.chains_report(cfg, P, names, traces, RD.kinds, nothing, nwarm ÷ spc, nprod ÷ spc, t0)
        ML.write_replica_outputs(cfg, P, RD.info, ctl, rtr, ptr, RD.restarts, snap)
        ML.write_replica_summary(cfg, P, ctl, rtr, ptr, RD.kinds, nwarm ÷ spc, res.passed, time() - t0)
        ML.HDF5.h5open(cfg.out, "r+") do f
            for (k, v) in RD.tim
                f["device_profile/$k"] = v / max(nst, 1)
            end
            for (k, v) in RD.tc
                f["device_profile/compile_$k"] = v
            end
            f["replicas/record_every"] = spc
            f["replicas/probe_slots"] = RD.Mp - RD.Mr
        end
        return (; res..., P, sets, RD, status=:finished, exitcode=0, restarts=RD.restarts)
    catch e
        e isa ML.CheckpointStop || rethrow()
        ML.logmsg(cfg, "stop file $(cfg.stop_file): stopped after the snapshot $(e.path) ($(e.where_)); exit code $(ML.EXIT_CHECKPOINTED)")
        return (; status=:stopped, exitcode=ML.EXIT_CHECKPOINTED, states=e.states, sets, RD, restarts=RD.restarts, P)
    end
end
