# Host loop of device runs (docs/reactant_design.md §5, phase 4).
#
# Per call of the device:
#   1. `run_sweeps`: `device_sweeps` sweeps of the operation families (compiled once; the
#      sweep count is traced);
#   2. `capacity_pass`: kscale and kpeak Metropolis passes over the investable units
#      (compiled once), on the same flat state reshaped to node arrays, then the
#      capacity-dependent chain parameters recomputed on the device;
#   3. a checkpoint on the host: download, monitored quantities per chain, online bias refits
#      at 2^j during the warmup (pooled over chains, frozen afterwards), then the bias
#      aggregates are re-packed and uploaded.
# Proposal normals and uniforms of the capacity moves are drawn on the host (a few thousand
# numbers per call); the operation rounds use Reactant's RNG.

"""
Proxy class of every (node, step), 1 = surplus, flattened (C, T·N): the CPU's `frame_injections`
p = renewable output − load, less the load-weighted share `w` (C, N) of the system total, is
positive. With renewables as load, renewable output − load = −dnet. A function of (x, k) only.
"""
function proxy_class(dnet, w)
    p = -dnet
    p = p .- reshape(w, size(w, 1), 1, size(w, 2)) .* sum(p; dims=3)
    return Float64.(reshape(p .> 0, size(dnet, 1), :))     # broadcast after the reshape: a materialized traced array
end

"Load-weighted node shares (C, N) of the chains' weather (the CPU's `frame_injections` weights)."
function load_shares(P::ML.Problem, ss::Vector{ML.State})
    N = P.E.sys.N
    w = zeros(length(ss), N)
    for (c, s) in enumerate(ss)
        Xn = ML.weather_nodes(P.W, s.ℓ, s.xs[end])
        d = vec(ML.mean(max.(Xn[1], 0.0); dims=2))
        w[c, P.E.wspecs[1].B.support] .= d ./ sum(d)
    end
    return w
end

"Capacity-dependent chain parameters from raw capacities (the formulas of `chain_params`, traceable)."
function chain_params_dev(M, kg, ks, cls, Cn::Val{C}) where {C}
    A = reshape(M.gav, 1, :)
    inv = reshape(M.ginv, 1, :)
    kc = ifelse.(inv, clamp.(kg, reshape(M.glo, 1, :), reshape(M.ghi, 1, :)), reshape(M.gkfix, 1, :))
    κg = A .* kc
    q = min.(reshape(M.gk0, 1, :), kc)
    oldfirst = reshape(M.gmc0 .<= M.gmc, 1, :)
    qlo = ifelse.(oldfirst, q, kc .- q)
    yk = ifelse.(kc .> 0, qlo ./ ifelse.(kc .> 0, kc, 1.0), 0.0)
    sinv = reshape(M.sinv, 1, :)
    ksc = ifelse.(sinv, clamp.(ks, reshape(M.slo, 1, :), reshape(M.shi, 1, :)), reshape(M.skfix, 1, :))
    return ChainParams(κg, kc .+ M.κ0, A .* qlo, yk, ksc, ksc .+ M.κ0, cls)
end

"""
Capacity moves on the flat loop state, in order: renewable scaling (`rq`), the peak sets
recomputed from the net load, scaling of the dispatchable (`gq`) and storage (`sq`) units,
peak-set moves (`gq`); Metropolis with given proposals, optional bias. Returns
(st, kg, ks, cp, BS, RS, dnet, cren, capren).
"""
function capacity_pass(st::FlatState, kg, ks, dnet, cren, capren, KP, Z, U, Zp, Up, gq, sq, σk, wsh,
                       BD, BS, gitem, sitem, RP, RS, Zr, Ur, rq, ritem, ::Val{dims}; σ, βk, βv, physical::Bool) where {dims}
    C, T, N, Ug, Us, Ul, mpk = dims
    m3(x, n) = reshape(x, C, T, n) .+ 0.0          # materialized (a lazy reshape of a traced array has another type)
    r = m3(st.r, N)
    if length(rq) > 0                              # renewables first: they move the net load and the peak sets
        rb = BD === nothing ? nothing : (BD, BS, ritem)
        out = kren_mh_loop(RP, RS, r, dnet, cren, capren, Zr, Ur, rq; σ, βk, βv, bias=rb)
        RS, r, dnet, cren, capren = out[1], out[2], out[3], out[4], out[5]
        rb === nothing || (BS = out[6])
    end
    inS = peak_sets_dev(dnet, KP.G, Val(mpk))
    S = DeviceState(m3(st.θ, N), m3(st.ω, Ug), m3(st.uc, Ug), m3(st.e, Us), m3(st.u, max(Ul, 1)), kg, ks, dnet, cren, capren)
    bias = BD === nothing ? nothing : (BD, BS, gitem, sitem)
    out = kscale_mh_loop(KP, S, r, Z, U, gq, sq; σ, βk, βv, physical, bias)
    S, r = out[1], out[2]
    bias === nothing || (bias = (BD, out[3], gitem, sitem))
    out = kpeak_mh_loop(KP, S, r, Zp, Up, gq, σk, inS; βk, βv, physical, bias)
    S, r = out[1], out[2]
    BSn = bias === nothing ? nothing : out[3]
    m2(x) = reshape(x, C, :) .+ 0.0
    st2 = FlatState(m2(S.θ), m2(S.ω), m2(S.uc), m2(S.e), m2(S.u), m2(r))
    cls = proxy_class(S.dnet, wsh)                 # from (x, k) after this pass, for the next operation sweeps
    cp = chain_params_dev(KP_model_view(KP), S.kg, S.ks, cls, Val(C))
    return st2, S.kg, S.ks, cp, BSn, RS, S.dnet, S.cren, S.capren
end

"""
One compiled program per checkpoint: `ns` operation sweeps (`run_sweeps`, the count traced,
Reactant's RNG), then the capacity pass on the same state (`capacity_pass`, host-drawn proposals).
`kparts` are the capacity-pass arguments after the state (`kargs(…)[2:end]`). Returns the outputs
of `capacity_pass`. Equal to the two separate calls to round-off (`test/test_program.jl`).
"""
function sweep_program(st::FlatState, cp, sp, k, tabs, ns, rng, fv::Val, kparts...; dims, σ, βk, βv, physical)
    st = run_sweeps(st, cp, sp, k, tabs, ns, rng, fv)
    return capacity_pass(st, kparts..., dims; σ, βk, βv, physical)
end

"""
Operation families of a device run: the plain families, then the compensated families (`:spair`;
LINEAR `:tsocL`/`:tpotL` when the model needs requirement tracking (line losses, lossy links),
else share-preserving `:tsoc`/`:tpot`) and the level-bucketed cell templates of `ci`.
"""
device_families(M::DeviceModel, ci) =
    (FAMILIES_FAST..., :spair, (tracking(M) ? (:tsocL, :tpotL) : (:tsoc, :tpot))..., (ci === nothing ? () : cell_families(ci))...)

"Write the renewable capacities of chain c (device order `ren_units`) into the State `s` (coefficients)."
function ren_to_host!(P::ML.Problem, kr::AbstractMatrix, c::Int, s::ML.State)
    E = P.E
    lk = P.lk[s.ℓ+1]
    units = ren_units(P)
    for (gi, g) in enumerate(E.gens)
        (g.vf == 0 && g.kf > 0) || continue
        qs = [q for (q, (a, i)) in enumerate(units) if a == gi]
        ML.fieldview(s.k, lk, E.kspecs, g.kf) .= ML.cols(E.kspecs[g.kf].B, lk.m[g.kf])' * reshape(kr[c, qs], :, 1)
    end
    return s
end

"The fields of the kmove parameters that `chain_params_dev` needs (a NamedTuple view)."
KP_model_view(KP) = KP

"""
    device_setup(cfg) -> NamedTuple

Everything a device run needs before compiling: the problem and the host chains, the packed
model and the plan, the device arrays and the two programs as plain functions with their
argument builder `kargs` and the host draws `draws()`. Shared by `run_device` and the profiler
(`device/scripts/profile_device.jl`).
"""
function device_setup(cfg::ML.Config; families=FAMILIES_FAST)
    rng = MersenneTwister(cfg.seed)
    P, sL, info = ML.setup(cfg; rng=rng)
    L = ML.nlevels(P) - 1
    if P.use_bias                                   # level-0 warmup and bias learning (CPU), as `run_chains`
        S0 = ML.new_sampler(P, sL, cfg)
        s0 = ML.predescend(P, ML.restrict_state(P, sL, 0), cfg)
        ML.logmsg(cfg, "level-0 warmup and bias learning ($(cfg.warmup0) its)")
        s0, _, _, _ = ML.run_level0!(rng, P, s0, S0, cfg.warmup0, cfg; learn=true, label="warmup")
        P.bias.active = false
        ML.lift_bias!(P, cfg)                       # as the CPU driver: bias at the chains' level
        sL = ML.evaluate!(P, sL)
    end
    K = cfg.nchains
    kinds = [cfg.chain_starts[mod1(c, length(cfg.chain_starts))] for c in 1:K]
    chains = [ML.chain_start(MersenneTwister(cfg.seed + 7 * c), P, sL, kinds[c], false) for c in 1:K]
    M, ix = pack(P, L)
    S = to_device(P, M, ix, chains)
    wsh = load_shares(P, chains)
    LM = ML.LocalModel(P, L)
    fams = families === :all ? device_families(M, cell_info(P, LM, ix)) : families
    pl = plan(M, S, P.βv; cls=proxy_class(S.dnet, wsh), families=fams, cells=cell_info(P, LM, ix))
    T, N = M.T, M.N
    Ug, Us, Ul = length(ix.gidx), length(ix.sidx), length(M.kcap)
    dims = Val((K, T, N, Ug, Us, Ul, max(1, round(Int, 0.05 * T))))    # last: peak-set size (q = 0.05, as the CPU)
    units = kmove_units(M)
    gq = [q for (k, q) in units if k == 1]; sq = [q for (k, q) in units if k == 2]
    nprop = cfg.kmove_nprop
    σk = cfg.kpeak_sigma .* max.(vec(maximum(S.kg[:, gq]; dims=1)), 100.0)     # fixed per-unit scales
    RP, RS = pack_ren(P, M, chains)
    rq = collect(1:length(ren_units(P)))
    usebias = P.use_bias
    packb() = usebias ? pack_bias(P, ix, chains) : (nothing, nothing, nothing, nothing, nothing)
    BD, BS, gitem, sitem, ritem = packb()
    # ---- device arrays and compiled programs
    Reactant.set_default_backend(cfg.device_backend)
    R = Reactant.to_rarray
    st, cp, sp, kk, tabs = R(pl.st), R(pl.cp), R(pl.sp), R(pl.k), R(pl.dev)
    kgd, ksd = R(S.kg), R(S.ks)
    dn, crd, cap = R(S.dnet), R(S.cren), R(S.capren)
    KPd = R(KParams(M))
    gqd, sqd, σkd, wshd = R(gq), R(sq), R(σk), R(wsh)
    BDd = BD === nothing ? nothing : R(BD); BSd = BS === nothing ? nothing : R(BS)
    gid = gitem === nothing ? nothing : R(gitem); sid = sitem === nothing ? nothing : R(sitem)
    rid = ritem === nothing ? nothing : R(ritem)
    RPd, RSd, rqd = R(RP), R(RS), R(rq)
    drng = Reactant.ReactantRNG(R(UInt64[cfg.seed, 0x9e3779b9]))
    fv = Val(pl.fams)
    sw(st, cp, sp, k, tabs, ns, rng) = run_sweeps(st, cp, sp, k, tabs, ns, rng, fv)
    hr = MersenneTwister(cfg.seed + 99)
    draws() = (R(randn(hr, K, length(units), nprop)), R(rand(hr, K, length(units), nprop)),
               R(randn(hr, K, length(gq), nprop)), R(rand(hr, K, length(gq), nprop)),
               R(randn(hr, K, max(length(rq), 1), nprop)), R(rand(hr, K, max(length(rq), 1), nprop)))
    Z, U, Zp, Up, Zr, Ur = draws()
    # every device array is an argument (captured arrays would be compiled in as constants)
    σ_, βk_, βv_, ph_ = cfg.kmove_sigma, P.βk, P.βv, P.E.measure == :physical
    kp(args...) = capacity_pass(args..., dims; σ=σ_, βk=βk_, βv=βv_, physical=ph_)
    fp(st, cp, sp, k, tabs, ns, rng, kparts...) = sweep_program(st, cp, sp, k, tabs, ns, rng, fv, kparts...;
                                                                 dims=dims, σ=σ_, βk=βk_, βv=βv_, physical=ph_)
    kargs(st, kg, ks, dn, crd, cap, Z, U, Zp, Up, BD, BS, RS, Zr, Ur) =
        (st, kg, ks, dn, crd, cap, KPd, Z, U, Zp, Up, gqd, sqd, σkd, wshd, BD, BS, gid, sid, RPd, RS, Zr, Ur, rqd, rid)
    return (; P, info, K, kinds, chains, M, ix, S, pl, LM, units, usebias, packb, R, sw, kp, fp, kargs, draws,
            st, cp, sp, kk, tabs, kgd, ksd, dn, crd, cap, BDd, BSd, RSd, drng, Z, U, Zp, Up, Zr, Ur,
            nsd=Reactant.ConcreteRNumber(cfg.device_sweeps))
end

"""
    run_device(cfg) -> results

Device run of `cfg.nchains` chains (see the module comment): `mlda_warmup` + `nsamples`
checkpoints, each after `device_sweeps` operation sweeps and one capacity pass.
"""
# merged = true compiles sweeps + capacity pass as one program (`sweep_program`, bitwise equal to
# the two calls, test/test_program.jl), but compiles ≈ 5× longer (295 s against 50 + 11 s at 40
# nodes daily) for one dispatch saved per checkpoint; the alternation is the default.
function run_device(cfg::ML.Config; merged::Bool=false, families=FAMILIES_FAST)
    t0 = time()
    D = device_setup(cfg; families=families)
    (; P, K, kinds, chains, M, ix, S, LM, usebias, packb, R, sw, kp, fp, kargs, draws, nsd) = D
    (; st, cp, sp, kk, tabs, kgd, ksd, dn, crd, cap, BDd, BSd, RSd, drng, Z, U, Zp, Up, Zr, Ur) = D
    kpart(st, kgd, ksd, dn, crd, cap, Z, U, Zp, Up, BDd, BSd, RSd, Zr, Ur) =
        kargs(st, kgd, ksd, dn, crd, cap, Z, U, Zp, Up, BDd, BSd, RSd, Zr, Ur)[2:end]
    if merged                                       # one program per checkpoint: sweeps, then the capacity pass
        tc1 = @elapsed ffp = @compile fp(st, cp, sp, kk, tabs, nsd, drng, kpart(st, kgd, ksd, dn, crd, cap, Z, U, Zp, Up, BDd, BSd, RSd, Zr, Ur)...)
        tc2 = 0.0
        ML.logmsg(cfg, @sprintf("device (%s): compiled one program (operation sweeps + capacity pass, %.0f s); %d chains",
                                cfg.device_backend, tc1, K))
    else
        tc1 = @elapsed fsw = @compile sw(st, cp, sp, kk, tabs, nsd, drng)
        tc2 = @elapsed fkp = @compile kp(kargs(st, kgd, ksd, dn, crd, cap, Z, U, Zp, Up, BDd, BSd, RSd, Zr, Ur)...)
        ML.logmsg(cfg, @sprintf("device (%s): compiled the operation sweeps (%.0f s) and the capacity pass (%.0f s); %d chains",
                                cfg.device_backend, tc1, tc2, K))
    end
    # ---- loop
    nwarm, nprod = cfg.mlda_warmup, cfg.nsamples
    online = usebias && cfg.bias_online
    OB = ML.OnlineBias(; every=cfg.bias_online_every, freeze=K * nwarm, nmin=cfg.bias_online_min, trust=cfg.bias_trust)
    PB = (usebias && cfg.bias_post) ? ML.PostBias() : nothing
    names = ML.monitor_names(P)
    traces = [Vector{Vector{Float64}}() for _ in 1:K]
    Sh = deepcopy(S)
    tim = Dict("operation" => 0.0, "capacity" => 0.0, "draws" => 0.0, "download" => 0.0, "host" => 0.0)
    sync(x) = (Reactant.synchronize(x); x)
    for it in 1:nwarm+nprod
        if merged                                   # "operation" then holds sweeps and capacity pass
            tim["operation"] += @elapsed begin
                st, kgd, ksd, cp, BSn, RSd, dn, crd, cap = ffp(st, cp, sp, kk, tabs, nsd, drng,
                                                               kpart(st, kgd, ksd, dn, crd, cap, Z, U, Zp, Up, BDd, BSd, RSd, Zr, Ur)...)
                sync(st.r); sync(kgd)
            end
            tim["draws"] += @elapsed (Z, U, Zp, Up, Zr, Ur = draws())
        else
            tim["operation"] += @elapsed (st = fsw(st, cp, sp, kk, tabs, nsd, drng); sync(st.r))
            tim["draws"] += @elapsed (Z, U, Zp, Up, Zr, Ur = draws())
            tim["capacity"] += @elapsed begin
                st, kgd, ksd, cp, BSn, RSd, dn, crd, cap = fkp(kargs(st, kgd, ksd, dn, crd, cap, Z, U, Zp, Up, BDd, BSd, RSd, Zr, Ur)...)
                sync(st.r); sync(kgd)
            end
        end
        BSn === nothing || (BSd = BSn)
        th0 = time()
        # checkpoint on the host
        tim["download"] += @elapsed begin
            unflat!(Sh, st)
            Sh.kg .= Array(kgd); Sh.ks .= Array(ksd)
            krh = Array(RSd.kr)
        end
        refit = false
        for c in 1:K
            ren_to_host!(P, krh, c, chains[c])
            chains[c] = to_host!(P, M, ix, Sh, c, chains[c])
            if online && it <= nwarm
                chains[c], upd = ML.online_bias_step!(P, chains[c], OB; monotone=ML.bias_monotone_args(P, cfg))
                refit |= upd
            end
            q = ML.monitor_quantities(P, chains[c]; LM=LM)
            push!(traces[c], q)
            (PB !== nothing && it > nwarm) && ML.post_record!(PB, P, chains[c], q)
        end
        if refit                                    # the target changed: re-pack the bias aggregates
            chains .= [ML.evaluate!(P, x) for x in chains]    # in place: `packb` reads this vector
            BD, BS, gitem, sitem, ritem = packb()
            BDd, BSd = R(BD), R(BS)                 # new values, same shapes: no recompile
        end
        tim["host"] += time() - th0
        if it % max(1, (nwarm + nprod) ÷ 10) == 0
            tot = [tr[end][length(P.E.kspecs)+3] for tr in traces]
            ML.logmsg(cfg, @sprintf("  checkpoint %4d/%d (%d sweeps each)  total per chain %s  [%.0fs]", it, nwarm + nprod,
                                    cfg.device_sweeps, join((@sprintf("%.4e", x) for x in tot), " "), time() - t0))
        end
    end
    ncall = nwarm + nprod
    tim["host"] -= tim["download"]
    ML.logmsg(cfg, "device profile per call (" * string(cfg.device_sweeps) * " operation sweeps, " * string(K) * " chains): " *
                   join((@sprintf("%s %.3f s", k, tim[k] / ncall) for k in ("operation", "capacity", "draws", "download", "host")), ", "))
    res = ML.chains_report(cfg, P, names, traces, kinds, PB, nwarm, nprod, t0)
    ML.HDF5.h5open(cfg.out, "r+") do f
        for (k, v) in tim
            f["device_profile/$k"] = v / ncall
        end
        f["device_profile/compile_operation"] = tc1; f["device_profile/compile_capacity"] = tc2
    end
    return res
end
