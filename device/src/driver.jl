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
function capacity_pass(st::FlatState, kg, ks, dnet, cren, capren, KP, Z, U, Zp, Up, gq, sq, σk, cls,
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
    cp = chain_params_dev(KP_model_view(KP), S.kg, S.ks, cls, Val(C))
    return st2, S.kg, S.ks, cp, BSn, RS, S.dnet, S.cren, S.capren
end

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
    run_device(cfg) -> results

Device run of `cfg.nchains` chains (see the module comment): `mlda_warmup` + `nsamples`
checkpoints, each after `device_sweeps` operation sweeps and one capacity pass.
"""
function run_device(cfg::ML.Config)
    t0 = time()
    rng = MersenneTwister(cfg.seed)
    P, sL, info = ML.setup(cfg; rng=rng)
    L = ML.nlevels(P) - 1
    if P.use_bias                                   # level-0 warmup and bias learning (CPU), as `run_chains`
        S0 = ML.new_sampler(P, sL, cfg)
        s0 = ML.predescend(P, ML.restrict_state(P, sL, 0), cfg)
        ML.logmsg(cfg, "level-0 warmup and bias learning ($(cfg.warmup0) its)")
        s0, _, _, _ = ML.run_level0!(rng, P, s0, S0, cfg.warmup0, cfg; learn=true, label="warmup")
        P.bias.active = false
    end
    K = cfg.nchains
    kinds = [cfg.chain_starts[mod1(c, length(cfg.chain_starts))] for c in 1:K]
    chains = [ML.chain_start(MersenneTwister(cfg.seed + 7 * c), P, sL, kinds[c], false) for c in 1:K]
    M, ix = pack(P, L)
    S = to_device(P, M, ix, chains)
    pl = plan(M, S, P.βv)
    T, N = M.T, M.N
    Ug, Us, Ul = length(ix.gidx), length(ix.sidx), length(M.kcap)
    dims = Val((K, T, N, Ug, Us, Ul, max(1, round(Int, 0.05 * T))))    # last: peak-set size (q = 0.05, as the CPU)
    units = kmove_units(M)
    gq = [q for (k, q) in units if k == 1]; sq = [q for (k, q) in units if k == 2]
    nprop = cfg.kmove_nprop
    σk = cfg.kpeak_sigma .* max.(vec(maximum(S.kg[:, gq]; dims=1)), 100.0)     # fixed per-unit scales
    LM = ML.LocalModel(P, L)
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
    gqd, sqd, σkd, clsd = R(gq), R(sq), R(σk), R(pl.cp.cls)
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
    kargs(st, kg, ks, dn, crd, cap, Z, U, Zp, Up, BD, BS, RS, Zr, Ur) =
        (st, kg, ks, dn, crd, cap, KPd, Z, U, Zp, Up, gqd, sqd, σkd, clsd, BD, BS, gid, sid, RPd, RS, Zr, Ur, rqd, rid)
    tc1 = @elapsed fsw = @compile sw(st, cp, sp, kk, tabs, Reactant.ConcreteRNumber(cfg.device_sweeps), drng)
    tc2 = @elapsed fkp = @compile kp(kargs(st, kgd, ksd, dn, crd, cap, Z, U, Zp, Up, BDd, BSd, RSd, Zr, Ur)...)
    ML.logmsg(cfg, @sprintf("device (%s): compiled the operation sweeps (%.0f s) and the capacity pass (%.0f s); %d chains",
                            cfg.device_backend, tc1, tc2, K))
    # ---- loop
    nwarm, nprod = cfg.mlda_warmup, cfg.nsamples
    online = usebias && cfg.bias_online
    OB = ML.OnlineBias(; every=cfg.bias_online_every, freeze=K * nwarm)
    PB = (usebias && cfg.bias_post) ? ML.PostBias() : nothing
    names = ML.monitor_names(P)
    traces = [Vector{Vector{Float64}}() for _ in 1:K]
    nsd = Reactant.ConcreteRNumber(cfg.device_sweeps)
    Sh = deepcopy(S)
    for it in 1:nwarm+nprod
        st = fsw(st, cp, sp, kk, tabs, nsd, drng)
        Z, U, Zp, Up, Zr, Ur = draws()
        st, kgd, ksd, cp, BSn, RSd, dn, crd, cap = fkp(kargs(st, kgd, ksd, dn, crd, cap, Z, U, Zp, Up, BDd, BSd, RSd, Zr, Ur)...)
        BSn === nothing || (BSd = BSn)
        # checkpoint on the host
        unflat!(Sh, st)
        Sh.kg .= Array(kgd); Sh.ks .= Array(ksd)
        krh = Array(RSd.kr)
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
            chains = [ML.evaluate!(P, x) for x in chains]
            BD, BS, gitem, sitem, ritem = packb()
            BDd, BSd = R(BD), R(BS)                 # new values, same shapes: no recompile
        end
        if it % max(1, (nwarm + nprod) ÷ 10) == 0
            tot = [tr[end][length(P.E.kspecs)+3] for tr in traces]
            ML.logmsg(cfg, @sprintf("  checkpoint %4d/%d (%d sweeps each)  total per chain %s  [%.0fs]", it, nwarm + nprod,
                                    cfg.device_sweeps, join((@sprintf("%.4e", x) for x in tot), " "), time() - t0))
        end
    end
    return ML.chains_report(cfg, P, names, traces, kinds, PB, nwarm, nprod, t0)
end
