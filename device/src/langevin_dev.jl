# Langevin capacity step of the replica sampler on the device (docs/replica_sampler.md §1,
# docs/system/device.md §5). The CPU reference until the sampler is in src/ is the prototype
# (scripts/replicas/replicas.jl: `lgv_knew`, `lgv_apply!`; scripts/langevin/langevin.jl).
#
# Per investable unit, in w = log(k + κ_s) (κ_s per carrier):
#     w' = w + clip(η_w ((k + κ_s) g + 1)) + √(2η_w) ξ,   g = −β_k ∂K + r s̄_k,  r = β_k/β_v,
# reflected at k = κ_min and (box = true) at the upper bound of the capacity box, s̄_k the
# weighted mean of the replicas' scores (`cap_score`). η_w = 0 freezes a unit.
#
# The dispatch follows the step in every replica, in the order of the prototype:
#   1. dispatchable units at fixed output: utilizations and commitment scaled by c(k0)/c(k1)
#      (the commit map of `kscale`); the residual takes the output lost to the clamp y ≤ 1;
#   2. renewables (one carrier at a time, its units at distinct nodes) and then storage at fixed
#      utilization (output A k, state-of-charge fractions), the node's dispatchable generators
#      following share-preservingly (the kop curve, `kop_term`/`kop_eval` with `commit`):
#      P1 = (Q − Δinj)₊ + e, y_i = σ_i P1/κ_i with σ_i = κ_i y_i/P, at every step of the node,
#      for the compensators with |A k| ≥ κ_min; a unit with k0 < κ_min moves alone.

"Per-unit parameters of the Langevin step (type-parametric: uploaded as device arrays)."
struct LgvParams{V,Mx}
    κg::V; ηg::Mx; hg::V         # dispatchable units: offset κ_s, step η_w per chain (C/M, U_g; 0: frozen), upper box bound
    κs::V; ηs::Mx; hs::V         # storage
    κr::V; ηr::Mx; hr::V         # renewables (`ren_units` order)
end

"""
    lgv_params(P, M, ix; κs, ηw) -> LgvParams

Per-unit arrays from per-k-field values: `κs` a vector over `P.E.kspecs`, `ηw` one such vector
for all chains or a vector of them, one per chain (the steps adapted per chain); units without
a k field (fixed capacities) get η = 0.
"""
function lgv_params(P::ML.Problem, M::DeviceModel, ix::DeviceIndex; κs, ηw)
    E = P.E
    gk = [E.gens[g].kf for (g, i) in ix.gidx]
    sk = [E.stors[s].kf for (s, i) in ix.sidx]
    rk = [E.gens[g].kf for (g, i) in ren_units(P)]
    ηc = eltype(ηw) <: Real ? [ηw] : ηw
    pick(v, kf) = [f > 0 ? Float64(v[f]) : 0.0 for f in kf]
    pickc(kf) = reduce(vcat, [reshape(pick(η, kf), 1, :) for η in ηc])
    hi(kf, idx) = [f > 0 ? Float64(E.kspecs[f].hi[i]) : 1.0 for (f, (_, i)) in zip(kf, idx)]    # one float type (F32: Float32)
    nz(v) = isempty(v) ? zeros(1) : v
    nzm(A) = size(A, 2) == 0 ? zeros(size(A, 1), 1) : A
    return LgvParams(nz(pick(κs, gk)), nzm(pickc(gk)), nz(hi(gk, ix.gidx)),
                     nz(pick(κs, sk)), nzm(pickc(sk)), nz(hi(sk, ix.sidx)),
                     nz(pick(κs, rk)), nzm(pickc(rk)), nz(hi(rk, ren_units(P))))
end

"""
    lgv_w(k, s, f, κs, η, hi, ξ; rr, clip, κmin, box) -> (k′, nclip)

The Langevin step of one unit class (C, U): scores s (r s̄ enters as `rr .* s`), forces f = β_k ∂K,
noise ξ ~ N(0, 1). Elementwise (`lgv_knew` of the prototype).
"""
function lgv_w(k, s, f, κs, η, hi, ξ; rr, clip, κmin, box::Bool)
    κ, ηr = reshape(κs, 1, :), η                          # η: (1 or C/M, U)
    g = -f .+ rr .* s
    w = log.(k .+ κ)
    dw0 = ηr .* ((k .+ κ) .* g .+ 1)
    nclip = sum(Float64.(abs.(dw0) .> clip); dims=2)
    w1 = w .+ clamp.(dw0, -clip, clip) .+ sqrt.(2 .* ηr) .* ξ
    wmin = log.(κmin .+ κ)
    w1 = ifelse.(w1 .< wmin, 2 .* wmin .- w1, w1)
    if box
        wmax = log.(max.(reshape(hi, 1, :), κmin + 1.0) .+ κ)
        w1 = ifelse.((w1 .> wmax) .& (w .<= wmax), max.(2 .* wmax .- w1, wmin), w1)
    end
    return ifelse.(ηr .== 0, k, exp.(w1) .- κ), vec(nclip)
end

# ---------------------------------------------------------------------------
# replicas: C = chains × M, the replicas of a chain contiguous (c = j + (chain − 1) M)
# ---------------------------------------------------------------------------

"Weighted sum over the M replicas of every chain: (C, U) → (C/M, U)."
rep_sum(x, w, ::Val{M}) where {M} = dropdims(sum(reshape(reshape(w, :, 1) .* x, M, :, size(x, 2)); dims=1); dims=1)
"The value of the first replica of every chain: (C, U) → (C/M, U)."
rep_first(x, ::Val{M}) where {M} = reshape(x, M, :, size(x, 2))[1, :, :]
"Every chain's value for each of its M replicas: (C/M, U) → (C, U) (exact: + 0)."
rep_bcast(x, ::Val{M}) where {M} = reshape(reshape(x, 1, size(x, 1), size(x, 2)) .+ zeros(M, 1, 1), M * size(x, 1), size(x, 2)) .+ 0.0

# ---------------------------------------------------------------------------
# dispatch following the capacities
# ---------------------------------------------------------------------------

"""
Dispatchable units at fixed output: capacities kg → kg1 (C, U_g), utilizations and commitment
scaled by c(k0)/c(k1), the residual r (C, T, N) takes the change of the clamped output.
Returns (ω, uc, kg1, r).
"""
function disp_follow(KP, ω, uc, kg, kg1, r)
    kc0 = clamp_cap(kg, KP.ginv, KP.glo, KP.ghi, KP.gkfix)
    kc1 = clamp_cap(kg1, KP.ginv, KP.glo, KP.ghi, KP.gkfix)
    sc = rowk(kc_ratio_d(kc0, kc1))
    ω1 = ω .* sc
    uc1 = ifelse.(rowv(KP.guc) .> 0, uc .* sc, uc)
    A = rowv(KP.gav)
    dp = A .* rowk(kc1) .* clamp.(ω1, 0.0, 1.0) .- A .* rowk(kc0) .* clamp.(ω, 0.0, 1.0)
    return ω1, uc1, kg1, r .- tmul(dp, KP.G)
end

"""
The node's dispatchable generators follow an injection change `dinj` (C, T, N) of the moved units
share-preservingly (the kop curve); `cnode` (C, 1, N) marks the nodes where they do (a moved unit
with k0 ≥ κ_min). Returns (ω, r) with r − dinj − (change of the compensators' output).
"""
function kop_follow(KP, ω, kg, r, dinj, cnode; κmin)
    kc = clamp_cap(kg, KP.ginv, KP.glo, KP.ghi, KP.gkfix)
    κ = rowv(KP.gav) .* rowk(kc)                                   # (C, 1, U_g): A k at every step
    sel = abs.(κ) .>= κmin
    yc = clamp.(ω, 0.0, 1.0)
    Pn = tmul(ifelse.(sel, κ .* ω, 0.0), KP.G)
    Qs = tmul(ifelse.(sel, κ .* yc, 0.0), KP.G)
    nu = tmul(ifelse.(sel, 1.0, 0.0), KP.G)                     # (C, 1, N): compensators per node
    act = cnode .& (nu .> 0) .& (abs.(Pn) .>= 1e-6)
    Q = r .+ Qs
    e = Pn .- max.(Q, 0.0)
    P1 = max.(Q .- dinj, 0.0) .+ e
    actq = tmulT(ifelse.(act, 1.0, 0.0), KP.G) .> 0.5
    upd = actq .& sel
    σ = κ .* ω ./ ifelse.(upd, tmulT(Pn, KP.G), 1.0)
    y1 = σ .* tmulT(P1, KP.G) ./ ifelse.(upd, κ, 1.0)
    ω1 = ifelse.(upd, y1, ω)
    dS = ifelse.(upd, κ .* (clamp.(ω1, 0.0, 1.0) .- yc), 0.0)
    return ω1, r .- dinj .- tmul(dS, KP.G)
end

"Charge/discharge utilizations (pc, mc) of every storage unit at fixed states of charge (`soc_power`), (C, T, U_s)."
function soc_pm(KP, e)
    h, ηin, ηout, qt = rowv(KP.sh), rowv(KP.sηin), rowv(KP.sηout), rowv(KP.sqt)
    ec, ec1 = clamp.(e, 0.0, 1.0), clamp.(nextτ(e), 0.0, 1.0)
    d = h .* (ec1 .- ec) ./ KP.Δt .- qt
    pos = d .>= 0
    pc = ifelse.(pos, ifelse.(ηin .> 0, d ./ ifelse.(ηin .> 0, ηin, 1.0), 0.0), 0.0)
    mc = ifelse.(pos, 0.0, min.(-d .* ηout, 1.0))
    return pc, mc
end

"""
    lgv_commit(KP, RP, st, kg1, ks1, kr1, RS, dnet, cren, capren, rgroups, sgroups; κmin)

Move every replica from its capacities to (kg1, ks1, kr1) (C, ·) with the dispatch following
(module comment). `rgroups`, `sgroups`: tuples of 0/1 unit masks (U_r,), (U_s,), one per carrier,
applied in order (the units of one mask at distinct nodes). Node arrays: ω, uc (C, T, U_g),
e (C, T, U_s), r, dnet (C, T, N). Returns (ω, uc, kg, ks, RS, r, dnet, cren, capren).
"""
function lgv_commit(KP, RP, ω, uc, e, kg, ks, r, kg1, ks1, kr1, RS, dnet, cren, capren, rgroups, sgroups; κmin)
    ω, uc, kg, r = disp_follow(KP, ω, uc, kg, kg1, r)
    kr, A = RS.kr, RS.A
    lo, hi, cap = reshape(RP.lo, 1, :), reshape(RP.hi, 1, :), reshape(RP.cap, 1, :)
    for grp in rgroups                                        # renewables, one carrier at a time
        g = reshape(grp, 1, :) .> 0
        k1 = ifelse.(g, kr1, kr)
        kc0, kc1 = clamp.(kr, lo, hi), clamp.(k1, lo, hi)
        dout = A .* rowk(kc1 .- kc0)                          # (C, T, U_r): output change (0 outside the group)
        dinj = tmul(dout, RP.H)
        cnode = tmul(rowk(ifelse.(g .& (kr .>= κmin), 1.0, 0.0)), RP.H) .> 0.5
        ω, r = kop_follow(KP, ω, kg, r, dinj, cnode; κmin)
        dnet = dnet .- dinj
        cren = cren .+ vec(sum(KP.Δt .* rowv(RP.mc) .* dout; dims=(2, 3)))
        capren = capren .+ vec(sum(cap .* (k1 .- kr) .+ RP.γk .* cap .* (abs.(k1 .- kc1) .- abs.(kr .- kc0)); dims=2))
        kr = k1
    end
    pc, mc = soc_pm(KP, e)
    for grp in sgroups                                        # storage, one carrier at a time
        g = reshape(grp, 1, :) .> 0
        k1 = ifelse.(g, ks1, ks)
        kc0 = clamp_cap(ks, KP.sinv, KP.slo, KP.shi, KP.skfix)
        kc1 = clamp_cap(k1, KP.sinv, KP.slo, KP.shi, KP.skfix)
        dinj = tmul((mc .- pc) .* rowk(kc1 .- kc0), KP.S)     # injection: discharge − charge
        cnode = tmul(rowk(ifelse.(g .& (ks .>= κmin), 1.0, 0.0)), KP.S) .> 0.5
        ω, r = kop_follow(KP, ω, kg, r, dinj, cnode; κmin)
        ks = k1
    end
    return ω, uc, kg, ks, RenState(kr, A), r, dnet, cren, capren
end

"Carrier masks of the renewable and storage units (tuples of 0/1 vectors), in the prototype's order (`lgv_units`)."
function lgv_groups(P::ML.Problem, ix::DeviceIndex)
    E = P.E
    ru = ren_units(P)
    rg = Tuple(Float64[a == gi for (a, _) in ru] for gi in unique(a for (a, _) in ru))
    sg = Tuple(Float64[a == si for (a, _) in ix.sidx] for si in unique(a for (a, _) in ix.sidx) if E.stors[si].kf > 0)
    for (grp, idx, bus) in ((rg, ru, (a, i) -> E.gens[a].bus[i]), (sg, ix.sidx, (a, i) -> E.stors[a].bus[i]))
        for m in grp
            b = [bus(idx[q]...) for q in eachindex(m) if m[q] > 0]
            allunique(b) || error("lgv_groups: the units of a carrier must sit at distinct nodes")
        end
    end
    return rg, sg
end

# ---------------------------------------------------------------------------
# the device programs of a replica step
# ---------------------------------------------------------------------------

"Running sums of the replicas' capacity scores (C, U) per unit class."
struct ScoreAcc{A}
    sg::A; ss::A; sr::A
end
ScoreAcc(C::Int, Ug::Int, Us::Int, Ur::Int) = ScoreAcc(zeros(C, max(Ug, 1)), zeros(C, max(Us, 1)), zeros(C, max(Ur, 1)))

"Node arrays (ω, uc, e, r), each (C, T, X), of a flat loop state."
function node_arrays(st::FlatState, ::Val{dims}) where {dims}
    C, T, N, Ug, Us = dims[1], dims[2], dims[3], dims[4], dims[5]
    m3(x, n) = reshape(x, C, T, n) .+ 0.0            # materialized (a lazy reshape of a traced array has another type)
    return m3(st.ω, Ug), m3(st.uc, Ug), m3(st.e, Us), m3(st.r, N)
end

"""
    score_acc(st, kg, ks, RS, KP, RP, acc, wt, Val(dims), Val(physical); βv, βk) -> acc

`acc` plus `wt` times the capacity scores of the current state (`cap_score` on the residual cache):
called after each operation sweep of a step, with wt = 1/inner.
"""
function score_acc(st::FlatState, kg, ks, RS, KP, RP, acc::ScoreAcc, wt, dv::Val, ph::Val; βv, βk)
    ω, uc, e, r = node_arrays(st, dv)
    sc = cap_score(KP, RP, ω, uc, e, kg, ks, RS.kr, RS.A, r, ph; βv, βk)
    return ScoreAcc(acc.sg .+ wt .* sc.sg, acc.ss .+ wt .* sc.ss, acc.sr .+ wt .* sc.sr)
end

"""
    lgv_pass(st, kg, ks, RS, dnet, cren, capren, KP, RP, LP, acc, W, ξg, ξs, ξr, wsh, rgroups, sgroups, Val(dims);
             rr, βk, clip, κmin, box) -> (st, kg, ks, cp, RS, dnet, cren, capren, diag)

The Langevin step of the shared capacities of every chain and the dispatch following it in every
replica. `acc` holds the replicas' scores (`score_acc`), `W` (C,) the replica weights (they sum to
1 over the replicas of a chain: 1/M_active, the gate, the tilt), ξ the noise per chain and unit
(C/M, ·). dims = (C, T, N, U_g, U_s, U_l, M[, nf]): M slots per chain, of which the last nf (default 0) are
frozen (the probe copies of the replica driver: weight 0 in W, their capacities, utilizations,
residual and net load kept). The capacity-dependent chain parameters are recomputed
(with the proxy class of the new net load). `diag`: the weighted scores s̄ and forces per chain and
the number of clipped drifts (nclip, per chain).
"""
function lgv_pass(st::FlatState, kg, ks, RS, dnet, cren, capren, KP, RP, LP::LgvParams, acc::ScoreAcc, W, ξg, ξs, ξr, wsh,
                  rgroups, sgroups, dv::Val{dims}; rr, βk, clip, κmin, box::Bool) where {dims}
    C, M = dims[1], dims[7]
    vm = Val(M)
    ω, uc, e, r = node_arrays(st, dv)
    sg, ss, sr = rep_sum(acc.sg, W, vm), rep_sum(acc.ss, W, vm), rep_sum(acc.sr, W, vm)
    kgc, ksc, krc = rep_first(kg, vm), rep_first(ks, vm), rep_first(RS.kr, vm)
    fg = capex_force(kgc, clamp_cap(kgc, KP.ginv, KP.glo, KP.ghi, KP.gkfix), KP.ginv, KP.gcap, KP.γk, βk)
    fs = capex_force(ksc, clamp_cap(ksc, KP.sinv, KP.slo, KP.shi, KP.skfix), KP.sinv, KP.scap, KP.γk, βk)
    fr = βk .* reshape(RP.cap, 1, :) .* (1 .+ RP.γk .* sign.(krc .- clamp.(krc, reshape(RP.lo, 1, :), reshape(RP.hi, 1, :))))
    kg1, ncg = lgv_w(kgc, sg, fg, LP.κg, LP.ηg, LP.hg, ξg; rr, clip, κmin, box)
    ks1, ncs = lgv_w(ksc, ss, fs, LP.κs, LP.ηs, LP.hs, ξs; rr, clip, κmin, box)
    kr1, ncr = lgv_w(krc, sr, fr, LP.κr, LP.ηr, LP.hr, ξr; rr, clip, κmin, box)
    out = lgv_commit(KP, RP, ω, uc, e, kg, ks, r, rep_bcast(kg1, vm), rep_bcast(ks1, vm), rep_bcast(kr1, vm), RS, dnet, cren,
                     capren, rgroups, sgroups; κmin)
    nf = length(dims) >= 8 ? dims[8] : 0
    if nf > 0              # frozen slots (the last nf of every chain's M: the probe copies) keep k and their operation
        fz = [mod1(c, M) > M - nf for c in 1:C]               # static
        keep(new, old) = ifelse.(reshape(fz, (C, ntuple(_ -> 1, ndims(old) - 1)...)), old, new)
        o0 = (ω, uc, kg, ks, RS.kr, r, dnet, cren, capren)
        o1 = (out[1], out[2], out[3], out[4], out[5].kr, out[6], out[7], out[8], out[9])
        ω, uc, kg, ks, kr, r, dnet, cren, capren = map(keep, o1, o0)
        RS = RenState(kr, out[5].A)
    else
        ω, uc, kg, ks, RS, r, dnet, cren, capren = out
    end
    m2(x) = reshape(x, C, :) .+ 0.0
    st2 = FlatState(st.θ, m2(ω), m2(uc), st.e, st.u, m2(r))
    cp = chain_params_dev(KP, kg, ks, proxy_class(dnet, wsh), Val(C))
    diag = (; sg, ss, sr, fg, fs, fr, nclip=ncg .+ ncs .+ ncr)
    return st2, kg, ks, cp, RS, dnet, cren, capren, diag
end

"`lgv_pass` with the noise drawn from Reactant's RNG (device runs)."
function lgv_pass_rng(st::FlatState, kg, ks, RS, dnet, cren, capren, KP, RP, LP::LgvParams, acc::ScoreAcc, W, rng, wsh,
                      rgroups, sgroups, dv::Val{dims}; kw...) where {dims}
    nch = dims[1] ÷ dims[7]
    ξg = randn(rng, Float64, nch, size(kg, 2))
    ξs = randn(rng, Float64, nch, size(ks, 2))
    ξr = randn(rng, Float64, nch, size(RS.kr, 2))
    return lgv_pass(st, kg, ks, RS, dnet, cren, capren, KP, RP, LP, acc, W, ξg, ξs, ξr, wsh, rgroups, sgroups, dv; kw...)
end

# ---------------------------------------------------------------------------
# weather of the replicas: uploaded from the host after each weather step
# ---------------------------------------------------------------------------

"""
    weather_arrays(P, M, ss) -> (D0, cren0, A)

Host arrays of the weather of the states `ss` (one per replica): D0 (C, T, N) the load less the
output of the fixed (non-investable) renewables, cren0 (C,) their running cost, A (C, T, U_r) the
clamped availability of the investable renewable units (`ren_units` order). With the renewable
capacities k_r on the device, dnet = D0 − H(A c(k_r)) (the arithmetic of `to_device`).
"""
function weather_arrays(P::ML.Problem, M::DeviceModel, ss::Vector{ML.State})
    E = P.E
    C, T, N = length(ss), M.T, M.N
    units = ren_units(P)
    D0, cren0, A = zeros(C, T, N), zeros(C), zeros(C, T, max(length(units), 1))
    for (c, s) in enumerate(ss)
        Xn = ML.weather_nodes(P.W, s.ℓ, s.xs[end])
        dn = max.(Xn[1], 0.0)
        for g in E.gens
            (g.vf == 0 && g.kf == 0) || continue
            Ag = g.wf > 0 ? clamp.(Xn[g.wf], 0.0, 1.0) : repeat(g.avail, 1, T)
            for t in 1:T, i in eachindex(g.bus)
                p = Ag[i, t] * g.kfix[i]
                dn[g.bus[i], t] -= p
                cren0[c] += M.Δt * g.mc[i] * p
            end
        end
        D0[c, :, :] .= dn'
        for (q, (gi, i)) in enumerate(units)
            A[c, :, q] .= clamp.(Xn[E.gens[gi].wf][i, :], 0.0, 1.0)
        end
    end
    return D0, cren0, A
end

"""
Compensated initialisation after a weather change (the prototype's `compensate!`): at every
(node, step) the dispatchable generators take up the change `need` (C, T, N) of the residual, an
increase in proportion to their headroom (up to it), a decrease in proportion to their output.
Returns (ω, r).
"""
function weather_comp(KP, ω, kg, r, need)
    kc = clamp_cap(kg, KP.ginv, KP.glo, KP.ghi, KP.gkfix)
    κ = rowv(KP.gav) .* rowk(kc)
    yc = clamp.(ω, 0.0, 1.0)
    ps = κ .* yc
    nq = tmulT(need, KP.G)
    wq = ifelse.(nq .> 0, max.(κ .- ps, 0.0), ps)
    tot = tmulT(tmul(wq, KP.G), KP.G)
    ok = (abs.(nq) .>= 1e-6) .& (tot .> 1e-9) .& (abs.(κ) .>= 1e-9)
    a = ifelse.(nq .> 0, min.(nq, tot), max.(nq, -tot)) ./ ifelse.(ok, tot, 1.0)
    ω1 = ω .+ ifelse.(ok, a .* wq ./ ifelse.(ok, κ, 1.0), 0.0)
    return ω1, r .- tmul(κ .* (clamp.(ω1, 0.0, 1.0) .- yc), KP.G)
end

"""
    weather_update(st, kg, ks, RS, dnet, cren, KP, RP, D0, cren0, A, chg, wsh, Val(dims), Val(comp))
        -> (st, RS, dnet, cren, cp)

New weather for the replicas with chg[c] > 0 (C,): the net load from the uploaded D0 and A and the
current renewable capacities, the renewable running cost, the residual shifted by the change of
the net load, and (comp = true) the compensated initialisation of the dispatch. The chain
parameters are recomputed (proxy class of the new net load, load shares `wsh`).
"""
function weather_update(st::FlatState, kg, ks, RS, dnet, cren, KP, RP, D0, cren0, A, chg, wsh, dv::Val{dims},
                        ::Val{comp}) where {dims, comp}
    C = dims[1]
    ω, uc, e, r = node_arrays(st, dv)
    ch3 = reshape(chg, :, 1, 1) .> 0
    A1 = ifelse.(ch3, A, RS.A)
    krc = clamp.(RS.kr, reshape(RP.lo, 1, :), reshape(RP.hi, 1, :))
    p = A1 .* rowk(krc)
    dn1 = ifelse.(ch3, D0 .- tmul(p, RP.H), dnet)
    cr1 = ifelse.(chg .> 0, cren0 .+ vec(sum(KP.Δt .* rowv(RP.mc) .* p; dims=(2, 3))), cren)
    need = dn1 .- dnet
    r = r .+ need
    comp && ((ω, r) = weather_comp(KP, ω, kg, r, need))
    m2(x) = reshape(x, C, :) .+ 0.0
    cp = chain_params_dev(KP, kg, ks, proxy_class(dn1, wsh), Val(C))
    return FlatState(st.θ, m2(ω), st.uc, st.e, st.u, m2(r)), RenState(RS.kr, A1), dn1, cr1, cp
end
