# Renewable capacity moves on the device (renewables as load; docs/reactant_design.md §3.6).
#
# The scaling move k → e^u k of one investable renewable unit has no utilization coordinate
# (the output A·k is exact), so the generalized-Gibbs Jacobian is e^{u}: log ratio
# −β_k Δcapex − β_v ΔC + u (− Δb̂). The output change d = A (k_c1 − k_c0) at every step of the
# unit's node moves the residual, the net load `dnet` (load − renewable output, used by the
# operation rounds and by the peak sets) and the renewable running cost `cren`; capex moves
# `capren`. The CPU reference is `kscale_sweep!(…; mh=true, kinds=:renewables)`.
#
# Peak sets (top fraction q of the net load per node) depend on the renewable capacities, so
# they are recomputed on the device after a renewable pass (`peak_sets_dev`).

"Static parameters of the renewable units (type-parametric scalars)."
struct RenParams{F,V,Mx}
    Δt::F; γk::F; c_shed::F; c_dump::F
    H::Mx                 # (U_r, N) node one-hot
    mc::V; lo::V; hi::V; cap::V
end

"Per-chain renewable state: raw capacities (C, U_r) and clamped availability (C, T, U_r)."
mutable struct RenState{A2,A3}
    kr::A2
    A::A3
end

"Renewable parameters and state of the chains `ss`."
function pack_ren(P::ML.Problem, M::DeviceModel, ss::Vector{ML.State})
    E = P.E
    L = ss[1].ℓ
    units = ren_units(P)
    N, T, C = M.N, M.T, length(ss)
    H = zeros(length(units), N)
    mc, lo, hi, cap = zeros(length(units)), zeros(length(units)), zeros(length(units)), zeros(length(units))
    for (q, (gi, i)) in enumerate(units)
        g = E.gens[gi]
        H[q, g.bus[i]] = 1.0
        mc[q], lo[q], hi[q], cap[q] = g.mc[i], E.kspecs[g.kf].lo[i], E.kspecs[g.kf].hi[i], g.capex[i]
    end
    kr, A = zeros(C, length(units)), zeros(C, T, length(units))
    lk = P.lk[L+1]
    for (c, s) in enumerate(ss)
        Xn = ML.weather_nodes(P.W, L, s.xs[end])
        for (q, (gi, i)) in enumerate(units)
            g = E.gens[gi]
            kr[c, q] = vec(ML.nodevals(s.k, lk, E.kspecs, g.kf))[i]
            A[c, :, q] .= clamp.(Xn[g.wf][i, :], 0.0, 1.0)
        end
    end
    return RenParams(M.Δt, M.γk, M.c_shed, M.c_dump, H, mc, lo, hi, cap), RenState(kr, A)
end

"Log ratio of the scaling move of renewable unit q by u (C,), the output change d (C, T) and the cost changes."
function kren_logratio(RP, RS, rn, q, u; βk, βv)
    kraw = reshape(RS.kr[:, q], :, 1)
    uu = reshape(u, :, 1)
    k1 = kraw .* exp.(uu)
    lo, hi, cap, mc = r1(RP.lo, q), r1(RP.hi, q), r1(RP.cap, q), r1(RP.mc, q)
    kc0, kc1 = clamp.(kraw, lo, hi), clamp.(k1, lo, hi)
    A = RS.A[:, :, q]
    p0, p1 = A .* kc0 .* 1.0, A .* kc1 .* 1.0
    c0, c1 = RP.Δt .* mc .* p0, RP.Δt .* mc .* p1
    d = p1 .- p0
    φ(x) = ifelse.(x .> 0, RP.c_shed .* x, -RP.c_dump .* x)
    Δop = vec(sum((c1 .- c0) .+ RP.Δt .* (φ(rn .- d) .- φ(rn)); dims=2))
    Δcap = vec(cap .* (k1 .- kraw) .+ RP.γk .* cap .* (abs.(k1 .- kc1) .- abs.(kraw .- kc0)))
    lr = -(βk .* Δcap .+ βv .* Δop) .+ vec(uu)
    return lr, d, vec(sum(c1 .- c0; dims=2)), Δcap
end

"""
    kren_mh_loop(RP, RS, r, dnet, cren, capren, Z, U, rq; σ, βk, βv, κmin, bias)

One Metropolis pass over the renewable units `rq` (indices into `ren_units`), proposals
u' = u + σ z, Z and U (C, length(rq), nprop). Returns (RS, r, dnet, cren, capren[, BS]).
"""
function kren_mh_loop(RP, RS, r, dnet, cren, capren, Z, U, rq; σ, βk, βv, κmin=1.0, bias=nothing)
    nr = length(rq)
    BD, BS, ritem = bias === nothing ? (nothing, nothing, nothing) : bias
    @trace for j in 1:nr
        q = Reactant.@allowscalar rq[j]
        RS, r, dnet, cren, capren, BS = kren_unit!(RP, RS, r, dnet, cren, capren, Z, U, q, j, BD, BS, ritem; σ, βk, βv, κmin)
    end
    return bias === nothing ? (RS, r, dnet, cren, capren) : (RS, r, dnet, cren, capren, BS)
end

function kren_unit!(RP, RS, r, dnet, cren, capren, Z, U, q, j, BD, BS, items; σ, βk, βv, κmin)
    nprop = size(Z, 3)
    rn = unit_node_r(r, RP.H, q)
    kraw = RS.kr[:, q]
    ucur = zero(kraw); lcur = zero(kraw); dcur = zero(rn); ccur = zero(kraw); kcur = zero(kraw)
    bcur = BD === nothing ? nothing : bias_fields(BS)
    for m in 1:nprop
        up = ucur .+ σ .* Z[:, j, m]
        lp, dp, Δc, Δk = kren_logratio(RP, RS, rn, q, up; βk, βv)
        Δb, bnew = bias_proposal(BD, BS, items, q, kraw .* (exp.(up) .- 1))
        Δb === nothing || (lp = lp .- Δb)
        ok = (kraw .>= κmin) .& (up .>= log.(κmin ./ kraw)) .& (log.(U[:, j, m]) .< lp .- lcur)
        ucur = ifelse.(ok, up, ucur); lcur = ifelse.(ok, lp, lcur)
        dcur = ifelse.(reshape(ok, :, 1), dp, dcur)
        ccur = ifelse.(ok, Δc, ccur); kcur = ifelse.(ok, Δk, kcur)
        bcur = select_new(ok, bnew, bcur)
    end
    BD === nothing || set_bias_fields!(BS, bcur)
    RS.kr[:, q] = kraw .* exp.(ucur)
    upd = reshape(dcur, size(dcur, 1), size(dcur, 2), 1) .* reshape(RP.H[q, :], 1, 1, :)
    r = r .- upd
    dnet = dnet .- upd
    return RS, r, dnet, cren .+ ccur, capren .+ kcur, BS
end

"""
Peak sets of every dispatchable unit (C, T, U_g): the steps with the top `m` net loads at the
unit's node (the CPU's `peak_set`; ties at the threshold step are included).
"""
function peak_sets_dev(dnet, G, ::Val{m}) where {m}
    srt = sort(dnet; dims=2, rev=true)
    thr = srt[:, m:m, :]
    ind = Float64.(dnet .>= thr)
    return tmul(ind, transpose(G)) .> 0.5
end
