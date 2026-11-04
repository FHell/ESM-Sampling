# Capacity moves on the device (docs/reactant_design.md §3.6, phase 4): the scaling group
# (k, y) ↦ (e^u k, y c(k)/c(e^u k)) of one investable unit (c the clamp to the capacity box;
# (e^u k, e^{−u} y) inside the box), with Metropolis and fixed-scale
# proposals u = σ z (the CPU reference is `kscale_sweep!(…; mh=true)`). Units run in a
# fixed order (they share the balance of their node); chains are the batch dimension.
#
# The per-step terms mirror the CPU expressions term by term (gen_cost, uc_cost,
# soc_cost and the balance difference), so they agree bitwise; the sums over the steps
# differ in their order (pairwise against sequential), i.e. the log ratios agree to
# round-off, and so do the accept decisions except at ties of probability zero.
#
# Scope: dispatchable and storage units (renewables as load need their availability on
# the device, phase 4b); no bias (α = 0); unit parameters enter as one-element slices,
# so the code stays traceable.

"Parameters of the capacity moves (type-parametric scalars: captured by device loops, they become traced)."
struct KParams{F,V,Mx,B}
    Δt::F; c_viol::F; κ0::F; γk::F; c_shed::F; c_dump::F
    G::Mx; S::Mx
    gav::V; gmc::V; gmc0::V; gk0::V; guc::V; ggmin::V; gsu::V; gnl::V; glo::V; ghi::V; gcap::V
    sh::V; sηin::V; sηout::V; sqt::V; smc::V; slo::V; shi::V; scap::V
    ginv::B; sinv::B
    gkfix::V; skfix::V
end
KParams(M::DeviceModel) = KParams(M.Δt, M.c_viol, M.κ0, M.γk, M.c_shed, M.c_dump, M.G, M.S,
    M.gav, M.gmc, M.gmc0, M.gk0, M.guc, M.ggmin, M.gsu, M.gnl, M.glo, M.ghi, M.gcap,
    M.sh, M.sηin, M.sηout, M.sqt, M.smc, M.slo, M.shi, M.scap, M.ginv, M.sinv, M.gkfix, M.skfix)

r1(v, q) = Reactant.@allowscalar v[q]          # one unit parameter (a traced scalar in a device loop)
nextτ2(A) = cat(A[:, 2:end], A[:, 1:1]; dims=2)

"Cost (C, T) and output of dispatchable unit q at capacity kc (C, 1), utilization Y and commitment Uc (C, T)."
function gen_unit_cost(M, q, kc, Y, Uc)
    Δt, cv, κ0 = M.Δt, M.c_viol, M.κ0
    A, mc, mc0, k0 = r1(M.gav, q), r1(M.gmc, q), r1(M.gmc0, q), r1(M.gk0, q)
    yc = clamp.(Y, 0.0, 1.0)
    p = A .* kc .* yc
    w = kc .+ κ0
    hasv = (k0 .> 0) .& (mc0 .!= mc)
    qv = min.(k0, kc)
    oldfirst = mc0 .<= mc
    plo = A .* ifelse.(oldfirst, qv, kc .- qv)
    clo, chi = ifelse.(oldfirst, mc0, mc), ifelse.(oldfirst, mc, mc0)
    cv_ = ifelse.(p .<= plo, clo .* p, clo .* plo .+ chi .* (p .- plo))
    c = ifelse.(hasv, Δt .* cv_, Δt .* mc .* p) .+ cv .* Δt .* w .* abs.(Y .- yc)
    # fleet unit commitment (uc_cost)
    ucm = r1(M.guc, q) .> 0
    gmin, su, nl = r1(M.ggmin, q), r1(M.gsu, q), r1(M.gnl, q)
    ucc, uc1 = clamp.(Uc, 0.0, 1.0), clamp.(nextτ2(Uc), 0.0, 1.0)
    κ = A .* kc
    cu = Δt .* nl .* κ .* ucc .+ su .* κ .* max.(uc1 .- ucc, 0.0) .+
         cv .* Δt .* w .* (max.(yc .- ucc, 0.0) .+ max.(gmin .* ucc .- yc, 0.0) .+ abs.(Uc .- ucc))
    return ifelse.(ucm, c .+ cu, c), p
end

"Cost (C, T) and injection k(p⁺ − p⁻) of storage unit q (SoC form, `soc_cost`)."
function stor_unit_cost(M, q, kc, Ye)
    Δt, cv, κ0 = M.Δt, M.c_viol, M.κ0
    h, ηin, ηout, qt, mcs = r1(M.sh, q), r1(M.sηin, q), r1(M.sηout, q), r1(M.sqt, q), r1(M.smc, q)
    Ye1 = nextτ2(Ye)
    ec, ec1 = clamp.(Ye, 0.0, 1.0), clamp.(Ye1, 0.0, 1.0)
    d = h .* (ec1 .- ec) ./ Δt .- qt
    pos = d .>= 0
    pc = ifelse.(pos, ifelse.(ηin .> 0, d ./ ifelse.(ηin .> 0, ηin, 1.0), 0.0), 0.0)
    mc = ifelse.(pos, 0.0, min.(-d .* ηout, 1.0))
    w = kc .+ κ0
    c = Δt .* mcs .* kc .* (pc .+ mc) .+ cv .* w .* h .* abs.(Ye .- ec)
    c = c .+ ifelse.(pc .> 1, cv .* Δt .* w .* (pc .- 1), 0.0)
    ex = -d .* ηout .- 1.0
    c = c .+ ifelse.(.!pos .& (qt .<= 0) .& (ex .> 0), cv .* Δt .* w .* ex, 0.0)
    c = c .+ ifelse.(pos .& (ηin .<= 0), cv .* Δt .* w .* d, 0.0)
    return c, kc .* (pc .- mc)
end

φbal_d(M, x) = ifelse.(x .> 0, M.c_shed .* x, -M.c_dump .* x)

"Utilization factor c(k0)/c(k1) of the capacity moves (1 where the clamped capacity does not change; `kc_ratio`)."
kc_ratio_d(kc0, kc1) = ifelse.(kc0 .== kc1, one.(kc0), kc0 ./ ifelse.(kc0 .== kc1, one.(kc1), kc1))
"log(c(k0)/c(k1)), 0 where the clamped capacity does not change."
kc_logratio_d(kc0, kc1) = log.(kc_ratio_d(kc0, kc1))

"""
Log-ratio of the scaling move of unit q (kind 1: dispatchable, 2: storage) by u (C,):
−β_k Δcapex − β_v ΔC + (1 − n) u (+ the physical-measure term), and the injection change
d (C, T) at the unit's node; `rn` (C, T) is the residual at that node.
"""
function kscale_logratio(M, S::DeviceState, rn, kind::Int, q, u; βk, βv, physical::Bool)
    T = size(S.ω, 2)
    kraw = reshape(kind == 1 ? S.kg[:, q] : S.ks[:, q], :, 1)                  # (C, 1)
    lo, hi, cap = kind == 1 ? (r1(M.glo, q), r1(M.ghi, q), r1(M.gcap, q)) : (r1(M.slo, q), r1(M.shi, q), r1(M.scap, q))
    uu = reshape(u, :, 1)
    k1 = kraw .* exp.(uu)
    kc0, kc1 = clamp.(kraw, lo, hi), clamp.(k1, lo, hi)
    sc = kc_ratio_d(kc0, kc1)
    if kind == 1
        isuc = r1(M.guc, q) .> 0
        Y, Uc = S.ω[:, :, q], S.uc[:, :, q]
        c0, p0 = gen_unit_cost(M, q, kc0, Y, Uc)
        c1, p1 = gen_unit_cost(M, q, kc1, Y .* sc, Uc .* sc)
        d = p1 .- p0
        n = T .* (1 .+ isuc)
    else
        Ye = S.e[:, :, q]
        c0, j0 = stor_unit_cost(M, q, kc0, Ye)
        c1, j1 = stor_unit_cost(M, q, kc1, Ye .* sc)
        d = -(j1 .- j0)
        n = T
    end
    Δop = vec(sum((c1 .- c0) .+ M.Δt .* (φbal_d(M, rn .- d) .- φbal_d(M, rn)); dims=2))
    Δcap = vec(cap .* (k1 .- kraw) .+ M.γk .* cap .* (abs.(k1 .- kc1) .- abs.(kraw .- kc0)))
    lr = -(βk .* Δcap .+ βv .* Δop) .+ vec(uu .+ n .* kc_logratio_d(kc0, kc1))
    physical && (lr = lr .+ vec(n .* (log.(kc1 .+ M.κ0) .- log.(kc0 .+ M.κ0))))
    return lr, d
end

"Residual (C, T) at the node of unit q (one-hot contraction; exact)."
unit_node_r(r, H, q) = dropdims(tmul(r, reshape(H[q, :], :, 1)); dims=3)

"""
    kscale_mh!(M, S, r, Z, U; units, σ, βk, βv, physical, κmin)

One Metropolis pass over `units` (a list of (kind, q)), nprop = size(Z, 3) proposals per
unit, proposal normals Z and uniforms U of shape (C, length(units), nprop). Updates the
capacities, the utilizations (commitment, state of charge) and the residual r in place
of the returned state; returns (S, r, accepted counts).
"""
function kscale_mh!(M, S::DeviceState, r, Z, U; units, σ, βk, βv, physical::Bool, κmin=1.0)
    nacc = zero(S.kg[:, 1])
    for (j, (kind, q)) in enumerate(units)
        H = kind == 1 ? M.G : M.S
        rn = unit_node_r(r, H, q)
        kraw = kind == 1 ? S.kg[:, q] : S.ks[:, q]
        ucur = zero(kraw)
        lcur = zero(kraw)
        dcur = zero(rn)
        for m in 1:size(Z, 3)
            up = ucur .+ σ .* Z[:, j, m]
            lp, dp = kscale_logratio(M, S, rn, kind, q, up; βk, βv, physical)
            ok = (kraw .>= κmin) .& (up .>= log.(κmin ./ kraw)) .& (log.(U[:, j, m]) .< lp .- lcur)
            ucur = ifelse.(ok, up, ucur)
            lcur = ifelse.(ok, lp, lcur)
            dcur = ifelse.(reshape(ok, :, 1), dp, dcur)
        end
        moved = ucur .!= 0
        nacc = nacc .+ moved
        lo, hi = kind == 1 ? (r1(M.glo, q), r1(M.ghi, q)) : (r1(M.slo, q), r1(M.shi, q))
        sc = reshape(kc_ratio_d(clamp.(kraw, lo, hi), clamp.(kraw .* exp.(ucur), lo, hi)), :, 1)
        if kind == 1
            S.kg[:, q] .= kraw .* exp.(ucur)
            S.ω[:, :, q] .= S.ω[:, :, q] .* sc
            S.uc[:, :, q] .= ifelse.(r1(M.guc, q) .> 0, S.uc[:, :, q] .* sc, S.uc[:, :, q])
        else
            S.ks[:, q] .= kraw .* exp.(ucur)
            S.e[:, :, q] .= S.e[:, :, q] .* sc
        end
        r = r .- reshape(dcur, size(dcur, 1), size(dcur, 2), 1) .* reshape(H[q, :], 1, 1, :)
    end
    return S, r, nacc
end

"Units of the device capacity moves in the CPU order (investable dispatchable units, then storage)."
kmove_units(M) = vcat([(1, q) for q in eachindex(M.ginv) if M.ginv[q]], [(2, q) for q in eachindex(M.sinv) if M.sinv[q]])

"""
Device-loop form of `kscale_mh!`: the dispatchable units `gq` and then the storage units
`sq` in traced loops (one program for any number of units). Z, U as in `kscale_mh!`,
columns ordered as `kmove_units`.
"""
function kscale_mh_loop(M::KParams, S::DeviceState, r, Z, U, gq, sq; σ, βk, βv, physical::Bool, κmin=1.0,
                        bias=nothing)
    ng, ns = length(gq), length(sq)
    ph = Val(physical)                                  # static flags as Val (captured Bools are traced in a loop)
    # bias = (BiasDev, BiasState, gitem, sitem) for α = 1, else nothing
    BD, BS, gitem, sitem = bias === nothing ? (nothing, nothing, nothing, nothing) : bias
    @trace for j in 1:ng
        q = Reactant.@allowscalar gq[j]                 # a traced unit index (one scalar per loop step)
        S, r, BS = kmove_unit!(M, S, r, Z, U, Val(1), q, j, ph, BD, BS, gitem; σ, βk, βv, κmin)
    end
    @trace for j in 1:ns
        q = Reactant.@allowscalar sq[j]
        S, r, BS = kmove_unit!(M, S, r, Z, U, Val(2), q, ng + j, ph, BD, BS, sitem; σ, βk, βv, κmin)
    end
    return bias === nothing ? (S, r) : (S, r, BS)
end

"Bias part of a proposal: Δb̂ and the aggregates after it (nothing without bias)."
function bias_proposal(BD, BS, items, q, δk)
    BD === nothing && return nothing, nothing
    ι = Reactant.@allowscalar items[q]
    return delta_bias(BD, BS, ι, δk)
end
select_new(ok, a, b) = b === nothing ? nothing : map((x, y) -> ifelse.(reshape(ok, :, 1), x, y), a, b)

"One unit of the device loop (traced unit index q, column j of the proposal arrays)."
function kmove_unit!(M, S, r, Z, U, ::Val{kind}, q, j, ::Val{physical}, BD, BS, items; σ, βk, βv, κmin) where {kind,physical}
    nprop = size(Z, 3)
    H = kind == 1 ? M.G : M.S
    rn = unit_node_r(r, H, q)
    kraw = kind == 1 ? S.kg[:, q] : S.ks[:, q]
    ucur = zero(kraw); lcur = zero(kraw); dcur = zero(rn)
    bcur = BD === nothing ? nothing : bias_fields(BS)
    for m in 1:nprop
        up = ucur .+ σ .* Z[:, j, m]
        lp, dp = kscale_logratio(M, S, rn, kind, q, up; βk, βv, physical)
        Δb, bnew = bias_proposal(BD, BS, items, q, kraw .* (exp.(up) .- 1))
        Δb === nothing || (lp = lp .- Δb)
        ok = (kraw .>= κmin) .& (up .>= log.(κmin ./ kraw)) .& (log.(U[:, j, m]) .< lp .- lcur)
        ucur = ifelse.(ok, up, ucur); lcur = ifelse.(ok, lp, lcur)
        dcur = ifelse.(reshape(ok, :, 1), dp, dcur)
        bcur = select_new(ok, bnew, bcur)
    end
    if BD !== nothing
        set_bias_fields!(BS, bcur)
    end
    lo, hi = kind == 1 ? (r1(M.glo, q), r1(M.ghi, q)) : (r1(M.slo, q), r1(M.shi, q))
    sc = reshape(kc_ratio_d(clamp.(kraw, lo, hi), clamp.(kraw .* exp.(ucur), lo, hi)), :, 1)
    if kind == 1
        S.kg[:, q] = kraw .* exp.(ucur)
        S.ω[:, :, q] = S.ω[:, :, q] .* sc
        S.uc[:, :, q] = ifelse.(r1(M.guc, q) .> 0, S.uc[:, :, q] .* sc, S.uc[:, :, q])
    else
        S.ks[:, q] = kraw .* exp.(ucur)
        S.e[:, :, q] = S.e[:, :, q] .* sc
    end
    r = r .- reshape(dcur, size(dcur, 1), size(dcur, 2), 1) .* reshape(H[q, :], 1, 1, :)
    return S, r, BS
end

# ---------------------------------------------------------------------------
# peak-set capacity moves (kpeak, absorb variant): k → k + δ; on the unit's peak steps the
# output rises by A (c(k + δ) − c(k)), elsewhere it stays (y ↦ (y c(k) + c(k+δ) − c(k))/c(k+δ),
# y ↦ y c(k)/c(k + δ), c the clamp to the box); the commitment follows the same affine map;
# Jacobian (1 + uc) T log(c(k)/c(k + δ)). The CPU
# reference is `kpeak_sweep!(…; mh=true)`. Peak sets come from (x, k_ren): host input inS
# (C, T, U_g) of Bools.
# ---------------------------------------------------------------------------

function kpeak_logratio(M, S::DeviceState, rn, q, δ, inS, ::Val{physical}; βk, βv) where {physical}
    T = size(S.ω, 2)
    k0 = reshape(S.kg[:, q], :, 1)
    dd = reshape(δ, :, 1)
    k1 = k0 .+ dd
    lo, hi, cap = r1(M.glo, q), r1(M.ghi, q), r1(M.gcap, q)
    kc0, kc1 = clamp.(k0, lo, hi), clamp.(k1, lo, hi)
    isuc = r1(M.guc, q) .> 0
    Y, Uc = S.ω[:, :, q], S.uc[:, :, q]
    same = kc0 .== kc1
    kd = ifelse.(same, one.(kc1), kc1)
    amap(X) = ifelse.(same, X, ifelse.(inS, (X .* kc0 .+ (kc1 .- kc0)) ./ kd, X .* kc0 ./ kd))
    Y1, U1 = amap(Y), amap(Uc)
    c0, p0 = gen_unit_cost(M, q, kc0, Y, Uc)
    c1, p1 = gen_unit_cost(M, q, kc1, Y1, U1)
    d = p1 .- p0
    Δop = vec(sum((c1 .- c0) .+ M.Δt .* (φbal_d(M, rn .- d) .- φbal_d(M, rn)); dims=2))
    Δcap = vec(cap .* dd .+ M.γk .* cap .* (abs.(k1 .- kc1) .- abs.(k0 .- kc0)))
    nu = 1 .+ isuc
    lr = -βv .* Δop .- βk .* Δcap .+ vec(nu .* T .* kc_logratio_d(kc0, kc1))
    physical && (lr = lr .+ vec(nu .* T .* (log.(kc1 .+ M.κ0) .- log.(kc0 .+ M.κ0))))
    return lr, d, Y1, U1
end

"""
    kpeak_mh_loop(KP, S, r, Z, U, gq, σk, inS; βk, βv, physical, κmin)

One Metropolis pass of peak-set moves over the investable dispatchable units `gq` (device
loop), proposals δ' = δ + σk[j] z, Z and U (C, length(gq), nprop), peak indicators inS
(C, T, U_g).
"""
function kpeak_mh_loop(M, S::DeviceState, r, Z, U, gq, σk, inS; βk, βv, physical::Bool, κmin=1.0, bias=nothing)
    ng = length(gq)
    ph = Val(physical)
    BD, BS, gitem, _ = bias === nothing ? (nothing, nothing, nothing, nothing) : bias
    @trace for j in 1:ng
        q = Reactant.@allowscalar gq[j]
        S, r, BS = kpeak_unit!(M, S, r, Z, U, q, j, σk, inS, ph, BD, BS, gitem; βk, βv, κmin)
    end
    return bias === nothing ? (S, r) : (S, r, BS)
end

function kpeak_unit!(M, S, r, Z, U, q, j, σk, inS, ph, BD, BS, items; βk, βv, κmin)
    nprop = size(Z, 3)
    rn = unit_node_r(r, M.G, q)
    k0 = S.kg[:, q]
    sj = Reactant.@allowscalar σk[j]
    Sq = inS[:, :, q]
    δ = zero(k0); lcur = zero(k0)
    dcur, Ycur, Ucur = zero(rn), S.ω[:, :, q], S.uc[:, :, q]
    bcur = BD === nothing ? nothing : bias_fields(BS)
    for m in 1:nprop
        δp = δ .+ sj .* Z[:, j, m]
        lp, dp, Y1, U1 = kpeak_logratio(M, S, rn, q, δp, Sq, ph; βk, βv)
        Δb, bnew = bias_proposal(BD, BS, items, q, δp)
        Δb === nothing || (lp = lp .- Δb)
        ok = (k0 .>= κmin) .& (k0 .+ δp .>= κmin) .& (log.(U[:, j, m]) .< lp .- lcur)
        okm = reshape(ok, :, 1)
        δ = ifelse.(ok, δp, δ); lcur = ifelse.(ok, lp, lcur)
        dcur = ifelse.(okm, dp, dcur); Ycur = ifelse.(okm, Y1, Ycur); Ucur = ifelse.(okm, U1, Ucur)
        bcur = select_new(ok, bnew, bcur)
    end
    if BD !== nothing
        set_bias_fields!(BS, bcur)
    end
    S.kg[:, q] = k0 .+ δ
    S.ω[:, :, q] = Ycur
    S.uc[:, :, q] = ifelse.(r1(M.guc, q) .> 0, Ucur, S.uc[:, :, q])
    r = r .- reshape(dcur, size(dcur, 1), size(dcur, 2), 1) .* reshape(M.G[q, :], 1, 1, :)
    return S, r, BS
end
