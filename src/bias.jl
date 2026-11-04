# Learned weather–investment bias (docs/system/ensemble.md §4.4).
#
#   b̂(x,k) = ĝ(x,k) − r Ē_x' ĝ(x',k) + (1 − r) ĥ(k),   r = β_k/β_v,
#
# evaluated at the level ℓ_b of the bias: level 0 during the level-0 warmup, the
# level of the finest chains after the lift (`lift_bias!`). ĝ models the weather
# dependence of log Z and ĥ the remaining k-dependence of Λ.
#
# ĝ is local in time: a sum over the steps of level ℓ_b (length Δt_b) of
# F₁(u₁) + F₂(u₂), with the system residual load u₁ = (D − R(k))/D̄ (load minus the
# availability of every generator except gas) and the shortfall u₂ = u₁ − firm(k)/D̄
# (firm: gas and storage power), F_g(u) = θ_g u + Σ_j θ_gj s softplus((u − τ_j)/s).
# F₁ and F₂ are evaluated from tables (`RampTable`: the piecewise-linear
# interpolant of the analytic ramp sum on a uniform grid with spacing s/res, linear
# beyond it with the end slopes); the tables define ĝ everywhere (global and
# incremental evaluation, CPU and device). The score regression uses the analytic
# basis: ∇ₓĝ is regressed on the per-sample scores −β_v ∇ₓC in the weather state
# coordinates of level ℓ_b (score identity, valid because Ω does not depend on x),
# ∂ₖĥ per coordinate on the residual capacity scores −β_v ∂ₖC + ∂ₖlog J − ∂ₖĝ. Both
# regressions are accumulated sums (running means with weight 1/n), refitted on a
# geometric schedule and frozen after the warmup. Ē is a fixed, weighted set of
# exact weather steps at level ℓ_b.

"Piecewise-linear table of a ramp sum F on the grid u0 + (j − 1) h, linear beyond it with the end slopes."
struct RampTable
    u0::Float64
    h::Float64
    F::Vector{Float64}
end

softplus(x) = x > 30 ? x : log1p(exp(x))
sigmoid(x) = 1 / (1 + exp(-x))

"Analytic ramp sum F(u) = θ_lin u + Σ_j θ_j s softplus((u − τ_j)/s)."
function ramp_analytic(θlin::Float64, θr::AbstractVector, τ::AbstractVector, s::Float64, u::Float64)
    v = θlin * u
    for j in eachindex(τ)
        v += θr[j] * s * softplus((u - τ[j]) / s)
    end
    return v
end

"""
Table of the ramp sum on [min τ − margin·s, max τ + margin·s] with spacing s/res. Beyond
the grid the interpolant continues with the end slopes, which equal the asymptotic
slopes of the ramp sum to e^{−margin}.
"""
function RampTable(θlin::Float64, θr::AbstractVector, τ::AbstractVector, s::Float64; res::Int=100, margin::Float64=20.0)
    h = s / res
    u0 = minimum(τ) - margin * s
    n = ceil(Int, (maximum(τ) + margin * s - u0) / h) + 1
    return RampTable(u0, h, [ramp_analytic(θlin, θr, τ, s, u0 + (j - 1) * h) for j in 1:n])
end

"Value of the table at u: F[j] + w (F[j+1] − F[j]) on the clamped segment j (linear extrapolation in the tails)."
@inline function tabval(tb::RampTable, u::Float64)
    F = tb.F
    n = length(F)
    x = (u - tb.u0) / tb.h
    i = x >= 0.0 ? (x < n - 1 ? unsafe_trunc(Int, x) : n - 2) : 0
    w = x - i
    @inbounds f0 = F[i+1]
    @inbounds f1 = F[i+2]
    return f0 + w * (f1 - f0)
end

mutable struct BiasModel
    τ1::Vector{Float64}
    τ2::Vector{Float64}
    s::Float64
    θ::Vector{Float64}
    a::Vector{Float64}             # ĥ coefficients per capacity coordinate of level ℓ_b
    c::Vector{Float64}
    κlo::Vector{Float64}           # ĥ: range of κ in the regression data per coordinate (±Inf: none)
    κhi::Vector{Float64}
    kscale::Vector{Float64}
    Dbar::Float64
    ratio::Float64                 # β_k / β_v
    ℓb::Int                        # level at which b̂ is evaluated (0-based)
    Δt::Float64                    # its step length (h)
    # Ē: fixed exact weather steps at level ℓ_b (node data after the data map)
    sD::Vector{Float64}            # system load per sample step
    sA::Vector{Matrix{Float64}}    # per generator group (not gas): availability, sample steps × support (0×0 for gas)
    wstep::Float64                 # Ē ĝ = wstep Σ_j (F₁ + F₂) over the sample steps
    nsamp::Int                     # exact weather draws the sample steps come from
    # tables of F₁, F₂ (rebuilt whenever θ changes)
    res::Int
    tab1::RampTable
    tab2::RampTable
    tabθ::Vector{Float64}
    # accumulated normal equations of the score regressions
    GtG::Matrix{Float64}
    Gts::Vector{Float64}
    hk::Matrix{Float64}            # per k coordinate: [Σ1 Σκ Σκ² Σy Σyκ min κ max κ]
    nobs::Int
    active::Bool
end

nθ(b::BiasModel) = 1 + length(b.τ1) + 1 + length(b.τ2)

"Rebuild the tables of F₁ and F₂ from θ."
function rebuild_tables!(b::BiasModel)
    n1 = length(b.τ1)
    b.tab1 = RampTable(b.θ[1], b.θ[2:1+n1], b.τ1, b.s; res=b.res)
    b.tab2 = RampTable(b.θ[2+n1], b.θ[3+n1:end], b.τ2, b.s; res=b.res)
    b.tabθ = copy(b.θ)
    return b
end

"The tables of F₁, F₂ for the current θ (rebuilt if θ changed since they were built)."
function tables(b::BiasModel)
    b.tabθ == b.θ || rebuild_tables!(b)
    return b.tab1, b.tab2
end

"Basis φ(u) and φ'(u) for the two feature groups at one step (analytic; the regression design)."
function bias_basis!(φ, dφ1, dφ2, b::BiasModel, u1, u2)
    fill!(dφ1, 0.0); fill!(dφ2, 0.0)
    φ[1] = u1; dφ1[1] = 1.0
    o = 1
    for τ in b.τ1
        o += 1
        φ[o] = b.s * softplus((u1 - τ) / b.s)
        dφ1[o] = sigmoid((u1 - τ) / b.s)
    end
    o += 1
    φ[o] = u2; dφ2[o] = 1.0
    for τ in b.τ2
        o += 1
        φ[o] = b.s * softplus((u2 - τ) / b.s)
        dφ2[o] = sigmoid((u2 - τ) / b.s)
    end
end

"Firm capacity (gas generators and storage power) of the clamped node capacities kc."
function firm_capacity(E::EnergyModel, kc)
    firm = 0.0
    for g in E.gens
        g.carrier == "gas" && (firm += sum(g.kf > 0 ? kc[g.kf] : g.kfix))
    end
    for s in E.stors
        firm += sum(s.kf > 0 ? kc[s.kf] : s.kfix)
    end
    return firm
end

"Non-gas availability R_t(k) and firm capacity for node weather W (after the data map)."
function residual_parts(E::EnergyModel, kc, W, T)
    R = zeros(T)
    for g in E.gens
        g.carrier == "gas" && continue
        kv = g.kf > 0 ? kc[g.kf] : g.kfix
        for t in 1:T, i in eachindex(g.bus)
            R[t] += (g.wf > 0 ? W[g.wf][i, t] : g.avail[i]) * kv[i]
        end
    end
    return R, firm_capacity(E, kc)
end

"Non-gas availability R_j(k) at the sample steps of the bias."
function sample_R(b::BiasModel, E::EnergyModel, kc)
    R = zeros(length(b.sD))
    for (gi, g) in enumerate(E.gens)
        g.carrier == "gas" && continue
        mul!(R, b.sA[gi], g.kf > 0 ? kc[g.kf] : g.kfix, 1.0, 1.0)
    end
    return R
end

"Clamped node capacities per k field at the level of `lk` and the clamp masks."
function node_capacities(E::EnergyModel, lk::Layout, zk)
    kc, mask = Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
    for (f, s) in enumerate(E.kspecs)
        y = vec(nodevals(zk, lk, E.kspecs, f))
        push!(kc, clamp.(y, s.lo, s.hi))
        push!(mask, iscomplete(lk, E.kspecs, f) ? ones(length(y)) : Float64.((s.lo .< y) .& (y .< s.hi)))
    end
    return kc, mask
end

"Features of ĝ for node weather `Xraw` (before the data map): u₁ per step and the firm capacity."
function bias_features(b::BiasModel, E::EnergyModel, lk::Layout, zk, Xraw)
    W = [map(x -> datamap(E.wspecs[f], x), Xraw[f]) for f in eachindex(Xraw)]
    T = size(W[1], 2)
    kc, _ = node_capacities(E, lk, zk)
    R, firm = residual_parts(E, kc, W, T)
    D = vec(sum(W[1]; dims=1))
    return (D .- R) ./ b.Dbar, firm
end

"ĝ from its features: Δt_b Σ_t F₁(u₁) + F₂(u₁ − firm/D̄) (tables)."
function ghat_features(b::BiasModel, U1::AbstractVector, firm::Float64)
    t1, t2 = tables(b)
    fd = firm / b.Dbar
    val = 0.0
    for t in eachindex(U1)
        val += tabval(t1, U1[t]) + tabval(t2, U1[t] - fd)
    end
    return b.Δt * val
end

"""
    ghat(b, E, lk, zk, Xraw; grad)

ĝ(x,k) for node weather `Xraw` (before the data map) at the level of the layout `lk`
(the level ℓ_b of the bias), from the tables. With `grad = true` it also returns the
gradients of the analytic ĝ w.r.t. the raw node weather and the k coordinates.
"""
function ghat(b::BiasModel, E::EnergyModel, lk::Layout, zk, Xraw; grad=false)
    U1, firm = bias_features(b, E, lk, zk, Xraw)
    val = ghat_features(b, U1, firm)
    grad || return val
    _, gX, gk = ghat_derivs(b, E, lk, zk, Xraw)
    return val, gX, gk
end

"""
Derivatives of the analytic ĝ: the per-step design rows Jt (T × nθ: ∂(Δt φ_j)/∂D_t),
the gradient w.r.t. the raw node weather and the gradient w.r.t. the k coordinates,
and the weather multipliers (∂u₁/∂X per field, node × step, times D̄).
"""
function ghat_derivs(b::BiasModel, E::EnergyModel, lk::Layout, zk, Xraw)
    W = [map(x -> datamap(E.wspecs[f], x), Xraw[f]) for f in eachindex(Xraw)]
    T = size(W[1], 2)
    Δt = b.Δt
    kc, mask = node_capacities(E, lk, zk)
    R, firm = residual_parts(E, kc, W, T)
    D = vec(sum(W[1]; dims=1))
    nb = nθ(b)
    φ, d1, d2 = zeros(nb), zeros(nb), zeros(nb)
    du = zeros(T)        # ∂ĝ/∂u₁ (u₂ shifts with u₁)
    du2 = zeros(T)       # ∂ĝ/∂u₂ only (for the firm-capacity dependence)
    Jt = zeros(T, nb)    # ∂(Δt Σ_t φ_j)/∂D_t
    for t in 1:T
        u1 = (D[t] - R[t]) / b.Dbar
        u2 = u1 - firm / b.Dbar
        bias_basis!(φ, d1, d2, b, u1, u2)
        du[t] = Δt * (dot(b.θ, d1) + dot(b.θ, d2))
        du2[t] = Δt * dot(b.θ, d2)
        @views Jt[t, :] .= Δt .* (d1 .+ d2) ./ b.Dbar
    end
    # ∂u₁·D̄ per raw node weather: load raises u, availability lowers it by the non-gas capacity on it
    Kw = weather_capacity(E, kc, length(Xraw), [nsupport(s.B) for s in E.wspecs])
    mult = [zeros(size(x)) for x in Xraw]
    for t in 1:T, n in axes(W[1], 1)
        mult[1][n, t] = datamap_deriv(E.wspecs[1], Xraw[1][n, t])
    end
    for f in 2:length(Xraw), t in 1:T, i in axes(Xraw[f], 1)
        mult[f][i, t] = -Kw[f][i] * datamap_deriv(E.wspecs[f], Xraw[f][i, t])
    end
    gX = [m .* reshape(du ./ b.Dbar, 1, :) for m in mult]
    # gradient w.r.t. k coordinates
    gk = zeros(lk.n)
    for g in E.gens
        g.kf > 0 || continue
        dk = zeros(length(g.bus))
        if g.carrier == "gas"
            dk .= -sum(du2) / b.Dbar
        else
            for t in 1:T, i in eachindex(g.bus)
                dk[i] -= du[t] / b.Dbar * (g.wf > 0 ? W[g.wf][i, t] : g.avail[i])
            end
        end
        add_field_grad!(gk, lk, E.kspecs, g.kf, reshape(mask[g.kf] .* dk, :, 1))
    end
    for s in E.stors
        s.kf > 0 || continue
        dk = fill(-sum(du2) / b.Dbar, length(s.bus))
        add_field_grad!(gk, lk, E.kspecs, s.kf, reshape(mask[s.kf] .* dk, :, 1))
    end
    return Jt, gX, gk, mult
end

"Per weather field: the summed clamped capacity of the non-gas generators on it, per support position."
function weather_capacity(E::EnergyModel, kc, nf::Int, sup::Vector{Int})
    Kw = [zeros(sup[f]) for f in 1:nf]
    for g in E.gens
        (g.wf > 0 && g.carrier != "gas") || continue
        Kw[g.wf] .+= g.kf > 0 ? kc[g.kf] : g.kfix
    end
    return Kw
end

"Ē ĝ at clamped node capacities kc with firm capacity `firm` (tables)."
function ghat_mean_kc(b::BiasModel, E::EnergyModel, kc, firm::Float64)
    Rs = sample_R(b, E, kc)
    t1, t2 = tables(b)
    fd = firm / b.Dbar
    val = 0.0
    for j in eachindex(b.sD)
        u1 = (b.sD[j] - Rs[j]) / b.Dbar
        val += tabval(t1, u1) + tabval(t2, u1 - fd)
    end
    return b.wstep * val
end

"Ē_x' ĝ(x',k) over the fixed sample steps (tables), and the k-gradient of the analytic form."
function ghat_mean(b::BiasModel, E::EnergyModel, lk::Layout, zk; grad=false)
    kc, mask = node_capacities(E, lk, zk)
    firm = firm_capacity(E, kc)
    val = ghat_mean_kc(b, E, kc, firm)
    grad || return val
    Rs = sample_R(b, E, kc)
    fd = firm / b.Dbar
    nb = nθ(b)
    φ, d1, d2 = zeros(nb), zeros(nb), zeros(nb)
    dkn = [zeros(length(x)) for x in kc]
    dfirm = 0.0
    for j in eachindex(b.sD)
        u1 = (b.sD[j] - Rs[j]) / b.Dbar
        bias_basis!(φ, d1, d2, b, u1, u1 - fd)
        du = b.wstep * (dot(b.θ, d1) + dot(b.θ, d2))
        dfirm -= b.wstep * dot(b.θ, d2) / b.Dbar
        for (gi, g) in enumerate(E.gens)
            (g.kf > 0 && g.carrier != "gas") || continue
            A = b.sA[gi]
            for i in eachindex(g.bus)
                dkn[g.kf][i] -= du / b.Dbar * A[j, i]
            end
        end
    end
    for g in E.gens
        g.kf > 0 && g.carrier == "gas" && (dkn[g.kf] .+= dfirm)
    end
    for s in E.stors
        s.kf > 0 && (dkn[s.kf] .+= dfirm)
    end
    gk = zeros(lk.n)
    for f in eachindex(E.kspecs)
        add_field_grad!(gk, lk, E.kspecs, f, reshape(mask[f] .* dkn[f], :, 1))
    end
    return val, gk
end

"""
Mode coefficients of the clamped node capacities, and a function applying the
transposed Jacobian. ĥ is evaluated on these so that, with regularized
constraints, it cannot outweigh the capacity penalty outside the box.
"""
function clamped_kcoeffs(E::EnergyModel, lk::Layout, zk)
    zc = zeros(lk.n)
    masks = Vector{Vector{Float64}}()
    for (f, s) in enumerate(E.kspecs)
        U = cols(s.B, lk.m[f])
        y = U * vec(fieldview(zk, lk, E.kspecs, f))
        fieldview(zc, lk, E.kspecs, f) .= reshape(U' * clamp.(y, s.lo, s.hi), :, 1)
        push!(masks, Float64.((s.lo .< y) .& (y .< s.hi)))
    end
    function vjp(g)
        out = zeros(lk.n)
        for (f, s) in enumerate(E.kspecs)
            U = cols(s.B, lk.m[f])
            fieldview(out, lk, E.kspecs, f) .= reshape(U' * (masks[f] .* (U * vec(fieldview(g, lk, E.kspecs, f)))), :, 1)
        end
        return out
    end
    return zc, vjp
end

"The fitted quadratic a κ + ½ c κ² of one ĥ coordinate, flat beyond its vertex κ* = −a/c when c < 0 (C¹)."
@inline function hsat(a::Float64, c::Float64, κ::Float64)
    c < 0 && κ > -a / c && (κ = -a / c)
    return a * κ + 0.5 * c * κ^2
end
@inline hsat_d(a::Float64, c::Float64, κ::Float64) = (c < 0 && κ > -a / c) ? 0.0 : a + c * κ

"""
ĥ of one coordinate: inside the range [lo, hi] of κ in the regression data the quadratic, flat
beyond its vertex when concave (`hsat`); outside the range continued linearly from the edge
with the edge slope clipped so that it never pushes away from the data (≥ 0 above hi, ≤ 0 below
lo). The fit can neither turn over inside the box nor run outward beyond its data.
"""
@inline function hcoord(a::Float64, c::Float64, lo::Float64, hi::Float64, κ::Float64)
    κc = κ > hi ? hi : (κ < lo ? lo : κ)
    s = κ > hi ? max(hsat_d(a, c, κc), 0.0) : (κ < lo ? min(hsat_d(a, c, κc), 0.0) : 0.0)
    return hsat(a, c, κc) + s * (κ - κc)
end
"Derivative of `hcoord` w.r.t. κ."
@inline hcoord_d(a::Float64, c::Float64, lo::Float64, hi::Float64, κ::Float64) =
    κ > hi ? max(hsat_d(a, c, hi), 0.0) : (κ < lo ? min(hsat_d(a, c, lo), 0.0) : hsat_d(a, c, κ))

"ĥ(k) on (clamped) k coordinates of level ℓ_b and its gradient."
function hhat(b::BiasModel, zk)
    val = 0.0
    for j in eachindex(zk)
        val += hcoord(b.a[j], b.c[j], b.κlo[j], b.κhi[j], zk[j] / b.kscale[j])
    end
    return val, [hcoord_d(b.a[j], b.c[j], b.κlo[j], b.κhi[j], zk[j] / b.kscale[j]) / b.kscale[j] for j in eachindex(zk)]
end

"Empty regression sums of ĥ for n coordinates (the min/max columns at ±Inf)."
hk_sums(n::Int) = hcat(zeros(n, 5), fill(Inf, n), fill(-Inf, n))

"""
The fields of the reference-measure term log J(k) = Σ_u n_u log(k_u + κ0) at the level of `lk`
(ensemble.md §5, `log_measure`), in its order: (k field, n_u) per counted unit group.
"""
function measure_fields(E::EnergyModel, lk::Layout)
    out = NTuple{2,Int}[]
    E.measure == :physical || return out
    lv = Layout(E.vspecs, lk.M, lk.h)
    for g in E.gens
        (g.kf > 0 && g.wf == 0 && g.vf > 0 && iscomplete(lv, E.vspecs, g.vf) && iscomplete(lk, E.kspecs, g.kf)) || continue
        push!(out, (g.kf, (g.ucf > 0 && iscomplete(lv, E.vspecs, g.ucf) ? 2 : 1) * lv.T))
    end
    for st in E.stors
        (st.kf > 0 && iscomplete(lv, E.vspecs, st.fe) && iscomplete(lk, E.kspecs, st.kf)) || continue
        push!(out, (st.kf, lv.T))
    end
    return out
end

"log J of the clamped node capacities kc (the value of `log_measure`)."
function logJ_kc(E::EnergyModel, mf, kc)
    val = 0.0
    for (f, n) in mf
        val += n * sum(log.(kc[f] .+ E.κ0))
    end
    return val
end

"""
    bias_value(b, E, lk, zk, Xraw; grad)

b̂ at the level ℓ_b of the bias (layout `lk`, raw node weather `Xraw` of that level):
ĝ − r Ē ĝ + (1 − r)(ĥ + log J), with the reference-measure term log J of the level exact.
Returns the value and, optionally, the gradients (analytic ĝ) w.r.t. the raw node
weather and the k coordinates.
"""
function bias_value(b::BiasModel, E::EnergyModel, lk::Layout, zk, Xraw; grad=false)
    lj = logJ_kc(E, measure_fields(E, lk), node_capacities(E, lk, zk)[1])
    if !grad
        g = ghat(b, E, lk, zk, Xraw)
        gm = ghat_mean(b, E, lk, zk)
        h, _ = hhat(b, clamped_kcoeffs(E, lk, zk)[1])
        return g - b.ratio * gm + (1 - b.ratio) * (h + lj)
    end
    g, gX, gk = ghat(b, E, lk, zk, Xraw; grad=true)
    gm, gkm = ghat_mean(b, E, lk, zk; grad=true)
    zc, vjp = clamped_kcoeffs(E, lk, zk)
    h, gkh = hhat(b, zc)
    gkh = vjp(gkh) .+ log_measure_grad(E, lk, Layout(E.vspecs, lk.M, lk.h), zk)
    return g - b.ratio * gm + (1 - b.ratio) * (h + lj), gX, gk .- b.ratio .* gkm .+ (1 - b.ratio) .* gkh
end

"""
Monotone-price fit of the ĝ coefficients (ensemble.md §4.4). ĝ ≈ log Z_x, so
dĝ/dD_t = −β_v Δt λ(u), with λ the implied price of the residual load u. With the
basis [u₁, ramps(u₁), u₂, ramps(u₂)] (u₂ = u₁ − firm/D̄ moves with u₁),

  λ(u) = −(θ_{u₁} + θ_{u₂} + Σ θ_ramp σ_ramp(u)) / (β_v D̄).

λ is non-decreasing iff every ramp coefficient is ≤ 0, and bounded in
[−c_dump, c_shed] iff θ_{u₁} + θ_{u₂} ≤ β_v D̄ c_dump and
θ_{u₁} + θ_{u₂} + Σ θ_ramp ≥ −β_v D̄ c_shed. The least-squares problem
min ½ θᵀAθ − θᵀg (the normal equations of the score regression) is solved under
these constraints as a small convex QP. ĝ can then grow at most at physical
price slopes, which removes the tail leak of the unconstrained fit. F₁ and F₂ are
then concave, so the secant slopes of their tables are non-increasing and the
implied price of the tabulated ĝ is monotone as well. In addition θ_{u₂} ≤ 0: the
shortfall part of the price is non-negative, so firm capacity never has a negative
value (∂ĝ/∂firm ≥ 0); the weather scores do not separate θ_{u₁} from θ_{u₂}.
"""
function monotone_fit(b::BiasModel, A::AbstractMatrix, g::AbstractVector, βv::Float64, c_shed::Float64, c_dump::Float64)
    n = length(g)
    i1 = 1
    r1 = 2:1+length(b.τ1)
    i2 = 2 + length(b.τ1)
    r2 = i2+1:n
    ramps = vcat(collect(r1), collect(r2))
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, θ[1:n])
    @constraint(model, [j in ramps], θ[j] <= 0.0)
    # firm capacity never has a negative value: F₂' ≤ 0 everywhere, i.e. its left-tail slope θ_{u₂} ≤ 0.
    # The weather scores do not separate θ_{u₁} and θ_{u₂} (their design columns are equal), so
    # this resolves a degeneracy without loss of fit.
    @constraint(model, θ[i2] <= 0.0)
    @constraint(model, θ[i1] + θ[i2] <= βv * b.Dbar * c_dump)
    @constraint(model, θ[i1] + θ[i2] + sum(θ[j] for j in ramps) >= -βv * b.Dbar * c_shed)
    As = Symmetric(Matrix(A))
    @objective(model, Min, 0.5 * sum(As[i, j] * θ[i] * θ[j] for i in 1:n, j in 1:n) - sum(g[i] * θ[i] for i in 1:n))
    optimize!(model)
    has_values(model) || return (A \ g)
    return value.(θ)
end

"Scales of the capacity coefficients of a layout (ĥ is quadratic in coefficient / scale)."
function coeff_scales(E::EnergyModel, lk::Layout)
    ks = zeros(lk.n)
    for (f, sp) in enumerate(E.kspecs)
        U = cols(sp.B, lk.m[f])
        fieldview(ks, lk, E.kspecs, f) .= reshape(abs.(U)' * sp.hi ./ 10 .+ 1.0, :, 1)
    end
    return ks
end

"""
    bias_samples(E, W, ℓ, rng; nsamples, nsteps) -> (sD, sA, wstep, Dmean)

Fixed exact weather steps for Ē at level ℓ: `nsamples` exact draws of the weather
at that level, of whose S·T steps `nsteps` are kept (uniformly without replacement;
all if `nsteps` ≤ 0 or ≥ S·T), each with the weight wstep = Δt T / nsteps, so that
Ē ĝ = wstep Σ_j (F₁ + F₂) is an unbiased estimate of the mean of ĝ over the draws.
`Dmean` is the mean system load of the draws.
"""
function bias_samples(E::EnergyModel, W::WeatherModel, ℓ::Int, rng; nsamples::Int=16, nsteps::Int=4096)
    lay = W.lays[ℓ+1]
    T, Δt = lay.T, Float64(lay.h)
    nall = nsamples * T
    sel = (nsteps <= 0 || nsteps >= nall) ? collect(1:nall) : sort(randperm(rng, nall)[1:nsteps])
    n = length(sel)
    sD = zeros(n)
    sA = [g.carrier == "gas" ? zeros(0, 0) : zeros(n, length(g.bus)) for g in E.gens]
    Dmean = 0.0
    j = 1
    for si in 1:nsamples
        m, z0, ws = sample_weather(rng, W, ℓ)
        x = weather_at_level(W, ℓ, m, z0, ws)
        Xn = weather_nodes(W, ℓ, x)
        Wd = [map(v -> datamap(E.wspecs[f], v), Xn[f]) for f in eachindex(Xn)]
        D = vec(sum(Wd[1]; dims=1))
        Dmean += mean(D) / nsamples
        while j <= n && sel[j] <= si * T
            t = sel[j] - (si - 1) * T
            sD[j] = D[t]
            for (gi, g) in enumerate(E.gens)
                g.carrier == "gas" && continue
                for i in eachindex(g.bus)
                    sA[gi][j, i] = g.wf > 0 ? Wd[g.wf][i, t] : g.avail[i]
                end
            end
            j += 1
        end
    end
    return sD, sA, Δt * T / n, Dmean
end

"""
    init_bias(E, W, lk, ratio; rng, nsamples, nsteps, τ1, τ2, s, res, ℓ)

A zero bias at level ℓ (layout `lk`, default level 0) with its sample steps for Ē and
the mean system load D̄ of the samples.
"""
function init_bias(E::EnergyModel, W::WeatherModel, lk::Layout, ratio; rng, nsamples=16, nsteps=4096,
                   τ1=collect(-1.5:0.25:0.75), τ2=collect(-0.6:0.1:0.4), s=0.05, res::Int=100, ℓ::Int=0)
    sD, sA, wstep, Dbar = bias_samples(E, W, ℓ, rng; nsamples=nsamples, nsteps=nsteps)
    nb = 2 + length(τ1) + length(τ2)
    θ = zeros(nb)
    b = BiasModel(τ1, τ2, s, θ, zeros(lk.n), zeros(lk.n), fill(-Inf, lk.n), fill(Inf, lk.n), coeff_scales(E, lk), Dbar,
                  ratio, ℓ, Float64(lk.h), sD, sA, wstep, nsamples, res, RampTable(0.0, 1.0, [0.0, 0.0]),
                  RampTable(0.0, 1.0, [0.0, 0.0]), Float64[], zeros(nb, nb), zeros(nb), hk_sums(lk.n), 0, true)
    return rebuild_tables!(b)
end

"""
    lift_bias!(b, E, W, lks, trks, ℓ; rng, nsamples, nsteps) -> b

Lift the bias from its level to level ℓ (the finest chains' level): θ and D̄ are
kept; a, c are prolongated (the coefficients of the old level are the prefix of each
field's coefficients, the new detail coefficients start at a = c = 0) and kscale is
that of the new layout (equal on the prefix); the regression sums are reset, so
that refits at level ℓ use level-ℓ data only; the sample steps of Ē are redrawn at
level ℓ.
"""
function lift_bias!(b::BiasModel, E::EnergyModel, W::WeatherModel, lks::Vector{Layout}, trks, ℓ::Int; rng,
                    nsamples::Int=b.nsamp, nsteps::Int=4096)
    ℓ >= b.ℓb || error("the bias can only be lifted to a finer level ($(b.ℓb) → $ℓ)")
    ℓ == b.ℓb && return b
    lk = lks[ℓ+1]
    b.a = prolong_to(b.a, lks, trks, E.kspecs, b.ℓb, ℓ)
    b.c = prolong_to(b.c, lks, trks, E.kspecs, b.ℓb, ℓ)
    b.κlo = prolong_to(b.κlo, lks, trks, E.kspecs, b.ℓb, ℓ)     # detail coefficients: a = c = 0, range [0, 0]
    b.κhi = prolong_to(b.κhi, lks, trks, E.kspecs, b.ℓb, ℓ)
    b.kscale = coeff_scales(E, lk)
    b.ℓb, b.Δt = ℓ, Float64(lk.h)
    nb = nθ(b)
    b.GtG, b.Gts, b.hk, b.nobs = zeros(nb, nb), zeros(nb), hk_sums(lk.n), 0
    b.sD, b.sA, b.wstep, _ = bias_samples(E, W, ℓ, rng; nsamples=nsamples, nsteps=nsteps)
    b.nsamp = nsamples
    return rebuild_tables!(b)
end

"Gradient w.r.t. the anomalies of level ℓ → gradient in the weather state coordinates (z0, ξ₁, …, ξ_ℓ) of that level."
function weather_coords_grad(W::WeatherModel, ℓ::Int, g::AbstractVector)
    parts = Vector{Vector{Float64}}()
    for l in ℓ:-1:1
        g, ξ = split_state(W.trs[l], W.lays[l], W.lays[l+1], W.specs, g)
        g .*= W.trs[l].r              # adjoint of the block-average prolongation (step fields)
        pushfirst!(parts, ξ)
    end
    return vcat(g, parts...)
end

"""
    bias_design(b, E, W, lk, zk, Xraw) -> (Jz, gk)

Design matrix of the weather score regression, ∇ĝ = Jz θ in the weather state
coordinates of level ℓ_b (analytic basis), and the k-gradient of the analytic ĝ.
"""
function bias_design(b::BiasModel, E::EnergyModel, W::WeatherModel, lk::Layout, zk, Xraw)
    Jt, _, gk, mult = ghat_derivs(b, E, lk, zk, Xraw)
    nb = nθ(b)
    ℓ = b.ℓb
    Jz = zeros(W.lays[ℓ+1].n, nb)
    for j in 1:nb
        cj = reshape(view(Jt, :, j), 1, :)
        Jz[:, j] .= weather_coords_grad(W, ℓ, weather_grad_to_x(W, ℓ, [m .* cj for m in mult]))
    end
    return Jz, gk
end

"""
Accumulate one learning sample with scores in the state coordinates of level ℓ_b:
sz = −β_v ∇C in the weather state coordinates (z0, ξ₁, …, ξ_ℓb) and
sk = −β_v ∂ₖC + ∂ₖ log J in the k coordinates of the level. The score identity holds
in any coordinates, so ∇ĝ is taken in the same ones. ĥ regresses what ĝ and the exact
measure term do not carry, sk − ∂ₖ log J − ∂ₖĝ. The sums go into `into` (the bias itself,
or the accumulators of the post-correction).
"""
function bias_observe!(b::BiasModel, E::EnergyModel, W::WeatherModel, lk::Layout, zk, Xraw, sz, sk; weight=1.0, into=b)
    Jz, gk = bias_design(b, E, W, lk, zk, Xraw)
    into.GtG .+= weight .* (Jz' * Jz)
    into.Gts .+= weight .* (Jz' * sz)
    κ = clamped_kcoeffs(E, lk, zk)[1] ./ b.kscale
    gJ = log_measure_grad(E, lk, Layout(E.vspecs, lk.M, lk.h), zk)
    y = (sk .- gJ .- gk) .* b.kscale     # target for ∂ĥ/∂κ
    @. into.hk[:, 1] += weight
    @. into.hk[:, 2] += weight * κ
    @. into.hk[:, 3] += weight * κ^2
    @. into.hk[:, 4] += weight * y
    @. into.hk[:, 5] += weight * y * κ
    @views into.hk[:, 6] .= min.(into.hk[:, 6], κ)
    @views into.hk[:, 7] .= max.(into.hk[:, 7], κ)
    into.nobs += 1
    return into
end

function bias_update!(b::BiasModel; ridge=1e-6, monotone=nothing)
    b.nobs < 1 && return b
    λ = ridge * tr_(b.GtG) / size(b.GtG, 1) + 1e-12
    if monotone === nothing
        b.θ .= (b.GtG + λ * I) \ b.Gts
    else
        b.θ .= monotone_fit(b, b.GtG + λ * I, b.Gts, monotone...)
    end
    for i in axes(b.hk, 1)
        n, sκ, sκ2, sy, syκ, κmin, κmax = b.hk[i, :]
        b.κlo[i], b.κhi[i] = n > 0 ? (κmin, κmax) : (-Inf, Inf)
        # the curvature is only identified once κ has varied; otherwise fit the slope
        varκ = sκ2 / n - (sκ / n)^2
        if varκ > 1e-6 * (1 + sκ2 / n)
            a, c = ([n sκ; sκ sκ2] + 1e-9 * n * I) \ [sy, syκ]
        else
            a, c = sy / n, 0.0
        end
        b.a[i], b.c[i] = a, c
    end
    return rebuild_tables!(b)
end

# ---------------------------------------------------------------------------
# incremental evaluation in the moves (docs/system/sampler.md §6, §7.2)
# ---------------------------------------------------------------------------
#
# The cache of a state at the level ℓ_b of the bias (spatially complete, so the
# capacities are the node capacities of every unit): the features per step for the
# chain's weather (D, R, u₁, F₁(u₁), F₂(u₂)) and per sample step (R, u₁, F₁, F₂), the
# firm capacity, the raw and clamped node capacities, the clamped coefficients of ĥ,
# and the sums ĝ, Ē ĝ, ĥ, log J. A capacity change of one unit moves R_t at every step
# (non-gas generators: A_{i,t} Δ) or the firm capacity (gas, storage), the clamped
# coefficients of its node and its term of log J; a weather change moves D_t and R_t at
# the steps of its window (Ē ĝ, ĥ and log J do not change).

mutable struct BiasCache
    T::Int
    tab1::RampTable
    tab2::RampTable
    At::Vector{Matrix{Float64}}     # per generator group (not gas, weather availability): chain's availability, T × support
    Kw::Vector{Vector{Float64}}     # per weather field: summed clamped non-gas capacity per support position
    Dx::Vector{Float64}
    Rx::Vector{Float64}
    U1x::Vector{Float64}
    F1x::Vector{Float64}
    F2x::Vector{Float64}
    Rs::Vector{Float64}
    U1s::Vector{Float64}
    F1s::Vector{Float64}
    F2s::Vector{Float64}
    firm::Float64
    kraw::Vector{Vector{Float64}}   # raw node capacities per k field
    kc::Vector{Vector{Float64}}     # clamped node capacities
    zc::Vector{Float64}             # clamped coefficients (ĥ)
    mf::Vector{NTuple{2,Int}}       # reference-measure term: (k field, n_u) in the order of `log_measure`
    nJ::Vector{Int}                 # n_u per k field (0: not in log J)
    κ0::Float64
    g::Float64
    gm::Float64
    h::Float64
    lj::Float64                     # log J
end

"Bias cache of capacity coordinates `zk` and raw node weather `Xraw` at the level ℓ_b of `b` (layout `lk`, complete)."
function BiasCache(b::BiasModel, E::EnergyModel, lk::Layout, zk, Xraw)
    all(f -> iscomplete(lk, E.kspecs, f), eachindex(E.kspecs)) || error("the bias cache needs complete capacity fields")
    W = [map(x -> datamap(E.wspecs[f], x), Xraw[f]) for f in eachindex(Xraw)]
    T = size(W[1], 2)
    kraw = [vec(nodevals(zk, lk, E.kspecs, f)) for f in eachindex(E.kspecs)]
    kc = [clamp.(kraw[f], s.lo, s.hi) for (f, s) in enumerate(E.kspecs)]
    R, firm = residual_parts(E, kc, W, T)
    D = vec(sum(W[1]; dims=1))
    At = [(g.wf > 0 && g.carrier != "gas") ? permutedims(W[g.wf]) : zeros(0, 0) for g in E.gens]
    Kw = weather_capacity(E, kc, length(Xraw), [size(x, 1) for x in Xraw])
    Rs = sample_R(b, E, kc)
    t1, t2 = tables(b)
    mf = measure_fields(E, lk)
    nJ = zeros(Int, length(E.kspecs))
    for (f, n) in mf
        nJ[f] = n
    end
    BC = BiasCache(T, t1, t2, At, Kw, D, R, zeros(T), zeros(T), zeros(T), Rs, zeros(length(Rs)), zeros(length(Rs)),
                   zeros(length(Rs)), firm, kraw, kc, clamped_kcoeffs(E, lk, zk)[1], mf, nJ, E.κ0, 0.0, 0.0, 0.0, 0.0)
    fd = firm / b.Dbar
    for t in 1:T
        u1 = (D[t] - R[t]) / b.Dbar
        BC.U1x[t] = u1; BC.F1x[t] = tabval(t1, u1); BC.F2x[t] = tabval(t2, u1 - fd)
    end
    for j in eachindex(Rs)
        u1 = (b.sD[j] - Rs[j]) / b.Dbar
        BC.U1s[j] = u1; BC.F1s[j] = tabval(t1, u1); BC.F2s[j] = tabval(t2, u1 - fd)
    end
    return bias_resum!(BC, b)
end

"Recompute the sums ĝ, Ē ĝ, ĥ, log J of the cache from its per-step values, coefficients and capacities."
function bias_resum!(BC::BiasCache, b::BiasModel)
    g = 0.0
    for t in 1:BC.T
        g += BC.F1x[t] + BC.F2x[t]
    end
    gm = 0.0
    for j in eachindex(BC.F1s)
        gm += BC.F1s[j] + BC.F2s[j]
    end
    lj = 0.0
    for (f, n) in BC.mf
        lj += n * sum(log.(BC.kc[f] .+ BC.κ0))
    end
    BC.g, BC.gm, BC.h, BC.lj = b.Δt * g, b.wstep * gm, hhat(b, BC.zc)[1], lj
    return BC
end

"b̂ of the cached state."
bias_value(BC::BiasCache, b::BiasModel) = BC.g - b.ratio * BC.gm + (1 - b.ratio) * (BC.h + BC.lj)

"k field of unit i of a generator group (kind 1) or storage group (kind 2), and whether the unit is firm."
@inline unit_kfield(E::EnergyModel, kind::Int, gi::Int) =
    kind == 1 ? (E.gens[gi].kf, E.gens[gi].carrier == "gas") : (E.stors[gi].kf, true)

"Change of ĥ when node i of k field kf changes its clamped capacity by Δc (optionally committed)."
function hhat_delta!(BC::BiasCache, b::BiasModel, E::EnergyModel, lk::Layout, kf::Int, i::Int, Δc::Float64, commit::Bool)
    Ut = E.kspecs[kf].B.Ut
    off = lk.off[kf]
    rv, nz = rowvals(Ut), nonzeros(Ut)
    dh = 0.0
    @inbounds for q in nzrange(Ut, i)
        j = off + rv[q]
        z0 = BC.zc[j]
        z1 = z0 + nz[q] * Δc
        dh += hcoord(b.a[j], b.c[j], b.κlo[j], b.κhi[j], z1 / b.kscale[j]) -
              hcoord(b.a[j], b.c[j], b.κlo[j], b.κhi[j], z0 / b.kscale[j])
        commit && (BC.zc[j] = z1)
    end
    return dh
end

"A constant vector (the availability of a generator without weather dependence)."
struct Fill1 <: AbstractVector{Float64}
    v::Float64
    n::Int
end
Base.size(x::Fill1) = (x.n,)
Base.getindex(x::Fill1, ::Int) = x.v

"""
Σ_t of the change of F₁(u₁) + F₂(u₂) when R_t grows by A_t Δc at every step (steps with
A_t = 0 are unchanged); with `commit` the per-step values are updated.
"""
function shift_steps!(R, D, U1, F1, F2, A::AbstractVector, Δc::Float64, D̄::Float64, fd::Float64,
                      t1::RampTable, t2::RampTable, commit::Bool)
    dg = 0.0
    @inbounds for t in eachindex(R)
        a = A[t]
        a == 0.0 && continue
        R1 = R[t] + a * Δc
        u1 = (D[t] - R1) / D̄
        f1, f2 = tabval(t1, u1), tabval(t2, u1 - fd)
        dg += f1 + f2 - F1[t] - F2[t]
        commit && (R[t] = R1; U1[t] = u1; F1[t] = f1; F2[t] = f2)
    end
    return dg
end

"""
    bias_delta_k(BC, b, E, lk, kind, gi, i, knew; commit=false) -> Δb̂

Change of b̂ when the raw node capacity of unit i of generator group gi (kind 1) or
storage group gi (kind 2) becomes `knew`. With `commit = true` the cache is updated.
"""
function bias_delta_k(BC::BiasCache, b::BiasModel, E::EnergyModel, lk::Layout, kind::Int, gi::Int, i::Int, knew::Float64;
                      commit::Bool=false)
    kf, firmunit = unit_kfield(E, kind, gi)
    sp = E.kspecs[kf]
    kc0 = BC.kc[kf][i]
    kc1 = clamp(knew, sp.lo[i], sp.hi[i])
    Δc = kc1 - kc0
    commit && (BC.kraw[kf][i] = knew; BC.kc[kf][i] = kc1)
    Δc == 0.0 && return 0.0
    t1, t2 = BC.tab1, BC.tab2
    D̄ = b.Dbar
    dg, dgm = 0.0, 0.0
    if firmunit
        fn = (BC.firm + Δc) / D̄
        @inbounds for t in 1:BC.T
            f2 = tabval(t2, BC.U1x[t] - fn)
            dg += f2 - BC.F2x[t]
            commit && (BC.F2x[t] = f2)
        end
        @inbounds for j in eachindex(BC.U1s)
            f2 = tabval(t2, BC.U1s[j] - fn)
            dgm += f2 - BC.F2s[j]
            commit && (BC.F2s[j] = f2)
        end
        commit && (BC.firm += Δc)
    else
        g = E.gens[gi]
        fd = BC.firm / D̄
        Ax = g.wf > 0 ? view(BC.At[gi], :, i) : Fill1(g.avail[i], BC.T)
        dg = shift_steps!(BC.Rx, BC.Dx, BC.U1x, BC.F1x, BC.F2x, Ax, Δc, D̄, fd, t1, t2, commit)
        dgm = shift_steps!(BC.Rs, b.sD, BC.U1s, BC.F1s, BC.F2s, view(b.sA[gi], :, i), Δc, D̄, fd, t1, t2, commit)
        commit && g.wf > 0 && (BC.Kw[g.wf][i] += Δc)
    end
    dh = hhat_delta!(BC, b, E, lk, kf, i, Δc, commit)
    dlj = BC.nJ[kf] > 0 ? BC.nJ[kf] * (log(kc1 + BC.κ0) - log(kc0 + BC.κ0)) : 0.0
    commit && bias_resum!(BC, b)
    return b.Δt * dg - b.ratio * b.wstep * dgm + (1 - b.ratio) * (dh + dlj)
end
