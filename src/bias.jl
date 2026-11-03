# Learned weather–investment bias (§6).
#
#   b̂(x,k) = ĝ(x,k) − (β_k/β_v) Ē_x' ĝ(x',k) + (1 − β_k/β_v) ĥ(k),
#
# which is b̂₁ + (1 − β_k/β_v) Λ̂ with b̂₁ = ĝ − Ēĝ and Λ̂ = Ēĝ + ĥ. Here ĝ models
# the weather dependence of log Z and ĥ the remaining k-dependence of Λ.
#
# ĝ is local in time: a sum over level-0 steps of a function of the system
# residual load u₁ = (D − R(k))/D̄ (load minus non-dispatchable availability)
# and of the shortfall u₂ = u₁ − firm(k)/D̄ (firm: gas and storage power),
# expanded in softplus ramps. ∇ₓĝ is regressed on the per-sample scores
# −β_v ∇ₓC (score identity, valid because Ω does not depend on x); ∇ₖĥ is
# regressed on the residual k-scores −β_v ∂ₖC − ∂ₖĝ. Both regressions are
# running least squares (SGD with 1/n steps), frozen after the warmup.

mutable struct BiasModel
    τ1::Vector{Float64}
    τ2::Vector{Float64}
    s::Float64
    θ::Vector{Float64}
    a::Vector{Float64}
    c::Vector{Float64}
    kscale::Vector{Float64}
    Dbar::Float64
    ratio::Float64                 # β_k / β_v
    # fixed exact weather samples for Ē (node-level data after the data map)
    sampD::Vector{Vector{Float64}}             # system load per step
    sampA::Vector{Vector{Matrix{Float64}}}     # availability per generator group (S × T)
    # running least squares
    GtG::Matrix{Float64}
    Gts::Vector{Float64}
    hk::Matrix{Float64}            # per k coordinate: [Σ1 Σκ Σκ² Σy Σyκ]
    nobs::Int
    active::Bool
end

nθ(b::BiasModel) = 1 + length(b.τ1) + 1 + length(b.τ2)

softplus(x) = x > 30 ? x : log1p(exp(x))
sigmoid(x) = 1 / (1 + exp(-x))

"Basis φ(u) and φ'(u) for the two feature groups at one step."
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

"Non-dispatchable availability R_t(k) and firm capacity for node weather W (after the data map)."
function residual_parts(E::EnergyModel, kc, W, T)
    R = zeros(T)
    firm = 0.0
    for g in E.gens
        kv = g.kf > 0 ? kc[g.kf] : g.kfix
        if g.carrier == "gas"
            firm += sum(kv)
            continue
        end
        for t in 1:T, i in eachindex(g.bus)
            R[t] += (g.wf > 0 ? W[g.wf][i, t] : g.avail[i]) * kv[i]
        end
    end
    for s in E.stors
        firm += sum(s.kf > 0 ? kc[s.kf] : s.kfix)
    end
    return R, firm
end

"Clamped node capacities per k field at level 0 and the clamp masks."
function node_capacities(E::EnergyModel, lk::Layout, zk)
    kc, mask = Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
    for (f, s) in enumerate(E.kspecs)
        y = vec(nodevals(zk, lk, E.kspecs, f))
        push!(kc, clamp.(y, s.lo, s.hi))
        push!(mask, iscomplete(lk, E.kspecs, f) ? ones(length(y)) : Float64.((s.lo .< y) .& (y .< s.hi)))
    end
    return kc, mask
end

"""
    ghat(b, E, lk, zk, Xraw, Δt; grad, design)

ĝ(x,k) for node weather `Xraw` (before the data map) at level 0. Optionally
returns the gradients w.r.t. the raw node weather and the level-0 k coordinates,
and the design matrix J with ∇ₓĝ = J θ (columns: raw node weather, flattened).
"""
function ghat(b::BiasModel, E::EnergyModel, lk::Layout, zk, Xraw, Δt; grad=false, design=false)
    W = [map(x -> datamap(E.wspecs[f], x), Xraw[f]) for f in eachindex(Xraw)]
    T = size(W[1], 2)
    kc, mask = node_capacities(E, lk, zk)
    R, firm = residual_parts(E, kc, W, T)
    D = vec(sum(W[1]; dims=1))
    nb = nθ(b)
    φ, d1, d2 = zeros(nb), zeros(nb), zeros(nb)
    val = 0.0
    du = zeros(T)        # ∂ĝ/∂u₁ (u₂ shifts with u₁)
    du2 = zeros(T)       # ∂ĝ/∂u₂ only (for the firm-capacity dependence)
    Jt = design ? zeros(T, nb) : nothing   # ∂(Σφ_j)/∂D_t per basis function
    for t in 1:T
        u1 = (D[t] - R[t]) / b.Dbar
        u2 = u1 - firm / b.Dbar
        bias_basis!(φ, d1, d2, b, u1, u2)
        val += Δt * dot(b.θ, φ)
        du[t] = Δt * (dot(b.θ, d1) + dot(b.θ, d2))
        du2[t] = Δt * dot(b.θ, d2)
        if design
            Jt[t, :] .= Δt .* (d1 .+ d2) ./ b.Dbar
        end
    end
    grad || design || return val
    # gradient w.r.t. raw node weather: load raises u, availability lowers it
    gX = [zeros(size(x)) for x in Xraw]
    for t in 1:T, n in axes(W[1], 1)
        gX[1][n, t] = du[t] / b.Dbar * datamap_deriv(E.wspecs[1], Xraw[1][n, t])
    end
    for g in E.gens
        g.wf > 0 || continue
        kv = g.kf > 0 ? kc[g.kf] : g.kfix
        for t in 1:T, i in eachindex(g.bus)
            gX[g.wf][i, t] -= du[t] / b.Dbar * kv[i] * datamap_deriv(E.wspecs[g.wf], Xraw[g.wf][i, t])
        end
    end
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
    if design
        # J θ = ∇ₓĝ: d/dD_t enters every load node; availability enters with −k
        J = [zeros(length(x), nb) for x in Xraw]
        for t in 1:T, n in axes(W[1], 1)
            dm = datamap_deriv(E.wspecs[1], Xraw[1][n, t])
            J[1][(t-1)*size(W[1], 1)+n, :] .= dm .* Jt[t, :]
        end
        for g in E.gens
            g.wf > 0 || continue
            kv = g.kf > 0 ? kc[g.kf] : g.kfix
            S = length(g.bus)
            for t in 1:T, i in 1:S
                dm = datamap_deriv(E.wspecs[g.wf], Xraw[g.wf][i, t])
                J[g.wf][(t-1)*S+i, :] .-= dm * kv[i] .* Jt[t, :]
            end
        end
        return val, gX, gk, J
    end
    return val, gX, gk
end

"Ē_x' ĝ(x',k) over the fixed exact weather samples, and its k-gradient."
function ghat_mean(b::BiasModel, E::EnergyModel, lk::Layout, zk, Δt; grad=false)
    kc, mask = node_capacities(E, lk, zk)
    nb = nθ(b)
    φ, d1, d2 = zeros(nb), zeros(nb), zeros(nb)
    S = length(b.sampD)
    val = 0.0
    dkn = [zeros(length(x)) for x in kc]
    for si in 1:S
        D, A = b.sampD[si], b.sampA[si]
        T = length(D)
        R = zeros(T)
        firm = 0.0
        for (gi, g) in enumerate(E.gens)
            kv = g.kf > 0 ? kc[g.kf] : g.kfix
            g.carrier == "gas" && (firm += sum(kv); continue)
            for t in 1:T, i in eachindex(g.bus)
                R[t] += A[gi][i, t] * kv[i]
            end
        end
        for s in E.stors
            firm += sum(s.kf > 0 ? kc[s.kf] : s.kfix)
        end
        dfirm = 0.0
        for t in 1:T
            u1 = (D[t] - R[t]) / b.Dbar
            u2 = u1 - firm / b.Dbar
            bias_basis!(φ, d1, d2, b, u1, u2)
            val += Δt * dot(b.θ, φ) / S
            if grad
                du = Δt * (dot(b.θ, d1) + dot(b.θ, d2)) / S
                dfirm -= Δt * dot(b.θ, d2) / S / b.Dbar
                for (gi, g) in enumerate(E.gens)
                    (g.kf > 0 && g.carrier != "gas") || continue
                    for i in eachindex(g.bus)
                        dkn[g.kf][i] -= du / b.Dbar * A[gi][i, t]
                    end
                end
            end
        end
        if grad
            for g in E.gens
                g.kf > 0 && g.carrier == "gas" && (dkn[g.kf] .+= dfirm)
            end
            for s in E.stors
                s.kf > 0 && (dkn[s.kf] .+= dfirm)
            end
        end
    end
    grad || return val
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

"ĥ(k) on (clamped) level-0 k coordinates and its gradient."
function hhat(b::BiasModel, zk)
    κ = zk ./ b.kscale
    val = sum(b.a .* κ .+ 0.5 .* b.c .* κ .^ 2)
    return val, (b.a .+ b.c .* κ) ./ b.kscale
end

"""
    bias_value(b, E, lk0, zk0, X0raw, Δt; grad)

b̂ at level 0 (raw node weather `X0raw`). Returns value and, optionally, the
gradients w.r.t. the raw node weather and the level-0 k coordinates.
"""
function bias_value(b::BiasModel, E::EnergyModel, lk0::Layout, zk0, X0raw, Δt; grad=false)
    if !grad
        g = ghat(b, E, lk0, zk0, X0raw, Δt)
        gm = ghat_mean(b, E, lk0, zk0, Δt)
        h, _ = hhat(b, clamped_kcoeffs(E, lk0, zk0)[1])
        return g - b.ratio * gm + (1 - b.ratio) * h
    end
    g, gX, gk = ghat(b, E, lk0, zk0, X0raw, Δt; grad=true)
    gm, gkm = ghat_mean(b, E, lk0, zk0, Δt; grad=true)
    zc, vjp = clamped_kcoeffs(E, lk0, zk0)
    h, gkh = hhat(b, zc)
    gkh = vjp(gkh)
    return g - b.ratio * gm + (1 - b.ratio) * h, gX, gk .- b.ratio .* gkm .+ (1 - b.ratio) .* gkh
end

"""
Monotone-price fit of the ĝ coefficients (docs/setup_overview.md §10). ĝ ≈ log Z_x,
so dĝ/dD_t = −β_v Δt λ(u), with λ the implied price of the residual load u. With
the basis [u₁, ramps(u₁), u₂, ramps(u₂)] (u₂ = u₁ − firm/D̄ moves with u₁),

  λ(u) = −(θ_{u₁} + θ_{u₂} + Σ θ_ramp σ_ramp(u)) / (β_v D̄).

λ is non-decreasing iff every ramp coefficient is ≤ 0, and bounded in
[−c_dump, c_shed] iff θ_{u₁} + θ_{u₂} ≤ β_v D̄ c_dump and
θ_{u₁} + θ_{u₂} + Σ θ_ramp ≥ −β_v D̄ c_shed. The least-squares problem
min ½ θᵀAθ − θᵀg (the normal equations of the score regression) is solved under
these constraints as a small convex QP. ĝ can then grow at most at physical
price slopes, which removes the tail leak of the unconstrained fit.
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
    @constraint(model, θ[i1] + θ[i2] <= βv * b.Dbar * c_dump)
    @constraint(model, θ[i1] + θ[i2] + sum(θ[j] for j in ramps) >= -βv * b.Dbar * c_shed)
    As = Symmetric(Matrix(A))
    @objective(model, Min, 0.5 * sum(As[i, j] * θ[i] * θ[j] for i in 1:n, j in 1:n) - sum(g[i] * θ[i] for i in 1:n))
    optimize!(model)
    has_values(model) || return (A \ g)
    return value.(θ)
end

function init_bias(E::EnergyModel, W::WeatherModel, lk0::Layout, ratio; rng, nsamples=16,
                   τ1=collect(-1.5:0.25:0.75), τ2=collect(-0.6:0.1:0.4), s=0.05)
    sampD, sampA = Vector{Float64}[], Vector{Matrix{Float64}}[]
    Dbar = 0.0
    for _ in 1:nsamples
        m, z0, _ = sample_weather(rng, W, 0)
        Xn = weather_nodes(W, 0, z0)
        Wd = [map(x -> datamap(E.wspecs[f], x), Xn[f]) for f in eachindex(Xn)]
        D = vec(sum(Wd[1]; dims=1))
        push!(sampD, D)
        Dbar += mean(D) / nsamples
        push!(sampA, [g.wf > 0 ? Wd[g.wf] : repeat(g.avail, 1, length(D)) for g in E.gens])
    end
    nb = 2 + length(τ1) + length(τ2)
    kscale = zeros(lk0.n)
    for (f, sp) in enumerate(E.kspecs)
        U = cols(sp.B, lk0.m[f])
        fieldview(kscale, lk0, E.kspecs, f) .= reshape(abs.(U)' * sp.hi ./ 10 .+ 1.0, :, 1)
    end
    return BiasModel(τ1, τ2, s, zeros(nb), zeros(lk0.n), zeros(lk0.n), kscale, Dbar, ratio,
                     sampD, sampA, zeros(nb, nb), zeros(nb), zeros(lk0.n, 5), 0, true)
end

"""
Accumulate one learning sample with scores in the state coordinates:
sz = −β_v ∇_{z0} C (level-0 standardized weather) and sk = −β_v ∇_{k0} C
(level-0 k coordinates). The score identity holds for the state coordinates,
so ∇ĝ is projected onto them as well.
"""
function bias_observe!(b::BiasModel, E::EnergyModel, W::WeatherModel, lk0::Layout, zk0, X0raw, Δt, sz, sk; weight=1.0)
    _, _, gk, J = ghat(b, E, lk0, zk0, X0raw, Δt; design=true)
    nb = nθ(b)
    Jz = zeros(length(sz), nb)
    for j in 1:nb
        Jz[:, j] .= weather_grad_to_x(W, 0, [reshape(view(J[f], :, j), size(X0raw[f])) for f in eachindex(J)])
    end
    b.GtG .+= weight .* (Jz' * Jz)
    b.Gts .+= weight .* (Jz' * sz)
    κ = clamped_kcoeffs(E, lk0, zk0)[1] ./ b.kscale
    y = (sk .- gk) .* b.kscale     # target for ∂ĥ/∂κ
    @. b.hk[:, 1] += weight
    @. b.hk[:, 2] += weight * κ
    @. b.hk[:, 3] += weight * κ^2
    @. b.hk[:, 4] += weight * y
    @. b.hk[:, 5] += weight * y * κ
    b.nobs += 1
end

function bias_update!(b::BiasModel; ridge=1e-6, monotone=nothing)
    b.nobs < 1 && return
    λ = ridge * tr_(b.GtG) / size(b.GtG, 1) + 1e-12
    if monotone === nothing
        b.θ .= (b.GtG + λ * I) \ b.Gts
    else
        b.θ .= monotone_fit(b, b.GtG + λ * I, b.Gts, monotone...)
    end
    for i in axes(b.hk, 1)
        n, sκ, sκ2, sy, syκ = b.hk[i, :]
        # the curvature is only identified once κ has varied; otherwise fit the slope
        varκ = sκ2 / n - (sκ / n)^2
        if varκ > 1e-6 * (1 + sκ2 / n)
            a, c = ([n sκ; sκ sκ2] + 1e-9 * n * I) \ [sy, syκ]
        else
            a, c = sy / n, 0.0
        end
        b.a[i], b.c[i] = a, c
    end
end

