# ---------------------------------------------------------------------------
# online bias learning with diminishing adaptation (docs/system/sampler.md §7.1)
# ---------------------------------------------------------------------------
#
# During the finest-level warmup, scores of the current state are added to the bias
# regression at the level ℓ_b of the bias (the chains' level after the lift):
# s_x = −β_v ∇C in the weather state coordinates (z0, ξ₁, …, ξ_ℓb) of that level (the
# gradient w.r.t. the anomalies of the chain's level, carried down to ℓ_b by the adjoint
# of the time transfers if the chain is finer, then split into the level transforms) and
# s_k = −β_v ∂ₖC + ∂ₖ log J in the capacity coordinates of the level (restriction =
# coefficient prefix). The accumulators are plain sums, i.e. running means with weight
# 1/n. The coefficients are refitted (monotone QP) only when the observation count
# reaches 2^j, and never after `freeze`, so the target changes on a geometric schedule
# with vanishing steps: the chain is a stochastic approximation whose limit is π with
# the final b̂.

mutable struct OnlineBias
    nobs::Int
    updates::Vector{Int}
    every::Int
    freeze::Int
    nmin::Int                      # no refit before this many observations (the lifted fit stays until then)
    trust::Float64                 # trust region of a refit (0: off), relative to the spread of b̂ over the recent states
    nrecent::Int                   # number of recent observed states it looks at
    recent::Vector{Any}            # their bias records (u₁ per step, clamped node capacities, clamped coefficients)
    steps::Vector{Float64}         # step fraction λ of every refit
end
OnlineBias(; every::Int=1, freeze::Int=typemax(Int), nmin::Int=4, trust::Float64=0.0, nrecent::Int=32) =
    OnlineBias(0, Int[], every, freeze, max(nmin, 4), trust, nrecent, Any[], Float64[])

"Whether the n-th observation is followed by a refit: n = 2^j ≥ `OB.nmin` (at least 4)."
refit_due(OB::OnlineBias, n::Int) = n >= OB.nmin && ispow2(n)

"""
    bias_scores(P, s) -> (Xb, s_x, s_k, kb)

Scores of a state for the bias regression at the level ℓ_b of the bias: the raw node
weather and the capacity coordinates of that level, s_x in its weather state coordinates
and s_k in its capacity coordinates.
"""
function bias_scores(P::Problem, s::State)
    E, W = P.E, P.W
    ℓ, ℓb = s.ℓ, P.bias.ℓb
    lk, lv = P.lk[ℓ+1], P.lv[ℓ+1]
    Xn = weather_nodes(W, ℓ, s.xs[end])
    _, (_, gko, _, gX) = energy_cost(E, lk, lv, s.k, s.v, Xn; grad=true)
    gx = weather_grad_to_x(W, ℓ, [-P.βv .* g for g in gX])
    for l in ℓ:-1:ℓb+1                       # anomalies of level ℓ → level ℓ_b (details fixed)
        gx, _ = split_state(W.trs[l], W.lays[l], W.lays[l+1], W.specs, gx)
        gx .*= W.trs[l].r
    end
    sx = weather_coords_grad(W, ℓb, gx)
    gk = -P.βv .* gko .+ log_measure_grad(E, lk, lv, s.k)
    sk = restrict_to(gk, P.lk, P.trk, E.kspecs, ℓ, ℓb)
    kb, xb = bias_restrict(P, ℓ, s.k, s.xs[end], s.z0)
    return weather_nodes(W, ℓb, xb), sx, sk, kb
end

"""
    online_bias_step!(P, s, OB; monotone) -> (s, updated)

Observe the scores of `s` (every `OB.every` calls) and refit b̂ at observation
counts 2^j ≥ `OB.nmin` (before `OB.freeze`). After a refit the state is re-evaluated.
"""
function online_bias_step!(P::Problem, s::State, OB::OnlineBias; monotone=nothing)
    P.use_bias || return s, false
    OB.nobs >= OB.freeze && return s, false
    OB.nobs += 1
    OB.nobs % OB.every == 0 || return s, false
    Xb, sx, sk, kb = bias_scores(P, s)
    lkb = P.lk[P.bias.ℓb+1]
    bias_observe!(P.bias, P.E, P.W, lkb, kb, Xb, sx, sk)
    if OB.trust > 0
        push!(OB.recent, bias_record(P.bias, P.E, lkb, kb, Xb))
        length(OB.recent) > OB.nrecent && popfirst!(OB.recent)
    end
    n = OB.nobs ÷ OB.every
    refit_due(OB, n) || return s, false
    old = bias_coeffs(P.bias)
    bias_update!(P.bias; monotone=monotone)
    λ = OB.trust > 0 ? trust_step!(P.bias, P.E, lkb, OB.recent, old, OB.trust) : 1.0
    push!(OB.updates, OB.nobs); push!(OB.steps, λ)
    return evaluate!(P, s), true
end

"Bias record of a state at level ℓ_b (layout `lk`): what `bias_from_record` needs."
function bias_record(b::BiasModel, E::EnergyModel, lk::Layout, kb, Xb)
    U1, _ = bias_features(b, E, lk, kb, Xb)
    return (U1, node_capacities(E, lk, kb)[1], clamped_kcoeffs(E, lk, kb)[1])
end

"The fitted coefficients (θ, a, c) of the bias."
bias_coeffs(b::BiasModel) = (copy(b.θ), copy(b.a), copy(b.c))

"Set the coefficients to (1 − λ) old + λ new (rebuilding the tables)."
function set_coeffs!(b::BiasModel, old, new, λ::Float64)
    b.θ .= (1 - λ) .* old[1] .+ λ .* new[1]
    b.a .= (1 - λ) .* old[2] .+ λ .* new[2]
    b.c .= (1 - λ) .* old[3] .+ λ .* new[3]
    return rebuild_tables!(b)
end

"""
    trust_step!(b, E, lk, recent, old, τ) -> λ

Trust region of a refit: the bias holds the new fit; replace it by the convex combination
(1 − λ) old + λ new with the largest λ ≤ 1 for which the change of b̂ minus its mean over the
recent states stays below τ max(S, n_k), S the spread (max − min) of the old b̂ over those
states and n_k the number of capacity coordinates (the floor for an unfitted bias). A convex
combination keeps the linear constraints of the monotone fit. ĝ, Ē ĝ and log J are linear in
the coefficients and ĥ nearly so (it saturates beyond its vertex), so λ = bound / deviation of
the full step, halved until the deviation is within the bound.
"""
function trust_step!(b::BiasModel, E::EnergyModel, lk::Layout, recent, old, τ::Float64)
    length(recent) < 2 && return 1.0
    new = bias_coeffs(b)
    vals() = [bias_from_record(b, E, lk, r...) for r in recent]
    dev(d) = maximum(abs.(d .- mean(d)))
    set_coeffs!(b, old, new, 0.0)
    b0 = vals()
    bound = τ * max(maximum(b0) - minimum(b0), Float64(lk.n))
    set_coeffs!(b, old, new, 1.0)
    d1 = dev(vals() .- b0)
    d1 <= bound && return 1.0
    λ = bound / d1
    for _ in 1:30
        set_coeffs!(b, old, new, λ)
        dev(vals() .- b0) <= bound && return λ
        λ /= 2
    end
    set_coeffs!(b, old, new, 0.0)
    return 0.0
end
