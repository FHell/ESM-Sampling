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
end
OnlineBias(; every::Int=1, freeze::Int=typemax(Int)) = OnlineBias(0, Int[], every, freeze)

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
counts 2^j (≥ 4, before `OB.freeze`). After a refit the state is re-evaluated.
"""
function online_bias_step!(P::Problem, s::State, OB::OnlineBias; monotone=nothing)
    P.use_bias || return s, false
    OB.nobs >= OB.freeze && return s, false
    OB.nobs += 1
    OB.nobs % OB.every == 0 || return s, false
    Xb, sx, sk, kb = bias_scores(P, s)
    bias_observe!(P.bias, P.E, P.W, P.lk[P.bias.ℓb+1], kb, Xb, sx, sk)
    n = OB.nobs ÷ OB.every
    (n >= 4 && ispow2(n)) || return s, false
    bias_update!(P.bias; monotone=monotone)
    push!(OB.updates, OB.nobs)
    return evaluate!(P, s), true
end
