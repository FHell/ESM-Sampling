# ---------------------------------------------------------------------------
# online bias learning with diminishing adaptation (docs/setup_overview.md §6)
# ---------------------------------------------------------------------------
#
# During production, scores of the current (fine-level) state are added to the bias
# regression: s_z = −β_v ∇_{z0} C (the fine weather gradient carried to the level-0
# weather coordinates by the adjoint of the time transfers, as for the LP warm
# start) and s_k = −β_v ∂_{k0} C (restriction = coefficient prefix). The accumulators
# are plain sums, i.e. running means with weight 1/n. The coefficients are refitted
# (monotone QP) only when the observation count reaches 2^j, and never after
# `freeze`, so the target changes on a geometric schedule with vanishing steps:
# the chain is a stochastic approximation whose limit is π with the final b̂.

mutable struct OnlineBias
    nobs::Int
    updates::Vector{Int}
    every::Int
    freeze::Int
end
OnlineBias(; every::Int=1, freeze::Int=typemax(Int)) = OnlineBias(0, Int[], every, freeze)

"Scores of a state for the bias regression, in level-0 coordinates: (X0, s_z, s_k, k0)."
function bias_scores(P::Problem, s::State)
    E, W = P.E, P.W
    ℓ = s.ℓ
    lk, lv = P.lk[ℓ+1], P.lv[ℓ+1]
    Xn = weather_nodes(W, ℓ, s.xs[end])
    _, (_, gko, _, gX) = energy_cost(E, lk, lv, s.k, s.v, Xn; grad=true)
    gz = weather_grad_to_x(W, ℓ, [-P.βv .* g for g in gX])
    for l in ℓ:-1:1
        gz, _ = split_state(W.trs[l], W.lays[l], W.lays[l+1], W.specs, gz)
        gz .*= W.trs[l].r
    end
    sk = restrict_to(-P.βv .* gko, P.lk, P.trk, E.kspecs, ℓ, 0)
    return weather_nodes(W, 0, s.z0), gz, sk, k_at0(P, ℓ, s.k)
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
    X0, sz, sk, k0 = bias_scores(P, s)
    bias_observe!(P.bias, P.E, P.W, P.lk[1], k0, X0, Float64(P.levels[1][2]), sz, sk)
    n = OB.nobs ÷ OB.every
    (n >= 4 && ispow2(n)) || return s, false
    bias_update!(P.bias; monotone=monotone)
    push!(OB.updates, OB.nobs)
    return evaluate!(P, s), true
end
