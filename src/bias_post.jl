# ---------------------------------------------------------------------------
# post-correction of the bias (docs/system/sampler.md §7.1)
# ---------------------------------------------------------------------------
#
# The chain samples π_b ∝ p(x) exp(−β_k capex − β_v C − b̂(x, k)) with the bias b̂ in
# use (frozen for production). After the run, b̃ is refitted from all scores (the
# warmup's accumulated sums plus the sufficient statistics of the production samples'
# scores) and every sample is reweighted to the target with b̃:
# w_i ∝ exp(b̂(x_i, k_i) − b̃(x_i, k_i)) (self-normalized; the normalizing constants
# cancel). The weight ESS says how far b̂ was from b̃; weights cannot repair regions
# the chain did not visit.
#
# Per sample only what b̃ needs is recorded, at the level ℓ_b of the bias: the features
# u₁ per step (load minus non-gas availability of the sample, over D̄), the clamped node
# capacities (firm capacity, availability of the sample steps of Ē) and the clamped
# capacity coefficients (ĥ); not the raw weather.

mutable struct PostBias
    U1::Vector{Vector{Float64}}             # per sample: u₁ per step of level ℓ_b
    kc::Vector{Vector{Vector{Float64}}}     # clamped node capacities per k field
    zc::Vector{Vector{Float64}}             # clamped capacity coefficients at level ℓ_b
    bused::Vector{Float64}
    Q::Vector{Vector{Float64}}
    D::Vector{Vector{Float64}}
    # sufficient statistics of the production scores (the normal equations of bias_observe!)
    GtG::Matrix{Float64}
    Gts::Vector{Float64}
    hk::Matrix{Float64}
    nobs::Int
    scores::Bool
end
PostBias(; scores::Bool=true) = PostBias(Vector{Float64}[], Vector{Vector{Float64}}[], Vector{Float64}[], Float64[],
                                         Vector{Float64}[], Vector{Float64}[], zeros(0, 0), Float64[], zeros(0, 0), 0, scores)

"b̂ of a recorded sample (features, clamped node capacities and coefficients at level ℓ_b, layout `lk`) under the bias `b`."
function bias_from_record(b::BiasModel, E::EnergyModel, lk::Layout, U1, kc, zc)
    firm = firm_capacity(E, kc)
    return ghat_features(b, U1, firm) - b.ratio * ghat_mean_kc(b, E, kc, firm) +
           (1 - b.ratio) * (hhat(b, zc)[1] + logJ_kc(E, measure_fields(E, lk), kc))
end

"""
Record one sample: its bias features at level ℓ_b, the bias in use, quantities and weather
diagnostics, and (with `PB.scores`) the sufficient statistics of its scores.
"""
function post_record!(PB::PostBias, P::Problem, s::State, q)
    b, E = P.bias, P.E
    lkb = P.lk[b.ℓb+1]
    kb, xb = bias_restrict(P, s.ℓ, s.k, s.xs[end], s.z0)
    Xb = weather_nodes(P.W, b.ℓb, xb)
    U1, _ = bias_features(b, E, lkb, kb, Xb)
    push!(PB.U1, U1)
    push!(PB.kc, node_capacities(E, lkb, kb)[1])
    push!(PB.zc, clamped_kcoeffs(E, lkb, kb)[1])
    push!(PB.bused, bias_term(P, s))
    push!(PB.Q, copy(q))
    push!(PB.D, weather_diagnostics(P, s.m, s.z0))
    if PB.scores
        if PB.nobs == 0
            nb = nθ(b)
            PB.GtG, PB.Gts, PB.hk = zeros(nb, nb), zeros(nb), hk_sums(lkb.n)
        end
        _, sx, sk, _ = bias_scores(P, s)
        bias_observe!(b, E, P.W, lkb, kb, Xb, sx, sk; into=PB)
    end
    return PB
end

"""
    post_correct(P, PB; monotone) -> (; w, ess, Qw, Dw, Q, D, btilde)

Refit b̃ from the accumulated scores of the warmup and the production samples (in place:
P.bias becomes b̃), weights, weight ESS and the weighted means of the recorded quantities
and weather diagnostics.
"""
function post_correct(P::Problem, PB::PostBias; monotone=nothing)
    n = length(PB.bused)
    n == 0 && return nothing
    b = P.bias
    if PB.nobs > 0
        b.GtG .+= PB.GtG; b.Gts .+= PB.Gts; b.nobs += PB.nobs
        @views b.hk[:, 1:5] .+= PB.hk[:, 1:5]
        @views b.hk[:, 6] .= min.(b.hk[:, 6], PB.hk[:, 6]); @views b.hk[:, 7] .= max.(b.hk[:, 7], PB.hk[:, 7])
    end
    bias_update!(b; monotone=monotone)
    lkb = P.lk[b.ℓb+1]
    bt = [bias_from_record(b, P.E, lkb, PB.U1[i], PB.kc[i], PB.zc[i]) for i in 1:n]
    lw = PB.bused .- bt
    w = exp.(lw .- maximum(lw))
    w ./= sum(w)
    ess = 1 / sum(abs2, w)
    Qw = sum(w[i] .* PB.Q[i] for i in 1:n)
    Dw = sum(w[i] .* PB.D[i] for i in 1:n)
    return (; w, ess, logw=lw, Qw, Dw, Q=mean(PB.Q), D=mean(PB.D), btilde=bt)
end
