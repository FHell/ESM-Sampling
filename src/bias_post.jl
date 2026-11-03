# ---------------------------------------------------------------------------
# post-correction of the bias (docs/todo.md item 4, docs/setup_overview.md §10)
# ---------------------------------------------------------------------------
#
# The chain samples π_b ∝ p(x) exp(−β_k capex − β_v C − b̂(x, k)) with the bias b̂ in
# use at each sample (online refits at 2^j change it during production). After the
# run, b̃ is refitted from all accumulated scores and every sample is reweighted to
# the target with b̃: w_i ∝ exp(b̂_i(x_i, k_i) − b̃(x_i, k_i)) (self-normalized; the
# normalizing constants cancel). The weight ESS says how far b̂ was from b̃; weights
# cannot repair regions the chain did not visit.

mutable struct PostBias
    z0s::Vector{Vector{Float64}}
    k0s::Vector{Vector{Float64}}
    bused::Vector{Float64}
    Q::Vector{Vector{Float64}}
    D::Vector{Vector{Float64}}
end
PostBias() = PostBias(Vector{Float64}[], Vector{Float64}[], Float64[], Vector{Float64}[], Vector{Float64}[])

"Record one sample: level-0 weather and capacities, the bias in use, quantities and weather diagnostics."
function post_record!(PB::PostBias, P::Problem, s::State, q)
    k0 = k_at0(P, s.ℓ, s.k)
    push!(PB.z0s, copy(s.z0)); push!(PB.k0s, copy(k0))
    push!(PB.bused, bias_term(P, k0, s.z0))
    push!(PB.Q, copy(q))
    push!(PB.D, weather_diagnostics(P, s.m, s.z0))
    return PB
end

"""
    post_correct(P, PB; monotone) -> (; w, ess, Qw, Dw, Q, D, btilde)

Refit b̃ from the accumulated scores (in place: P.bias becomes b̃), weights, weight ESS
and the weighted means of the recorded quantities and weather diagnostics.
"""
function post_correct(P::Problem, PB::PostBias; monotone=nothing)
    n = length(PB.bused)
    n == 0 && return nothing
    bias_update!(P.bias; monotone=monotone)
    bt = [bias_term(P, PB.k0s[i], PB.z0s[i]) for i in 1:n]
    lw = PB.bused .- bt
    w = exp.(lw .- maximum(lw))
    w ./= sum(w)
    ess = 1 / sum(abs2, w)
    Qw = sum(w[i] .* PB.Q[i] for i in 1:n)
    Dw = sum(w[i] .* PB.D[i] for i in 1:n)
    return (; w, ess, logw=lw, Qw, Dw, Q=mean(PB.Q), D=mean(PB.D), btilde=bt)
end
