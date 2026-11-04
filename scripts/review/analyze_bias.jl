# Analysis functions for the α = 1 bias review (used from a session that has `P` set up by
# scripts/mixing/common.jl's `mix_setup` or equivalently, and the outputs of
# scripts/review/reproduce_a1_bias.jl). Everything here evaluates states and scores; the only
# sampling is `score_series` (operation sweeps at fixed weather and capacities).
import ESMSampling as ML
using Random, LinearAlgebra, SparseArrays, Statistics, Printf, Serialization

"Set the fields of a bias snapshot (NamedTuple from reproduce_a1_bias.jl) into P.bias."
function load_bias!(P, snap)
    for f in keys(snap)
        f == :tabθ && continue
        setfield!(P.bias, f, deepcopy(getfield(snap, f)))
    end
    ML.rebuild_tables!(P.bias)
    return P.bias
end

"Node-value matrix U of a k field at the complete level (node values = U * coefficients)."
Unode(P, f) = ML.cols(P.E.kspecs[f].B, P.lk[end].m[f])

"Coefficient gradient → node gradient per k field (U orthonormal on the complete level)."
function node_grad(P, g)
    lk = P.lk[end]
    [Unode(P, f) * vec(ML.fieldview(g, lk, P.E.kspecs, f)) for f in eachindex(P.E.kspecs)]
end

"Capacity scores of a state in node coordinates: s_k = −β_v ∂_k C + ∂_k log J, the capex force β_k ∂_k K, and the clamp mask."
function node_scores(P, s)
    E = P.E
    lk, lv = P.lk[s.ℓ+1], P.lv[s.ℓ+1]
    Xn = ML.weather_nodes(P.W, s.ℓ, s.xs[end])
    parts, (gkc, gko, gv, gX) = ML.energy_cost(E, lk, lv, s.k, s.v, Xn; grad=true)
    sk = -P.βv .* gko .+ ML.log_measure_grad(E, lk, lv, s.k)
    fk = P.βk .* gkc
    kc, mask = ML.node_capacities(E, lk, s.k)
    return (; sk=node_grad(P, sk), fk=node_grad(P, fk), skc=sk, fkc=fk, kc, mask, parts)
end

"""
    score_series(rng, P, S, s, n) -> (sk, fk, sx, states)

`n` operation sweeps (local sweep, storage multiscale with probability 1/2, as in the fine
step) at fixed weather and capacities, recording after each the node capacity scores, the
capex force and the weather scores in the state coordinates of the bias level.
"""
function score_series(rng, P, S, s, n; storage=true)
    Kk = S.loc
    LMk = ML.local_model(Kk, P, s.ℓ)
    s = copy(s)
    sks, fks, sxs = Vector{Vector{Vector{Float64}}}(), Vector{Vector{Vector{Float64}}}(), Vector{Vector{Float64}}()
    for i in 1:n
        s, _ = ML.local_sweep!(rng, P, LMk, s; wsets=Kk.wsets, extra=Kk.extra)
        if storage && rand(rng) < 0.5 && LMk.complete[P.E.fθ]
            isempty(Kk.pools) && append!(Kk.pools, [ML.BlockPool(P, 140) for _ in 1:Threads.maxthreadid()])
            s, _ = ML.storage_ms_sweep!(rng, P, LMk, s; pools=Kk.pools)
        end
        ns = node_scores(P, s)
        _, sx, _, _ = ML.bias_scores(P, s)
        push!(sks, ns.sk); push!(fks, ns.fk); push!(sxs, sx)
    end
    return (; sk=sks, fk=fks, sx=sxs, state=s)
end

"Lag-1 autocorrelation of a series."
acf1(x) = (n = length(x); m = mean(x); v = var(x); v > 0 ? sum((x[1:n-1] .- m) .* (x[2:n] .- m)) / ((n - 1) * v) : 0.0)

"""
Summary of a score series for one k field: per unit the mean, the sd and the lag-1 autocorrelation
of the score (nats/MW), the capex force, and the number of independent samples needed to pin the
mean score to 10 % of the capex force.
"""
function score_summary(ser, f; mask=nothing)
    M = reduce(hcat, [x[f] for x in ser.sk])          # units × sweeps
    F = reduce(hcat, [x[f] for x in ser.fk])
    μ = vec(mean(M; dims=2)); σ = vec(std(M; dims=2)); ρ = [acf1(M[i, :]) for i in axes(M, 1)]
    fk = vec(mean(F; dims=2))
    need = (σ ./ (0.1 .* abs.(fk))).^2 .* (1 .+ ρ) ./ (1 .- ρ)
    return (; μ, σ, ρ, fk, need)
end

"""
Gradient of b̂ in node coordinates split into its parts: ∂ĝ/∂k, −r ∂Ēĝ/∂k, (1−r) ∂ĥ/∂k, (1−r) ∂log J/∂k.
"""
function bias_grad_parts(P, s)
    b, E = P.bias, P.E
    lk = P.lk[b.ℓb+1]
    kb, xb = ML.bias_restrict(P, s.ℓ, s.k, s.xs[end], s.z0)
    Xb = ML.weather_nodes(P.W, b.ℓb, xb)
    g, gX, gk = ML.ghat(b, E, lk, kb, Xb; grad=true)
    gm, gkm = ML.ghat_mean(b, E, lk, kb; grad=true)
    zc, vjp = ML.clamped_kcoeffs(E, lk, kb)
    h, gkh = ML.hhat(b, zc)
    gkh = vjp(gkh)
    gJ = ML.log_measure_grad(E, lk, ML.Layout(E.vspecs, lk.M, lk.h), kb)
    r = b.ratio
    return (; g=node_grad(P, gk), gm=node_grad(P, -r .* gkm), h=node_grad(P, (1 - r) .* gkh), J=node_grad(P, (1 - r) .* gJ),
            total=node_grad(P, gk .- r .* gkm .+ (1 - r) .* (gkh .+ gJ)), vals=(; g, gm, h, lj=ML.logJ_kc(E, ML.measure_fields(E, lk), ML.node_capacities(E, lk, kb)[1])))
end

"The implied price curve λ(u) = −(F₁' + F₂'(u − fd))/(β_v D̄) of the bias at firm fraction fd, on a grid."
function implied_price(P, fd; us=-2.5:0.05:1.0)
    b = P.bias
    n1 = length(b.τ1)
    [(-(ramp_deriv(b.θ[1], b.θ[2:1+n1], b.τ1, b.s, u) + ramp_deriv(b.θ[2+n1], b.θ[3+n1:end], b.τ2, b.s, u - fd))) / (P.βv * b.Dbar) for u in us]
end

"Derivative of the analytic ramp sum."
function ramp_deriv(θlin::Float64, θr::AbstractVector, τ::AbstractVector, s::Float64, u::Float64)
    v = θlin
    for j in eachindex(τ)
        v += θr[j] * ML.sigmoid((u - τ[j]) / s)
    end
    return v
end

"""
Refit ĥ from the observation records with the k-gradient of ĝ taken at a given θ (consistent) or at
the θ in force at observation time (as the code does). Returns per coordinate (a, c) for both.
"""
function hhat_refit(P, obs, θ_final, θ_at_obs)
    b, E, W = P.bias, P.E, P.W
    lkb = P.lk[b.ℓb+1]
    n = lkb.n
    hk_c, hk_s = ML.hk_sums(n), ML.hk_sums(n)
    θ0 = copy(b.θ)
    for (i, o) in enumerate(obs)
        κ = ML.clamped_kcoeffs(E, lkb, o.kb)[1] ./ b.kscale
        gJ = ML.log_measure_grad(E, lkb, ML.Layout(E.vspecs, lkb.M, lkb.h), o.kb)
        b.θ .= θ_final; ML.rebuild_tables!(b)
        _, gk_c = ML.bias_design(b, E, W, lkb, o.kb, o.Xb)
        b.θ .= θ_at_obs[i]; ML.rebuild_tables!(b)
        _, gk_s = ML.bias_design(b, E, W, lkb, o.kb, o.Xb)
        for (hk, gk) in ((hk_c, gk_c), (hk_s, gk_s))
            y = (o.sk .- gJ .- gk) .* b.kscale
            @. hk[:, 1] += 1; @. hk[:, 2] += κ; @. hk[:, 3] += κ^2; @. hk[:, 4] += y; @. hk[:, 5] += y * κ
            @views hk[:, 6] .= min.(hk[:, 6], κ); @views hk[:, 7] .= max.(hk[:, 7], κ)
        end
    end
    b.θ .= θ0; ML.rebuild_tables!(b)
    fit(hk) = begin
        a, c = zeros(n), zeros(n)
        for i in 1:n
            nn, sκ, sκ2, sy, syκ = hk[i, 1:5]
            varκ = sκ2 / nn - (sκ / nn)^2
            if varκ > 1e-6 * (1 + sκ2 / nn)
                a[i], c[i] = ([nn sκ; sκ sκ2] + 1e-9 * nn * I) \ [sy, syκ]
            else
                a[i], c[i] = sy / nn, 0.0
            end
        end
        (; a, c)
    end
    return (; consistent=fit(hk_c), stale=fit(hk_s), hk_c, hk_s)
end

"θ in force at each observation, from the lifted θ and the refit history (refit after observation `nobs`)."
function theta_at_obs(θ_lift, hist, nobs_total)
    out = Vector{Vector{Float64}}(undef, nobs_total)
    θ = copy(θ_lift)
    j = 1
    for i in 1:nobs_total
        out[i] = copy(θ)
        while j <= length(hist) && hist[j].nobs == i
            θ = copy(hist[j].bias.θ); j += 1
        end
    end
    return out
end

"""
Residual of the weather-score regression on the field-mean rows of z0, summed over the year:
the first-order force on the annual mean of each field, in nats per unit shift of all its
field-mean coordinates, for the score vector `sx` (e.g. a sweep average) and the design of state `s`.
"""
function annual_mean_residual(P, s, sx)
    b, W = P.bias, P.W
    lkb = P.lk[b.ℓb+1]
    kb, xb = ML.bias_restrict(P, s.ℓ, s.k, s.xs[end], s.z0)
    Xb = ML.weather_nodes(W, b.ℓb, xb)
    Jz, _ = ML.bias_design(b, P.E, W, lkb, kb, Xb)
    res = sx .- Jz * b.θ
    lay = W.lays[1]
    out = Float64[]; fit = Float64[]; sc = Float64[]
    for f in eachindex(W.specs)
        idx = [lay.off[f] + (τ - 1) * lay.m[f] + 1 for τ in 1:lay.T]      # mode 1 (field mean) at every level-0 step
        push!(out, sum(res[idx])); push!(fit, sum((Jz * b.θ)[idx])); push!(sc, sum(sx[idx]))
    end
    return (; residual=out, fitted=fit, score=sc)
end
