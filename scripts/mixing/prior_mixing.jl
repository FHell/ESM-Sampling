# Prior-only mixing of the level-0 weather (docs/mixing_diagnosis.md): the coordinate moves
# of the fine sampler (one z0 coordinate, exact Gaussian conditional) against block shifts of
# one row over B consecutive level-0 steps, on the regime-switching VAR prior alone (no cost:
# the case of an exact bias with perfect compensation), with FFBS of the regimes given z0
# after every sweep. Measures the integrated autocorrelation time of the annual and seasonal
# means of the field-mean rows, in sweeps and per coordinate update.

include(joinpath(@__DIR__, "wblock.jl"))

"Exact Gaussian draw of the shift t of z0 along direction φ on row o over steps a..b."
function prior_shift!(rng, W, m, Z, o, a, b, φ)
    U0 = z0_nlogp_range(W, m, Z, a, b)
    Z[o, a:b] .+= φ; U1 = z0_nlogp_range(W, m, Z, a, b)
    Z[o, a:b] .-= 2 .* φ; Um = z0_nlogp_range(W, m, Z, a, b)
    Z[o, a:b] .+= φ
    κ, β = U1 + Um - 2U0, 0.5 * (U1 - Um)
    t = -β / κ + randn(rng) / sqrt(κ)
    Z[o, a:b] .+= t .* φ
    return t
end

"Integrated autocorrelation time (Geyer initial monotone sequence) of a single series."
function iact(x::AbstractVector)
    n = length(x)
    y = x .- mean(x)
    v = sum(abs2, y) / n
    v == 0 && return NaN
    ρ(t) = sum(y[i] * y[i+t] for i in 1:n-t) / (n * v)
    τ = 1.0
    prev = Inf
    for t in 1:2:n-2
        Γ = ρ(t) + ρ(t + 1)
        Γ <= 0 && break
        Γ = min(Γ, prev); prev = Γ
        τ += 2Γ
    end
    return τ
end

"""
    prior_run(rng, W, nsweeps; mode, B, frac, rows) -> (traces, nupdates)

mode = :single: frac·n0·T0 single-coordinate draws per sweep (the fine sampler's z0 moves);
mode = :block: the same number of coordinate-equivalents spent on block shifts of length B
(each block shift counts B coordinate updates), on the rows `rows`, uniform positions.
"""
function prior_run(rng, W, nsweeps::Int; mode::Symbol=:single, B::Int=16, frac::Float64=0.05, rows=1:ML.n0(W),
                   ffbs::Bool=true, m0=nothing, Z0=nothing)
    lay = W.lays[1]
    n, T0 = ML.n0(W), lay.T
    m, z0, _ = ML.sample_weather(rng, W, 0)
    m = m0 === nothing ? m : copy(m0)
    Z = Z0 === nothing ? ML.z0matrix(z0, lay, W.specs) : copy(Z0)
    fieldrows = [z0row(W, f, 1) for f in eachindex(W.specs)]
    tr = zeros(length(fieldrows) * 5 + W.K, nsweeps)
    nupd = round(Int, frac * n * T0)
    edges = round.(Int, range(0, T0, length=5))
    for it in 1:nsweeps
        if mode == :single
            for _ in 1:nupd
                o = rand(rng, rows); τ = rand(rng, 1:T0)
                prior_shift!(rng, W, m, Z, o, τ, τ, [1.0])
            end
        else
            for _ in 1:max(1, nupd ÷ B)
                o = rand(rng, rows); a = rand(rng, 1:T0); b = min(a + B - 1, T0)
                prior_shift!(rng, W, m, Z, o, a, b, ones(b - a + 1))
            end
        end
        if ffbs
            z0v = zeros(lay.n)
            for f in eachindex(W.specs)
                ML.fieldview(z0v, lay, W.specs, f) .= Z[z0row(W, f, 1):z0row(W, f, lay.m[f]), :]
            end
            m = ML.ffbs(rng, ML.regime_loglik(W, z0v), W.Ptr, W.π0)
        end
        q = 0
        for o in fieldrows
            tr[q+=1, it] = mean(Z[o, :])
            for s in 1:4
                tr[q+=1, it] = mean(Z[o, edges[s]+1:edges[s+1]])
            end
        end
        for k in 1:W.K
            tr[q+=1, it] = count(==(k), m) / T0
        end
    end
    return tr
end
