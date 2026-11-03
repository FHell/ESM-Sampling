# Convergence diagnostics over several chains (docs/validation_plan.md Part A):
# rank-normalized split-R̂, bulk and tail ESS and the MCSE of the mean, following
# Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021), "Rank-normalization, folding,
# and localization: an improved R̂ for assessing convergence of MCMC".

"Split every chain into halves (the second half of an odd chain drops its first draw)."
function split_chains(chains::Vector{<:AbstractVector})
    out = Vector{Vector{Float64}}()
    for c in chains
        n = length(c)
        h = n ÷ 2
        push!(out, Float64.(c[1:h]))
        push!(out, Float64.(c[n-h+1:n]))
    end
    return out
end

"Rank-normalize the pooled draws: z = Φ⁻¹((rank − 3/8)/(S + 1/4)), average ranks for ties."
function rank_normalize(chains::Vector{<:AbstractVector})
    x = reduce(vcat, chains)
    S = length(x)
    p = sortperm(x)
    r = zeros(S)
    i = 1
    while i <= S
        j = i
        while j < S && x[p[j+1]] == x[p[i]]
            j += 1
        end
        rk = (i + j) / 2
        for q in i:j
            r[p[q]] = rk
        end
        i = j + 1
    end
    z = [-sqrt(2) * erfcinv(2 * (ri - 0.375) / (S + 0.25)) for ri in r]
    out = Vector{Vector{Float64}}()
    o = 0
    for c in chains
        push!(out, z[o+1:o+length(c)])
        o += length(c)
    end
    return out
end

"Plain R̂ of equal-length chains (between/within variance)."
function rhat_basic(chains::Vector{<:AbstractVector})
    M = length(chains)
    n = minimum(length, chains)
    (M < 2 || n < 2) && return NaN
    ms = [mean(c[1:n]) for c in chains]
    vs = [var(c[1:n]) for c in chains]
    W = mean(vs)
    B = n * var(ms)
    W == 0 && return B == 0 ? 1.0 : Inf
    vplus = (n - 1) / n * W + B / n
    return sqrt(vplus / W)
end

"Rank-normalized split-R̂: the maximum of the bulk and the folded (tail) version."
function rhat_rank(chains::Vector{<:AbstractVector})
    sc = split_chains(chains)
    bulk = rhat_basic(rank_normalize(sc))
    med = median(reduce(vcat, sc))
    tail = rhat_basic(rank_normalize([abs.(c .- med) for c in sc]))
    return max(bulk, tail)
end

"Autocovariances of x at lags 0..n−1 (biased estimator, by direct sums; n is small here)."
function autocov(x::AbstractVector)
    n = length(x)
    μ = mean(x)
    y = x .- μ
    return [sum(y[i] * y[i+t] for i in 1:n-t; init=0.0) / n for t in 0:n-1]
end

"""
Multi-chain effective sample size (Stan's estimator): the autocorrelations combined
across chains through the variance estimate var⁺, truncated by Geyer's initial
monotone sequence. Chains are cut to the shortest length.
"""
function ess_multi(chains::Vector{<:AbstractVector})
    M = length(chains)
    n = minimum(length, chains)
    n < 4 && return Float64(M * n)
    cs = [Float64.(c[1:n]) for c in chains]
    acs = [autocov(c) for c in cs]
    W = mean(var(c) for c in cs)
    W == 0 && return Float64(M * n)
    ms = [mean(c) for c in cs]
    vplus = (n - 1) / n * W + (M > 1 ? var(ms) : 0.0)
    ρ = [1 - (W - mean(ac[t+1] for ac in acs)) / vplus for t in 0:n-1]      # acov with denominator n, as Stan
    ρ[1] = 1.0
    # Geyer: sums of consecutive pairs P_k = ρ_{2k} + ρ_{2k+1}, positive and monotone
    τ = -1.0
    prev = Inf
    k = 0
    while 2k + 1 < n
        Pk = ρ[2k+1] + ρ[2k+2]
        Pk <= 0 && break
        Pk = min(Pk, prev)
        τ += 2 * Pk
        prev = Pk
        k += 1
    end
    τ = max(τ, 1 / log10(M * n))
    return M * n / τ
end

"Bulk ESS (rank-normalized split chains)."
ess_bulk(chains) = ess_multi(rank_normalize(split_chains(chains)))

"Tail ESS: the smaller ESS of the indicators of the 5 % and 95 % quantiles (split chains)."
function ess_tail(chains)
    sc = split_chains(chains)
    x = reduce(vcat, sc)
    vals = Float64[]
    for q in (0.05, 0.95)
        qv = quantile(x, q)
        push!(vals, ess_multi([Float64.(c .<= qv) for c in sc]))
    end
    return minimum(vals)
end

"MCSE of the mean, relative to the posterior standard deviation: 1/√ESS_mean."
function mcse_rel(chains)
    e = ess_multi(split_chains(chains))
    return 1 / sqrt(max(e, 1.0))
end

"""
    convergence_summary(chains; rhat_max, ess_min, mcse_max) -> NamedTuple

Diagnostics of one quantity over several chains, and whether it passes.
"""
function convergence_summary(chains::Vector{<:AbstractVector}; rhat_max=1.01, ess_min=400.0, mcse_max=0.05)
    x = reduce(vcat, chains)
    if all(==(x[1]), x)                          # a constant quantity carries no information
        return (; rhat=1.0, ess_bulk=Float64(length(x)), ess_tail=Float64(length(x)), mcse_rel=0.0,
                mean=x[1], sd=0.0, pass=true, constant=true)
    end
    r = rhat_rank(chains)
    eb, et = ess_bulk(chains), ess_tail(chains)
    mr = mcse_rel(chains)
    pass = isfinite(r) && r < rhat_max && eb >= ess_min && et >= ess_min && mr <= mcse_max
    return (; rhat=r, ess_bulk=eb, ess_tail=et, mcse_rel=mr, mean=mean(x), sd=std(x), pass, constant=false)
end

"""
    warmup_end(traces; threshold, every, minlen) -> Int

First sweep n (checked every `every` sweeps) from which the rank-normalized split-R̂ of
every quantity over the second half of the sweeps so far stays below `threshold` up to
the end. `traces[q][k]` is the trace of quantity q in chain k. Returns 0 if never.
"""
function warmup_end(traces; threshold=1.05, every=10, minlen=20)
    n = minimum(length(c) for tr in traces for c in tr)
    ok = Int[]
    for m in minlen:every:n
        r = maximum((rhat_rank([c[m÷2+1:m] for c in tr]) for tr in traces if !all(==(tr[1][1]), reduce(vcat, [c[1:m] for c in tr])));
                    init=1.0)
        push!(ok, r < threshold ? m : 0)
    end
    # the first checkpoint after which every later checkpoint passes
    last_fail = findlast(==(0), ok)
    last_fail === nothing && return isempty(ok) ? 0 : ok[1] ÷ 2
    last_fail == length(ok) && return 0
    return ok[last_fail+1] ÷ 2
end
