# Single-chain effective sample size and sampled shadow prices (multi-chain
# diagnostics: convergence.jl).

"""
Effective sample size of a scalar chain (Geyer's initial monotone sequence on
the autocorrelations, FFT-free O(n·lag)).
"""
function ess(x::AbstractVector{<:Real})
    n = length(x)
    n < 4 && return Float64(n)
    μ = mean(x)
    v = sum(abs2, x .- μ) / n
    v == 0 && return Float64(n)
    ρ(k) = sum((x[i] - μ) * (x[i+k] - μ) for i in 1:n-k) / (n * v)
    s = 0.0
    prev = Inf
    k = 0
    while 2k + 1 < n
        Γ = ρ(2k) + ρ(2k + 1)
        Γ <= 0 && break
        Γ = min(Γ, prev)          # monotone
        s += Γ
        prev = Γ
        k += 1
    end
    τ = -1 + 2s
    return n / max(τ, 1.0 / n)
end

"""
Sampled nodal prices. In the regularized model the free energy
F = −β_v⁻¹ log Z satisfies ∂F/∂d_{n,τ} = E_π[∂C/∂d_{n,τ}] (score identity), and
∂C/∂d_{n,τ} = Δt φ'(r̄_{c,τ}) with r̄ the cell-mean residual of the node's
balance cell. `PriceAcc` accumulates φ'(r̄) (€/MWh) per node and step.
"""
mutable struct PriceAcc
    sum::Matrix{Float64}
    n::Int
end
PriceAcc(N::Int, T::Int) = PriceAcc(zeros(N, T), 0)

function accumulate_prices!(acc::PriceAcc, P::Problem, LM::LocalModel, LC::LocalCost)
    E = P.E
    for τ in 1:LM.T, n in 1:E.sys.N
        c = LM.cell[n]
        x = LC.S[c, τ]
        acc.sum[n, τ] += x > 0 ? E.c_shed : (x < 0 ? -E.c_dump : 0.0)
    end
    acc.n += 1
    return acc
end

prices(acc::PriceAcc) = acc.sum ./ max(acc.n, 1)

# ---------------------------------------------------------------------------
# aggregated sampled prices (per region and season)
# ---------------------------------------------------------------------------

"""
Accumulator of the sampled prices aggregated per region (any node labelling,
e.g. countries or the cells of a Haar partition) and meteorological season. Per
sample it records, for every (region, season):

* `mean`: the time- and node-averaged marginal price φ'(r̄) (€/MWh);
* `lw`: the load-weighted price Σ φ'·d / Σ d (what consumers pay on average);
* `pshed`, `pdump`: the fraction of node-steps in shortage and in surplus.

The traces are kept per sample, so ESS and standard errors follow from them.
"""
mutable struct RegionPriceAcc
    region::Vector{Int}
    names::Vector{String}
    mean::Vector{Matrix{Float64}}
    lw::Vector{Matrix{Float64}}
    pshed::Vector{Matrix{Float64}}
    pdump::Vector{Matrix{Float64}}
end

function RegionPriceAcc(labels::AbstractVector)
    names = unique(string.(labels))
    idx = Dict(n => i for (i, n) in enumerate(names))
    return RegionPriceAcc([idx[string(l)] for l in labels], names, Matrix{Float64}[], Matrix{Float64}[],
                          Matrix{Float64}[], Matrix{Float64}[])
end

function accumulate_region_prices!(acc::RegionPriceAcc, P::Problem, LM::LocalModel, LC::LocalCost, s::State)
    E = P.E
    R = length(acc.names)
    h = P.lv[LM.ℓ+1].h
    Xn = weather_nodes(P.W, LM.ℓ, s.xs[end])
    sp, n, lw, ld, ns, nd = zeros(R, 4), zeros(R, 4), zeros(R, 4), zeros(R, 4), zeros(R, 4), zeros(R, 4)
    for τ in 1:LM.T
        q = season(τ, h)
        for i in 1:E.sys.N
            x = LC.S[LM.cell[i], τ]
            p = x > 0 ? E.c_shed : (x < 0 ? -E.c_dump : 0.0)
            g = acc.region[i]
            d = max(Xn[1][i, τ], 0.0)
            sp[g, q] += p; n[g, q] += 1
            lw[g, q] += p * d; ld[g, q] += d
            x > 0 && (ns[g, q] += 1)
            x < 0 && (nd[g, q] += 1)
        end
    end
    push!(acc.mean, sp ./ max.(n, 1))
    push!(acc.lw, lw ./ max.(ld, 1e-9))
    push!(acc.pshed, ns ./ max.(n, 1))
    push!(acc.pdump, nd ./ max.(n, 1))
    return acc
end

"""
Summary of a `RegionPriceAcc`: posterior means, standard errors (from the ESS)
and the per-sample traces, as a Dict of arrays (region × season [× sample]).
"""
function region_price_summary(acc::RegionPriceAcc)
    out = Dict{String,Any}()
    isempty(acc.mean) && return out
    for (name, tr) in (("mean", acc.mean), ("load_weighted", acc.lw), ("p_shed", acc.pshed), ("p_dump", acc.pdump))
        A = cat(tr...; dims=3)
        R, S, n = size(A)
        μ = dropdims(sum(A; dims=3); dims=3) ./ n
        se = zeros(R, S)
        for r in 1:R, q in 1:S
            x = A[r, q, :]
            sd = n > 1 ? std(x) : 0.0
            se[r, q] = sd == 0 ? 0.0 : sd / sqrt(max(ess(x), 1.0))
        end
        out["$name/mean"] = μ
        out["$name/se"] = se
        out["$name/trace"] = A
    end
    out["regions"] = acc.names
    out["seasons"] = ["DJF", "MAM", "JJA", "SON"]
    return out
end
