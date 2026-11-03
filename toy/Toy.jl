# Toy model of docs/toy_mlda.md:  min_x w·x  s.t.  Lx ≤ c  on a grid graph.
#
# Everything is written in slack coordinates s = c − Lx with Σs = S = Σc; a move
# along a direction d (Σd = 0) changes the potentials x implicitly. The Gibbs
# measure is ∝ exp(−β r·s) on {s ≥ 0} (hard) or exp(−β[r·s + μ Σ(s)_−]) (regularized),
# r = a_max − a, a = L⁺w. Every move samples its 1-D line conditional exactly.
module Toy

using LinearAlgebra, SparseArrays, Random, Statistics

export ToyProblem, grid_problem, exact_sample, line_move!, potential, quadtree_cells, block_cells,
       merit_cells, local_dir, cell_dir, ess

struct ToyProblem
    N::Int
    n::Int                          # grid side
    nbrs::Vector{Vector{Int}}
    a::Vector{Float64}
    r::Vector{Float64}
    istar::Int
    β::Float64
    μ::Float64                      # Inf = hard constraints
end

hard(P::ToyProblem) = isinf(P.μ)

"Grid graph n×n; a = west–east gradient + noise; β and μ relative to the mean reduced cost."
function grid_problem(n; βrel=200.0, μrel=Inf, noise=0.3, seed=1)
    rng = MersenneTwister(seed)
    N = n * n
    id(i, j) = (j - 1) * n + i
    nbrs = [Int[] for _ in 1:N]
    for i in 1:n, j in 1:n
        i < n && (push!(nbrs[id(i, j)], id(i + 1, j)); push!(nbrs[id(i + 1, j)], id(i, j)))
        j < n && (push!(nbrs[id(i, j)], id(i, j + 1)); push!(nbrs[id(i, j + 1)], id(i, j)))
    end
    a = [(i - 1) / (n - 1) + noise * randn(rng) for j in 1:n for i in 1:n]
    istar = argmax(a)
    r = maximum(a) .- a
    rbar = mean(r[r.>0])
    μ = isinf(μrel) ? Inf : μrel * maximum(r)
    return ToyProblem(N, n, nbrs, a, r, istar, βrel / rbar, μ)
end

"Potential U(s) − const (Inf if infeasible under hard constraints)."
function potential(P::ToyProblem, s)
    U = P.β * dot(P.r, s)
    if hard(P)
        any(<(0), s) && return Inf
    else
        U += P.β * P.μ * sum(x -> max(-x, 0.0), s)
    end
    return U
end

"Exact equilibrium draw for total slack S (rejection on the slack node)."
function exact_sample(rng, P::ToyProblem, S)
    while true
        s = zeros(P.N)
        for j in 1:P.N
            j == P.istar && continue
            λp = P.β * P.r[j]
            if hard(P)
                s[j] = randexp(rng) / λp
            else
                λm = P.β * (P.μ - P.r[j])
                s[j] = rand(rng) < (1 / λp) / (1 / λp + 1 / λm) ? randexp(rng) / λp : -randexp(rng) / λm
            end
        end
        s[P.istar] = S - sum(s)
        if hard(P)
            s[P.istar] >= 0 && return s
        else
            rand(rng) < exp(-P.β * P.μ * max(-s[P.istar], 0.0)) && return s
        end
    end
end

# ---------------------------------------------------------------------------
# exact line sampling of a piecewise-linear convex potential
# ---------------------------------------------------------------------------

"Sample t on [lo, hi] from density ∝ exp(−g t)."
function trexp(rng, lo, hi, g)
    if isinf(hi)
        @assert g > 0
        return lo + randexp(rng) / g
    elseif isinf(lo)
        @assert g < 0
        return hi - randexp(rng) / (-g)
    end
    Δ = hi - lo
    abs(g * Δ) < 1e-12 && return lo + rand(rng) * Δ
    u = rand(rng)
    return g > 0 ? lo - log1p(-u * (-expm1(-g * Δ))) / g :
                   hi + log1p(-u * (-expm1(g * Δ))) / (-g)
end

logsumexp(v) = (m = maximum(v); isinf(m) ? m : m + log(sum(exp.(v .- m))))

"""
    hinge_sample(rng, g, v, c, ω) -> t

Exact draw from the density ∝ exp(−U(t)) with
U(t) = g t + Σ_k ω_k (v_k + c_k t)_−; ω_k = Inf makes term k a hard
constraint v_k + c_k t ≥ 0. U must be coercive.
"""
function hinge_sample(rng, g, v, c, ω)
    lo, hi = -Inf, Inf
    bp = Float64[]
    for k in eachindex(v)
        c[k] == 0 && continue
        τ = -v[k] / c[k]
        if isinf(ω[k])
            c[k] > 0 ? (lo = max(lo, τ)) : (hi = min(hi, τ))
        else
            push!(bp, τ)
        end
    end
    lo >= hi && return 0.5 * (lo + hi)          # (numerically) empty interval
    # slope at t: g − Σ_{soft, active} ω c, active ⇔ v + c t < 0
    slope(t) = g - sum((!isinf(ω[k]) && c[k] != 0 && v[k] + c[k] * t < 0) ? ω[k] * c[k] : 0.0 for k in eachindex(v); init=0.0)
    edges = vcat(lo, sort!(filter(τ -> lo < τ < hi, bp)), hi)
    ns = length(edges) - 1
    gs = [slope(isinf(edges[q]) ? edges[q+1] - 1.0 : isinf(edges[q+1]) ? edges[q] + 1.0 : 0.5 * (edges[q] + edges[q+1])) for q in 1:ns]
    # potential at the finite edges, relative to the first finite edge
    Ue = fill(NaN, length(edges))
    q0 = isinf(edges[1]) ? 2 : 1
    Ue[q0] = 0.0
    for q in q0+1:length(edges)
        isinf(edges[q]) && break
        Ue[q] = Ue[q-1] + gs[q-1] * (edges[q] - edges[q-1])
    end
    logm = zeros(ns)
    for q in 1:ns
        a, b, gq = edges[q], edges[q+1], gs[q]
        if isinf(a)
            gq < 0 || error("potential not coercive to the left")
            logm[q] = -Ue[q+1] - log(-gq)
        elseif isinf(b)
            gq > 0 || error("potential not coercive to the right")
            logm[q] = -Ue[q] - log(gq)
        else
            Δ = b - a
            logm[q] = abs(gq * Δ) < 1e-12 ? -Ue[q] + log(Δ) :
                      gq > 0 ? -Ue[q] + log(-expm1(-gq * Δ)) - log(gq) :
                               -Ue[q+1] + log(-expm1(gq * Δ)) - log(-gq)
        end
    end
    w = exp.(logm .- logsumexp(logm))
    u, acc, q = rand(rng), 0.0, ns
    for i in eachindex(w)
        acc += w[i]
        u <= acc && (q = i; break)
    end
    return trexp(rng, edges[q], edges[q+1], gs[q])
end

"""
    line_move!(rng, P, s, idx, coef; zone) -> t

Move s[idx] .+= t .* coef with t drawn exactly from its conditional under P
(Σ coef = 0, so Σ s is preserved). `zone = (inW, b)` adds the interface
constraint Σ_{i∈W} s_i ≥ b (hard or priced like the node constraints).
"""
function line_move!(rng, P::ToyProblem, s, idx, coef; zone=nothing)
    g = -P.β * sum(P.a[idx[k]] * coef[k] for k in eachindex(idx))   # U = −β a·s + penalties
    ωn = hard(P) ? Inf : P.β * P.μ
    v = [s[i] for i in idx]
    c = collect(Float64, coef)
    ω = fill(ωn, length(idx))
    if zone !== nothing
        inW, bnd, σW = zone
        cz = sum(coef[k] for k in eachindex(idx) if inW[idx[k]]; init=0.0)
        if cz != 0
            push!(v, σW[] - bnd); push!(c, cz); push!(ω, ωn)
        end
    end
    t = hinge_sample(rng, g, v, c, ω)
    @inbounds for k in eachindex(idx)
        s[idx[k]] += t * coef[k]
    end
    if zone !== nothing
        inW, bnd, σW = zone
        σW[] += t * sum(coef[k] for k in eachindex(idx) if inW[idx[k]]; init=0.0)
    end
    return t
end

# ---------------------------------------------------------------------------
# directions and cells
# ---------------------------------------------------------------------------

"Δs = −L e_i: moving the potential of node i."
local_dir(P::ToyProblem, i) = (vcat(i, P.nbrs[i]), vcat(-Float64(length(P.nbrs[i])), ones(length(P.nbrs[i]))))

"Δs = −L 1_C: shifting the potentials of a cell."
function cell_dir(P::ToyProblem, C::Vector{Int})
    inC = falses(P.N); inC[C] .= true
    d = Dict{Int,Float64}()
    for i in C, j in P.nbrs[i]
        inC[j] && continue
        d[i] = get(d, i, 0.0) - 1.0
        d[j] = get(d, j, 0.0) + 1.0
    end
    k = collect(keys(d))
    return k, [d[x] for x in k]
end

"Square blocks of side b (geographic cells); the slack node gets its own cell."
function block_cells(P::ToyProblem, b)
    n = P.n
    cells = Vector{Int}[]
    for bi in 0:b:n-1, bj in 0:b:n-1
        C = [(j - 1) * n + i for i in bi+1:min(bi + b, n) for j in bj+1:min(bj + b, n)]
        filter!(!=(P.istar), C)
        isempty(C) || push!(cells, C)
    end
    return cells
end

"Cells of equal size grouped by reduced cost (merit-order cells); slack node excluded."
function merit_cells(P::ToyProblem, ncell)
    order = sortperm(P.r)
    filter!(!=(P.istar), order)
    m = cld(length(order), ncell)
    return [order[q:min(q + m - 1, end)] for q in 1:m:length(order)]
end

"All quadtree blocks (sides n/2, n/4, …, 2) for multigrid directions."
quadtree_cells(P::ToyProblem) = vcat([block_cells(P, b) for b in (P.n ÷ 2^k for k in 1:round(Int, log2(P.n)) - 1)]...)

"Effective sample size (Geyer's initial monotone sequence)."
function ess(x)
    n = length(x)
    y = x .- mean(x)
    v = sum(abs2, y) / n
    v == 0 && return Float64(n)
    ρ(k) = sum(y[1:n-k] .* y[1+k:n]) / (n * v)
    s, prev = 0.0, Inf
    for k in 0:2:n-2
        Γ = ρ(k) + ρ(k + 1)
        Γ <= 0 && break
        Γ = min(Γ, prev); s += Γ; prev = Γ
    end
    return n / max(2s - 1, 1.0)
end

end
