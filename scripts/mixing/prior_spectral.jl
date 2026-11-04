# Spectral analysis of random-scan Gibbs moves on the level-0 weather prior z0 | m
# (docs/mixing_diagnosis.md). For a Gaussian target with precision Λ and exact conditional
# draws along directions d_j chosen with probabilities p_j, the expected deviation from the
# mean contracts per move by I − GΛ with G = Σ_j p_j d_j d_jᵀ / (d_jᵀΛd_j). The slowest
# relaxation rate per move is λ_min(GΛ) (GΛ is similar to a symmetric PSD matrix). We get it
# by power iteration on (GΛ)⁻¹ = Λ⁻¹G⁻¹ with sparse Cholesky solves. Also the relaxation rate
# of particular linear functionals (annual means): a Rayleigh quotient on the same operator.

include(joinpath(@__DIR__, "wblock.jl"))
using SparseArrays, LinearAlgebra

"Precision matrix of vec(Z) (Z n×T0, column-major) under the VAR prior given regimes m."
function var_precision(W, m)
    n, T0 = ML.n0(W), W.lays[1].T
    idx(τ) = (τ-1)*n+1:τ*n
    I_, J_, V_ = Int[], Int[], Float64[]
    function addblock!(a, b, M)
        for j in axes(M, 2), i in axes(M, 1)
            M[i, j] == 0 && continue
            push!(I_, first(idx(a)) + i - 1); push!(J_, first(idx(b)) + j - 1); push!(V_, M[i, j])
        end
    end
    addblock!(1, 1, Matrix(W.Σ0inv))
    for τ in 2:T0
        k = m[τ]
        Φ = W.Bm[k][:, 1:n]
        Si = Matrix(W.Σinv[k])
        addblock!(τ, τ, Si)
        addblock!(τ - 1, τ - 1, Φ' * Si * Φ)
        addblock!(τ, τ - 1, -Si * Φ)
        addblock!(τ - 1, τ, -Φ' * Si)
    end
    Λ = sparse(I_, J_, V_, n * T0, n * T0)
    return (Λ + Λ') / 2
end

"Directions of a move family as sparse columns: (:single) unit vectors, (:block, B) shifts of one row over B steps, (:vblock, B, V) shifts along the columns of V (n × p) over B steps."
function family_dirs(W, fam)
    n, T0 = ML.n0(W), W.lays[1].T
    cols = SparseVector{Float64,Int}[]
    if fam[1] == :single
        for τ in 1:T0, o in 1:n
            push!(cols, sparsevec([(τ-1)*n+o], [1.0], n * T0))
        end
    elseif fam[1] == :block
        B = fam[2]
        rows = length(fam) >= 3 ? fam[3] : 1:n
        for a in 1:T0, o in rows
            b = min(a + B - 1, T0)
            push!(cols, sparsevec([(τ-1)*n+o for τ in a:b], ones(b - a + 1), n * T0))
        end
    elseif fam[1] == :vblock
        B, V = fam[2], fam[3]
        for a in 1:T0, j in axes(V, 2)
            b = min(a + B - 1, T0)
            ii = Int[]; vv = Float64[]
            for τ in a:b, o in 1:n
                V[o, j] == 0 && continue
                push!(ii, (τ-1)*n+o); push!(vv, V[o, j])
            end
            push!(cols, sparsevec(ii, vv, n * T0))
        end
    end
    return cols
end

"""
Expected-squared-jump rate of the functional aᵀx per move of a family, relative to its
variance: rate = E[(Δ aᵀx)²] / (2 Var aᵀx) = mean_j (d_jᵀa)²/(d_jᵀΛd_j) / (aᵀΛ⁻¹a) at
stationarity (exact conditional draws along d_j, directions chosen uniformly). 1/rate is
the number of moves the family needs to move aᵀx by about its standard deviation.
"""
function esjd_rate(Λ, FΛ, a, dirs)
    va = dot(a, FΛ \ a)
    acc = 0.0
    for d in dirs
        acc += dot(d, a)^2 / dot(d, Λ * d)
    end
    return acc / length(dirs) / va, va
end

"Linear functionals: annual and seasonal means of rows `rows`."
function mean_functionals(W, rows)
    n, T0 = ML.n0(W), W.lays[1].T
    edges = round.(Int, range(0, T0, length=5))
    out = Pair{String,Vector{Float64}}[]
    for o in rows
        a = zeros(n * T0)
        for τ in 1:T0
            a[(τ-1)*n+o] = 1 / T0
        end
        push!(out, "annual row $o" => a)
        for q in 1:4
            a = zeros(n * T0)
            for τ in edges[q]+1:edges[q+1]
                a[(τ-1)*n+o] = 1 / (edges[q+1] - edges[q])
            end
            push!(out, "season $q row $o" => a)
        end
    end
    return out
end
