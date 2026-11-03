# Laplace modes of the network and nested mode bases on node supports.
#
# Every node-indexed field (utilizations, capacities, potentials, load,
# availability) is expanded in modes of the graph Laplacian. A field with node
# support S uses the Gram-Schmidt orthonormalization of the global modes
# restricted to S, taken in order of increasing eigenvalue. Level ℓ keeps the
# basis vectors spanned by the first M_ℓ global modes, so the bases are nested
# and the coarse coordinates of a field are a prefix of its fine coordinates.

"""
    laplace_modes(N, edges)

Eigen-decomposition of the weighted graph Laplacian with `edges` given as
`(a, b, w)`. Returns `(U, λ)` with eigenvalues in ascending order.
"""
function laplace_modes(N::Int, edges)
    L = zeros(N, N)
    for (a, b, w) in edges
        L[a, a] += w
        L[b, b] += w
        L[a, b] -= w
        L[b, a] -= w
    end
    F = eigen(Symmetric(L))
    U = F.vectors
    # fix signs so that every mode has a positive sum (or first nonzero entry)
    for j in 1:N
        s = sum(U[:, j])
        if abs(s) < 1e-10
            s = U[findfirst(x -> abs(x) > 1e-10, U[:, j]), j]
        end
        s < 0 && (U[:, j] .*= -1)
    end
    return U, F.values
end

"""
Orthonormal, nested basis of a field on its node `support`, stored sparse.

`rank_at[M]` is the number of basis vectors spanned by the first `M` global
modes, so the level-ℓ basis of the field is `U[:, 1:rank_at[M_ℓ]]`. For graph
Haar bases, `cells[m]` gives the cell label of every support node when the
first `m` vectors span exactly the cell-constant fields of a partition; the
orthogonal projection onto them is then the cell mean (O(support) work).
`Ut` is the transpose (fast row access: the coefficients touched by one node).
"""
struct ModeBasis
    U::SparseMatrixCSC{Float64,Int}
    Ut::SparseMatrixCSC{Float64,Int}
    support::Vector{Int}
    rank_at::Vector{Int}
    cells::Dict{Int,Vector{Int}}
    csize::Dict{Int,Vector{Int}}
    colcache::Dict{Int,SparseMatrixCSC{Float64,Int}}
    lk::ReentrantLock
end

function ModeBasis(U::AbstractMatrix, support::Vector{Int}, rank_at::Vector{Int};
                   cells::Dict{Int,Vector{Int}}=Dict{Int,Vector{Int}}())
    Us = sparse(U)
    droptol!(Us, 1e-14)
    csize = Dict(m => [count(==(c), lab) for c in 1:maximum(lab; init=0)] for (m, lab) in cells)
    return ModeBasis(Us, sparse(Us'), support, rank_at, cells, csize, Dict{Int,SparseMatrixCSC{Float64,Int}}(), ReentrantLock())
end

nsupport(B::ModeBasis) = length(B.support)
nmodes(B::ModeBasis) = size(B.U, 2)
modes_at(B::ModeBasis, M::Int) = B.rank_at[clamp(M, 1, length(B.rank_at))]

"The first `m` basis vectors as a (cached) sparse matrix."
function cols(B::ModeBasis, m::Int)
    m == size(B.U, 2) && return B.U
    lock(B.lk) do
        get!(() -> B.U[:, 1:m], B.colcache, m)
    end
end

"""
Orthogonal projection of `r` (on the support) onto the first `m` basis vectors,
written to `out`: the cell mean for Haar levels, `U Uᵀ r` otherwise. `tmp` needs
length ≥ max(m, number of cells).
"""
function project!(out::AbstractVector, B::ModeBasis, m::Int, r::AbstractVector, tmp::AbstractVector)
    if m == size(B.U, 2)
        out .= r
    elseif haskey(B.cells, m)
        c = B.cells[m]
        cnt = B.csize[m]
        s = view(tmp, 1:length(cnt))
        fill!(s, 0.0)
        @inbounds for i in eachindex(c)
            s[c[i]] += r[i]
        end
        @inbounds for i in eachindex(c)
            out[i] = s[c[i]] / cnt[c[i]]
        end
    else
        U = cols(B, m)
        t = view(tmp, 1:m)
        mul!(t, U', r)
        mul!(out, U, t)
    end
    return out
end

"""
    nested_basis(Ug, support; deflate)

Sequential Gram-Schmidt of the global modes `Ug[support, j]`, `j = 1, 2, …`,
after projecting out the orthonormal columns of `deflate` (used to remove the
gauge modes of the potentials).
"""
function nested_basis(Ug::Matrix{Float64}, support::Vector{Int};
                      deflate::Matrix{Float64}=zeros(length(support), 0), tol=1e-7)
    S, N = length(support), size(Ug, 2)
    target = S - size(deflate, 2)
    cols = Vector{Vector{Float64}}()
    rank_at = zeros(Int, N)
    for j in 1:N
        length(cols) == target && (rank_at[j] = target; continue)
        v = Ug[support, j]
        for _ in 1:2
            v .-= deflate * (deflate' * v)
            for c in cols
                v .-= dot(c, v) .* c
            end
        end
        nv = norm(v)
        nv > tol && push!(cols, v ./ nv)
        rank_at[j] = length(cols)
    end
    @assert length(cols) == target "mode basis incomplete on support of size $S"
    return ModeBasis(reduce(hcat, cols; init=zeros(S, 0)), support, rank_at)
end

"Identity basis: the field is not coarsened in space (used for DC links)."
identity_basis(S::Int, N::Int) = ModeBasis(sparse(1.0I, S, S), collect(1:S), fill(S, N))

"""
Connected components of the AC network, as orthonormal indicator columns over
all nodes. Potentials are defined modulo one constant per component.
"""
function component_indicators(N::Int, bus0, bus1)
    parent = collect(1:N)
    find(i) = (while parent[i] != i; i = parent[i]; end; i)
    for (a, b) in zip(bus0, bus1)
        ra, rb = find(a), find(b)
        ra != rb && (parent[ra] = rb)
    end
    roots = unique(find.(1:N))
    Z = zeros(N, length(roots))
    for i in 1:N
        Z[i, findfirst(==(find(i)), roots)] = 1.0
    end
    return Z ./ sqrt.(sum(Z; dims=1))
end

"""
    haar_basis(partitions, support; edges=nothing, drop_top=false)

Sparse orthonormal graph Haar basis on `support` for a nested sequence of
partitions of all nodes (label vectors, coarse to fine; the last is normally the
identity). Order: the (normalized) indicators of the top-level cells (omitted
with `drop_top`, e.g. for gauge-free potentials), then, level by level, for
every parent cell the contrasts between its children arranged as a balanced
binary tree: the children are ordered along the Fiedler vector of their
quotient graph (`edges` = (a, b, w) over all nodes, optional), split into two
halves of similar size, contrasted (size-weighted), and the halves are split
recursively. Every vector lives in one parent cell, the vectors of one parent
sum to O(size · log #children) nonzeros, and the first `counts[q]` vectors span
the cell-constant fields of partition q (minus the top cells with `drop_top`).
Returns `(U, counts, labels)` with `labels[q]` the cell index (1..n_q) of every
support node at level q.
"""
function haar_basis(partitions::Vector{<:AbstractVector}, support::Vector{Int}=collect(1:length(partitions[1]));
                    edges=nothing, drop_top::Bool=false, weights=nothing)
    wts = weights === nothing ? ones(length(support)) : Float64.(weights)
    S = length(support)
    pos = Dict(n => i for (i, n) in enumerate(support))
    # cell index per support node and level (cells as sets of support positions)
    labels = Vector{Vector{Int}}()
    key = Dict{Any,Int}()
    prev = fill(0, S)
    for lab in partitions
        key = Dict{Any,Int}()
        cur = [get!(key, (prev[i], lab[support[i]]), length(key) + 1) for i in 1:S]
        push!(labels, cur)
        prev = cur
    end
    I_, J_, V_ = Int[], Int[], Float64[]
    ncol = 0
    # weighted Haar: cell "indicators" are w∘1_c; contrasts w∘1_A/‖w_A‖² − w∘1_B/‖w_B‖² (orthogonal
    # to w∘1_{A∪B}); uniform weights give the ordinary graph Haar basis
    function addvec!(plus::Vector{Int}, minus::Vector{Int})
        ncol += 1
        if isempty(minus)
            a = 1 / sqrt(sum(abs2, wts[plus]))
            for i in plus; push!(I_, i); push!(J_, ncol); push!(V_, a * wts[i]); end
            return
        end
        sp, sm = sum(abs2, wts[plus]), sum(abs2, wts[minus])
        nrm = sqrt(1 / sp + 1 / sm)
        for i in plus; push!(I_, i); push!(J_, ncol); push!(V_, wts[i] / sp / nrm); end
        for i in minus; push!(I_, i); push!(J_, ncol); push!(V_, -wts[i] / sm / nrm); end
    end
    counts = Int[]
    if !drop_top
        for c in 1:maximum(labels[1]; init=0)
            addvec!(findall(==(c), labels[1]), Int[])
        end
    end
    push!(counts, ncol)
    # adjacency among support nodes, for ordering children
    W = Dict{Tuple{Int,Int},Float64}()
    if edges !== nothing
        for (a, b, w) in edges
            (haskey(pos, a) && haskey(pos, b)) || continue
            i, j = pos[a], pos[b]
            i == j && continue
            W[(min(i, j), max(i, j))] = get(W, (min(i, j), max(i, j)), 0.0) + w
        end
    end
    for q in 2:length(labels)
        par, ch = labels[q-1], labels[q]
        for p in 1:maximum(par; init=0)
            members = findall(==(p), par)
            kids = unique(ch[members])
            length(kids) < 2 && continue
            groups = [members[ch[members].==c] for c in kids]
            groups = order_groups(groups, W)
            bisect_contrasts!(addvec!, groups)
        end
        push!(counts, ncol)
    end
    U = sparse(I_, J_, V_, S, ncol)
    return U, counts, labels
end

"Order cells (vectors of support positions) along the Fiedler vector of their quotient graph."
function order_groups(groups::Vector{Vector{Int}}, W::Dict{Tuple{Int,Int},Float64})
    K = length(groups)
    (K <= 2 || isempty(W)) && return groups
    g = Dict(i => k for (k, grp) in enumerate(groups) for i in grp)
    L = zeros(K, K)
    for ((i, j), w) in W
        (haskey(g, i) && haskey(g, j)) || continue
        a, b = g[i], g[j]
        a == b && continue
        L[a, a] += w; L[b, b] += w; L[a, b] -= w; L[b, a] -= w
    end
    all(iszero, L) && return groups
    F = eigen(Symmetric(L + 1e-9 * I))
    return groups[sortperm(F.vectors[:, 2])]
end

"Balanced binary-tree contrasts between consecutive groups of cells."
function bisect_contrasts!(addvec!, groups::Vector{Vector{Int}})
    length(groups) < 2 && return
    sz = cumsum(length.(groups))
    tot = sz[end]
    h = argmin([abs(2 * sz[j] - tot) for j in 1:length(groups)-1])
    A, B = groups[1:h], groups[h+1:end]
    addvec!(reduce(vcat, A), reduce(vcat, B))
    bisect_contrasts!(addvec!, A)
    bisect_contrasts!(addvec!, B)
end

"""
    haar_vectors(partitions)

Graph Haar basis of ℝᴺ (all nodes) for nested partitions; see `haar_basis`.
Returns `(U, counts)`.
"""
function haar_vectors(partitions::Vector{<:AbstractVector}; edges=nothing)
    U, counts, _ = haar_basis(partitions; edges=edges)
    return U, counts
end

"""
    haar_field_basis(partitions, gcounts, support; edges, drop_top)

ModeBasis of a field on `support` from global nested partitions. `gcounts[q]`
is the number of global basis vectors at partition level q (the M of a level),
so `rank_at[M]` is the field's count at the finest q with `gcounts[q] ≤ M`.
"""
function haar_field_basis(partitions, gcounts::Vector{Int}, support::Vector{Int}; edges=nothing, drop_top=false, weights=nothing)
    U, counts, labels = haar_basis(partitions, support; edges=edges, drop_top=drop_top, weights=weights)
    N = length(partitions[1])
    rank_at = zeros(Int, N)
    for M in 1:N
        q = something(findlast(<=(M), gcounts), 1)
        rank_at[M] = counts[q]
    end
    cells = Dict{Int,Vector{Int}}()
    if !drop_top && weights === nothing
        for q in eachindex(counts)
            cells[counts[q]] = labels[q]
        end
    end
    @assert size(U, 2) == length(support) - (drop_top ? maximum(labels[1]) : 0)
    return ModeBasis(U, support, rank_at; cells=cells)
end
