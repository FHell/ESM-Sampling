# Flow-based hierarchy, offline part (docs/flow_hierarchy.md §2, §5): proxy
# capacities and injections from the weather data, congestion projection by a
# QP per step on the unextended network, transport modes from a PCA of the line
# loadings, and the intra-block storage shapes. Run-time frames (§3) at the end.

const RENEWABLE_FIELDS = (:onwind, :offwind, :solar)

struct FlowBasis
    K::Int
    modes::Matrix{Float64}                         # N × K angle modes (orthonormal, ⟂ component constants)
    explained::Vector{Float64}                     # cumulative explained loading variance of modes 1..K
    khat::Vector{Float64}                          # proxy capacity per generator unit (sys.gen order)
    bindfreq::Vector{Float64}                      # binding frequency per line in the proxy ensemble
    h::Int                                         # proxy time resolution (hours)
    shapes::Dict{Tuple{Int,Int,Int},Matrix{Float64}}   # (h, L, cell) → (L+1) × nshape, zero at both ends
    cell::Vector{Int}                              # storage-shape cell (country) of every node
    resid::Dict{Int,Array{Float64,3}}              # h → proxy residual load per cell (cells × T × years)
    lk::ReentrantLock
end

"Proxy capacities: fixed fleet as is; investable renewables scaled from their potentials to a target energy share."
function proxy_capacities(sys::System; share=0.7)
    G = sys.gen
    k = copy(G[:k])
    load = sys.weather[:load][2]
    Ebar = mean(Float64, load) * size(load, 1)               # mean system load (MW)
    ren = Int[]
    potential = 0.0
    for (u, c) in enumerate(G[:carrier])
        s = Symbol(c)
        (s in RENEWABLE_FIELDS && G[:invest][u] > 0.5 && haskey(sys.weather, s)) || continue
        nodes, data = sys.weather[s]
        i = findfirst(==(G[:bus][u]), nodes)
        i === nothing && continue
        push!(ren, u)
        potential += G[:kbar][u] * mean(Float64, view(data, i, :, :))
    end
    fixed_ren = 0.0
    for (u, c) in enumerate(G[:carrier])
        s = Symbol(c)
        (s in RENEWABLE_FIELDS && !(u in ren) && haskey(sys.weather, s)) || continue
        nodes, data = sys.weather[s]
        i = findfirst(==(G[:bus][u]), nodes)
        i === nothing || (fixed_ren += k[u] * mean(Float64, view(data, i, :, :)))
    end
    α = potential > 0 ? clamp((share * Ebar - fixed_ren) / potential, 0.0, 1.0) : 0.0
    for u in ren
        k[u] = α * G[:kbar][u]
    end
    return k
end

"Proxy injections p[n, t, y] at resolution h: renewable output at k̂ minus load, balanced copper-plate."
function proxy_injections(sys::System, khat; h=24, years=1:length(sys.years))
    N = sys.N
    nodesL, L = sys.weather[:load]
    D = aggregate(L[:, :, years], h)
    T, Y = size(D, 2), size(D, 3)
    R = zeros(N, T, Y)
    G = sys.gen
    for s in RENEWABLE_FIELDS
        haskey(sys.weather, s) || continue
        nodes, data = sys.weather[s]
        A = aggregate(data[:, :, years], h)
        for (u, c) in enumerate(G[:carrier])
            Symbol(c) == s || continue
            i = findfirst(==(G[:bus][u]), nodes)
            i === nothing && continue
            R[G[:bus][u], :, :] .+= khat[u] .* A[i, :, :]
        end
    end
    Dn = zeros(N, T, Y)
    Dn[nodesL, :, :] .= D
    Δ = R .- Dn
    dbar = vec(mean(Dn; dims=(2, 3)))
    w = dbar ./ sum(dbar)
    p = similar(Δ)
    for y in 1:Y, t in 1:T
        tot = sum(view(Δ, :, t, y))
        p[:, t, y] .= view(Δ, :, t, y) .- w .* tot
    end
    return p, Dn, R
end

"""
Batched congestion projection (array-first, Reactant-traceable: dense matmuls,
broadcasts and clamps only, static shapes). For every column p_t of `Pm`
(N × T injections) solve

  min_{θ,u} ½‖Lθ + C u − p_t‖² + ε/2‖x‖²   s.t.  |b_e(θ_a − θ_b)| ≤ s_e, |u_l| ≤ ū_l

by ADMM on y = A x = (Dθ, u) with x = (θ, u):
  x ← M⁻¹ (Gᵀp + ρ Aᵀ(y − λ)),  y ← clamp(A x + λ, lo, hi),  λ ← λ + A x − y,
with G = [L C], M = GᵀG + ρAᵀA + εI inverted once. Returns (θ, loading, mask)
with loading = Dθ/s and the binding mask |loading| ≥ 1 − tol.
"""
struct CongestionADMM
    Minv::Matrix{Float64}
    G::Matrix{Float64}
    A::Matrix{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
    s::Vector{Float64}
    N::Int
    nl::Int
    ρ::Float64
end

function CongestionADMM(sys::System; ρ=1.0, ε=1e-6)
    N, nl, nk = sys.N, length(sys.line_s), length(sys.link_cap)
    D = zeros(nl, N)
    for e in 1:nl
        D[e, sys.line_bus0[e]] += sys.line_b[e]; D[e, sys.line_bus1[e]] -= sys.line_b[e]
    end
    L = D' * Diagonal(1 ./ sys.line_b) * D                  # = K diag(b) Kᵀ (flows out of each node)
    C = zeros(N, nk)
    for l in 1:nk
        C[sys.link_bus0[l], l] += 1.0; C[sys.link_bus1[l], l] -= 1.0
    end
    G = hcat(L, C)
    A = [D zeros(nl, nk); zeros(nk, N) Matrix(1.0I, nk, nk)]
    # scale: work in units where the typical flow limit is 1
    M = G' * G + ρ * (A' * A) + ε * I
    lo = vcat(-sys.line_s, -sys.link_cap)
    hi = vcat(sys.line_s, sys.link_cap)
    return CongestionADMM(inv(Symmetric(M)), G, A, lo, hi, sys.line_s, N, nl, ρ)
end

function project_congestion(Q::CongestionADMM, Pm::AbstractMatrix; iters::Int=400, tol=1e-3)
    T = size(Pm, 2)
    Gp = Q.G' * Pm
    Y = zeros(size(Q.A, 1), T)
    Λ = zeros(size(Q.A, 1), T)
    X = Q.Minv * Gp
    for _ in 1:iters
        X = Q.Minv * (Gp .+ Q.ρ .* (Q.A' * (Y .- Λ)))
        AX = Q.A * X
        Y = clamp.(AX .+ Λ, Q.lo, Q.hi)
        Λ = Λ .+ AX .- Y
    end
    θ = X[1:Q.N, :]
    load = (Q.A[1:Q.nl, 1:Q.N] * θ) ./ Q.s
    load = clamp.(load, -1.0, 1.0) .* (abs.(load) .<= 1 + 10tol) .+ load .* (abs.(load) .> 1 + 10tol)
    mask = abs.(Y[1:Q.nl, :] ./ Q.s) .>= 1 - tol
    return θ, load, mask
end

"Line loadings ℓ_e = b_e(θ_a − θ_b)/s_e."
loadings(sys::System, θ) = [sys.line_b[e] * (θ[sys.line_bus0[e]] - θ[sys.line_bus1[e]]) / sys.line_s[e] for e in eachindex(sys.line_s)]

"""
    flow_basis(sys; K_max, explain, h, nyears, share) -> FlowBasis

Proxy ensemble on `nyears` evenly spaced weather years at resolution h,
congestion projection per step, PCA of the loadings (second moment, so the mean
pattern is included), loading patterns mapped back to angles by the
loading-weighted Laplacian, orthonormalized against the component constants.
"""
function flow_basis(sys::System; K_max::Int=12, explain::Float64=0.95, h::Int=24, nyears::Int=4, share=0.7)
    N = sys.N
    khat = proxy_capacities(sys; share=share)
    Yall = length(sys.years)
    years = (min(nyears, Yall) <= 1 ? [1] : unique(round.(Int, range(1, Yall; length=min(nyears, Yall)))))
    p, _, _ = proxy_injections(sys, khat; h=h, years=years)
    T, Y = size(p, 2), size(p, 3)
    nl = length(sys.line_s)
    nbind = zeros(nl)
    Pm = reshape(p, sys.N, T * Y)
    Q = CongestionADMM(sys)
    _, ℓs, mask = project_congestion(Q, Pm)
    nbind .= vec(sum(mask; dims=2)) ./ size(mask, 2)
    S = ℓs * ℓs' ./ size(ℓs, 2)
    F = eigen(Symmetric(S))
    λ = reverse(F.values); Q = reverse(F.vectors; dims=2)
    cum = cumsum(max.(λ, 0)) ./ sum(max.(λ, 0))
    K = min(K_max, something(findfirst(>=(explain), cum), length(cum)), N - 1)
    # back to angles: d = (WᵀW)⁺ Wᵀ q with W = diag(b/s) Kᵀ
    W = zeros(nl, N)
    for e in 1:nl
        w = sys.line_b[e] / sys.line_s[e]
        W[e, sys.line_bus0[e]] += w; W[e, sys.line_bus1[e]] -= w
    end
    Zc = component_indicators(N, sys.line_bus0, sys.line_bus1)
    M = W' * W + Zc * Zc'
    D = M \ (W' * Q[:, 1:K])
    D .-= Zc * (Zc' * D)
    # orthonormalize (Gram–Schmidt in order of explained variance)
    modes = zeros(N, 0)
    for k in 1:K
        v = D[:, k]
        for _ in 1:2
            v .-= modes * (modes' * v)
        end
        nv = norm(v)
        nv > 1e-10 && (modes = hcat(modes, v ./ nv))
    end
    cidx = Dict(c => i for (i, c) in enumerate(unique(sys.country)))
    return FlowBasis(size(modes, 2), modes, cum[1:size(modes, 2)], khat, nbind, h,
                     Dict{Tuple{Int,Int,Int},Matrix{Float64}}(), [cidx[c] for c in sys.country],
                     Dict{Int,Array{Float64,3}}(), ReentrantLock())
end

"""
Angle basis of the flow design (§2.5): the transport modes first, then the
graph-Haar angle vectors with the modes projected out. `rank_at[M]` = K at
spatially coarse levels (M < N) and N − n_comp at the complete level.
"""
function flow_theta_basis(FB::FlowBasis, Bhaar::ModeBasis, N::Int)
    cols = [FB.modes[:, k] for k in 1:FB.K]
    H = Matrix(Bhaar.U)
    for j in 1:size(H, 2)
        v = H[:, j]
        for _ in 1:2
            for c in cols
                v .-= dot(c, v) .* c
            end
        end
        nv = norm(v)
        nv > 1e-8 && push!(cols, v ./ nv)
    end
    U = reduce(hcat, cols)
    @assert size(U, 2) == size(H, 2) "flow angle basis incomplete"
    rank_at = [M < N ? FB.K : size(U, 2) for M in 1:N]
    return ModeBasis(U, collect(1:N), rank_at)
end

"""
Intra-block storage shapes (§5) for blocks of L steps of h hours in cell c:
leading principal components (second moment) of the proxy SoC profile within
blocks, the cumulative sum of the cell's proxy residual minus its block mean.
Each column has length L+1 (instants of the block), zero at both ends and
max |φ| = 1. Computed on demand and cached.
"""
function storage_shapes(FB::FlowBasis, sys::System, h::Int, L::Int, c::Int; nshape::Int=2, nyears::Int=4)
    key = (h, L, c)
    lock(FB.lk) do
        haskey(FB.shapes, key) && return FB.shapes[key]
        RC = get!(FB.resid, h) do
            Yall = length(sys.years)
            years = (min(nyears, Yall) <= 1 ? [1] : unique(round.(Int, range(1, Yall; length=min(nyears, Yall)))))
            _, Dn, R = proxy_injections(sys, FB.khat; h=h, years=years)
            nc = maximum(FB.cell)
            out = zeros(nc, size(Dn, 2), size(Dn, 3))
            for n in 1:sys.N
                out[FB.cell[n], :, :] .+= Dn[n, :, :] .- R[n, :, :]
            end
            out
        end
        res = RC[c, :, :]                                                       # T × Y
        T, Y = size(res)
        X = Vector{Vector{Float64}}()
        for y in 1:Y, b in 0:(T÷L)-1
            seg = res[b*L+1:(b+1)*L, y]
            dev = seg .- mean(seg)
            push!(X, vcat(0.0, cumsum(dev)))          # SoC deviation at instants 0..L (ends 0)
        end
        Xm = reduce(hcat, X)
        F = eigen(Symmetric(Xm * Xm' ./ size(Xm, 2)))
        Φ = reverse(F.vectors; dims=2)[:, 1:min(nshape, L - 1)]
        for k in 1:size(Φ, 2)
            Φ[1, k] = 0.0; Φ[end, k] = 0.0
            Φ[:, k] ./= maximum(abs.(Φ[:, k]))
        end
        FB.shapes[key] = Φ
        return Φ
    end
end

