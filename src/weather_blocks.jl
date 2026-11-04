# Joint low-frequency moves of the level-0 weather (docs/system/moves/weather.md).
#
# The coordinate moves of z0 change one row at one level-0 step. The rows of z0 are strongly
# coupled (the load modes are almost collinear, the VAR is persistent), so under the prior
# the annual and seasonal means move by a tiny amount per coordinate move. A block move
# shifts z0 by t·v over the level-0 steps a..b, v a fixed joint direction of all rows:
#
#   z0[:, a:b] → z0[:, a:b] + t v 1ᵀ,
#
# with the innovations of the finer levels fixed, so the finest anomalies change by the
# windowed propagation through the detail regressions (affine in t). The VAR prior on the
# line is the exact quadratic of the emissions of the steps a..b+1; the dispatchable units
# at every node whose requirement changes follow share-preservingly with requirement
# tracking (as for the coordinate moves), the line is drawn exactly with its Gaussian prior,
# and the pro-rata Jacobian and, at α = 1, the bias change e^{−Δb̂} enter one Metropolis
# step. Lengths, positions and directions are drawn independently of the state, and the
# directions are fixed by the weather model, so the move is invariant.

"""
    z0_block_directions(W; p=6) -> V (n0 × p)

Fixed joint directions of the block moves: the leading eigenvectors of the long-run
covariance of the level-0 VAR, Σ_k π_k (I − Φ_k)⁻¹ Σ_k (I − Φ_k)⁻ᵀ, averaged over the regimes
with their stationary frequencies π. The long-run covariance is the covariance of block means
over many correlation times, so these are the directions in which the annual and seasonal
means fluctuate most.
"""
function z0_block_directions(W::WeatherModel; p::Int=6)
    n = n0(W)
    ev = eigen(Matrix(W.Ptr'))
    πs = real(ev.vectors[:, argmax(real(ev.values))])
    πs ./= sum(πs)
    S = zeros(n, n)
    for k in 1:W.K
        A = inv(I - W.Bm[k][:, 1:n])
        S .+= πs[k] .* (A * inv(Symmetric(Matrix(W.Σinv[k]))) * A')
    end
    return eigen(Symmetric(S)).vectors[:, end:-1:end-min(p, n)+1]
end

"−log p(z0 | m) terms (VAR emissions) of the level-0 steps a..b+1, without the log-determinants."
function z0_nlogp_steps(W::WeatherModel, m, Z::AbstractMatrix, a::Int, b::Int)
    n, T0 = size(Z)
    val = 0.0
    e = zeros(n)
    for τ in a:min(b + 1, T0)
        if τ == 1
            val += 0.5 * dot(view(Z, :, 1), W.Σ0inv * view(Z, :, 1))
        else
            B = W.Bm[m[τ]]
            c, cs, sn = harmonics(τ, T0)
            e .= view(Z, :, τ) .- view(B, :, 1:n) * view(Z, :, τ - 1) .- c .* view(B, :, n + 1) .- cs .* view(B, :, n + 2) .-
                 sn .* view(B, :, n + 3)
            val += 0.5 * dot(e, W.Σinv[m[τ]] * e)
        end
    end
    return val
end

"Quadratic (κ, β) of −log p(z0 + t v 1ᵀ on steps a..b | m) − const in t (exact: three evaluations)."
function z0_block_quad(W::WeatherModel, m, z0, a::Int, b::Int, v::AbstractVector)
    Z = z0matrix(z0, W.lays[1], W.specs)
    U0 = z0_nlogp_steps(W, m, Z, a, b)
    Z[:, a:b] .+= v; U1 = z0_nlogp_steps(W, m, Z, a, b)
    Z[:, a:b] .-= 2 .* v; Um = z0_nlogp_steps(W, m, Z, a, b)
    return (U1 + Um - 2U0, 0.5 * (U1 - Um))
end

"Change of z0 (as a level-0 coefficient vector) per unit t of a block move: v on the steps a..b."
function z0_block_delta(W::WeatherModel, a::Int, b::Int, v::AbstractVector)
    lay = W.lays[1]
    dz = zeros(lay.n)
    row = 0
    for f in eachindex(W.specs)
        fieldview(dz, lay, W.specs, f)[:, a:b] .= v[row+1:row+lay.m[f]]
        row += lay.m[f]
    end
    return dz
end

"Number of finest-level steps a block move of `B` level-0 steps can reach (pool size)."
function z0_block_span(W::WeatherModel, L::Int, B::Int)
    span = B
    for ℓ in 1:L
        span = (span + 2) * W.trs[ℓ].r
    end
    return span
end

"""
    z0_block_prepare!(bm, P, LM, LC, s, jac, a, b, v; comp) -> (ch, dz, quad) or nothing

Prepare the block move of direction v on the level-0 steps a..b: the chain of windowed
changes to the finest level, the change of z0 per unit t, the prior quadratic, and the
block move `bm` with its compensation (`jac` collects the pro-rata Jacobian terms).
"""
function z0_block_prepare!(bm::BlockMove, P::Problem, LM::LocalModel, LC::LocalCost, s::State, jac, a::Int, b::Int,
                           v::AbstractVector; comp::Bool=true)
    W = P.W
    lay = W.lays[1]
    Dm = Matrix{Float64}[]
    row = 0
    for f in eachindex(W.specs)
        push!(Dm, repeat(v[row+1:row+lay.m[f]], 1, b - a + 1))
        row += lay.m[f]
    end
    ch = wdelta_chain(W, s.m, WDelta(0, a, b, Dm), s.ℓ)
    prep_weather_move!(bm, P, LM, LC, ch, jac; comp=comp) === nothing && return nothing
    return ch, z0_block_delta(W, a, b, v), z0_block_quad(W, s.m, s.z0, a, b, v)
end

"Commit a prepared block move at t: operation, caches, z0 and the anomalies of every level, and the bias cache."
function z0_block_commit!(P::Problem, s::State, bm::BlockMove, ch, dz, t::Float64; BC=nothing)
    commit_block!(bm, s.v, t)
    BC === nothing || bias_delta_weather(BC, P.bias, P.E, bm, t; commit=true)
    s.z0 .+= t .* dz
    s.xs[1] === s.z0 || (s.xs[1] .+= t .* dz)
    apply_wdelta!(s, P.W, ch[2:end], t)
    return s
end

"""
One block move of direction v on the level-0 steps a..b: exact draw of the bias-free line
conditional (cost and VAR prior), then a Metropolis step with the pro-rata Jacobian and, at
α = 1 (`BC` the bias cache), e^{−Δb̂}. Returns t (0 if not moved).
"""
function z0_block_move!(rng, P::Problem, LM::LocalModel, LC::LocalCost, s::State, bm::BlockMove, jac, secprev,
                        a::Int, b::Int, v::AbstractVector; comp::Bool=true, BC=nothing)
    pr = z0_block_prepare!(bm, P, LM, LC, s, jac, a, b, v; comp=comp)
    pr === nothing && return 0.0
    ch, dz, quad = pr
    t = sample_line!(rng, bm, P.βv, secprev; quad=quad)
    t == 0.0 && return 0.0
    logα = isempty(jac) ? 0.0 : logjac(jac, t, bm)
    BC === nothing || (logα -= bias_delta_weather(BC, P.bias, P.E, bm, t))
    (isfinite(logα) && log(rand(rng)) < logα) || return 0.0
    z0_block_commit!(P, s, bm, ch, dz, t; BC=BC)
    return t
end

"""
    z0_block_sweep!(rng, P, LM, s, nmoves; LC, BC, pool, lengths, comp) -> (s, stats)

`nmoves` joint block moves of the level-0 weather in sequence, each with a length B drawn
from `lengths` with probabilities ∝ 1/B (about equal CPU time per length), a start uniform
over the level-0 steps (the block clipped at the end of the year) and a direction drawn
uniformly from `z0_block_directions(P.W)`. The caches `LC` and `BC` (built if not given)
follow the accepted moves; the sums of `BC` are recomputed at the end. stats: (moves, moved).
"""
function z0_block_sweep!(rng, P::Problem, LM::LocalModel, s::State, nmoves::Int; LC=nothing, BC=nothing, pool=nothing,
                         lengths=(16, 64, 365), comp::Bool=true, V=nothing)
    W = P.W
    nmoves > 0 || return s, (0, 0)
    LC = LC === nothing ? LocalCost(P, LM, s) : LC
    BC = (P.use_bias && BC === nothing) ? BiasCache(P, s) : BC
    V = V === nothing ? z0_block_directions(W) : V
    T0 = W.lays[1].T
    span = maximum(z0_block_span(W, s.ℓ, min(B, T0)) for B in lengths)
    pool = pool === nothing ? BlockPool(P, span) : pool
    while length(pool.wss) < span
        push!(pool.wss, MoveWS(P))
    end
    bm = BlockMove(pool.wss, 0, Int[], P, LM, LC, Float64[], Float64[])
    jac, sp = NTuple{5,Float64}[], Float64[]
    pw = cumsum([1 / B for B in lengths]); pw ./= pw[end]
    moved = 0
    for _ in 1:nmoves
        B = lengths[findfirst(>=(rand(rng)), pw)]
        a = rand(rng, 1:T0)
        b = min(a + B - 1, T0)
        v = view(V, :, rand(rng, axes(V, 2)))
        moved += z0_block_move!(rng, P, LM, LC, s, bm, jac, sp, a, b, v; comp=comp, BC=BC) != 0.0
    end
    BC === nothing || bias_resum!(BC, P.bias)
    return evaluate!(P, s), (nmoves, moved)
end
