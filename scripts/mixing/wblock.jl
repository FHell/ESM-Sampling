# Collective weather moves (prototypes for docs/mixing_diagnosis.md).
#
# 1. z0 block shift: one level-0 row o (field f, mode jm) moves by t·φ over the level-0 steps
#    a..b (φ a fixed profile), with the exact VAR prior quadratic of the steps a..b+1, the
#    windowed propagation to the finest level (innovations fixed), share-preserving
#    compensation with requirement tracking at every node whose requirement changes, the
#    exact line draw with the Gaussian prior on the line, and the pro-rata Jacobian in a
#    Metropolis step: `weather_coord_move!` generalized from one step to a block.
# 2. Non-centered regime block move: the regimes of the level-0 steps a..b are proposed from
#    their exact conditional under the prior given z0 and the regimes outside (FFBS inside
#    the window), the innovations stay fixed, so the fine anomalies change with the regimes;
#    the operation follows by the pro-rata map (σ, e fixed), and the move is accepted with
#    exp(-β_v ΔC) times the pro-rata Jacobian (the prior and the proposal cancel).

"−log p(z0 | m) terms (VAR emissions) of the steps a..b+1, without the log-determinants."
function z0_nlogp_range(W, m, Z::AbstractMatrix, a::Int, b::Int)
    n, T0 = size(Z)
    val = 0.0
    for τ in a:min(b + 1, T0)
        if τ == 1
            val += 0.5 * dot(view(Z, :, 1), W.Σ0inv * view(Z, :, 1))
        else
            k = m[τ]
            B = W.Bm[k]
            c, cs, sn = ML.harmonics(τ, T0)
            e = Z[:, τ] .- B[:, 1:n] * Z[:, τ-1] .- c .* B[:, n+1] .- cs .* B[:, n+2] .- sn .* B[:, n+3]
            val += 0.5 * dot(e, W.Σinv[k] * e)
        end
    end
    return val
end

"Row index o of the stacked z0 matrix for (field f, mode jm)."
z0row(W, f, jm) = sum(W.lays[1].m[1:f-1]; init=0) + jm

"""
One z0 block move along the direction Dz (n0 × (b−a+1), rows of the stacked z0 matrix) on
the level-0 steps a..b. Returns t (0 if rejected).
"""
function z0_dir_move!(rng, P, LM, LC, s, bm, jac, secprev, a::Int, b::Int, Dz::AbstractMatrix;
                      comp::Bool=true, check::Bool=false)
    W = P.W
    L = s.ℓ
    lay = W.lays[1]
    Z = ML.z0matrix(s.z0, lay, W.specs)
    U0 = z0_nlogp_range(W, s.m, Z, a, b)
    Z[:, a:b] .+= Dz; U1 = z0_nlogp_range(W, s.m, Z, a, b)
    Z[:, a:b] .-= 2 .* Dz; Um = z0_nlogp_range(W, s.m, Z, a, b)
    Z[:, a:b] .+= Dz
    quad = (U1 + Um - 2U0, 0.5 * (U1 - Um))
    Dm = Matrix{Float64}[]
    row = 0
    for ff in eachindex(W.specs)
        push!(Dm, Matrix(Dz[row+1:row+lay.m[ff], :])); row += lay.m[ff]
    end
    d0 = ML.WDelta(0, a, b, Dm)
    ch = ML.wdelta_chain(W, s.m, d0, L)
    ML.prep_weather_move!(bm, P, LM, LC, ch, jac; comp=comp) === nothing && return 0.0
    t = ML.sample_line!(rng, bm, P.βv, secprev; quad=quad)
    t == 0.0 && return 0.0
    logα = isempty(jac) ? 0.0 : ML.logjac(jac, t, bm)
    dz = zeros(lay.n)
    for ff in eachindex(W.specs)
        ML.fieldview(dz, lay, W.specs, ff)[:, a:b] .= Dm[ff]
    end
    if P.use_bias
        k0 = ML.k_at0(P, L, s.k)
        logα -= ML.bias_term(P, k0, s.z0 .+ t .* dz) - ML.bias_term(P, k0, s.z0)
    end
    (isfinite(logα) && log(rand(rng)) < logα) || return 0.0
    ΔC = check ? ML.mv_eval!(bm, t) : 0.0
    L0 = s.L
    ML.commit_block!(bm, s.v, t)
    s.z0 .+= t .* dz
    s.xs[1] === s.z0 || (s.xs[1] .+= t .* dz)
    ML.apply_wdelta!(s, W, ch[2:end], t)
    if check
        ML.evaluate!(P, s)
        err = (s.L - L0) - (-P.βv * ΔC)
        abs(err) > 1e-6 * max(1.0, abs(P.βv * ΔC)) && @warn "z0 block: incremental and global cost differ" err ΔC
        xs2 = ML.weather_at_level(W, L, s.m, s.z0, s.ws)
        ex = maximum(abs.(xs2 .- s.xs[end]))
        ex > 1e-9 && @warn "z0 block: anomalies differ from a fresh evaluation" ex
    end
    return t
end

"One z0 block move on row (f, jm) with profile φ over the steps a..b (single-row special case)."
function z0_block_move!(rng, P, LM, LC, s, bm, jac, secprev, f::Int, jm::Int, a::Int, b::Int, φ::Vector{Float64}; kw...)
    Dz = zeros(ML.n0(P.W), b - a + 1)
    Dz[z0row(P.W, f, jm), :] .= φ
    return z0_dir_move!(rng, P, LM, LC, s, bm, jac, secprev, a, b, Dz; kw...)
end

"Block profile: :const (1 on the block) or :tent (triangular)."
function block_profile(B::Int, kind::Symbol)
    kind == :const && return ones(B)
    kind == :tent && return [1 - abs((2q - 1) / B - 1) for q in 1:B]
    error("profile $kind")
end

"""
    z0_block_sweep!(rng, P, LM, s; B, nmoves, V, profile, pool, check) -> (s, ts)

`nmoves` z0 block moves of length B (start uniform, clipped at the year end). Directions:
the columns of `V` (n0 × p, joint directions over all rows), one at random per move; or, with
`V = nothing`, the field-mean row of a random field (`rows` pairs (f, jm) if given).
"""
function z0_block_sweep!(rng, P, LM, s; B::Int=8, nmoves::Int=10, V=nothing, rows=nothing, profile::Symbol=:const,
                         pool=nothing, check::Bool=false, comp::Bool=true)
    W = P.W
    T0 = W.lays[1].T
    n = ML.n0(W)
    rows = rows === nothing ? [(f, 1) for f in eachindex(W.specs)] : rows
    s = copy(s)
    LC = ML.LocalCost(P, LM, s)
    span = (B + 2) * prod((W.trs[l].r for l in 1:s.ℓ); init=1) + 4
    pool = pool === nothing ? ML.BlockPool(P, span) : pool
    while length(pool.wss) < span
        push!(pool.wss, ML.MoveWS(P))
    end
    bm = ML.BlockMove(pool.wss, 0, Int[], P, LM, LC, Float64[], Float64[])
    jac, sp = NTuple{5,Float64}[], Float64[]
    ts = zeros(nmoves)
    for q in 1:nmoves
        a = rand(rng, 1:T0)
        b = min(a + B - 1, T0)
        φ = block_profile(b - a + 1, profile)
        if V === nothing
            f, jm = rand(rng, rows)
            Dz = zeros(n, b - a + 1); Dz[z0row(W, f, jm), :] .= φ
        else
            Dz = V[:, rand(rng, axes(V, 2))] * φ'
        end
        ts[q] = z0_dir_move!(rng, P, LM, LC, s, bm, jac, sp, a, b, Dz; check=check, comp=comp)
    end
    ML.evaluate!(P, s)
    if check
        LC2 = ML.LocalCost(P, LM, s)
        errY = maximum(maximum(abs.(LC.Y[f] .- LC2.Y[f]); init=0.0) for f in eachindex(LC.Y) if f != P.E.fθ)
        errS = maximum(abs.(LC.S .- LC2.S)); errX = maximum(maximum(abs.(LC.X[f] .- LC2.X[f])) for f in eachindex(LC.X))
        @info "z0 block cache check" errY errS errX
    end
    return s, ts
end

# ---------------------------------------------------------------------------
# non-centered regime block move
# ---------------------------------------------------------------------------

"Draw m[a..b] from p(m_{a..b} | m_{a-1}, m_{b+1}, z0) (FFBS inside the window); returns (m', log q ratio = 0)."
function regime_window_draw(rng, W, ll, m, a::Int, b::Int)
    K, T = size(ll)
    lP = log.(W.Ptr)
    la = zeros(K, b - a + 1)
    for k in 1:K
        la[k, 1] = (a == 1 ? log(W.π0[k]) : lP[m[a-1], k]) + ll[k, a]
    end
    for τ in a+1:b, k in 1:K
        la[k, τ-a+1] = ML.logsumexp(la[:, τ-a] .+ lP[:, k]) + ll[k, τ]
    end
    m2 = copy(m)
    lw = b < T ? la[:, end] .+ lP[:, m[b+1]] : la[:, end]
    m2[b] = ML.sample_log(rng, lw)
    for τ in b-1:-1:a
        m2[τ] = ML.sample_log(rng, la[:, τ-a+1] .+ lP[:, m2[τ+1]])
    end
    return m2
end

"""
One non-centered regime move on the level-0 steps a..b. Returns (changed, accepted, ΔC·β_v, logJ).
"""
function regime_block_move!(rng, P, LM, LC, s, bm, jac, ll, a::Int, b::Int; comp::Bool=true, check::Bool=false)
    W = P.W
    L = s.ℓ
    m2 = regime_window_draw(rng, W, ll, s.m, a, b)
    m2 == s.m && return (false, false, 0.0, 0.0)
    x2 = ML.weather_at_level(W, L, m2, s.z0, s.ws)
    lay = W.lays[L+1]
    h0 = W.lays[1].h
    # finest steps that descend from a..b (the change is confined to them)
    fa = (a - 1) * (h0 ÷ lay.h) + 1; fb = b * (h0 ÷ lay.h)
    D = [Matrix(ML.fieldview(x2, lay, W.specs, f)[:, fa:fb] .- ML.fieldview(s.xs[end], lay, W.specs, f)[:, fa:fb]) for f in eachindex(W.specs)]
    d = ML.WDelta(L, fa, fb, D)
    while length(bm.wss) < fb - fa + 1
        push!(bm.wss, ML.MoveWS(P))
    end
    ML.prep_weather_move!(bm, P, LM, LC, [d], jac; comp=comp) === nothing && return (true, false, 0.0, 0.0)
    ΔC = ML.mv_eval!(bm, 1.0)
    lj = isempty(jac) ? 0.0 : ML.logjac(jac, 1.0, bm)
    logα = -P.βv * ΔC + lj
    if P.use_bias
        logα -= 0.0                      # the bias depends on (z0, k0) only: unchanged by a regime move
    end
    (isfinite(logα) && log(rand(rng)) < logα) || return (true, false, P.βv * ΔC, lj)
    L0 = s.L
    ML.commit_block!(bm, s.v, 1.0)
    s.m = m2
    s.xs[end] .= x2
    for ℓ in 1:L-1                       # intermediate levels (none for two-level hierarchies)
        s.xs[ℓ+1] .= ML.weather_at_level(W, ℓ, m2, s.z0, s.ws)
    end
    if check
        ML.evaluate!(P, s)
        err = (s.L - L0) - (-P.βv * ΔC)
        abs(err) > 1e-6 * max(1.0, abs(P.βv * ΔC)) && @warn "regime block: incremental and global cost differ" err ΔC
    end
    return (true, true, P.βv * ΔC, lj)
end

"""
    regime_block_sweep!(rng, P, LM, s; B, nmoves, check) -> (s, stats)

`nmoves` non-centered regime moves on windows of B level-0 steps at uniform positions.
stats = (proposed changes, accepted, mean |β ΔC| of proposals).
"""
function regime_block_sweep!(rng, P, LM, s; B::Int=1, nmoves::Int=50, check::Bool=false, comp::Bool=true)
    W = P.W
    T0 = W.lays[1].T
    s = copy(s)
    LC = ML.LocalCost(P, LM, s)
    pool = ML.BlockPool(P, B * (W.lays[1].h ÷ W.lays[s.ℓ+1].h) + 2)
    bm = ML.BlockMove(pool.wss, 0, Int[], P, LM, LC, Float64[], Float64[])
    jac = NTuple{5,Float64}[]
    ll = ML.regime_loglik(W, s.z0)
    nch, nacc, dcs = 0, 0, Float64[]
    for q in 1:nmoves
        a = rand(rng, 1:T0); b = min(a + B - 1, T0)
        ch, acc, dc, lj = regime_block_move!(rng, P, LM, LC, s, bm, jac, ll, a, b; comp=comp, check=check)
        nch += ch; nacc += acc
        ch && push!(dcs, abs(dc))
    end
    ML.evaluate!(P, s)
    if check
        LC2 = ML.LocalCost(P, LM, s)
        errY = maximum(maximum(abs.(LC.Y[f] .- LC2.Y[f]); init=0.0) for f in eachindex(LC.Y) if f != P.E.fθ)
        errS = maximum(abs.(LC.S .- LC2.S)); errX = maximum(maximum(abs.(LC.X[f] .- LC2.X[f])) for f in eachindex(LC.X))
        @info "regime block cache check" errY errS errX
    end
    return s, (nch, nacc, isempty(dcs) ? NaN : median(dcs))
end
