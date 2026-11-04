# Fine-level weather moves (docs/system/moves/weather.md).
#
# The fine chain moves the weather itself. The weather
# model is non-centered: coordinates (m, z0, w_1..w_L), with the anomalies of level
# ℓ an affine function of (z0, w_1..w_ℓ) given the regimes m. Three moves, each
# π-invariant at the finest level:
#
# * regimes, centered: with the anomalies x of every level held fixed, m is drawn
#   exactly by FFBS from p(m | x) (level-0 VAR emissions plus the detail densities
#   N(ξ; Aρ, LLᵀ) of all descendants of each level-0 step), and the innovations are
#   recomputed, w = L⁻¹(ξ − Aρ). Cost and bias do not change.
# * innovations w_ℓ (any level ℓ ≥ 1) and level-0 anomalies z0, non-centered: one
#   coordinate moves by t; the change of the finest anomalies is linear in t (exact
#   windowed propagation through the detail regressions), so the raw node weather
#   is affine in t and the cost is piecewise linear (data-map kinks as primary
#   breakpoints). Dispatchable generators follow the requirement with exact T2
#   (requirement-tracking kinks), so a weather move is not blocked by the balance.
#   The line conditional with the Gaussian prior (w: N(0,1); z0: the VAR terms of
#   its step and the next) is sampled exactly (truncated Gaussian segments); the T2
#   Jacobian and, at α = 1, the bias change enter an independence Metropolis step.
#   The bias is evaluated at the finest level, so every weather coordinate moves it;
#   its change is local to the steps of the move's window (bias cache, bias.jl).

"Windowed change of the standardized anomalies at weather level ℓ (0-based): steps a..b."
struct WDelta
    ℓ::Int
    a::Int
    b::Int
    D::Vector{Matrix{Float64}}     # per weather field: m_f(ℓ) × (b − a + 1)
end

"Detail group of coarse block j of the transfer from level ℓc to ℓc+1."
function blockgroup(W::WeatherModel, ℓc::Int, j::Int, m)
    lc = W.lays[ℓc+1]
    h0 = W.lays[1].h
    τ0 = ancestor(j, lc.h, h0)
    return groupindex(m[τ0], season(τ0, h0), phase(j, lc.h), W.details[ℓc+1].P)
end

"Decode the flat index q of the innovations of level ℓ into (f, part, i, b, j); part 1: time details, 2: new modes."
function innovation_coord(W::WeatherModel, ℓ::Int, q::Int)
    tr, lc, lf = W.trs[ℓ], W.lays[ℓ], W.lays[ℓ+1]
    r = tr.r
    f = something(findlast(o -> o < q, tr.doff), 1)
    o = q - tr.doff[f] - 1
    mc, mf = lc.m[f], lf.m[f]
    if o < tr.dT[f]
        i = o % mc + 1; o ÷= mc
        b = o % (r - 1) + 1
        return (f, 1, i, b, o ÷ (r - 1) + 1)
    end
    o -= tr.dT[f]
    ns = mf - mc
    i = o % ns + 1; o ÷= ns
    return (f, 2, i, o % r + 1, o ÷ r + 1)
end

"Change of the level-ℓ anomalies per unit change of innovation (f, part, i, b, j) of level ℓ."
function wdelta_innovation(W::WeatherModel, m, ℓ::Int, f::Int, part::Int, i::Int, b::Int, j::Int)
    D, lc, lf, tr = W.details[ℓ], W.lays[ℓ], W.lays[ℓ+1], W.trs[ℓ]
    r, H = tr.r, tr.H
    g = blockgroup(W, ℓ - 1, j, m)
    Dm = [zeros(lf.m[ff], r) for ff in eachindex(W.specs)]
    mc = lc.m[f]
    if part == 1
        L = view(D.LT[f], :, :, i, g)
        for o in 1:r, k in b:r-1
            Dm[f][i, o] += H[o, k] * L[k, b]
        end
    else
        L = view(D.LS[f], :, :, i, g)
        for o in b:r
            Dm[f][mc+i, o] = L[o, b]
        end
    end
    return WDelta(ℓ, (j - 1) * r + 1, j * r, Dm)
end

"Propagate a level-ℓ change to level ℓ+1 (linear part of the detail regressions; innovations fixed)."
function propagate(W::WeatherModel, m, d::WDelta)
    ℓ = d.ℓ
    D, tr, lc, lf = W.details[ℓ+1], W.trs[ℓ+1], W.lays[ℓ+1], W.lays[ℓ+2]
    r, H, Tc = tr.r, tr.H, lc.T
    ja, jb = max(d.a - 1, 1), min(d.b + 1, Tc)
    n = (jb - ja + 1) * r
    out = Matrix{Float64}[]
    ΔZ(f, i, jj) = (d.a <= jj <= d.b) ? d.D[f][i, jj-d.a+1] : 0.0
    for (f, s) in enumerate(W.specs)
        mc, mf = lc.m[f], lf.m[f]
        Dn = zeros(mf, n)
        for j in ja:jb
            g = blockgroup(W, ℓ, j, m)
            t0 = (j - ja) * r
            for i in 1:mc
                z1, z2, z3 = ΔZ(f, i, max(j - 1, 1)), ΔZ(f, i, j), ΔZ(f, i, min(j + 1, Tc))
                (z1 == 0.0 && z2 == 0.0 && z3 == 0.0) && continue
                for o in 1:r
                    acc = z2
                    for k in 1:r-1
                        ξ = D.AT[f][k, 2, i, g] * z1 + D.AT[f][k, 3, i, g] * z2 + D.AT[f][k, 4, i, g] * z3
                        acc += H[o, k] * ξ
                    end
                    Dn[i, t0+o] = acc
                end
            end
            z = mc >= 1 ? ΔZ(f, 1, j) : 0.0
            if z != 0.0
                for ii in 1:mf-mc, a in 1:r
                    Dn[mc+ii, t0+a] = D.AS[f][a, 2, ii, g] * z
                end
            end
        end
        push!(out, Dn)
    end
    return WDelta(ℓ + 1, (ja - 1) * r + 1, jb * r, out)
end

"The chain of windowed changes from level d.ℓ to level L."
function wdelta_chain(W::WeatherModel, m, d::WDelta, L::Int)
    ch = [d]
    while ch[end].ℓ < L
        push!(ch, propagate(W, m, ch[end]))
    end
    return ch
end

"Node weather changes at the finest level per step: (field, support position, dX) lists, steps d.a..d.b."
function node_changes(W::WeatherModel, d::WDelta)
    lay = W.lays[d.ℓ+1]
    out = [NTuple{3,Float64}[] for _ in d.a:d.b]
    for (f, s) in enumerate(W.specs)
        U = cols(s.B, lay.m[f])
        Dm = d.D[f]
        any(!iszero, Dm) || continue
        S = size(U, 1)
        acc = zeros(S)
        rv, nz = rowvals(U), nonzeros(U)
        for q in axes(Dm, 2)
            fill!(acc, 0.0)
            for j in axes(Dm, 1)
                c = Dm[j, q]
                c == 0.0 && continue
                for p in nzrange(U, j)
                    acc[rv[p]] += nz[p] * c
                end
            end
            for pos in 1:S
                a = W.scale[f] * acc[pos]
                abs(a) > 1e-14 && push!(out[q], (Float64(f), Float64(pos), a))
            end
        end
    end
    return out
end

"Apply t·Δ to the anomalies of the state (levels d.ℓ..L) for a chain of changes."
function apply_wdelta!(s::State, W::WeatherModel, ch, t::Float64)
    for d in ch
        x = s.xs[d.ℓ+1]
        lay = W.lays[d.ℓ+1]
        for f in eachindex(W.specs)
            Z = fieldview(x, lay, W.specs, f)
            @views Z[:, d.a:d.b] .+= t .* d.D[f]
        end
    end
end

"VAR terms of log p(z0 | m) that involve step τ0: the emissions of τ0 and τ0+1."
function z0_local_logp(W::WeatherModel, m, Z::AbstractMatrix, τ0::Int)
    n, T0 = size(Z)
    val = 0.0
    for τ in τ0:min(τ0 + 1, T0)
        if τ == 1
            val -= 0.5 * dot(view(Z, :, 1), W.Σ0inv * view(Z, :, 1))
        else
            k = m[τ]
            B = W.Bm[k]
            c, cs, sn = harmonics(τ, T0)
            e = Z[:, τ] .- B[:, 1:n] * Z[:, τ-1] .- c .* B[:, n+1] .- cs .* B[:, n+2] .- sn .* B[:, n+3]
            val -= 0.5 * dot(e, W.Σinv[k] * e)
        end
    end
    return val
end

"Quadratic (κ, b) of the potential −log p(z0 + t e_o) − const along coordinate row o at step τ0."
function z0_quad(W::WeatherModel, m, z0, o::Int, τ0::Int)
    Z = z0matrix(z0, W.lays[1], W.specs)
    U(t) = (Z[o, τ0] += t; v = -z0_local_logp(W, m, Z, τ0); Z[o, τ0] -= t; v)
    u0, u1, um = U(0.0), U(1.0), U(-1.0)
    return (u1 + um - 2u0, 0.5 * (u1 - um))
end

# ---------------------------------------------------------------------------
# regimes (centered FFBS) and innovations from the anomalies
# ---------------------------------------------------------------------------

"Detail log densities per regime and level-0 step: ll[k, τ0] += Σ log N(ξ; Aρ, LLᵀ) of its descendants."
function detail_loglik!(ll::Matrix{Float64}, W::WeatherModel, xs, L::Int)
    K = W.K
    h0 = W.lays[1].h
    for ℓ in 1:L
        D, tr, lc, lf = W.details[ℓ], W.trs[ℓ], W.lays[ℓ], W.lays[ℓ+1]
        r = tr.r
        _, ξ = split_state(tr, lc, lf, W.specs, xs[ℓ+1])
        xc = xs[ℓ]
        for (f, s) in enumerate(W.specs)
            Zc = fieldview(xc, lc, W.specs, f)
            ξT, ξS = detail_views(ξ, tr, lc, lf, W.specs, f)
            mc, mf = lc.m[f], lf.m[f]
            e = zeros(r)
            for j in 1:lc.T
                τ0 = ancestor(j, lc.h, h0)
                sea, ph = season(τ0, h0), phase(j, lc.h)
                for k in 1:K
                    g = groupindex(k, sea, ph, D.P)
                    acc = 0.0
                    if r > 1
                        for i in 1:mc
                            ρ = (1.0, Zc[i, max(j - 1, 1)], Zc[i, j], Zc[i, min(j + 1, lc.T)])
                            A = view(D.AT[f], :, :, i, g); Lm = view(D.LT[f], :, :, i, g)
                            acc += gauss_detail(e, view(ξT, i, :, j), A, Lm, ρ, r - 1)
                        end
                    end
                    for i in 1:mf-mc
                        ρ = (1.0, Zc[1, j], 0.0, 0.0)
                        A = view(D.AS[f], :, :, i, g); Lm = view(D.LS[f], :, :, i, g)
                        acc += gauss_detail(e, view(ξS, i, :, j), A, Lm, ρ, r)
                    end
                    ll[k, τ0] += acc
                end
            end
        end
    end
    return ll
end

"log N(ξ; Aρ, LLᵀ) up to the constant (lower-triangular solve)."
@inline function gauss_detail(e, ξ, A, Lm, ρ, d)
    val = 0.0
    @inbounds for a in 1:d
        acc = ξ[a]
        for c in 1:size(A, 2)
            acc -= A[a, c] * ρ[c]
        end
        for b in 1:a-1
            acc -= Lm[a, b] * e[b]
        end
        e[a] = acc / Lm[a, a]
        val -= 0.5 * e[a]^2 + log(Lm[a, a])
    end
    return val
end

"Innovations of level ℓ from the anomalies and the regimes: w = L⁻¹(ξ − Aρ)."
function innovations_from_x(W::WeatherModel, m, xc, xf, ℓ::Int)
    D, tr, lc, lf = W.details[ℓ], W.trs[ℓ], W.lays[ℓ], W.lays[ℓ+1]
    r = tr.r
    h0 = W.lays[1].h
    _, ξ = split_state(tr, lc, lf, W.specs, xf)
    w = zeros(tr.n)
    for (f, s) in enumerate(W.specs)
        Zc = fieldview(xc, lc, W.specs, f)
        ξT, ξS = detail_views(ξ, tr, lc, lf, W.specs, f)
        wT, wS = detail_views(w, tr, lc, lf, W.specs, f)
        mc, mf = lc.m[f], lf.m[f]
        e = zeros(r)
        for j in 1:lc.T
            g = blockgroup(W, ℓ - 1, j, m)
            if r > 1
                for i in 1:mc
                    ρ = (1.0, Zc[i, max(j - 1, 1)], Zc[i, j], Zc[i, min(j + 1, lc.T)])
                    gauss_detail(e, view(ξT, i, :, j), view(D.AT[f], :, :, i, g), view(D.LT[f], :, :, i, g), ρ, r - 1)
                    wT[i, :, j] .= e[1:r-1]
                end
            end
            for i in 1:mf-mc
                gauss_detail(e, view(ξS, i, :, j), view(D.AS[f], :, :, i, g), view(D.LS[f], :, :, i, g), (1.0, Zc[1, j], 0.0, 0.0), r)
                wS[i, :, j] .= e
            end
        end
    end
    return w
end

"""
Centered regime update at the level of `s`: m ~ p(m | x_0..x_L) by FFBS (level-0
emissions plus detail densities), then the innovations are recomputed.
"""
function fine_regime_step!(rng, P::Problem, s::State)
    W = P.W
    L = s.ℓ
    ll = regime_loglik(W, s.z0)
    detail_loglik!(ll, W, s.xs, L)
    s.m = ffbs(rng, ll, W.Ptr, W.π0)
    for ℓ in 1:L
        s.ws[ℓ] = innovations_from_x(W, s.m, s.xs[ℓ], s.xs[ℓ+1], ℓ)
    end
    return s
end

# ---------------------------------------------------------------------------
# non-centered coordinate moves of z0 and of the innovations
# ---------------------------------------------------------------------------

"""
Prepare the block move of a weather change chain at the finest level: weather
entries per step and exact T2 compensation (requirement-tracking kinks) at every
node whose requirement changes. Returns the BlockMove (or nothing).
"""
function prep_weather_move!(bm::BlockMove, P::Problem, LM::LocalModel, LC::LocalCost, ch, jac; comp::Bool=true, κmin=1.0)
    W = P.W
    d = ch[end]
    T = LM.T
    nch = node_changes(W, d)
    n = d.b - d.a + 1
    length(bm.wss) >= n || return nothing
    bm.n = n
    resize!(bm.τs, n)
    empty!(bm.bps)
    empty!(jac)
    any(!isempty, nch) || return nothing
    E = P.E
    for q in 1:n
        τ = d.a + q - 1
        ws = bm.wss[q]
        clear!(ws)
        ws.follow = true
        bm.τs[q] = τ
        nodes = Int[]
        for (ff, pp, dX) in nch[q]
            f, pos = Int(ff), Int(pp)
            push!(ws.wf, f); push!(ws.wpos, pos); push!(ws.wX0, LC.X[f][pos, τ]); push!(ws.wdX, dX)
            push!(nodes, E.wspecs[f].B.support[pos])
        end
        if comp
            for nd in unique(nodes)
                prorata!(ws, jac, P, LM, LC, nd, 1.0, τ; κmin=κmin, dyn=true, skipweather=true)
            end
        end
        finalize!(ws, P, LM, LC, τ)
        append!(bm.bps, ws.bps)
    end
    return bm
end

"""
Change of b̂ (of ĝ: Ē ĝ and ĥ do not depend on x) when the weather move `bm` is taken to t:
per step of its window, D_t and R_t change by the data-mapped weather entries. With
`commit = true` the cache follows the move (per-step values only; `bias_resum!` after).
"""
function bias_delta_weather(BC::BiasCache, b::BiasModel, E::EnergyModel, bm::BlockMove, t::Float64; commit::Bool=false)
    D̄ = b.Dbar
    fd = BC.firm / D̄
    dg = 0.0
    @inbounds for q in 1:bm.n
        ws = bm.wss[q]
        isempty(ws.wf) && continue
        τ = bm.τs[q]
        dD, dR = 0.0, 0.0
        for e in eachindex(ws.wf)
            f, pos = ws.wf[e], ws.wpos[e]
            X0 = ws.wX0[e]
            X1 = X0 + t * ws.wdX[e]
            if f == 1                                   # load
                dD += max(X1, 0.0) - max(X0, 0.0)
            else                                        # availability, times the non-gas capacity on it
                a1 = clamp(X1, 0.0, 1.0)
                dR += BC.Kw[f][pos] * (a1 - clamp(X0, 0.0, 1.0))
                if commit
                    for (gi, g) in enumerate(E.gens)
                        (g.wf == f && g.carrier != "gas") && (BC.At[gi][τ, pos] = a1)
                    end
                end
            end
        end
        D1, R1 = BC.Dx[τ] + dD, BC.Rx[τ] + dR
        u1 = (D1 - R1) / D̄
        f1, f2 = tabval(BC.tab1, u1), tabval(BC.tab2, u1 - fd)
        dg += f1 + f2 - BC.F1x[τ] - BC.F2x[τ]
        commit && (BC.Dx[τ] = D1; BC.Rx[τ] = R1; BC.U1x[τ] = u1; BC.F1x[τ] = f1; BC.F2x[τ] = f2)
    end
    return b.Δt * dg
end

"""
One exact weather coordinate move (innovation of level ℓ ≥ 1, or z0 with ℓ = 0). Returns t.
At α = 1 (`BC` the bias cache of the state) the exact draw of the bias-free line
conditional is followed by a Metropolis correction with e^{−Δb̂}; on acceptance the cache
follows the move (per-step values; the sums are recomputed by the caller).
"""
function weather_coord_move!(rng, P::Problem, LM::LocalModel, LC::LocalCost, s::State, bm::BlockMove, jac,
                             secprev, ℓ::Int, q::Int; comp::Bool=true, BC=nothing)
    W = P.W
    L = s.ℓ
    if ℓ == 0
        lay = W.lays[1]
        T0 = lay.T
        n0_ = n0(W)
        o = (q - 1) % n0_ + 1                # row of the stacked z0 matrix
        τ0 = (q - 1) ÷ n0_ + 1
        f = 1
        rows = 0
        while rows + lay.m[f] < o
            rows += lay.m[f]; f += 1
        end
        jm = o - rows
        Dm = [zeros(lay.m[ff], 1) for ff in eachindex(W.specs)]
        Dm[f][jm, 1] = 1.0
        d0 = WDelta(0, τ0, τ0, Dm)
        quad = z0_quad(W, s.m, s.z0, o, τ0)
        iz = lay.off[f] + (τ0 - 1) * lay.m[f] + jm
    else
        f, part, i, b, j = innovation_coord(W, ℓ, q)
        d0 = wdelta_innovation(W, s.m, ℓ, f, part, i, b, j)
        quad = (1.0, s.ws[ℓ][q])
    end
    ch = wdelta_chain(W, s.m, d0, L)
    prep_weather_move!(bm, P, LM, LC, ch, jac; comp=comp) === nothing && return 0.0
    t = sample_line!(rng, bm, P.βv, secprev; quad=quad)
    t == 0.0 && return 0.0
    logα = isempty(jac) ? 0.0 : logjac(jac, t, bm)
    BC === nothing || (logα -= bias_delta_weather(BC, P.bias, P.E, bm, t))
    (isfinite(logα) && log(rand(rng)) < logα) || return 0.0
    commit_block!(bm, s.v, t)
    BC === nothing || bias_delta_weather(BC, P.bias, P.E, bm, t; commit=true)
    if ℓ == 0
        s.z0[iz] += t
        s.xs[1] === s.z0 || (s.xs[1][iz] += t)
        apply_wdelta!(s, W, ch[2:end], t)
    else
        s.ws[ℓ][q] += t
        apply_wdelta!(s, W, ch, t)
    end
    return t
end

"""
    fine_weather_sweep!(rng, P, LM, s; frac, frac0, comp, pools) -> (s, stats)

Weather moves at the level of `s` (complete spatial level): the centered regime
update, then `frac` of the innovation coordinates of every level ℓ = 1..L and `frac0`
of the level-0 coordinates (random draws), each an exact compensated line move with,
at α = 1, the Metropolis bias correction from the bias cache `BC` (built from `s` if
not given; it follows the accepted moves). Moves in coarse blocks j with equal j mod M
at their level are independent (their windows at the finest level are separated; M
from the spread of the windows; the bias change of a move is local to its window) and
run in parallel. Then `zblocks` joint block moves of z0 in expectation (weather_blocks.jl),
in sequence, of the lengths whose window spans at most `zmaxspan` finest-level steps.
"""
function fine_weather_sweep!(rng, P::Problem, LM::LocalModel, s::State; frac::Float64=0.05, frac0::Float64=0.05,
                             comp::Bool=true, pools=nothing, LC=nothing, BC=nothing, zblocks::Float64=0.0,
                             zmaxspan::Int=600)
    W = P.W
    L = s.ℓ
    @assert LM.ℓ == L
    s = copy(s)
    fine_regime_step!(rng, P, s)
    LC = LC === nothing ? LocalCost(P, LM, s) : LC
    BC = (P.use_bias && BC === nothing) ? BiasCache(P, s) : BC
    nth = Threads.maxthreadid()
    # pool size: the longest window at the finest level (z0 at a level-0 step)
    lays = W.lays
    span = 1
    for ℓ in 1:L
        span = (span + 2) * W.trs[ℓ].r
    end
    if pools === nothing
        pools = [BlockPool(P, span) for _ in 1:nth]
    elseif isempty(pools) || length(pools[1].wss) < span
        empty!(pools); append!(pools, [BlockPool(P, span) for _ in 1:nth])
    end
    stats = zeros(L + 1, 2)
    for ℓ in vcat(collect(L:-1:1), 0)
        ncoord = ℓ == 0 ? lays[1].n : W.trs[ℓ].n
        nm = round(Int, (ℓ == 0 ? frac0 : frac) * ncoord)
        nm == 0 && continue
        qs = rand(rng, 1:ncoord, nm)
        blk(q) = ℓ == 0 ? (q - 1) ÷ n0(W) + 1 : innovation_coord(W, ℓ, q)[5]
        byblock = Dict{Int,Vector{Int}}()
        for q in qs
            push!(get!(byblock, blk(q), Int[]), q)
        end
        # windows at the finest level: ± one step of every level the change passes through
        hblk = ℓ == 0 ? lays[1].h : lays[ℓ].h
        spread = sum((lays[l+1].h for l in ℓ:L-1); init=0)
        M = 2 * cld(spread, hblk) + 2
        for c in shuffle(rng, collect(0:M-1))
            blocks = [j for j in keys(byblock) if j % M == c]
            isempty(blocks) && continue
            seeds = rand(rng, UInt64, length(blocks))
            cnt = zeros(Int, length(blocks), 2)
            pch = Channel{Any}(length(pools))       # workspaces handed out explicitly (no reliance on threadid)
            foreach(p_ -> put!(pch, p_), pools)
            Threads.@threads for bi in eachindex(blocks)
                local r = Xoshiro(seeds[bi])
                local pool = take!(pch)
                try                                   # task-local names (the sequential branch uses the same ones)
                    local bmt = BlockMove(pool.wss, 0, Int[], P, LM, LC, Float64[], Float64[])
                    local jact, spt = NTuple{5,Float64}[], Float64[]
                    for q in byblock[blocks[bi]]
                        local tt = weather_coord_move!(r, P, LM, LC, s, bmt, jact, spt, ℓ, q; comp=comp, BC=BC)
                        cnt[bi, 1] += 1; cnt[bi, 2] += tt != 0.0
                    end
                finally
                    put!(pch, pool)
                end
            end
            stats[ℓ+1, :] .+= vec(sum(cnt; dims=1))
        end
    end
    # joint low-frequency block moves of z0 (weather_blocks.jl), `zblocks` per sweep in expectation
    nb = zblocks > 0 ? floor(Int, zblocks) + (rand(rng) < zblocks - floor(zblocks)) : 0
    nb > 0 && z0_block_sweep!(rng, P, LM, s, nb; LC=LC, BC=BC, pool=pools[1], comp=comp, maxspan=zmaxspan)
    BC === nothing || bias_resum!(BC, P.bias)
    return evaluate!(P, s), stats
end
