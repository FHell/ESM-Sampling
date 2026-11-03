# Coordinate Gibbs base kernel for level 0 (alternative to reflective HMC).
#
# With all constraints priced (regularized Ω) the level-0 potential
#
#   U = β_k capex(k) + β_v Σ_τ cost_τ(k, v, x) + b̂(z0, k) − log p(z0 | m)
#
# is a sum of per-step terms term_τ = β_v cost_τ + ĝ_τ − log p_τ plus terms in k
# only. A coordinate of a step field at τ enters term_τ only, an instant field
# (state of charge) at τ enters term_{τ-1} and term_τ, a weather coordinate at τ
# enters term_τ and term_{τ+1} (AR(1)). Each 1-D conditional is therefore
# evaluated locally, and steps with |τ − τ'| ≥ 5 can be updated in parallel.
#
# Coordinates are updated by slice sampling (stepping out + shrinkage), which is
# exact for the conditional for any width. The sweep visits the moves (colour
# classes, the k coordinates, FFBS, carrier scaling) in a uniformly random
# order, and coordinates within a block in random order, so the kernel is a
# mixture over orderings of reversible updates and is itself π₀-reversible, as
# MLDA requires of the coarse kernel.

const NCOLOURS = 5

struct StepWS
    y::Vector{Vector{Float64}}     # node values per v field at τ
    y1::Vector{Vector{Float64}}    # instant fields at τ + 1
    x::Vector{Vector{Float64}}     # raw node weather at τ
    r::Vector{Float64}
    re::Vector{Float64}
    tm::Vector{Float64}
    zτ::Vector{Float64}
    zprev::Vector{Float64}
    e::Vector{Float64}
    Se::Vector{Float64}
    φ::Vector{Float64}
    d1::Vector{Float64}
    d2::Vector{Float64}
end

function StepWS(P::Problem)
    E = P.E
    N, n, nb = E.sys.N, n0(P.W), nθ(P.bias)
    StepWS([zeros(nsupport(s.B)) for s in E.vspecs], [zeros(nsupport(s.B)) for s in E.vspecs],
           [zeros(nsupport(s.B)) for s in E.wspecs], zeros(N), zeros(N), zeros(N), zeros(n), zeros(n), zeros(n),
           zeros(n), zeros(nb), zeros(nb), zeros(nb))
end

"Quantities fixed during a sweep over (v, z0): clamped capacities and firm capacity."
struct SweepCtx
    kc::Vector{Vector{Float64}}
    kgen::Vector{Vector{Float64}}
    kst::Vector{Vector{Float64}}
    firm::Float64
end

function SweepCtx(P::Problem, k; ℓ::Int=0)
    E = P.E
    kc, _ = node_capacities(E, P.lk[ℓ+1], k)
    kgen = [g.kf > 0 ? kc[g.kf] : g.kfix for g in E.gens]
    kst = [s.kf > 0 ? kc[s.kf] : s.kfix for s in E.stors]
    firm = sum(sum(kgen[gi]) for (gi, g) in enumerate(E.gens) if g.carrier == "gas"; init=0.0) +
           sum(sum(x) for x in kst; init=0.0)
    return SweepCtx(kc, kgen, kst, firm)
end

@inline function colvals!(out, U::SparseMatrixCSC, Z, m, τ)
    fill!(out, 0.0)
    rv, nz = rowvals(U), nonzeros(U)
    @inbounds for j in 1:m
        c = Z[j, τ]
        c == 0.0 && continue
        for q in nzrange(U, j)
            out[rv[q]] += nz[q] * c
        end
    end
    return out
end

"""
term_τ = β_v cost_τ + ĝ_τ − log p_τ(z0 | m) at level 0. At level ℓ ≥ 1 (`x` the
standardized level-ℓ weather) only β_v cost_τ: the weather is held fixed there.
"""
step_term(P::Problem, ctx::SweepCtx, ws::StepWS, zv, z0, m, τ::Int) = step_term(P, 0, ctx, ws, zv, z0, z0, m, τ)

function step_term(P::Problem, ℓ::Int, ctx::SweepCtx, ws::StepWS, zv, x, z0, m, τ::Int)
    E, W = P.E, P.W
    sys = E.sys
    lv, lw = P.lv[ℓ+1], W.lays[ℓ+1]
    T, Δt = lv.T, Float64(lv.h)
    τ1 = mod(τ, T) + 1
    vs = E.vspecs
    for (f, s) in enumerate(vs)
        U, m_ = s.B.U, lv.m[f]
        Z = fieldview(zv, lv, vs, f)
        colvals!(ws.y[f], U, Z, m_, τ)
        s.kind == INSTANT && colvals!(ws.y1[f], U, Z, m_, τ1)
    end
    for (f, s) in enumerate(W.specs)
        U, m_ = s.B.U, lw.m[f]
        Z = fieldview(x, lw, W.specs, f)
        Xb = fieldview(W.xbar[ℓ+1], lw, W.specs, f)
        xf = ws.x[f]
        fill!(xf, 0.0)
        rv, nz = rowvals(U), nonzeros(U)
        @inbounds for j in 1:m_
            c = Xb[j, τ] + W.scale[f] * Z[j, τ]
            for q in nzrange(U, j)
                xf[rv[q]] += nz[q] * c
            end
        end
    end
    c_viol, κ0 = E.c_viol, E.κ0
    r = ws.r
    @inbounds for n in 1:sys.N
        r[n] = max(ws.x[1][n], 0.0)
    end
    cost = 0.0
    θ = ws.y[E.fθ]
    θpen = !ishard(E, lv, vs, E.fθ)
    LT = E.lines
    @inbounds for e in eachindex(sys.line_s)
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        fl = sys.line_b[e] * (θ[a] - θ[b])
        ov, loss = line_eval(LT, e, sys.line_s[e], fl)
        r[a] += fl + 0.5 * loss
        r[b] += -fl + 0.5 * loss
        θpen && (cost += c_viol * Δt * ov)
    end
    if E.flink > 0
        y = ws.y[E.flink]
        hard = ishard(E, lv, vs, E.flink)
        @inbounds for l in eachindex(sys.link_cap)
            yc = hard ? y[l] : clamp(y[l], -1.0, 1.0)
            u = sys.link_cap[l] * yc
            j0, j1 = link_inj(E.link_eta[l], u)
            r[sys.link_bus0[l]] += j0
            r[sys.link_bus1[l]] += j1
            hard || (cost += c_viol * Δt * sys.link_cap[l] * abs(y[l] - yc))
        end
    end
    R = 0.0
    @inbounds for (gi, g) in enumerate(E.gens)
        fixed = g.vf == 0                       # renewables as negative load: full output
        y = fixed ? ws.r : ws.y[g.vf]
        kv = ctx.kgen[gi]
        hard = fixed || ishard(E, lv, vs, g.vf)
        for i in eachindex(g.bus)
            yc = fixed ? 1.0 : (hard ? y[i] : clamp(y[i], 0.0, 1.0))
            A = g.wf > 0 ? clamp(ws.x[g.wf][i], 0.0, 1.0) : g.avail[i]
            p = A * kv[i] * yc
            cost += Δt * gen_vint(g, i, A, kv[i], yc)[1]
            r[g.bus[i]] -= p
            hard || (cost += c_viol * Δt * (kv[i] + κ0) * abs(y[i] - yc))
            g.ucf > 0 && (cost += uc_cost(g, i, A, kv[i], kv[i] + κ0, Δt, c_viol, yc, ws.y[g.ucf][i], ws.y1[g.ucf][i]))
            g.carrier == "gas" || (R += A * kv[i])
        end
    end
    @inbounds for (si, s) in enumerate(E.stors)
        if issoc(E)
            ye, ye1 = ws.y[s.fe], ws.y1[s.fe]
            kv = ctx.kst[si]
            for i in eachindex(s.bus)
                c, _, j = soc_cost(s, i, kv[i], kv[i] + κ0, Δt, c_viol, ye[i], ye1[i])
                cost += c
                r[s.bus[i]] += j
            end
            continue
        end
        yp, ym, ye, ye1 = ws.y[s.fp], ws.y[s.fm], ws.y[s.fe], ws.y1[s.fe]
        kv = ctx.kst[si]
        hard = ishard(E, lv, vs, s.fp)
        for i in eachindex(s.bus)
            pc, mc = hard ? (yp[i], ym[i]) : (clamp(yp[i], 0.0, 1.0), clamp(ym[i], 0.0, 1.0))
            cost += Δt * s.mc[i] * kv[i] * (pc + mc)
            r[s.bus[i]] += kv[i] * (pc - mc)
            if !hard
                w = kv[i] + κ0
                ec, ec1 = clamp(ye[i], 0.0, 1.0), clamp(ye1[i], 0.0, 1.0)
                cost += c_viol * Δt * w * (abs(yp[i] - pc) + abs(ym[i] - mc))
                cost += c_viol * w * s.hours[i] * abs(ye[i] - ec)
                gd = s.hours[i] * (ec1 - ec) / Δt - s.eta_in[i] * pc + mc / s.eta_out[i] - s.qt[i]
                gd > 0 && (cost += c_viol * Δt * w * gd)
            end
        end
    end
    # priced balance on the level's load-mode projection of the residual
    Ub = E.wspecs[1].B
    mr = modes_at(Ub, lv.M)
    D = 0.0
    @inbounds for n in 1:sys.N
        D += max(ws.x[1][n], 0.0)
    end
    re = ws.re
    project!(re, Ub, mr, r, ws.tm)
    @inbounds for n in 1:sys.N
        rn = re[n]
        cost += Δt * (rn > 0 ? E.c_shed * rn : -E.c_dump * rn)
    end
    val = P.βv * cost
    ℓ == 0 || return val
    # bias: the per-step part of ĝ (the k-only parts of b̂ are constant here)
    if P.use_bias
        b = P.bias
        φ, d1, d2 = ws.φ, ws.d1, ws.d2
        u1 = (D - R) / b.Dbar
        bias_basis!(φ, d1, d2, b, u1, u1 - ctx.firm / b.Dbar)
        val += Δt * dot(b.θ, φ)
    end
    # weather prior, AR(1) term of step τ
    zτ = ws.zτ
    o = 0
    for (f, s) in enumerate(W.specs)
        Z = fieldview(z0, lw, W.specs, f)
        for j in 1:lw.m[f]
            o += 1
            zτ[o] = Z[j, τ]
        end
    end
    if τ == 1
        mul!(ws.Se, W.Σ0inv, zτ)
        val += 0.5 * dot(zτ, ws.Se)
    else
        zp = ws.zprev
        o = 0
        for (f, s) in enumerate(W.specs)
            Z = fieldview(z0, lw, W.specs, f)
            for j in 1:lw.m[f]
                o += 1
                zp[o] = Z[j, τ-1]
            end
        end
        k = m[τ]
        B = W.Bm[k]
        n = length(zτ)
        c, cs, sn = harmonics(τ, lw.T)
        e = ws.e
        mul!(e, view(B, :, 1:n), zp)
        @inbounds for a in 1:n
            e[a] = zτ[a] - e[a] - c * B[a, n+1] - cs * B[a, n+2] - sn * B[a, n+3]
        end
        mul!(ws.Se, W.Σinv[k], e)
        val += 0.5 * dot(e, ws.Se)
    end
    return val
end

"Neal's slice sampler (stepping out, shrinkage) on A[i] for the log density lf()."
function slice_coord!(rng, A, i, lf, w; maxsteps=16)
    x0 = A[i]
    logy = lf() - randexp(rng)
    L = x0 - w * rand(rng)
    R = L + w
    J = floor(Int, maxsteps * rand(rng))
    K = maxsteps - 1 - J
    A[i] = L
    while J > 0 && lf() > logy
        L -= w; A[i] = L; J -= 1
    end
    A[i] = R
    while K > 0 && lf() > logy
        R += w; A[i] = R; K -= 1
    end
    for _ in 1:200
        x1 = L + rand(rng) * (R - L)
        A[i] = x1
        lf() > logy && return abs(x1 - x0)
        x1 < x0 ? (L = x1) : (R = x1)
    end
    A[i] = x0
    return 0.0
end

mutable struct GibbsState
    wv::Vector{Float64}      # slice widths per v field
    wz::Vector{Float64}      # per weather field
    wk::Vector{Float64}      # per k coordinate
    adapt::Bool
    nmove::Vector{Float64}   # accumulators for width adaptation (per v field, then weather fields)
    smove::Vector{Float64}
    wss::Vector{StepWS}
    sweeps::Int
end

function GibbsState(P::Problem)
    nv, nw = length(P.E.vspecs), length(P.W.specs)
    wv = fill(1e-3, nv)
    wv[P.E.fθ] = 10.0
    GibbsState(wv, fill(0.05, nw), max.(1e-3 .* P.bias.kscale, 1.0), false, zeros(nv + nw), zeros(nv + nw),
               [StepWS(P) for _ in 1:Threads.maxthreadid()], 0)
end

"""
Update the coordinates of step τ in random order: (v, z0) at level 0, v only at
level ℓ ≥ 1 (with the level-ℓ weather `x` fixed).
"""
function update_step!(rng, P::Problem, G::GibbsState, ctx, ws, zv, z0, m, τ, mv, sv; ℓ::Int=0, x=z0, weather::Bool=true,
                      vmoves::Bool=true)
    E, W = P.E, P.W
    lv, lw = P.lv[ℓ+1], W.lays[ℓ+1]
    T = lv.T
    coords = Tuple{Int,Int,Int}[]   # (kind, field, index): kind 1 = v, 2 = weather
    if vmoves
        for (f, s) in enumerate(E.vspecs), j in 1:lv.m[f]
            push!(coords, (1, f, lv.off[f] + (τ - 1) * lv.m[f] + j))
        end
    end
    if ℓ == 0 && weather
        for (f, s) in enumerate(W.specs), j in 1:lw.m[f]
            push!(coords, (2, f, lw.off[f] + (τ - 1) * lw.m[f] + j))
        end
    end
    shuffle!(rng, coords)
    τm = mod(τ - 2, T) + 1
    term(t) = step_term(P, ℓ, ctx, ws, zv, x, z0, m, t)
    for (kind, f, i) in coords
        if kind == 1
            if E.vspecs[f].kind == INSTANT
                lf = () -> -(term(τm) + term(τ))
            else
                lf = () -> -term(τ)
            end
            d = slice_coord!(rng, zv, i, lf, G.wv[f])
            mv[f] += 1; sv[f] += d
        else
            lf = τ < T ? () -> -(step_term(P, ctx, ws, zv, z0, m, τ) + step_term(P, ctx, ws, zv, z0, m, τ + 1)) :
                         () -> -step_term(P, ctx, ws, zv, z0, m, τ)
            d = slice_coord!(rng, z0, i, lf, G.wz[f])
            o = length(E.vspecs)
            mv[o+f] += 1; sv[o+f] += d
        end
    end
end

"One colour class: all steps τ ≡ c (mod NCOLOURS), in parallel."
function update_colour!(rng, P::Problem, G::GibbsState, ctx, zv, z0, m, c; ℓ::Int=0, x=z0, weather::Bool=true,
                        vmoves::Bool=true)
    T = P.lv[ℓ+1].T
    steps = collect(c:NCOLOURS:T)
    seeds = rand(rng, UInt64, length(steps))
    nacc = length(G.nmove)
    mvs = [zeros(nacc) for _ in steps]
    svs = [zeros(nacc) for _ in steps]
    Threads.@threads :static for q in eachindex(steps)
        ws = G.wss[Threads.threadid()]
        update_step!(Xoshiro(seeds[q]), P, G, ctx, ws, zv, z0, m, steps[q], mvs[q], svs[q]; ℓ=ℓ, x=x, weather=weather,
                     vmoves=vmoves)
    end
    G.nmove .+= sum(mvs)
    G.smove .+= sum(svs)
end

"Slice updates of the level-0 k coordinates (full evaluation of L₀), in random order."
function update_k!(rng, P::Problem, G::GibbsState, s::State)
    for i in shuffle(rng, collect(eachindex(s.k)))
        lf = () -> logtarget(P, 0, s.k, s.v, s.z0, s.z0)[1]
        d = slice_coord!(rng, s.k, i, lf, G.wk[i])
        G.adapt && (G.wk[i] = 0.9 * G.wk[i] + 0.1 * max(3d, 1e-3 * G.wk[i]))
    end
end

"""
One reversible Gibbs sweep at level 0: the colour classes, the k block, FFBS
and the carrier scaling move, in uniformly random order.
"""
function gibbs_step(rng, P::Problem, s::State, S::MLDASampler, G::GibbsState)
    @assert P.E.regularized "the coordinate Gibbs kernel needs regularized constraints"
    s = copy(s)
    moves = shuffle(rng, vcat(collect(1:NCOLOURS), [-1, -2, -3]))
    for mv in moves
        if mv > 0
            ctx = SweepCtx(P, s.k)
            update_colour!(rng, P, G, ctx, s.v, s.z0, s.m, mv)
        elseif mv == -1
            update_k!(rng, P, G, s)
        elseif mv == -2
            s.m = ffbs(rng, regime_loglik(P.W, s.z0), P.W.Ptr, P.W.π0)
        else
            s.xs = [s.z0]
            evaluate!(P, s)
            s = scale_step(rng, P, s, S)
        end
    end
    s.xs = [s.z0]
    G.sweeps += 1
    if G.adapt && G.sweeps % 5 == 0
        nv = length(P.E.vspecs)
        for f in eachindex(G.nmove)
            G.nmove[f] > 0 || continue
            w = max(3 * G.smove[f] / G.nmove[f], 1e-8)
            f <= nv ? (G.wv[f] = w) : (G.wz[f-nv] = w)
        end
        fill!(G.nmove, 0.0); fill!(G.smove, 0.0)
    end
    return evaluate!(P, s)
end

"""
Gibbs sweep of the operation at level ℓ ≥ 1 with (k, x) fixed, colour classes in
random order: π_ℓ-reversible, mixed with MLDA steps at that level (v2 §5.2).
Slice widths are those adapted at level 0, rescaled per level by the adaptation
accumulators of that sweep.
"""
function fine_gibbs_step(rng, P::Problem, s::State, G::GibbsState)
    s = copy(s)
    ℓ = s.ℓ
    ctx = SweepCtx(P, s.k; ℓ=ℓ)
    adapt = G.adapt
    G.adapt = false
    for c in shuffle(rng, collect(1:NCOLOURS))
        update_colour!(rng, P, G, ctx, s.v, s.z0, s.m, c; ℓ=ℓ, x=s.xs[end])
    end
    G.adapt = adapt
    return evaluate!(P, s)
end

"Slice sampling of t for the move v[idx] .+= t (a direction with unit entries)."
function slice_dir!(rng, v, idx::Vector{Int}, lf, w; maxsteps=16)
    base = v[idx]
    setto!(t) = (@inbounds for q in eachindex(idx); v[idx[q]] = base[q] + t; end)
    logy = lf() - randexp(rng)
    L = -w * rand(rng)
    R = L + w
    J = floor(Int, maxsteps * rand(rng))
    K = maxsteps - 1 - J
    setto!(L)
    while J > 0 && lf() > logy
        L -= w; setto!(L); J -= 1
    end
    setto!(R)
    while K > 0 && lf() > logy
        R += w; setto!(R); K -= 1
    end
    for _ in 1:200
        t = L + rand(rng) * (R - L)
        setto!(t)
        lf() > logy && return abs(t)
        t < 0 ? (L = t) : (R = t)
    end
    setto!(0.0)
    return 0.0
end

"""
    temporal_multigrid_sweep!(rng, P, G, s, widths, stats; lengths, mcap)

Multigrid directions in time at the level of `s`, with (k, x) fixed: for every
dyadic block length L, every block and every v field, shift each of the first
`mcap` spatial coefficients by a common t over the block (for state-of-charge
fields this is the intertemporal storage shift), sampled along the line under
the level's cost. Blocks of one length are coloured into non-interacting
classes and updated in parallel. `widths[f, iL]` are the slice widths and
`stats[f, iL, 1:2]` accumulate (count, Σ|t|) for their adaptation.
"""
function temporal_multigrid_sweep!(rng, P::Problem, G::GibbsState, s::State, widths, stats;
                                   lengths=[2, 4, 8, 16, 32, 64, 128, 256], mcap::Int=10)
    ℓ = s.ℓ
    E = P.E
    lv = P.lv[ℓ+1]
    T = lv.T
    ctx = SweepCtx(P, s.k; ℓ=ℓ)
    x = s.xs[end]
    for iL in shuffle(rng, collect(eachindex(lengths)))
        L = lengths[iL]
        nb = cld(T, L)
        evens = [b for b in 0:2:nb-1 if !(isodd(nb) && b == nb - 1)]
        classes = [evens, collect(1:2:nb-1)]
        isodd(nb) && nb > 1 && push!(classes, [nb - 1])
        for cls in shuffle(rng, classes)
            isempty(cls) && continue
            seeds = rand(rng, UInt64, length(cls))
            loc = [zeros(size(stats)) for _ in cls]
            Threads.@threads :static for q in eachindex(cls)
                r = Xoshiro(seeds[q])
                ws = G.wss[Threads.threadid()]
                a = cls[q] * L + 1
                b = min(a + L - 1, T)
                dirs = [(f, j) for f in eachindex(E.vspecs) for j in 1:min(lv.m[f], mcap)]
                shuffle!(r, dirs)
                for (f, j) in dirs
                    idx = [lv.off[f] + (τ - 1) * lv.m[f] + j for τ in a:b]
                    taus = E.vspecs[f].kind == INSTANT ? vcat(mod(a - 2, T) + 1, collect(a:b)) : collect(a:b)
                    lf = () -> -sum(step_term(P, ℓ, ctx, ws, s.v, x, s.z0, s.m, τ) for τ in taus)
                    d = slice_dir!(r, s.v, idx, lf, widths[f, iL])
                    loc[q][f, iL, 1] += 1
                    loc[q][f, iL, 2] += d
                end
            end
            stats .+= sum(loc)
        end
    end
    return evaluate!(P, s)
end
