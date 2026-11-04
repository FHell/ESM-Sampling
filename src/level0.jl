# Level-0 slice updates of the bias warmup chain (used by `local_base_step`).
#
# With all constraints priced the level-0 potential
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
# exact for the conditional for any width.

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
    P.use_bias && tables(P.bias)            # the step terms read the tables in parallel: bring them up to date first
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
    LT = E.lines
    @inbounds for e in eachindex(sys.line_s)
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        fl = sys.line_b[e] * (θ[a] - θ[b])
        ov, loss = line_eval(LT, e, sys.line_s[e], fl)
        r[a] += fl + 0.5 * loss
        r[b] += -fl + 0.5 * loss
        cost += c_viol * Δt * ov
    end
    if E.flink > 0
        y = ws.y[E.flink]
        @inbounds for l in eachindex(sys.link_cap)
            yc = clamp(y[l], -1.0, 1.0)
            u = sys.link_cap[l] * yc
            j0, j1 = link_inj(E.link_eta[l], u)
            r[sys.link_bus0[l]] += j0
            r[sys.link_bus1[l]] += j1
            cost += c_viol * Δt * sys.link_cap[l] * abs(y[l] - yc)
        end
    end
    R = 0.0
    @inbounds for (gi, g) in enumerate(E.gens)
        fixed = g.vf == 0                       # renewables as negative load: full output
        y = fixed ? ws.r : ws.y[g.vf]
        kv = ctx.kgen[gi]
        for i in eachindex(g.bus)
            yc = fixed ? 1.0 : clamp(y[i], 0.0, 1.0)
            A = g.wf > 0 ? clamp(ws.x[g.wf][i], 0.0, 1.0) : g.avail[i]
            p = A * kv[i] * yc
            cost += Δt * gen_vint(g, i, A, kv[i], yc)[1]
            r[g.bus[i]] -= p
            fixed || (cost += c_viol * Δt * (kv[i] + κ0) * abs(y[i] - yc))
            g.ucf > 0 && (cost += uc_cost(g, i, A, kv[i], kv[i] + κ0, Δt, c_viol, yc, ws.y[g.ucf][i], ws.y1[g.ucf][i]))
            g.carrier == "gas" || (R += A * kv[i])
        end
    end
    @inbounds for (si, s) in enumerate(E.stors)               # state-of-charge form
        ye, ye1 = ws.y[s.fe], ws.y1[s.fe]
        kv = ctx.kst[si]
        for i in eachindex(s.bus)
            c, _, j = soc_cost(s, i, kv[i], kv[i] + κ0, Δt, c_viol, ye[i], ye1[i])
            cost += c
            r[s.bus[i]] += j
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
    # bias: the per-step part of ĝ (tables; the k-only parts of b̂ are constant here)
    if P.use_bias
        b = P.bias
        b.ℓb == 0 || error("the level-0 step terms need the bias at level 0")
        u1 = (D - R) / b.Dbar
        val += b.Δt * (tabval(b.tab1, u1) + tabval(b.tab2, u1 - ctx.firm / b.Dbar))
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
    GibbsState(wv, fill(0.05, nw), max.(1e-3 .* coeff_scales(P.E, P.lk[1]), 1.0), false, zeros(nv + nw), zeros(nv + nw),
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
