# Capacity moves with their operation (`kop`, docs/system/moves/capacity.md).
#
# `kscale` moves a capacity at fixed physical dispatch, so a capacity can move only about one
# capex scale 1/(β_k c) beyond what its dispatch uses, and a renewable capacity only by the
# thermal slack of its node balance (a few MW). `kop` moves the capacity at fixed
# utilization coordinates instead, so that the unit's operation scales with it:
#
#   k_u → k_u e^u for one unit, or for all units of a carrier with a common u;
#   renewables (as load): the output A k scales; storage (SoC form): the state-of-charge
#   fractions are fixed, so the stored energy and the charge/discharge powers scale;
#   at the unit's node the other dispatchable generators follow the requirement change
#   −Δinj_τ at every step share-preservingly, P(u) = (Q − Δinj)₊ + e, p_i = σ_i P with
#   (σ, e) fixed: the coordinate line of the pro-rata reparametrization (sampler.md §5.2),
#   at all steps of the node at once.
#
# Line density in u (w.r.t. du): π(k(u), v(u)) · Π_units k_i(u)/k_i(0) · Π_{n,τ} |P_{n,τ}(u)/P_{n,τ}(0)|^{m_{n,τ}−1}:
# the Jacobian of the scaling group on the moved capacities and that of the pro-rata
# coordinates; π includes capex, the box penalty, every cost term of the unit and its
# compensators, the balance, the reference measure and, at α = 1, the bias. u is drawn by
# slice sampling with a fixed width (u is a group parameter, so the stepping-out rule is
# the same from every point of the curve), truncated so that every moved k stays ≥ κmin.
# Units are selected by kind and k ≥ κmin, compensators by |A k| ≥ κmin at fixed k: both
# selections are invariant along the curve.

"Data of one unit of a kop move, fixed along the curve."
struct KopTerm
    kind::Int                  # 1 generator (renewable as load), 2 storage
    gi::Int
    i::Int
    kf::Int
    n::Int                     # node
    c::Int                     # balance cell
    nc::Int
    k0::Float64                # raw node capacity at u = 0
    lo::Float64
    hi::Float64
    cap::Float64               # capex per MW
    nmeas::Int                 # exponent of the reference measure term (0: none)
    j::Vector{Float64}         # injection per MW of clamped capacity, per step
    c0::Vector{Float64}        # own cost per step at k0
    S0::Vector{Float64}        # cell residual sums at u = 0
    Q::Vector{Float64}         # pro-rata coordinates per step: requirement, excess, P, |S| − 1
    e::Vector{Float64}
    Pm::Vector{Float64}
    nm::Vector{Float64}
    comp::Vector{Vector{NTuple{5,Float64}}}   # compensators per step: (field, index, group, κ, σ)
    compc0::Vector{Float64}    # their cost per step at u = 0
end

"A prepared kop move: its terms and the lower bound of u (k ≥ κmin for every term)."
struct KopMove
    terms::Vector{KopTerm}
    lo::Float64
end

"Own cost of the unit at step τ with clamped capacity kc, and its injection per MW of capacity."
@inline function kop_own(P::Problem, LM::LocalModel, LC::LocalCost, kind::Int, gi::Int, i::Int, kc::Float64, τ::Int)
    E = P.E
    Δt, cv, κ0, T = LM.Δt, E.c_viol, E.κ0, LM.T
    if kind == 1
        g = E.gens[gi]
        A = LC.A[gi][i, τ]
        return gen_cost(g, i, A, kc, kc + κ0, Δt, cv, 1.0)[1], A
    end
    st = E.stors[gi]
    τ1 = mod(τ, T) + 1
    ye, ye1 = LC.Y[st.fe][i, τ], LC.Y[st.fe][i, τ1]
    c = soc_cost(st, i, kc, kc + κ0, Δt, cv, ye, ye1)[1]
    pc, mc = soc_power(st, i, Δt, clamp(ye, 0.0, 1.0), clamp(ye1, 0.0, 1.0))
    return c, mc - pc                              # injection: discharge minus charge
end

"""
Commitment cost of compensator i of generator group gj at step τ with output utilization y and
its commitment fixed (0 without unit commitment): the walls see the output.
"""
@inline function kop_uc(P::Problem, LM::LocalModel, LC::LocalCost, gj::Int, i::Int, k::Float64, τ::Int, y::Float64)
    g = P.E.gens[gj]
    g.ucf > 0 || return 0.0
    U = LC.Y[g.ucf]
    return uc_cost(g, i, LC.A[gj][i, τ], k, k + P.E.κ0, LM.Δt, P.E.c_viol, clamp(y, 0.0, 1.0), U[i, τ], U[i, mod(τ, LM.T)+1])
end

"Terms of unit (kind, gi, i) of a kop move from the current caches."
function kop_term(P::Problem, LM::LocalModel, LC::LocalCost, Kraw, kind::Int, gi::Int, i::Int; κmin::Float64=1.0)
    E = P.E
    T = LM.T
    kf, n = kind == 1 ? (E.gens[gi].kf, E.gens[gi].bus[i]) : (E.stors[gi].kf, E.stors[gi].bus[i])
    nmeas = (kind == 2 && E.measure == :physical) ? T : 0     # renewables as load have no dispatch coordinate
    sp = E.kspecs[kf]
    k0 = Kraw[kf][i]
    kc0 = clamp(k0, sp.lo[i], sp.hi[i])
    c = LM.cell[n]
    Δt, cv, κ0 = LM.Δt, E.c_viol, E.κ0
    j = zeros(T); c0 = zeros(T); S0 = zeros(T)
    Q = zeros(T); e = zeros(T); Pm = zeros(T); nm = zeros(T)
    comp = [NTuple{5,Float64}[] for _ in 1:T]
    compc0 = zeros(T)
    for τ in 1:T
        c0[τ], j[τ] = kop_own(P, LM, LC, kind, gi, i, kc0, τ)
        S0[τ] = LC.S[c, τ]
        Pτ, Qs, nu = 0.0, 0.0, 0
        for (f, ii) in LM.genunits[n]
            E.gens[LM.role[f][2]].wf > 0 && continue
            κ = κunit(P, LM, LC, f, ii, τ)
            abs(κ) >= κmin || continue
            ω = LC.Y[f][ii, τ]
            Pτ += κ * ω; Qs += κ * clamp(ω, 0.0, 1.0); nu += 1
        end
        (nu == 0 || abs(Pτ) < 1e-6) && continue
        Qτ = LC.r[n, τ] + Qs
        Q[τ] = Qτ; Pm[τ] = Pτ; e[τ] = Pτ - max(Qτ, 0.0); nm[τ] = nu - 1
        for (f, ii) in LM.genunits[n]
            gj = LM.role[f][2]
            gg = E.gens[gj]
            gg.wf > 0 && continue
            κ = κunit(P, LM, LC, f, ii, τ)
            abs(κ) >= κmin || continue
            push!(comp[τ], (Float64(f), Float64(ii), Float64(gj), κ, κ * LC.Y[f][ii, τ] / Pτ))
            kj = LC.kgen[gj][ii]
            compc0[τ] += gen_cost(gg, ii, LC.A[gj][ii, τ], kj, kj + κ0, Δt, cv, LC.Y[f][ii, τ])[1] +
                         kop_uc(P, LM, LC, gj, ii, kj, τ, LC.Y[f][ii, τ])
        end
    end
    return KopTerm(kind, gi, i, kf, n, c, LM.ncell[c], k0, sp.lo[i], sp.hi[i], f_capex(E, kf)[i], nmeas,
                   j, c0, S0, Q, e, Pm, nm, comp, compc0)
end

"""
Log line density of one term at raw capacity k1 without the group Jacobian and the bias:
(capex, cost, balance, measure and pro-rata Jacobian; the pro-rata log Jacobian alone).
With `commit` the compensators, the residuals and the coefficients of `s.v` are written.
"""
function kop_eval(P::Problem, LM::LocalModel, LC::LocalCost, kt::KopTerm, k1::Float64; commit::Bool=false, s=nothing)
    E = P.E
    Δt, cv, κ0, T = LM.Δt, E.c_viol, E.κ0, LM.T
    kc0, kc1 = clamp(kt.k0, kt.lo, kt.hi), clamp(k1, kt.lo, kt.hi)
    Δcap = kt.cap * (k1 - kt.k0) + E.γk * kt.cap * (abs(k1 - kc1) - abs(kt.k0 - kc0))
    dkc = kc1 - kc0
    lv = P.lv[LM.ℓ+1]
    Δop, lj = 0.0, 0.0
    for τ in 1:T
        Δop += kop_own(P, LM, LC, kt.kind, kt.gi, kt.i, kc1, τ)[1] - kt.c0[τ]
        dinj = kt.j[τ] * dkc
        dS = -dinj
        if !isempty(kt.comp[τ])
            P1 = max(kt.Q[τ] - dinj, 0.0) + kt.e[τ]
            lj += kt.nm[τ] * log(abs(P1 / kt.Pm[τ]))
            cc = 0.0
            for (ff, fi, fg, κ, σ) in kt.comp[τ]
                f, ii, gj = Int(ff), Int(fi), Int(fg)
                kj = LC.kgen[gj][ii]
                y0 = LC.Y[f][ii, τ]
                y1 = σ * P1 / κ
                cc += gen_cost(E.gens[gj], ii, LC.A[gj][ii, τ], kj, kj + κ0, Δt, cv, y1)[1] + kop_uc(P, LM, LC, gj, ii, kj, τ, y1)
                dS -= κ * (clamp(y1, 0.0, 1.0) - clamp(y0, 0.0, 1.0))
                if commit
                    LC.Y[f][ii, τ] = y1
                    m = lv.m[f]; o = lv.off[f] + (τ - 1) * m
                    Ut = E.vspecs[f].B.Ut
                    for q in nzrange(Ut, ii)
                        jj = rowvals(Ut)[q]
                        jj <= m && (s.v[o+jj] += (y1 - y0) * nonzeros(Ut)[q])
                    end
                end
            end
            Δop += cc - kt.compc0[τ]
        end
        S1 = kt.S0[τ] + dS
        Δop += Δt * kt.nc * (φbal(E, S1 / kt.nc) - φbal(E, kt.S0[τ] / kt.nc))
        if commit
            LC.r[kt.n, τ] += dS
            LC.S[kt.c, τ] = S1
        end
    end
    lm = kt.nmeas > 0 ? kt.nmeas * (log(kc1 + κ0) - log(kc0 + κ0)) : 0.0
    return -P.βk * Δcap - P.βv * Δop + lm + lj, lj
end

"Investable renewable (as load) and storage units at a complete level, one group per carrier."
function kop_units(P::Problem, LM::LocalModel)
    E = P.E
    lk = P.lk[LM.ℓ+1]
    groups = Vector{NTuple{3,Int}}[]
    for (gi, g) in enumerate(E.gens)
        (g.kf > 0 && g.vf == 0 && iscomplete(lk, E.kspecs, g.kf)) || continue
        push!(groups, [(1, gi, i) for i in eachindex(g.bus)])
    end
    for (si, st) in enumerate(E.stors)
        (st.kf > 0 && iscomplete(lk, E.kspecs, st.kf) && LM.complete[st.fe]) || continue
        push!(groups, [(2, si, i) for i in eachindex(st.bus)])
    end
    return groups
end

"""
Prepare the kop move of `group` (units with k ≥ κmin, at distinct balance cells); nothing if
empty. `Kraw` are the raw node capacities of `s` (computed if not given).
"""
function kop_prepare(P::Problem, LM::LocalModel, LC::LocalCost, s::State, group; κmin::Float64=1.0, Kraw=nothing)
    E = P.E
    Kraw = Kraw === nothing ? knodes(P, LM.ℓ, s.k) : Kraw
    terms = KopTerm[]
    for (kind, gi, i) in group
        kf = kind == 1 ? E.gens[gi].kf : E.stors[gi].kf
        Kraw[kf][i] >= κmin || continue
        push!(terms, kop_term(P, LM, LC, Kraw, kind, gi, i; κmin=κmin))
    end
    (isempty(terms) || !allunique(kt.c for kt in terms)) && return nothing
    return KopMove(terms, maximum(log(κmin / kt.k0) for kt in terms))
end

"A copy of a bias cache for trial evaluations (the tables and the weather availabilities are shared: capacity moves do not change them)."
bias_cache_trial(BC::BiasCache) =
    BiasCache((f in (:tab1, :tab2, :At) ? getfield(BC, f) : deepcopy(getfield(BC, f)) for f in fieldnames(BiasCache))...)

"Change of b̂ when the terms of `mv` take the capacities k0 e^u (one unit: incremental; several: in sequence on a copy, or on BC itself with `commit`)."
function kop_bias_delta(P::Problem, LM::LocalModel, BC, mv::KopMove, u::Float64; commit::Bool=false)
    BC === nothing && return 0.0
    lk = P.lk[LM.ℓ+1]
    if length(mv.terms) == 1 && !commit
        kt = mv.terms[1]
        return bias_delta_k(BC, P.bias, P.E, lk, kt.kind, kt.gi, kt.i, kt.k0 * exp(u))
    end
    B = commit ? BC : bias_cache_trial(BC)
    d = 0.0
    for kt in mv.terms
        d += bias_delta_k(B, P.bias, P.E, lk, kt.kind, kt.gi, kt.i, kt.k0 * exp(u); commit=true)
    end
    return d
end

"""
    kop_logdensity(P, LM, LC, mv, u; BC) -> (lf, lj, Δb)

Log line density of the prepared move at u relative to u = 0 (everything, including the group
Jacobian N u, the pro-rata Jacobian lj and −Δb̂), the pro-rata log Jacobian, and Δb̂.
"""
function kop_logdensity(P::Problem, LM::LocalModel, LC::LocalCost, mv::KopMove, u::Float64; BC=nothing)
    u < mv.lo && return -Inf, 0.0, 0.0
    lf, lj = 0.0, 0.0
    for kt in mv.terms
        a, b = kop_eval(P, LM, LC, kt, kt.k0 * exp(u))
        lf += a; lj += b
    end
    Δb = kop_bias_delta(P, LM, BC, mv, u)
    return lf + length(mv.terms) * u - Δb, lj, Δb
end

"""
Commit the prepared move at u: capacities, compensators, coefficients, local caches, the bias
cache and, if given, the raw node capacities `Kraw`.
"""
function kop_commit!(P::Problem, LM::LocalModel, LC::LocalCost, s::State, mv::KopMove, u::Float64; BC=nothing, Kraw=nothing)
    E = P.E
    lk = P.lk[LM.ℓ+1]
    kop_bias_delta(P, LM, BC, mv, u; commit=true)
    for kt in mv.terms
        k1 = kt.k0 * exp(u)
        kop_eval(P, LM, LC, kt, k1; commit=true, s=s)
        Ut = E.kspecs[kt.kf].B.Ut
        Δ = k1 - kt.k0
        for q in nzrange(Ut, kt.i)
            s.k[lk.off[kt.kf]+rowvals(Ut)[q]] += Δ * nonzeros(Ut)[q]
        end
        kc1 = clamp(k1, kt.lo, kt.hi)
        kt.kind == 1 ? (LC.kgen[kt.gi][kt.i] = kc1) : (LC.kst[kt.gi][kt.i] = kc1)
        Kraw === nothing || (Kraw[kt.kf][kt.i] = k1)
    end
    return s
end

"""
    kop_sweep!(rng, P, LM, s; LC, BC, mode, w, κmin) -> (s, nmoved)

Capacity moves with their operation at a spatially complete level: one move per investable
renewable and storage unit and (`mode = :both`, the default) one per such carrier with a
common scale, in random order; `mode = :unit` or `:carrier` for one kind only. At α = 1 the
bias change comes from the bias cache `BC` (built from `s` if not given), which follows the
committed moves; `LC` likewise for the local caches.
"""
function kop_sweep!(rng, P::Problem, LM::LocalModel, s::State; LC=nothing, BC=nothing, mode::Symbol=:both,
                    w::Float64=0.3, κmin::Float64=1.0)
    LM.complete[P.E.fθ] || return s, 0               # node balances needed: spatially complete levels only
    LC = LC === nothing ? LocalCost(P, LM, s) : LC
    BC = (P.use_bias && BC === nothing) ? BiasCache(P, s) : BC
    groups = kop_units(P, LM)
    moves = Vector{Vector{NTuple{3,Int}}}()
    mode in (:unit, :both) && append!(moves, [[u] for g in groups for u in g])
    mode in (:carrier, :both) && append!(moves, groups)
    Kraw = knodes(P, LM.ℓ, s.k)                       # raw node capacities, kept current by the commits
    nmoved = 0
    for group in shuffle(rng, moves)
        mv = kop_prepare(P, LM, LC, s, group; κmin=κmin, Kraw=Kraw)
        mv === nothing && continue
        u = slice_scalar(rng, x -> kop_logdensity(P, LM, LC, mv, x; BC=BC)[1], 0.0, w, mv.lo)
        u == 0.0 && continue
        kop_commit!(P, LM, LC, s, mv, u; BC=BC, Kraw=Kraw)
        nmoved += 1
    end
    return evaluate!(P, s), nmoved
end
