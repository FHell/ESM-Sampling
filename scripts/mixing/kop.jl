# Capacity-with-operation moves ("kop", prototype for docs/mixing_diagnosis.md).
#
# kscale moves a capacity at fixed physical dispatch (k, ω) -> (e^u k, e^-u ω), so a capacity
# can only move about 1/(β_k c) beyond what its current dispatch uses, and the dispatch can
# only grow into the current capacity: a ratchet. kop instead moves the capacity at fixed
# *utilization* coordinates, so the unit's operation scales with it, and the dispatchable
# generators at its node absorb the injection change share-preservingly (pro rata):
#
#   k_u -> k_u e^u (one unit, or all units of a carrier with a shared u),
#   ω_u, e_u fixed (renewables as load: output A k; storage: SoC fraction fixed, so the
#   stored energy and the charge/discharge powers scale with k),
#   at the unit's node n and every step τ the requirement of the other dispatchable
#   generators changes by -Δinj_τ; they follow with P(t) = (Q + g)_+ + e, p_i = σ_i P,
#   (σ, e) fixed (the coordinate line of the pro-rata reparametrization).
#
# Line density in u (w.r.t. du):
#   π(k(u), v(u)) · Π_units k_i(u)/k_i(0) · Π_{n,τ} |P_{n,τ}(u)/P_{n,τ}(0)|^{m_{n,τ}-1}
# (the first product is the Jacobian of the scaling group acting on the scaled capacities,
# the second the Jacobian of the pro-rata coordinates). Sampled by slice sampling in u with
# a fixed width (u is a group parameter, so the interval rule is the same from every point).
# Units are selected by kind and k ≥ κmin (truncation u ≥ log(κmin/k) keeps it invariant);
# compensators by |κ| = A k ≥ κmin with fixed k, so the selection does not move.

const KOP_KMIN = 1.0

"Per-unit data of a kop move, fixed along the line."
struct KopTerm
    kind::Int                  # 1 generator, 2 storage
    gi::Int
    i::Int
    kf::Int
    n::Int                     # node
    c::Int                     # balance cell
    nc::Int
    k0::Float64                # raw node capacity
    lo::Float64
    hi::Float64
    cap::Float64
    nmeas::Int                 # physical-measure exponent (0 if none)
    j::Vector{Float64}         # injection per MW of clamped capacity, per step
    c0::Vector{Float64}        # own cost per step at k0
    S0::Vector{Float64}        # cell residual sums at u = 0
    # pro-rata compensation per step: Q, e, Pm, m-1, and the compensators (f, i, gi, κ, σ)
    Q::Vector{Float64}
    e::Vector{Float64}
    Pm::Vector{Float64}
    nm::Vector{Float64}
    comp::Vector{Vector{NTuple{5,Float64}}}
    compc0::Vector{Float64}    # compensators' cost at u = 0, per step
end

"Own cost and injection per MW of the unit at step τ with clamped capacity kc."
@inline function kop_own(P, LM, LC, kind, gi, i, kc, τ)
    E = P.E
    Δt, cv, κ0, T = LM.Δt, E.c_viol, E.κ0, LM.T
    if kind == 1
        g = E.gens[gi]
        y = g.vf > 0 ? LC.Y[g.vf][i, τ] : 1.0
        A = LC.A[gi][i, τ]
        c, p = ML.gen_cost(g, i, A, kc, kc + κ0, Δt, cv, y)
        return c, A * clamp(y, 0.0, 1.0)
    else
        st = E.stors[gi]
        τ1 = mod(τ, T) + 1
        ye, ye1 = LC.Y[st.fe][i, τ], LC.Y[st.fe][i, τ1]
        c, _, _ = ML.soc_cost(st, i, kc, kc + κ0, Δt, cv, ye, ye1)
        pc, mc = ML.soc_power(st, i, Δt, clamp(ye, 0.0, 1.0), clamp(ye1, 0.0, 1.0))
        return c, -(pc - mc)                       # injection per MW: discharge minus charge
    end
end

"Build the kop term of unit (kind, gi, i) from the current caches."
function kop_term(P, LM, LC, Kraw, kind, gi, i; κmin=KOP_KMIN)
    E = P.E
    T = LM.T
    if kind == 1
        g = E.gens[gi]; kf = g.kf; n = g.bus[i]
        phys = E.measure == :physical && g.wf == 0 && g.vf > 0
        nmeas = phys ? T * (g.ucf > 0 ? 2 : 1) : 0
        self = g.vf > 0 ? (g.vf, i) : (0, 0)
    else
        st = E.stors[gi]; kf = st.kf; n = st.bus[i]
        nmeas = E.measure == :physical ? (ML.issoc(E) ? T : 3T) : 0
        self = (0, 0)
    end
    sp = E.kspecs[kf]
    k0 = Kraw[kf][i]
    kc0 = clamp(k0, sp.lo[i], sp.hi[i])
    cap = ML.f_capex(E, kf)[i]
    c = LM.cell[n]; nc = LM.ncell[c]
    j = zeros(T); c0 = zeros(T); S0 = zeros(T)
    Q = zeros(T); e = zeros(T); Pm = zeros(T); nm = zeros(T)
    comp = [NTuple{5,Float64}[] for _ in 1:T]
    compc0 = zeros(T)
    Δt, cv, κ0 = LM.Δt, E.c_viol, E.κ0
    for τ in 1:T
        cc, jj = kop_own(P, LM, LC, kind, gi, i, kc0, τ)
        c0[τ] = cc; j[τ] = jj
        S0[τ] = LC.S[c, τ]
        Pτ, Qs, nu = 0.0, 0.0, 0
        for (f, ii) in LM.genunits[n]
            (f, ii) == self && continue
            gg = E.gens[LM.role[f][2]]
            gg.wf > 0 && continue
            κ = ML.κunit(P, LM, LC, f, ii, τ)
            abs(κ) >= κmin || continue
            ω = LC.Y[f][ii, τ]
            Pτ += κ * ω; Qs += κ * clamp(ω, 0.0, 1.0); nu += 1
        end
        (nu == 0 || abs(Pτ) < 1e-6) && continue
        Qτ = LC.r[n, τ] + Qs
        Q[τ] = Qτ; Pm[τ] = Pτ; e[τ] = Pτ - max(Qτ, 0.0); nm[τ] = nu - 1
        for (f, ii) in LM.genunits[n]
            (f, ii) == self && continue
            gj = LM.role[f][2]
            gg = E.gens[gj]
            gg.wf > 0 && continue
            κ = ML.κunit(P, LM, LC, f, ii, τ)
            abs(κ) >= κmin || continue
            σ = κ * LC.Y[f][ii, τ] / Pτ
            push!(comp[τ], (Float64(f), Float64(ii), Float64(gj), κ, σ))
            kj = LC.kgen[gj][ii]
            compc0[τ] += ML.gen_cost(gg, ii, LC.A[gj][ii, τ], kj, kj + κ0, Δt, cv, LC.Y[f][ii, τ])[1]
        end
    end
    return KopTerm(kind, gi, i, kf, n, c, nc, k0, sp.lo[i], sp.hi[i], cap, nmeas, j, c0, S0, Q, e, Pm, nm, comp, compc0)
end

"""
Log line density change of one kop term at capacity k1 (raw), split as
(lf, logJ_prorata): lf includes capex, operation, balance, measure and the pro-rata Jacobian.
"""
function kop_eval(P, LM, LC, kt::KopTerm, k1::Float64; commit::Bool=false, s=nothing)
    E = P.E
    Δt, cv, κ0, T = LM.Δt, E.c_viol, E.κ0, LM.T
    kc0 = clamp(kt.k0, kt.lo, kt.hi); kc1 = clamp(k1, kt.lo, kt.hi)
    Δcap = kt.cap * (k1 - kt.k0) + E.γk * kt.cap * (abs(k1 - kc1) - abs(kt.k0 - kc0))
    Δop = 0.0
    lj = 0.0
    dkc = kc1 - kc0
    lv = P.lv[LM.ℓ+1]
    for τ in 1:T
        c1, _ = kop_own(P, LM, LC, kt.kind, kt.gi, kt.i, kc1, τ)
        Δop += c1 - kt.c0[τ]
        dinj = kt.j[τ] * dkc
        dS = -dinj
        if !isempty(kt.comp[τ])
            Q1 = kt.Q[τ] - dinj
            P1 = max(Q1, 0.0) + kt.e[τ]
            lj += kt.nm[τ] * log(abs(P1 / kt.Pm[τ]))
            cc = 0.0
            for (ff, fi, fg, κ, σ) in kt.comp[τ]
                f, ii, gj = Int(ff), Int(fi), Int(fg)
                gg = E.gens[gj]
                kj = LC.kgen[gj][ii]
                A = LC.A[gj][ii, τ]
                y0 = LC.Y[f][ii, τ]
                y1 = σ * P1 / κ
                cj, _ = ML.gen_cost(gg, ii, A, kj, kj + κ0, Δt, cv, y1)
                cc += cj
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
        Δop += Δt * kt.nc * (ML.φbal(E, S1 / kt.nc) - ML.φbal(E, kt.S0[τ] / kt.nc))
        if commit
            LC.r[kt.n, τ] += dS
            LC.S[kt.c, τ] = S1
        end
    end
    lm = kt.nmeas > 0 ? kt.nmeas * (log(kc1 + κ0) - log(kc0 + κ0)) : 0.0
    lf = -P.βk * Δcap - P.βv * Δop + lm + lj
    return lf, lj
end

"Write the capacity change of unit (kf, i) into s.k (node change Δ) and the cache."
function kop_setk!(P, LM, LC, s, kt::KopTerm, k1::Float64)
    E = P.E
    lk = P.lk[LM.ℓ+1]
    Ut = E.kspecs[kt.kf].B.Ut
    Δ = k1 - kt.k0
    for q in nzrange(Ut, kt.i)
        s.k[lk.off[kt.kf]+rowvals(Ut)[q]] += Δ * nonzeros(Ut)[q]
    end
    kc1 = clamp(k1, kt.lo, kt.hi)
    kt.kind == 1 ? (LC.kgen[kt.gi][kt.i] = kc1) : (LC.kst[kt.gi][kt.i] = kc1)
end

"Units eligible for kop: renewables as load (vf = 0) and storage, optionally by carrier."
function kop_units(P, LM; carriers=nothing, kinds=(:renewable, :storage))
    E = P.E
    lk = P.lk[LM.ℓ+1]
    units = Vector{NTuple{3,Int}}[]
    for (gi, g) in enumerate(E.gens)
        (g.kf > 0 && ML.iscomplete(lk, E.kspecs, g.kf)) || continue
        (g.vf == 0 && :renewable in kinds) || (g.vf > 0 && :dispatch in kinds && LM.complete[g.vf]) || continue
        carriers === nothing || g.carrier in carriers || continue
        push!(units, [(1, gi, i) for i in eachindex(g.bus)])
    end
    for (si, st) in enumerate(E.stors)
        (st.kf > 0 && ML.iscomplete(lk, E.kspecs, st.kf) && LM.complete[st.fe] && :storage in kinds) || continue
        carriers === nothing || st.carrier in carriers || continue
        push!(units, [(2, si, i) for i in eachindex(st.bus)])
    end
    return units                                     # one group per carrier
end

"""
    kop_move!(rng, P, LM, LC, s, group; w) -> (u, nunits)

One kop move of the units in `group` (a list of (kind, gi, i)) with a shared log-scale u.
Units with k < κmin are left out (fixed selection); u is truncated to keep k ≥ κmin.
"""
function kop_move!(rng, P, LM, LC, s, group; w::Float64=0.3, κmin::Float64=KOP_KMIN, check::Bool=false)
    Kraw = ML.knodes(P, LM.ℓ, s.k)
    terms = KopTerm[]
    for (kind, gi, i) in group
        kf = kind == 1 ? P.E.gens[gi].kf : P.E.stors[gi].kf
        Kraw[kf][i] >= κmin || continue
        push!(terms, kop_term(P, LM, LC, Kraw, kind, gi, i; κmin=κmin))
    end
    isempty(terms) && return 0.0, 0
    @assert allunique(kt.c for kt in terms) "kop: one unit per balance cell in a group"
    lo = maximum(log(κmin / kt.k0) for kt in terms)
    N = length(terms)
    # α = 1: the bias change, evaluated on the level-0 restriction of the moved capacities
    X0b = P.use_bias ? ML.weather_nodes(P.W, 0, s.z0) : nothing          # fixed along the move
    bterm(k) = ML.bias_value(P.bias, P.E, P.lk[1], ML.k_at0(P, LM.ℓ, k), X0b, Float64(P.levels[1][2]))
    b0 = P.use_bias ? bterm(s.k) : 0.0
    kbase = P.use_bias ? copy(s.k) : s.k
    function dbias(u)
        P.use_bias || return 0.0
        lk = P.lk[LM.ℓ+1]
        for kt in terms
            Ut = P.E.kspecs[kt.kf].B.Ut
            Δ = kt.k0 * (exp(u) - 1)
            for q in nzrange(Ut, kt.i)
                s.k[lk.off[kt.kf]+rowvals(Ut)[q]] += Δ * nonzeros(Ut)[q]
            end
        end
        b1 = bterm(s.k)
        copyto!(s.k, kbase)
        return b1 - b0
    end
    lf(u) = u < lo ? -Inf : sum(kop_eval(P, LM, LC, kt, kt.k0 * exp(u))[1] for kt in terms) + N * u - dbias(u)
    u = ML.slice_scalar(rng, lf, 0.0, w, lo)
    u == 0.0 && return 0.0, N
    L0 = s.L
    pred = check ? sum(kop_eval(P, LM, LC, kt, kt.k0 * exp(u))[1] - kop_eval(P, LM, LC, kt, kt.k0 * exp(u))[2] for kt in terms) - dbias(u) : 0.0
    for kt in terms
        k1 = kt.k0 * exp(u)
        kop_eval(P, LM, LC, kt, k1; commit=true, s=s)
        kop_setk!(P, LM, LC, s, kt, k1)
    end
    if check
        ML.evaluate!(P, s)
        err = (s.L - L0) - pred
        abs(err) > 1e-6 * max(1.0, abs(pred)) && @warn "kop: incremental and global log target differ" err pred
    end
    return u, N
end

"""
    kop_sweep!(rng, P, LM, s; mode, carriers, w, check) -> (s, stats)

mode = :unit: one kop move per eligible unit (random order); :carrier: one move per carrier
group with a shared u; :both: both, in random order.
"""
function kop_sweep!(rng, P, LM, s; mode::Symbol=:unit, carriers=nothing, kinds=(:renewable, :storage), w::Float64=0.3,
                    check::Bool=false)
    s = copy(s)
    LC = ML.LocalCost(P, LM, s)
    groups = kop_units(P, LM; carriers=carriers, kinds=kinds)
    moves = Vector{Vector{NTuple{3,Int}}}()
    mode in (:unit, :both) && append!(moves, [[u] for g in groups for u in g])
    mode in (:carrier, :both) && append!(moves, groups)
    shuffle!(rng, moves)
    us = Float64[]
    for g in moves
        u, _ = kop_move!(rng, P, LM, LC, s, g; w=w, check=check)
        push!(us, u)
    end
    ML.evaluate!(P, s)
    if check
        LC2 = ML.LocalCost(P, LM, s)
        errY = maximum(maximum(abs.(LC.Y[f] .- LC2.Y[f]); init=0.0) for f in eachindex(LC.Y) if f != P.E.fθ)
        errS = maximum(abs.(LC.S .- LC2.S)); errr = maximum(abs.(LC.r .- LC2.r))
        errk = maximum(maximum(abs.(LC.kst[q] .- LC2.kst[q]); init=0.0) for q in eachindex(LC.kst))
        errg = maximum(maximum(abs.(LC.kgen[q] .- LC2.kgen[q]); init=0.0) for q in eachindex(LC.kgen))
        @info "kop cache check" errY errS errr errk errg
    end
    return s, us
end
