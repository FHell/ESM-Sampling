# ---------------------------------------------------------------------------
# long-range storage moves (SoC form): slong, slevel (docs/system/moves/storage.md)
# ---------------------------------------------------------------------------
#
# Moving stored energy from one step to any other changes the cost only at the two end
# steps (round-trip losses, prices, the walls where the node's dispatch saturates), and the
# level of a whole cyclic trajectory changes the cost only through the box. A long-range
# plateau raises the state of charge of one unit at the cyclic interval of instants
# a+1 … a+L and moves energy from step a to step b = a+L; the level move raises every instant.
# The interior steps (both ends raised) are a compact list of storage terms, not step
# workspaces: for t in [ilo, ihi], where every raised instant stays in [0, 1], their cost
# change is exactly zero and their secondaries are constant; outside, each is evaluated
# exactly (storage term and the balance of its cell). A walk evaluation inside the interval
# costs the two end steps.

"""
Prepared long-range storage move of unit (si, i): end-step workspaces `wa` (step a, with the
compensation and the linked storage term), `wb` (step b, its own state-of-charge entry and the
compensation), the carrier `wn` of the change at instant a+1 (the next step of `wa`), and the
interior steps `τi` with their terms at t = 0. `L == T`: the level move (no end steps).
"""
mutable struct SLong
    wa::MoveWS
    wb::MoveWS
    wn::MoveWS
    P::Problem
    LM::LocalModel
    LC::LocalCost
    si::Int
    i::Int
    a::Int
    L::Int
    ilo::Float64
    ihi::Float64
    τi::Vector{Int}                  # interior steps (instants τ and τ+1 raised)
    qlo::Vector{Float64}             # per interior step: both its instants stay in [0, 1] for t in [qlo, qhi]
    qhi::Vector{Float64}
    c0::Vector{Float64}              # their storage cost, injection and secondaries at t = 0
    j0::Vector{Float64}
    sec0::Vector{Float64}            # 4 per interior step: d, charge limit, discharge limit, cell sum
    idr::Vector{Float64}             # injection change of each interior step at the last evaluation
    sec::Vector{Float64}
    bps::Vector{Float64}
    lb::LineBuf
end
SLong(wa, wb, wn, P, LM, LC) = SLong(wa, wb, wn, P, LM, LC, 0, 0, 0, 0, Inf, -Inf, Int[], Float64[], Float64[], Float64[],
                                     Float64[], Float64[], Float64[], Float64[], Float64[], LineBuf())

linebuf(m::SLong) = m.lb
mv_bps(m::SLong) = m.bps
mv_sec(m::SLong) = m.sec
mv_wss(m::SLong) = m.L < m.LM.T ? (m.wb, m.wa) : ()
mv_hasstor(m::SLong) = true
mv_fill!(m::SLong) = mv_eval!(m, 0.0; mode=1)
mv_evalc!(m::SLong, t) = mv_eval!(m, t; mode=2)

"Storage-term secondaries in the evaluation order of `mv_eval!(::SLong)`: wb, the interior steps, wa."
function storage_mask!(mask::Vector{Bool}, m::SLong)
    empty!(mask)
    m.L < m.LM.T && stepmask!(mask, m.wb)
    for _ in eachindex(m.τi)
        push!(mask, true, true, true, false)
    end
    m.L < m.LM.T && stepmask!(mask, m.wa)
    return mask
end

"""
    prep_slong!(m, si, i, a, L, jac; κmin, learn) -> Bool

Long-range plateau of storage unit (si, i): its state of charge rises by t at the instants
a+1 … a+L (cyclic, 1 ≤ L ≤ T−1), energy moved from step a to step a+L; the dispatch at the
unit's node compensates at both end steps (share-preserving pro rata with its Jacobian terms in
`jac`, or the learned rule). With L = T every instant rises (the level move): no power changes
inside the box and no compensation.
"""
function prep_slong!(m::SLong, si::Int, i::Int, a::Int, L::Int, jac; κmin=1.0, learn=nothing)
    P, LM, LC = m.P, m.LM, m.LC
    E = P.E
    T = LM.T
    s = E.stors[si]
    LM.complete[s.fe] || return false
    1 <= L <= T || return false
    κs = LC.kst[si][i] * s.hours[i] / LM.Δt
    κs < κmin && return false
    m.si, m.i, m.a, m.L = si, i, a, L
    clear!(m.wa); clear!(m.wb); clear!(m.wn)
    empty!(jac)
    empty!(m.bps)
    Y = LC.Y[s.fe]
    if L < T
        τa, τb = mod1(a, T), mod1(a + L, T)
        prep_node!(m.wb, P, LM, s.fe, i, 1.0, τb)                   # instant b rises (the start of step b)
        L >= 2 && add_entry!(m.wn, s.fe, i, 1.0)                     # instant a+1: the end value of step a
        node = s.bus[i]
        u = ((si, i),)
        prorata!(m.wa, jac, P, LM, LC, node, κs, τa; κmin=κmin, learn=learn, qadj=soc_qadj(P, LM, LC, u, node, τa))
        prorata!(m.wb, jac, P, LM, LC, node, -κs, τb; κmin=κmin, learn=learn, qadj=soc_qadj(P, LM, LC, u, node, τb))
        finalize!(m.wa, P, LM, LC, τa)
        finalize!(m.wb, P, LM, LC, τb)
        link_soc!(m.wa, L >= 2 ? m.wn : m.wb, LM)
        append!(m.bps, m.wa.bps); append!(m.bps, m.wb.bps)
        ni = L - 1
        τ0 = a                                                       # interior steps a+1 … a+L−1
    else
        ni = T
        τ0 = 0                                                       # every step
    end
    # interior steps: terms at t = 0, the clamp points of their raised start instants, and the
    # interval over which every raised instant (the interior ones and b) stays in the box
    resize!(m.τi, ni); resize!(m.c0, ni); resize!(m.j0, ni); resize!(m.sec0, 4ni); resize!(m.idr, ni)
    resize!(m.qlo, ni); resize!(m.qhi, ni)
    Δt, cv = LM.Δt, E.c_viol
    k = LC.kst[si][i]; w = k + E.κ0
    node = s.bus[i]; c = LM.cell[node]
    lo, hi = -Inf, Inf
    inbox = true
    @inbounds for q in 1:ni
        τ = mod1(τ0 + q, T)
        τ1 = mod1(τ + 1, T)
        m.τi[q] = τ
        y, y1 = Y[i, τ], Y[i, τ1]
        c0, d0, j0 = soc_cost(s, i, k, w, Δt, cv, y, y1)
        m.c0[q], m.j0[q] = c0, j0
        m.sec0[4q-3] = d0
        m.sec0[4q-2] = s.eta_in[i] > 0 ? d0 - s.eta_in[i] : 1.0
        m.sec0[4q-1] = -d0 * s.eta_out[i] - 1.0
        m.sec0[4q] = LC.S[c, τ]
        push!(m.bps, -y, 1.0 - y)
        inbox &= 0.0 <= y <= 1.0
        lo = max(lo, -y); hi = min(hi, 1.0 - y)
        if 0.0 <= y <= 1.0 && 0.0 <= y1 <= 1.0
            m.qlo[q], m.qhi[q] = max(-y, -y1), min(1.0 - y, 1.0 - y1)
        else
            m.qlo[q], m.qhi[q] = Inf, -Inf
        end
    end
    if L < T                                                         # instant b (the end value of the last interior step)
        yb = Y[i, mod1(a + L, T)]
        inbox &= 0.0 <= yb <= 1.0
        lo = max(lo, -yb); hi = min(hi, 1.0 - yb)
    end
    m.ilo, m.ihi = inbox ? (lo, hi) : (Inf, -Inf)
    return true
end

"Cost change ΔC(t) of the prepared long-range move; fills `m.sec` (order: wb, interior steps, wa)."
function mv_eval!(m::SLong, t; mode::Int=0)
    P, LM, LC = m.P, m.LM, m.LC
    E = P.E
    T = LM.T
    Δ = 0.0
    empty!(m.sec)
    ends = m.L < T
    if ends
        Δ += eval_move!(m.wb, P, LM, LC, mod1(m.a + m.L, T), t; mode=mode)
        append!(m.sec, m.wb.sec)
    end
    if m.ilo <= t <= m.ihi
        append!(m.sec, m.sec0)
    else
        s = E.stors[m.si]; i = m.i
        Y = LC.Y[s.fe]
        Δt, cv = LM.Δt, E.c_viol
        k = LC.kst[m.si][i]; w = k + E.κ0
        c = LM.cell[s.bus[i]]; nc = LM.ncell[c]
        @inbounds for q in eachindex(m.τi)
            if m.qlo[q] <= t <= m.qhi[q]             # this step's instants stay in the box: no change
                m.idr[q] = 0.0
                push!(m.sec, m.sec0[4q-3], m.sec0[4q-2], m.sec0[4q-1], m.sec0[4q])
                continue
            end
            τ = m.τi[q]
            c1, d1, j1 = soc_cost(s, i, k, w, Δt, cv, Y[i, τ] + t, Y[i, mod1(τ + 1, T)] + t)
            dr = j1 - m.j0[q]
            m.idr[q] = dr
            S0 = LC.S[c, τ]; S1 = S0 + dr
            Δ += (c1 - m.c0[q]) + Δt * nc * (φbal(E, S1 / nc) - φbal(E, S0 / nc))
            push!(m.sec, d1, s.eta_in[i] > 0 ? d1 - s.eta_in[i] : 1.0, -d1 * s.eta_out[i] - 1.0, S1)
        end
    end
    if ends
        Δ += eval_move!(m.wa, P, LM, LC, mod1(m.a, T), t; next=(m.L >= 2 ? m.wn : m.wb), mode=mode)
        append!(m.sec, m.wa.sec)
    end
    return Δ
end

"Add a node-space change a of entry i of field f at step τ to the coefficients v (as `addz_node!`)."
function addv_node!(v::AbstractVector, lv::Layout, E::EnergyModel, f::Int, τ::Int, i::Int, a::Float64)
    Ut = E.vspecs[f].B.Ut
    mf = lv.m[f]
    o = lv.off[f] + (τ - 1) * mf
    rv, nz = rowvals(Ut), nonzeros(Ut)
    @inbounds for q in nzrange(Ut, i)
        j = rv[q]
        j <= mf && (v[o+j] += a * nz[q])
    end
end

"Apply the prepared long-range move with parameter t to the coefficients and the caches."
function commit_slong!(m::SLong, v::AbstractVector, t::Float64)
    t == 0.0 && return
    P, LM, LC = m.P, m.LM, m.LC
    E = P.E
    T = LM.T
    mv_eval!(m, t; mode=2)
    s = E.stors[m.si]; i = m.i
    if m.L < T
        m.wa.trec = false; m.wb.trec = false
        apply_step!(m.wa, P, LC, v, mod1(m.a, T), t)
        apply_step!(m.wb, P, LC, v, mod1(m.a + m.L, T), t)        # includes instant b
    end
    lv = P.lv[LM.ℓ+1]
    Y = LC.Y[s.fe]
    @inbounds for τ in m.τi                                       # the raised start instants of the interior steps
        Y[i, τ] += t
        addv_node!(v, lv, E, s.fe, τ, i, t)
    end
    if !(m.ilo <= t <= m.ihi)
        node = s.bus[i]; c = LM.cell[node]
        @inbounds for q in eachindex(m.τi)
            τ = m.τi[q]
            LC.r[node, τ] += m.idr[q]
            LC.S[c, τ] += m.idr[q]
        end
    end
    return
end

"""
    storage_long_sweep!(rng, P, LM, s; nmoves, p_level, hmin, nxmoves, LC, pools, learn, κmin) -> (s, stats)

Long-range storage sweep at a complete level in the SoC form: for every storage unit with at
least `hmin` hours of storage and κ_s ≥ κmin, a level move with probability `p_level` and
`nmoves` long-range plateaus (`slong`) with start a uniform on 1…T and length L uniform on
1…T−1 (cyclic), in random order. The moves of one cell run on one thread and the cells in
parallel (a move touches only its unit, the dispatch at its node and the balance of the node's
cell; the coefficient changes, which share coarse basis coefficients across cells, are
accumulated per thread and added at the end). Then `nxmoves` interval storage pairs (`sxlong`,
storage_cross.jl) per unit with at least the pair's storage hours (`LM.sxunits`), in rounds of
disjoint footprints. Selection uses only (k, x) and fixed probabilities. `stats`: rows (slong,
slevel, sxlong); columns (moves, accepted, Σ |t|·E in MWh).
"""
function storage_long_sweep!(rng, P::Problem, LM::LocalModel, s::State; nmoves::Int=4, p_level::Float64=1.0,
                             hmin::Float64=24.0, nxmoves::Int=0, LC=nothing, pools=nothing, learn=nothing, κmin::Float64=1.0)
    E = P.E
    T = LM.T
    LCc = LC === nothing ? LocalCost(P, LM, s) : LC
    pl::Vector{BlockPool} = pools === nothing ? [BlockPool(P, 3) for _ in 1:Threads.maxthreadid()] : pools
    ensure_pools!(pl, P, 3)
    bycell = Dict{Int,Vector{NTuple{2,Int}}}()
    for (si, st) in enumerate(E.stors)
        LM.complete[st.fe] || continue
        for i in eachindex(st.bus)
            (st.hours[i] >= hmin && LCc.kst[si][i] * st.hours[i] / LM.Δt >= κmin) || continue
            push!(get!(bycell, LM.cell[st.bus[i]], NTuple{2,Int}[]), (si, i))
        end
    end
    cells = sort!(collect(keys(bycell)))
    seeds = rand(rng, UInt64, length(cells))
    loc = [zeros(2, 3) for _ in cells]
    # coefficient changes per thread, summed after the parallel phase: the node values of
    # different cells share coarse basis coefficients (the caches of LC are per node and cell)
    dvs = [zeros(length(s.v)) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for c in eachindex(cells)
        r = Xoshiro(seeds[c])
        pool = pl[Threads.threadid()]
        dv = dvs[Threads.threadid()]
        m = SLong(pool.wss[1], pool.wss[2], pool.wss[3], P, LM, LCc)
        secprev = pool.wss[1].secprev
        jac = pool.wss[1].jac
        mvl = NTuple{3,Int}[]                                    # (unit, L); L = T: level move
        for (si, i) in bycell[cells[c]]
            rand(r) < p_level && push!(mvl, (si, i, T))
            for _ in 1:nmoves
                push!(mvl, (si, i, rand(r, 1:T-1)))
            end
        end
        shuffle!(r, mvl)
        for (si, i, L) in mvl
            prep_slong!(m, si, i, rand(r, 1:T), L, jac; κmin=κmin, learn=learn) || continue
            t = sample_line!(r, m, P.βv, secprev)
            (t != 0.0 && !isempty(jac) && rand(r) >= exp(min(0.0, logjac(jac, t, m)))) && (t = 0.0)
            commit_slong!(m, dv, t)
            row = L == T ? 2 : 1
            loc[c][row, 1] += 1
            loc[c][row, 2] += t != 0.0
            loc[c][row, 3] += abs(t) * LCc.kst[si][i] * E.stors[si].hours[i]
        end
        clear!(m.wa); clear!(m.wb); clear!(m.wn)
    end
    stx = storage_cross_phase!(rng, P, LM, LCc, dvs, pl; nmoves=nxmoves, learn=learn, κmin=κmin)
    for dv in dvs
        s.v .+= dv
    end
    return evaluate!(P, s), vcat(sum(loc; init=zeros(2, 3)), stx)
end
