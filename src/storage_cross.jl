# ---------------------------------------------------------------------------
# cross-node storage pairs (SoC form): sxpair, sxlong (docs/system/moves/storage.md)
# ---------------------------------------------------------------------------
#
# "Store more here, less there, with the flows compensating." Unit A's state of charge rises by t
# (as a fraction of its energy E_A = k h) and unit B's, at the same node or at a partner node of the
# local re-routing templates, falls by the same nominal energy (by ρ t, ρ = E_A/E_B). At the steps
# whose power changes, the nominal injection changes at A and B are equal and opposite; the flow
# between them is re-routed on the ball template of the pair (`xloc_directions`) and the leak at
# the ball's edge is compensated share-preservingly (with the Jacobian step), as in `xlocc`. The
# dispatch at A's and B's nodes takes at most the expected losses, so the end steps no longer stop
# where the node's own dispatch saturates; they stop at the lines of the ball, the power limits and
# boxes of the two units and, for the part that leaks, at the leak nodes. The actual injection of a unit differs from the
# nominal one by its losses: for a unit that only discharges (reservoir hydro) the fixed loss share
# 1 − η_out of the nominal change is compensated by the dispatch at its node (`sx_step!`), the
# losses of units that can charge go into the residual of their node.
#
# `sxpair` is the single-instant form (instant τ, steps τ−1 and τ; a family of the local sweep),
# `sxlong` the interval form on the cyclic instants a+1 … a+L (steps a and a+L; the long-range
# storage sweep), built as `slong` with compact interior terms for both units.

"""
Storage-pair templates of a level: per node n the partner nodes (n itself first, then the xloc
partners within `rp` hops), and per partner the ball re-routing of a unit dipole (angles, leak
nodes and their requirement changes; empty for n itself) and the balance cells it touches
(n, the partner, the nodes with an angle change, the other ends of their lines, the leak nodes).
`reuse`: the xloc templates of the level when they have the same hops. Depends only on the network.
"""
function sx_templates(sys, GS::GridSolver, nodelines, cell, θcomplete::Bool, reuse; rp::Int=1, rb::Int=1)
    N = sys.N
    part, xnod, xval, xlkn, xlkg = if !θcomplete
        ([Int[] for _ in 1:N], [Vector{Int}[] for _ in 1:N], [Vector{Float64}[] for _ in 1:N],
         [Vector{Int}[] for _ in 1:N], [Vector{Float64}[] for _ in 1:N])
    else
        reuse === nothing ? xloc_directions(sys, GS, nodelines; rp=rp, rb=rb) : reuse
    end
    sxpart = [vcat([n], part[n]) for n in 1:N]
    sxnod = [vcat([Int[]], xnod[n]) for n in 1:N]
    sxval = [vcat([Float64[]], xval[n]) for n in 1:N]
    sxlkn = [vcat([Int[]], xlkn[n]) for n in 1:N]
    sxlkg = [vcat([Float64[]], xlkg[n]) for n in 1:N]
    sxfoot = [Vector{Int}[] for _ in 1:N]
    for n in 1:N, (j, m) in enumerate(sxpart[n])
        nd = Int[n, m]
        for i in sxnod[n][j]
            push!(nd, i)
            for e in nodelines[i]
                push!(nd, sys.line_bus0[e], sys.line_bus1[e])
            end
        end
        append!(nd, sxlkn[n][j])
        push!(sxfoot[n], sort!(unique!([cell[i] for i in nd])))
    end
    return sxpart, sxnod, sxval, sxlkn, sxlkg, sxfoot
end

"""
    sx_partner(rng, P, LM, LC, si, i; κmin) -> (j, sj, jj)

Partner of storage unit (si, i) for a storage pair: a unit (sj, jj) ≠ (si, i) with at least the
pair's storage hours (`LM.sxnodeu`) and κ = k h/Δt ≥ κmin at the unit's node or one of its partner
nodes (slot j of `LM.sxpart`), with probability ∝ min(k_A, k_B) (the end steps of a pair stop at
the power headroom of the weaker unit, so the energy a move shifts scales with it). A function of k
only. (0, 0, 0) if there is none.
"""
function sx_partner(rng, P::Problem, LM::LocalModel, LC::LocalCost, si::Int, i::Int; κmin::Float64=1.0)
    E = P.E
    s = E.stors[si]
    n = s.bus[i]
    kA = LC.kst[si][i]
    W = 0.0
    @inbounds for m in LM.sxpart[n], (sj, jj) in LM.sxnodeu[m]
        (sj == si && jj == i) && continue
        kB = LC.kst[sj][jj]
        kB * E.stors[sj].hours[jj] / LM.Δt >= κmin || continue
        W += min(kA, kB)
    end
    W > 0.0 || return (0, 0, 0)
    u = rand(rng) * W
    acc = 0.0
    last = (0, 0, 0)
    @inbounds for (j, m) in enumerate(LM.sxpart[n]), (sj, jj) in LM.sxnodeu[m]
        (sj == si && jj == i) && continue
        kB = LC.kst[sj][jj]
        kB * E.stors[sj].hours[jj] / LM.Δt >= κmin || continue
        acc += min(kA, kB)
        last = (j, sj, jj)
        u <= acc && return last
    end
    return last
end

"Loss share λ of a storage unit in the pair compensation: 1 − η_out for a unit that only discharges (reservoir hydro), else 0."
@inline sx_loss(s::StorGroup, i::Int) = s.eta_in[i] > 0 ? 0.0 : 1.0 - s.eta_out[i]

"""
Requirement shift of the loss compensation of storage unit (si, i) at step τ: its actual injection
minus (1 − λ) times its nominal injection N = −k h (Y_{τ+1} − Y_τ)/Δt (unclamped Y). With the flows
exporting the nominal change, the requirement at the unit's node plus this shift changes by exactly
λ ΔN along the move (as `soc_qadj` for λ = 0).
"""
function sx_qloss(P::Problem, LM::LocalModel, LC::LocalCost, si::Int, i::Int, λ::Float64, τ::Int)
    s = P.E.stors[si]
    τ1 = mod(τ, LM.T) + 1
    Ye = LC.Y[s.fe]
    k = LC.kst[si][i]
    pc, mc = soc_power(s, i, LM.Δt, clamp(Ye[i, τ], 0.0, 1.0), clamp(Ye[i, τ1], 0.0, 1.0))
    return k * (mc - pc) + (1.0 - λ) * k * s.hours[i] * (Ye[i, τ1] - Ye[i, τ]) / LM.Δt
end

"""
One power-changing step τ of a storage pair us = (si, i, sj, jj) with template slot j of A's node n:
the nominal injection changes by +c t at A and −c t at B. The flow is re-routed (angles c times the
template), the leak is compensated share-preservingly at the leak nodes, and the expected losses
of the two units (`sx_loss`: λ_A c and −λ_B c) by the dispatch at their nodes, with the requirement
`sx_qloss` (Jacobian terms in `jac`; or the learned rule). In the discharge regime of a reservoir the
residuals at A's and B's nodes then do not change.
"""
function sx_step!(ws, jac, P::Problem, LM::LocalModel, LC::LocalCost, us::NTuple{4,Int}, j::Int, c::Float64, τ::Int;
                  κmin::Float64=1.0, learn=nothing)
    E = P.E
    fθ = E.fθ
    si, i, sj, jj = us
    n = E.stors[si].bus[i]
    xn, xv = LM.sxnod[n][j], LM.sxval[n][j]
    @inbounds for q in eachindex(xn)
        prep_node!(ws, P, LM, fθ, xn[q], c * xv[q], τ)
    end
    ln, lg = LM.sxlkn[n][j], LM.sxlkg[n][j]
    @inbounds for q in eachindex(ln)
        prorata!(ws, jac, P, LM, LC, ln[q], c * lg[q], τ; κmin=κmin, learn=learn)
    end
    m = E.stors[sj].bus[jj]
    λA, λB = sx_loss(E.stors[si], i), sx_loss(E.stors[sj], jj)
    if m == n
        g = (λA - λB) * c
        g != 0.0 && prorata!(ws, jac, P, LM, LC, n, g, τ; κmin=κmin, learn=learn,
                             qadj=sx_qloss(P, LM, LC, si, i, λA, τ) + sx_qloss(P, LM, LC, sj, jj, λB, τ))
    else
        λA != 0.0 && prorata!(ws, jac, P, LM, LC, n, λA * c, τ; κmin=κmin, learn=learn, qadj=sx_qloss(P, LM, LC, si, i, λA, τ))
        λB != 0.0 && prorata!(ws, jac, P, LM, LC, m, -λB * c, τ; κmin=κmin, learn=learn, qadj=sx_qloss(P, LM, LC, sj, jj, λB, τ))
    end
    return
end

"""
    prep_sxpair!(rng, wa, ws, P, LM, LC, τm, τ, jac; κmin) -> Bool

Single-instant storage pair (`sxpair`, a family of the local sweep): unit A uniform among the
storage units with at least the pair's storage hours, partner B by `sx_partner`; A's state of
charge at instant τ rises by t and B's falls by ρ t (ρ = E_A/E_B, equal nominal energy). At step
τ−1 (`wa`) the nominal injection changes by −κ_A t at A's node and +κ_A t at B's, at step τ (`ws`)
the opposite (κ_A = k h/Δt); the flow is re-routed on the pair's ball template and the leak is
compensated share-preservingly (Jacobian terms in `jac`).
"""
function prep_sxpair!(rng, wa, ws, P::Problem, LM::LocalModel, LC::LocalCost, τm::Int, τ::Int, jac; κmin::Float64=1.0)
    E = P.E
    isempty(LM.sxunits) && return false
    si, i = rand(rng, LM.sxunits)
    s = E.stors[si]
    EA = LC.kst[si][i] * s.hours[i]
    κA = EA / LM.Δt
    κA < κmin && return false
    j, sj, jj = sx_partner(rng, P, LM, LC, si, i; κmin=κmin)
    j == 0 && return false
    empty!(jac)
    sB = E.stors[sj]
    ρ = EA / (LC.kst[sj][jj] * sB.hours[jj])
    prep_node!(ws, P, LM, s.fe, i, 1.0, τ)
    prep_node!(ws, P, LM, sB.fe, jj, -ρ, τ)
    sx_step!(ws, jac, P, LM, LC, (si, i, sj, jj), j, κA, τ; κmin=κmin)
    sx_step!(wa, jac, P, LM, LC, (si, i, sj, jj), j, -κA, τm; κmin=κmin)
    return true
end

"""
    sxpair_move!(rng, wa, ws, P, LM, LC, τ, jac; κmin) -> BlockMove or nothing

A complete `sxpair` line move at instant τ, as the local sweep builds it: the discrete choices
(unit, partner) from `rng`, then the workspaces of steps τ−1 (`wa`) and τ (`ws`) finalized and
linked. Sample it with `sample_line!(rng, bm, β, ws.secprev)`, its Jacobian is `logjac(jac, t, bm)`,
commit it with `commit_block!`. The same `rng` state gives the same choices at every point of the
curve.
"""
function sxpair_move!(rng, wa::MoveWS, ws::MoveWS, P::Problem, LM::LocalModel, LC::LocalCost, τ::Int, jac;
                      κmin::Float64=1.0)
    T = LM.T
    τm = mod(τ - 2, T) + 1
    clear!(ws); clear!(wa)
    prep_sxpair!(rng, wa, ws, P, LM, LC, τm, τ, jac; κmin=κmin) || return nothing
    finalize!(ws, P, LM, LC, τ); finalize!(wa, P, LM, LC, τm); link_soc!(wa, ws, LM)
    bm = BlockMove([wa, ws], 2, [τm, τ], P, LM, LC, Float64[], Float64[])
    append!(bm.bps, wa.bps); append!(bm.bps, ws.bps)
    return bm
end

"""
Prepared interval storage pair (`sxlong`) of units A = (si, i) and B = (sj, jj): the end-step
workspaces `wa` (step a: re-routing, leak compensation, the linked storage terms) and `wb` (step
b = a+L: both units' state-of-charge entries at instant b, re-routing, leak compensation), the
carrier `wn` of the change at instant a+1, and the interior steps with both units' terms at t = 0.
Unit u's instants change by cu[u]·t, cu = (1, −ρ).
"""
mutable struct SXLong
    wa::MoveWS
    wb::MoveWS
    wn::MoveWS
    P::Problem
    LM::LocalModel
    LC::LocalCost
    us::NTuple{4,Int}                # (si, i, sj, jj)
    cu::NTuple{2,Float64}
    j::Int                           # partner slot of the template
    a::Int
    L::Int
    ilo::Float64                     # every raised instant of both units stays in [0, 1] for t in [ilo, ihi]
    ihi::Float64
    ncs::Int                         # cell sums per interior step (1: both units in one balance cell)
    τi::Vector{Int}                  # interior steps a+1 … a+L−1
    qlo::Vector{Float64}             # per interior step and unit: both its instants stay in [0, 1] for t in [qlo, qhi]
    qhi::Vector{Float64}
    c0::Vector{Float64}              # per interior step and unit: storage cost and injection at t = 0
    j0::Vector{Float64}
    sec0::Vector{Float64}            # per interior step: 3 storage secondaries per unit, then the cell sum(s)
    idr::Vector{Float64}             # per interior step and unit: injection change at the last evaluation
    sec::Vector{Float64}
    bps::Vector{Float64}
    lb::LineBuf
end
SXLong(wa, wb, wn, P, LM, LC) = SXLong(wa, wb, wn, P, LM, LC, (0, 0, 0, 0), (1.0, 0.0), 0, 0, 0, Inf, -Inf, 1, Int[],
                                       Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
                                       Float64[], LineBuf())

linebuf(m::SXLong) = m.lb
mv_bps(m::SXLong) = m.bps
mv_sec(m::SXLong) = m.sec
mv_wss(m::SXLong) = (m.wb, m.wa)
mv_hasstor(m::SXLong) = true
mv_fill!(m::SXLong) = mv_eval!(m, 0.0; mode=1)
mv_evalc!(m::SXLong, t) = mv_eval!(m, t; mode=2)

"Storage-term secondaries in the evaluation order of `mv_eval!(::SXLong)`: wb, the interior steps, wa."
function storage_mask!(mask::Vector{Bool}, m::SXLong)
    empty!(mask)
    stepmask!(mask, m.wb)
    for _ in eachindex(m.τi)
        for _ in 1:6; push!(mask, true); end
        for _ in 1:m.ncs; push!(mask, false); end
    end
    stepmask!(mask, m.wa)
    return mask
end

"Interval of t over which y + c t stays in [0, 1] (c ≠ 0)."
@inline sx_boxint(y::Float64, c::Float64) = c > 0 ? (-y / c, (1.0 - y) / c) : ((1.0 - y) / c, -y / c)

@inline sx_unit(m::SXLong, u::Int) = u == 1 ? (m.us[1], m.us[2]) : (m.us[3], m.us[4])

"""
    prep_sxlong!(m, si, i, sj, jj, j, a, L, jac; κmin, learn) -> Bool

Interval storage pair: A = (si, i) rises by t and B = (sj, jj) (at partner slot j of A's node) falls
by ρ t at the cyclic instants a+1 … a+L (1 ≤ L ≤ T−1): energy moves from step a to step b = a+L at
A and from b to a at B, the flows between them re-routed at both end steps, the leak compensated
share-preservingly (Jacobian terms in `jac`) or by the learned rule.
"""
function prep_sxlong!(m::SXLong, si::Int, i::Int, sj::Int, jj::Int, j::Int, a::Int, L::Int, jac;
                      κmin::Float64=1.0, learn=nothing)
    P, LM, LC = m.P, m.LM, m.LC
    E = P.E
    T = LM.T
    1 <= L <= T - 1 || return false
    sA, sB = E.stors[si], E.stors[sj]
    (LM.complete[sA.fe] && LM.complete[sB.fe]) || return false
    EA, EB = LC.kst[si][i] * sA.hours[i], LC.kst[sj][jj] * sB.hours[jj]
    κA = EA / LM.Δt
    (κA < κmin || EB / LM.Δt < κmin) && return false
    ρ = EA / EB
    m.us = (si, i, sj, jj); m.cu = (1.0, -ρ); m.j = j; m.a = a; m.L = L
    clear!(m.wa); clear!(m.wb); clear!(m.wn)
    empty!(jac)
    empty!(m.bps)
    τa, τb = mod1(a, T), mod1(a + L, T)
    n = sA.bus[i]
    prep_node!(m.wb, P, LM, sA.fe, i, 1.0, τb)                   # instant b (the start of step b)
    prep_node!(m.wb, P, LM, sB.fe, jj, -ρ, τb)
    if L >= 2                                                    # instant a+1: the end value of step a
        add_entry!(m.wn, sA.fe, i, 1.0)
        add_entry!(m.wn, sB.fe, jj, -ρ)
    end
    sx_step!(m.wa, jac, P, LM, LC, m.us, j, -κA, τa; κmin=κmin, learn=learn)
    sx_step!(m.wb, jac, P, LM, LC, m.us, j, κA, τb; κmin=κmin, learn=learn)
    finalize!(m.wa, P, LM, LC, τa)
    finalize!(m.wb, P, LM, LC, τb)
    link_soc!(m.wa, L >= 2 ? m.wn : m.wb, LM)
    append!(m.bps, m.wa.bps); append!(m.bps, m.wb.bps)
    # interior steps a+1 … a+L−1: both units' terms at t = 0, the clamp points of their raised start
    # instants, and the interval over which every raised instant (interior and b) stays in the box
    ni = L - 1
    cA, cB = LM.cell[sA.bus[i]], LM.cell[sB.bus[jj]]
    m.ncs = cA == cB ? 1 : 2
    nsq = 6 + m.ncs
    resize!(m.τi, ni); resize!(m.c0, 2ni); resize!(m.j0, 2ni); resize!(m.idr, 2ni)
    resize!(m.qlo, 2ni); resize!(m.qhi, 2ni); resize!(m.sec0, nsq * ni)
    Δt, cv = LM.Δt, E.c_viol
    lo, hi = -Inf, Inf
    inbox = true
    @inbounds for q in 1:ni
        τ = mod1(a + q, T)
        τ1 = mod1(τ + 1, T)
        m.τi[q] = τ
        for u in 1:2
            sg, iu = sx_unit(m, u)
            s = E.stors[sg]
            c = m.cu[u]
            Y = LC.Y[s.fe]
            y, y1 = Y[iu, τ], Y[iu, τ1]
            k = LC.kst[sg][iu]
            c0, d0, j0 = soc_cost(s, iu, k, k + E.κ0, Δt, cv, y, y1)
            p = 2(q - 1) + u
            m.c0[p], m.j0[p] = c0, j0
            o = nsq * (q - 1) + 3(u - 1)
            m.sec0[o+1] = d0
            m.sec0[o+2] = s.eta_in[iu] > 0 ? d0 - s.eta_in[iu] : 1.0
            m.sec0[o+3] = -d0 * s.eta_out[iu] - 1.0
            l0, h0 = sx_boxint(y, c)
            push!(m.bps, l0, h0)
            inbox &= 0.0 <= y <= 1.0
            lo = max(lo, l0); hi = min(hi, h0)
            if 0.0 <= y <= 1.0 && 0.0 <= y1 <= 1.0
                l1, h1 = sx_boxint(y1, c)
                m.qlo[p], m.qhi[p] = max(l0, l1), min(h0, h1)
            else
                m.qlo[p], m.qhi[p] = Inf, -Inf
            end
        end
        o = nsq * (q - 1) + 6
        m.sec0[o+1] = LC.S[cA, τ]
        m.ncs == 2 && (m.sec0[o+2] = LC.S[cB, τ])
    end
    for u in 1:2                                                 # instant b (the end value of the last interior step)
        sg, iu = sx_unit(m, u)
        yb = LC.Y[E.stors[sg].fe][iu, τb]
        inbox &= 0.0 <= yb <= 1.0
        l0, h0 = sx_boxint(yb, m.cu[u])
        lo = max(lo, l0); hi = min(hi, h0)
    end
    m.ilo, m.ihi = inbox ? (lo, hi) : (Inf, -Inf)
    return true
end

"""
    sxlong_choice(rng, P, LM, LC, si, i; κmin) -> (si, i, sj, jj, j, a, L) or nothing

The discrete choices of one `sxlong` move of unit (si, i) as the long-range sweep draws them:
the partner (`sx_partner`), the start a uniform on 1…T and the length L uniform on 1…T−1.
"""
function sxlong_choice(rng, P::Problem, LM::LocalModel, LC::LocalCost, si::Int, i::Int; κmin::Float64=1.0)
    T = LM.T
    j, sj, jj = sx_partner(rng, P, LM, LC, si, i; κmin=κmin)
    j == 0 && return nothing
    return (si, i, sj, jj, j, rand(rng, 1:T), rand(rng, 1:T-1))
end

"""
    prep_sxlong_random!(rng, m, jac; κmin, learn) -> Bool

A complete `sxlong` line move with seeded discrete choices: unit A uniform among `LM.sxunits`,
then `sxlong_choice`, then `prep_sxlong!`. Sample it with `sample_line!(rng, m, β, m.wa.secprev)`,
its Jacobian is `logjac(jac, t, m)`, commit it with `commit_sxlong!`.
"""
function prep_sxlong_random!(rng, m::SXLong, jac; κmin::Float64=1.0, learn=nothing)
    isempty(m.LM.sxunits) && return false
    si, i = rand(rng, m.LM.sxunits)
    sel = sxlong_choice(rng, m.P, m.LM, m.LC, si, i; κmin=κmin)
    sel === nothing && return false
    return prep_sxlong!(m, sel..., jac; κmin=κmin, learn=learn)
end

"Cost change ΔC(t) of the prepared storage pair; fills `m.sec` (order: wb, interior steps, wa)."
function mv_eval!(m::SXLong, t; mode::Int=0)
    P, LM, LC = m.P, m.LM, m.LC
    E = P.E
    T = LM.T
    Δ = eval_move!(m.wb, P, LM, LC, mod1(m.a + m.L, T), t; mode=mode)
    empty!(m.sec)
    append!(m.sec, m.wb.sec)
    if m.ilo <= t <= m.ihi
        append!(m.sec, m.sec0)
    else
        Δt, cv = LM.Δt, E.c_viol
        nsq = 6 + m.ncs
        si, i, sj, jj = m.us
        cA, cB = LM.cell[E.stors[si].bus[i]], LM.cell[E.stors[sj].bus[jj]]
        @inbounds for q in eachindex(m.τi)
            τ = m.τi[q]
            τ1 = mod1(τ + 1, T)
            for u in 1:2
                p = 2(q - 1) + u
                o = nsq * (q - 1) + 3(u - 1)
                if m.qlo[p] <= t <= m.qhi[p]                 # both instants of this unit stay in the box: no change
                    m.idr[p] = 0.0
                    push!(m.sec, m.sec0[o+1], m.sec0[o+2], m.sec0[o+3])
                    continue
                end
                sg, iu = sx_unit(m, u)
                s = E.stors[sg]
                c = m.cu[u]
                Y = LC.Y[s.fe]
                k = LC.kst[sg][iu]
                c1, d1, j1 = soc_cost(s, iu, k, k + E.κ0, Δt, cv, Y[iu, τ] + c * t, Y[iu, τ1] + c * t)
                m.idr[p] = j1 - m.j0[p]
                Δ += c1 - m.c0[p]
                push!(m.sec, d1, s.eta_in[iu] > 0 ? d1 - s.eta_in[iu] : 1.0, -d1 * s.eta_out[iu] - 1.0)
            end
            if m.ncs == 1
                nc = LM.ncell[cA]
                S0 = LC.S[cA, τ]; S1 = S0 + (m.idr[2q-1] + m.idr[2q])
                Δ += Δt * nc * (φbal(E, S1 / nc) - φbal(E, S0 / nc))
                push!(m.sec, S1)
            else
                for (u, c) in ((1, cA), (2, cB))
                    nc = LM.ncell[c]
                    S0 = LC.S[c, τ]; S1 = S0 + m.idr[2(q-1)+u]
                    Δ += Δt * nc * (φbal(E, S1 / nc) - φbal(E, S0 / nc))
                    push!(m.sec, S1)
                end
            end
        end
    end
    Δ += eval_move!(m.wa, P, LM, LC, mod1(m.a, T), t; next=(m.L >= 2 ? m.wn : m.wb), mode=mode)
    append!(m.sec, m.wa.sec)
    return Δ
end

"Apply the prepared storage pair with parameter t to the coefficients `v` and the caches."
function commit_sxlong!(m::SXLong, v::AbstractVector, t::Float64)
    t == 0.0 && return
    P, LM, LC = m.P, m.LM, m.LC
    E = P.E
    T = LM.T
    mv_eval!(m, t; mode=2)
    m.wa.trec = false; m.wb.trec = false
    apply_step!(m.wa, P, LC, v, mod1(m.a, T), t)
    apply_step!(m.wb, P, LC, v, mod1(m.a + m.L, T), t)          # includes instant b
    lv = P.lv[LM.ℓ+1]
    out = !(m.ilo <= t <= m.ihi)
    @inbounds for u in 1:2
        sg, iu = sx_unit(m, u)
        s = E.stors[sg]
        Y = LC.Y[s.fe]
        dy = m.cu[u] * t
        node = s.bus[iu]; c = LM.cell[node]
        for q in eachindex(m.τi)                                 # the raised start instants of the interior steps
            τ = m.τi[q]
            Y[iu, τ] += dy
            addv_node!(v, lv, E, s.fe, τ, iu, dy)
            if out
                dr = m.idr[2(q-1)+u]
                LC.r[node, τ] += dr
                LC.S[c, τ] += dr
            end
        end
    end
    return
end

"""
Rounds of storage-pair moves with disjoint footprints (the balance cells of their templates,
which contain every node, line and cell the move reads or writes at any step), by first fit in
the order of `mv`; the moves of a round commute and run in parallel.
"""
function sx_rounds(P::Problem, LM::LocalModel, mv)
    nc = length(LM.ncell)
    marks = Vector{Vector{Bool}}()
    rounds = Vector{Vector{Int}}()
    for (k, x) in enumerate(mv)
        foot = LM.sxfoot[P.E.stors[x[1]].bus[x[2]]][x[5]]
        r = findfirst(mk -> !any(c -> mk[c], foot), marks)
        if r === nothing
            push!(marks, fill(false, nc)); push!(rounds, Int[])
            r = length(rounds)
        end
        for c in foot
            marks[r][c] = true
        end
        push!(rounds[r], k)
    end
    return rounds
end

"""
    storage_cross_phase!(rng, P, LM, LC, dvs, pools; nmoves, learn, κmin) -> stats

The `sxlong` moves of the long-range storage sweep: for every storage unit with at least the pair's
storage hours (`LM.sxunits`) and κ ≥ κmin, `nmoves` interval pairs with a partner by `sx_partner`,
start a uniform on 1…T and length L uniform on 1…T−1, in random order, scheduled in rounds of
disjoint footprints (`sx_rounds`) whose moves run in parallel. Coefficient changes go into the
per-thread vectors `dvs` (the node values of different cells share coarse basis coefficients).
Selection uses only (k, x) and fixed probabilities. `stats`: (moves, accepted, Σ |t|·E_A in MWh).
"""
function storage_cross_phase!(rng, P::Problem, LM::LocalModel, LC::LocalCost, dvs, pools; nmoves::Int=8,
                              learn=nothing, κmin::Float64=1.0)
    E = P.E
    T = LM.T
    st = zeros(1, 3)
    (nmoves > 0 && T >= 2) || return st
    mv = NTuple{7,Int}[]                                         # (si, i, sj, jj, j, a, L)
    for (si, i) in LM.sxunits
        LC.kst[si][i] * E.stors[si].hours[i] / LM.Δt >= κmin || continue
        for _ in 1:nmoves
            sel = sxlong_choice(rng, P, LM, LC, si, i; κmin=κmin)
            sel === nothing || push!(mv, sel)
        end
    end
    isempty(mv) && return st
    shuffle!(rng, mv)
    seeds = rand(rng, UInt64, length(mv))
    rounds = sx_rounds(P, LM, mv)
    nth = Threads.maxthreadid()
    ms = [SXLong(pools[t].wss[1], pools[t].wss[2], pools[t].wss[3], P, LM, LC) for t in 1:nth]
    jacs = [NTuple{5,Float64}[] for _ in 1:nth]
    loc = [zeros(1, 3) for _ in 1:nth]
    for rd in rounds
        Threads.@threads :static for q in eachindex(rd)
            tid = Threads.threadid()
            si, i, sj, jj, j, a, L = mv[rd[q]]
            r = Xoshiro(seeds[rd[q]])
            m = ms[tid]
            jac = jacs[tid]
            prep_sxlong!(m, si, i, sj, jj, j, a, L, jac; κmin=κmin, learn=learn) || continue
            t = sample_line!(r, m, P.βv, m.wa.secprev)
            (t != 0.0 && !isempty(jac) && rand(r) >= exp(min(0.0, logjac(jac, t, m)))) && (t = 0.0)
            commit_sxlong!(m, dvs[tid], t)
            loc[tid][1, 1] += 1
            loc[tid][1, 2] += t != 0.0
            loc[tid][1, 3] += abs(t) * LC.kst[si][i] * E.stors[si].hours[i]
        end
    end
    for m in ms
        clear!(m.wa); clear!(m.wb); clear!(m.wn)
    end
    return sum(loc)
end
