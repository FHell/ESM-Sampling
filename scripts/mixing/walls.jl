# Walls of storage line moves (ledger E21, E27): after a line draw, the kink on each side of the
# walk where the slope rises most before the potential exceeds its minimum by `nats`, and which
# terms kink there: a line limit, a generator at a clamp (gas or other; at the move's own nodes or
# at a leak node), a compensation kink, a storage box, a storage power limit or charge/discharge
# switch, or a balance cell turning into shed. Include after `import ESMSampling as ML`.

"End-step workspaces (ws, step) in the evaluation order of the move, and the interior instants (field, index, coefficient, steps)."
function wall_parts(m)
    T = m.LM.T
    E = m.P.E
    if m isa ML.SXLong
        si, i, sj, jj = m.us
        return [(m.wb, mod1(m.a + m.L, T)), (m.wa, mod1(m.a, T))],
               [(E.stors[si].fe, i, m.cu[1], m.τi), (E.stors[sj].fe, jj, m.cu[2], m.τi)]
    elseif m isa ML.SLong
        ints = [(E.stors[m.si].fe, m.i, 1.0, m.τi)]
        m.L == T && return Tuple{ML.MoveWS,Int}[], ints
        return [(m.wb, mod1(m.a + m.L, T)), (m.wa, mod1(m.a, T))], ints
    else                                     # BlockMove of the local sweep: evaluated later step first
        return [(m.wss[q], m.τs[q]) for q in m.n:-1:1], Tuple{Int,Int,Float64,Vector{Int}}[]
    end
end

"Kinds of the terms that kink at t = p (several if they coincide); `own`: the nodes of the moved storage units."
function wall_kinds(m, p, own)
    P, LM, LC = m.P, m.LM, m.LC
    E = P.E
    sys = E.sys
    tol = 1e-6
    kinds = String[]
    ends, ints = wall_parts(m)
    δ = 1e-7 * max(abs(p), 1e-6)
    ML.mv_eval!(m, p - δ); sa = copy(ML.mv_sec(m))
    ML.mv_eval!(m, p + δ); sb = copy(ML.mv_sec(m))
    ML.mv_eval!(m, p)                        # the kinks (kφ) at p
    off = 0
    function secs!(nst, ncell, nall)         # one block of secondaries: storage (3 per term) first, cell sums last
        for q in 1:nall
            j = off + q
            (sa[j] < 0) != (sb[j] < 0) || continue
            if q <= 3nst
                push!(kinds, mod1(q, 3) == 1 ? "storage charge/discharge switch (hydro: zero discharge)" : "storage power limit")
            elseif q > nall - ncell
                sb[j] > sa[j] && push!(kinds, "shed (cell turns into deficit)")
            end
        end
        off += nall
    end
    interior = (m isa ML.SXLong || m isa ML.SLong) && !isempty(m.τi)
    for (k, (ws, τ)) in enumerate(ends)
        if k == 2 && interior                # the interior steps come between wb and wa
            nst = m isa ML.SXLong ? 2 : 1
            nsq = m isa ML.SXLong ? 6 + m.ncs : 4
            for _ in eachindex(m.τi); secs!(nst, nsq - 3nst, nsq); end
        end
        secs!(length(ws.stu), length(ws.cells), length(ws.sec))
        for e in ws.lines
            f1 = LC.F[e, τ] + p * ws.dfl[e]
            abs(abs(f1) - sys.line_s[e]) <= tol * sys.line_s[e] && push!(kinds, "line limit")
        end
        for (f, i) in ws.ent
            rl, g = LM.role[f]
            y1 = LC.Y[f][i, τ] + ML.entryΔ(ws, f, i, p)
            (abs(y1) <= tol || abs(y1 - 1) <= tol) || continue
            if rl == ML.ROLE_GEN
                wh = LM.bus[f][i] in own ? "own node" : "leak node"
                push!(kinds, E.gens[g].carrier * (abs(y1 - 1) <= tol ? " at capacity" : " at zero") * " ($wh)")
            elseif rl == ML.ROLE_SE
                push!(kinds, "storage box")
            end
        end
        for q in eachindex(ws.kQ)
            ws.kdyn[q] && continue
            abs(ws.kQ[q] + ws.kg[q] * p) <= tol * (abs(ws.kQ[q]) + abs(ws.kg[q] * p) + 1) &&
                push!(kinds, ws.knode[q] in own ? "compensation kink (own node)" : "compensation kink (leak node)")
        end
    end
    for (f, i, c, τs) in ints, τ in τs
        y1 = LC.Y[f][i, τ] + c * p
        (abs(y1) <= tol || abs(y1 - 1) <= tol) && push!(kinds, "storage box")
    end
    return unique!(kinds)
end

"""
Walls of the prepared move `m` after `sample_line!`: per side, the walk point with the largest rise
of the slope among the points before the potential exceeds its minimum by `nats`, with the kinds of
the terms that kink there. Returns [(side, t, kinds)].
"""
function walls(m, own; nats=4.0)
    lb = ML.linebuf(m)
    pts, cs = copy(lb.pts), copy(lb.cs)
    i0 = findfirst(==(0.0), pts)
    cmin = minimum(cs)
    out = Tuple{Int,Float64,Vector{String}}[]
    for sgn in (1, -1)
        best, bj = -Inf, 0
        for j in (sgn == 1 ? (i0+1:length(pts)-1) : (i0-1:-1:2))
            cs[j] > cmin + nats && break
            sl_in = (cs[j] - cs[j-sgn]) / abs(pts[j] - pts[j-sgn])
            sl_out = (cs[j+sgn] - cs[j]) / abs(pts[j+sgn] - pts[j])
            d = sl_out - sl_in
            d > best && (best = d; bj = j)
        end
        if bj == 0
            push!(out, (sgn, NaN, ["none before the walk end"]))
            continue
        end
        k = wall_kinds(m, pts[bj], own)
        push!(out, (sgn, pts[bj], isempty(k) ? ["unidentified"] : k))
    end
    return out
end
