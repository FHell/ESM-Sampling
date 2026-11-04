# Slope-jump form of the exact line sampler (docs/system/device.md §3).
#
# The dense form evaluates every term column at every point of the line: C·B·points·columns, with
# points growing with the columns. The slope-jump form evaluates every column only at its own
# breakpoints:
#   1. per column: 0, its primary points (clamp points of its arguments, both pieces of a kinked
#      argument, the kink point) and the roots of its secondary quantities (found on the column's
#      own points: the quantity depends on the column's arguments only), sorted (local points);
#   2. per column: the cost change at its local points (plus one point beyond each end), the slope
#      on every local segment, the slope jump at every local point and the slope just right of 0;
#   3. per balance slot: the local points of every column that injects into it (and 0), the slot
#      residual at those points, its roots, then the balance cost and its jumps in the same way;
#   4. all (point, jump) pairs sorted per site (one payload sort); the slope of every segment as the
#      slope right of 0 plus the jumps passed from 0 outwards, the potential by integrating the
#      slopes outwards from c(0) = 0 (no cancellation near 0), then the inverse-CDF draw of the dense
#      form (`sample_core`).
# The distinct points are the dense form's (0, the primaries, the kept roots); duplicates add
# zero-length segments without mass, so with the same uniforms both forms give the same t up to
# round-off. The exact guard is the same: a quantity with more sign changes than kept → t = 0.

"Local points p (C, B, L, n) extended by one point beyond each end (as `extend`)."
function extend4(p)
    L = size(p, 3)
    lo, hi = p[:, :, 1:1, :], p[:, :, L:L, :]
    return cat(lo .- (1.0 .+ absd.(lo)), p, hi .+ (1.0 .+ absd.(hi)); dims=3)
end

"(C, B, 1, n) zeros shaped like the stacked field X."
zcols(X) = 0.0 .* X

"(Q, g) of the kink of every column (C, B, 1, n), kid 0 → (0, 0) (φ ≡ 0); nothing without kinked columns."
function kinkcols(kk, kid::Vector{Int}, zc)
    (kk === nothing || !any(>(0), kid)) && return nothing, nothing
    nk = size(kk.Q, 4)
    idx = [q == 0 ? nk + 1 : q for q in kid]
    return cat(kk.Q, zc; dims=4)[:, :, :, idx], cat(kk.g, zc; dims=4)[:, :, :, idx]
end

"x0 + a t + ak φ(t) at the points t (C, B, L, n or 1), φ from the column's kink (Qc, gc); as `argval`."
function argloc(x0, a, ak, Qc, gc, t)
    v = a === nothing ? x0 .+ 0.0 .* t : x0 .+ a .* t
    (ak === nothing || Qc === nothing) && return v
    return v .+ ak .* (max.(Qc .+ gc .* t, 0.0) .- max.(Qc, 0.0))
end

"""
A candidate point num/den (0 where den = 0, as `sdiv`) into P, and into V the indicator of a real
candidate at exactly 0 (den ≠ 0, `mask` > 0): padded candidates are at 0, real ones may be too.
"""
function cand!(P, V, num, den; mask=nothing)
    x = sdiv.(num, den)
    ok = den .!= 0.0
    if mask !== nothing
        x = ifelse.(mask .> 0.0, x, 0.0)
        ok = ok .& (mask .> 0.0)
    end
    push!(P, x)
    push!(V, ifelse.(ok .& (x .== 0.0), 1.0, 0.0))
    return P
end

"""
Primary points of an argument x0 + a t + ak φ(t) of every column, as `primaries`: the clamp values
`cs` on both pieces of a kinked argument (and the kink point if `kink`), `mask` > 0 where they count.
"""
function argprims!(P, V, x0, a, ak, Qc, gc, cs; mask=nothing, kink=true)
    if ak === nothing || Qc === nothing
        a === nothing && return P
        for c in cs; cand!(P, V, c .- x0, a; mask=mask); end
        return P
    end
    Qp = max.(Qc, 0.0)
    A0 = a === nothing ? 0.0 : a
    for c in cs
        cand!(P, V, c .- x0 .- ak .* (Qc .- Qp), A0 .+ ak .* gc; mask=mask)   # piece Q + g t > 0
        a === nothing || cand!(P, V, c .+ (.-x0 .+ ak .* Qp), a; mask=mask)   # piece Q + g t ≤ 0
    end
    kink && cand!(P, V, .-Qc, gc)
    return P
end

"""
Slope jumps of every column of v (values at the sorted extended points pe, both (C, B, L+2, n)):
the jumps at the L inner points (C, B, L, n) and the slope just right of t = 0 (C, B, 1, n).
"""
function coljumps(pe, v)
    n = size(pe, 3)
    pa, pb = pe[:, :, 1:n-1, :], pe[:, :, 2:n, :]
    L = pb .- pa
    g = ifelse.(L .> 0.0, (v[:, :, 2:n, :] .- v[:, :, 1:n-1, :]) ./ ifelse.(L .> 0.0, L, 1.0), 0.0)
    J = g[:, :, 2:n-1, :] .- g[:, :, 1:n-2, :]
    s0 = sum(ifelse.((pa .<= 0.0) .& (pb .> 0.0), g, 0.0); dims=3)
    return J, s0
end

"Root count kept for a quantity of columns with moving-argument counts `cnt`: 1 if all are monotone."
nrkeep(cnt, NR) = any(cnt .>= 2) ? NR : (any(cnt .== 1) ? min(1, NR) : 0)

"Gather the local points X (C, B, L, n) of the columns of every slot: (C, B, L·mx, R); zero-padded."
function slotgather(X, slots::Vector{Int}, R::Int)
    n = size(X, 4)
    cols = [findall(==(j), slots) for j in 1:R]
    mx = maximum(length.(cols); init=0)
    mx == 0 && return nothing
    Xz = cat(X, 0.0 .* X[:, :, :, 1:1]; dims=4)
    idx = reduce(vcat, [vcat(c, fill(n + 1, mx - length(c))) for c in cols])
    C, B, L = size(X, 1), size(X, 2), size(X, 3)
    return rs(Xz[:, :, :, idx], C, B, L * mx, R)
end

"Sort the points p (C, B, N) with the payload v along dimension 3 (stable, `isless`)."
function sortpair(p, v)
    perm = sortperm(p; dims=3)
    return p[perm], v[perm]
end
function sortpair(p::Reactant.TracedRArray, v::Reactant.TracedRArray)
    a, b = Reactant.Ops.sort(p, v; comparator=(x, y, _, _) -> isless(x, y), dimension=3, is_stable=true)
    return a, b
end

"""
    jump_parts(tb, k; K, sz) -> (points, jumps, s0, over, nz0, s0abs)

Local points and slope jumps of every column and balance slot (vectors of (C, B, m) arrays), the
slope of ΔC just right of 0 (C, B, 1), the guard (C, B, 1), the number of real candidates at
exactly 0 among the points (C, B, 1; with the copies in the slot lists) and the sum of the
columns' absolute slopes right of 0 (the scale of the round-off of s0). `sz` (a Ref) receives the
largest number of elements per site of an evaluated tensor.
"""
function jump_parts(tb::TermBatch, k::Consts; K::Val{NR}=Val(4), sz=nothing) where {NR}
    S = tb.st
    Δt, cv = k.Δt, k.cv
    z = S.z
    C, B = size(z, 1), size(z, 2)
    zc = rs(z, C, B, 1, 1)
    over = z .> 0.0
    pts, jmp = Any[], Any[]
    s0 = z
    s0abs = z
    nz0 = z
    roles = Any[]                                       # (local points, slots, injection at points)
    hasslots = S.nr > 0
    # real candidates at 0 of a group (V: indicators), counted once per list its points enter: the
    # group's own and the list of every slot a column injects into (`copies`, per column)
    addz!(V, copies::Vector{Float64}) =
        (nz0 = nz0 .+ rs(sum(sum(cat(V...; dims=3); dims=3) .* reshape(copies, 1, 1, 1, :); dims=4), C, B, 1))
    copies(slotsets...) = Float64[1 + (hasslots ? sum(sl[j] > 0 for sl in slotsets) : 0) for j in eachindex(slotsets[1])]
    track(x) = (sz === nothing || (sz[] = max(sz[], size(x, 3) * size(x, 4))); x)
    function addgroup!(P, f)
        pe = extend4(P)
        v = track(f(pe))
        J, s = coljumps(pe, v)
        push!(pts, flat3(P)); push!(jmp, flat3(J))
        s0 = s0 .+ rs(sum(s; dims=4), C, B, 1)
        s0abs = s0abs .+ rs(sum(absd.(s); dims=4), C, B, 1)
    end
    function roots!(Rv, zq, pe, cnt)
        nr = nrkeep(cnt, NR)
        nr == 0 && return
        track(zq)
        rts, cn = rootsK(zq, pe, nr)
        push!(Rv, rts)
        over = over .| (rs(maximum(cn; dims=4), C, B, 1) .> nr)
    end
    if S.gen !== nothing && (S.gen.a !== nothing || S.gen.ak !== nothing)
        g = S.gen
        Qc, gc = kinkcols(S.kk, g.kid, zc)
        P, V = Any[zcols(g.y0)], Any[]
        argprims!(P, V, g.y0, g.a, g.ak, Qc, gc, (0.0, 1.0))
        argprims!(P, V, g.y0, g.a, g.ak, Qc, gc, (g.yk,); mask=g.hv, kink=false)     # vintage kink
        addz!(V, copies(g.slot))
        P = sort(cat(P...; dims=3); dims=3)
        y(t) = argloc(g.y0, g.a, g.ak, Qc, gc, t)
        addgroup!(P, pe -> gen_c.(Δt, cv, y(pe), g.κ, g.w, g.mc, g.clo, g.chi, g.plo, g.hv) .-
                           gen_c.(Δt, cv, g.y0, g.κ, g.w, g.mc, g.clo, g.chi, g.plo, g.hv))
        push!(roles, (P, g.slot, pe -> .-(gen_p.(y(pe), g.κ) .- gen_p.(g.y0, g.κ))))
    end
    q = S.uc
    if q !== nothing && q.on !== nothing && any(x -> x !== nothing, (q.ay, q.aky, q.au, q.aku, q.au1, q.aku1))
        args = ((q.y0, q.ay, q.aky, q.kidy), (q.u0, q.au, q.aku, q.kidu), (q.u10, q.au1, q.aku1, q.kidu1))
        kc = [kinkcols(S.kk, a[4], zc) for a in args]
        P, V = Any[zcols(q.y0)], Any[]
        for (j, (x0, a, ak, _)) in enumerate(args)
            argprims!(P, V, x0, a, ak, kc[j]..., (0.0, 1.0))
        end
        isempty(V) || addz!(V, ones(length(q.kidy)))
        P = sort(cat(P...; dims=3); dims=3)
        av(j, t) = argloc(args[j][1], args[j][2], args[j][3], kc[j]..., t)
        pe = extend4(P)
        yc, ucl, uc1 = clamp01.(av(1, pe)), clamp01.(av(2, pe)), clamp01.(av(3, pe))
        Rv = Any[]
        roots!(Rv, q.on .* (yc .- ucl), pe, q.cy .+ q.cu)
        roots!(Rv, q.on .* (q.gmin .* ucl .- yc), pe, q.cy .+ q.cu)
        roots!(Rv, q.on .* (uc1 .- ucl), pe, q.cu .+ q.cu1)
        P = isempty(Rv) ? P : sort(cat(P, Rv...; dims=3); dims=3)
        addgroup!(P, pe -> q.on .* (uc_val.(Δt, cv, clamp01.(av(1, pe)), av(2, pe), av(3, pe), q.κ, q.w, q.gmin, q.su, q.nl) .-
                                    uc_val.(Δt, cv, clamp01.(q.y0), q.u0, q.u10, q.κ, q.w, q.gmin, q.su, q.nl)))
    end
    if S.link !== nothing && S.link.a !== nothing
        l = S.link
        P, V = Any[zcols(l.y0)], Any[]
        for c in (-1.0, 1.0, 0.0); cand!(P, V, c .- l.y0, l.a); end
        addz!(V, copies(l.slot0, l.slot1))
        P = sort(cat(P...; dims=3); dims=3)
        yl(t) = l.y0 .+ l.a .* t
        addgroup!(P, pe -> link_c.(Δt, cv, yl(pe), l.cap) .- link_c.(Δt, cv, l.y0, l.cap))
        push!(roles, (P, l.slot0, pe -> link_j0.(yl(pe), l.cap, l.η) .- link_j0.(l.y0, l.cap, l.η)))
        push!(roles, (P, l.slot1, pe -> link_j1.(yl(pe), l.cap, l.η) .- link_j1.(l.y0, l.cap, l.η)))
    end
    if S.line !== nothing && S.line.df !== nothing
        ln = S.line
        U4 = rs(k.ψ.U, 1, 1, length(k.ψ.U), 1)
        P, V = Any[zcols(ln.f0)], Any[]
        cand!(P, V, .-ln.f0, ln.df)
        cand!(P, V, U4 .* ln.s .- ln.f0, ln.df)
        cand!(P, V, .-U4 .* ln.s .- ln.f0, ln.df)
        addz!(V, copies(ln.near, ln.far))
        P = sort(cat(P...; dims=3); dims=3)
        fl(t) = ln.f0 .+ ln.df .* t
        u0 = absd.(ln.f0) ./ ln.s
        dloss(f) = ln.α .* ln.s .^ 2 .* (hinge(k.Lq, absd.(f) ./ ln.s) .- hinge(k.Lq, u0))
        addgroup!(P, pe -> (cv * Δt) .* (ln.s .* (hinge(k.ψ, absd.(fl(pe)) ./ ln.s) .- hinge(k.ψ, u0))))
        push!(roles, (P, ln.near, pe -> (f = fl(pe); (f .- ln.f0) .+ 0.5 .* dloss(f))))
        push!(roles, (P, ln.far, pe -> (f = fl(pe); .-(f .- ln.f0) .+ 0.5 .* dloss(f))))
    end
    if S.soc !== nothing && (S.soc.ae !== nothing || S.soc.ae1 !== nothing)
        so = S.soc
        P, V = Any[zcols(so.e0)], Any[]
        for (x0, a) in ((so.e0, so.ae), (so.e10, so.ae1))
            a === nothing && continue
            cand!(P, V, 0.0 .- x0, a); cand!(P, V, 1.0 .- x0, a)
        end
        addz!(V, copies(so.slot))
        P = sort(cat(P...; dims=3); dims=3)
        ye(t) = so.e0 .+ (so.ae === nothing ? 0.0 .* t : so.ae .* t)
        ye1(t) = so.e10 .+ (so.ae1 === nothing ? 0.0 .* t : so.ae1 .* t)
        dd(t) = soc_d.(Δt, ye(t), ye1(t), so.h, so.qt)
        pe = extend4(P)
        d1 = dd(pe)
        Rv = Any[]
        for zq in (d1, ifelse.(so.ηin .> 0.0, d1 .- so.ηin, 1.0), .-d1 .* so.ηout .- 1.0)
            roots!(Rv, zq, pe, so.cnt)
        end
        P = isempty(Rv) ? P : sort(cat(P, Rv...; dims=3); dims=3)
        addgroup!(P, pe -> soc_c.(Δt, cv, ye(pe), ye1(pe), so.k, so.w, so.h, so.ηin, so.ηout, so.qt, so.mc) .-
                           soc_c.(Δt, cv, so.e0, so.e10, so.k, so.w, so.h, so.ηin, so.ηout, so.qt, so.mc))
        d0 = soc_d.(Δt, so.e0, so.e10, so.h, so.qt)
        push!(roles, (P, so.slot, pe -> soc_j.(dd(pe), so.k, so.ηin, so.ηout) .- soc_j.(d0, so.k, so.ηin, so.ηout)))
    end
    R = S.nr
    if R > 0 && !isempty(roles)
        Ps = Any[rs(z, C, B, 1, 1) .+ 0.0 .* S.r0]
        for (P, sl, _) in roles
            G = slotgather(P, sl, R)
            G === nothing || push!(Ps, G)
        end
        r1(pe) = begin                                   # residual of every slot at its points pe (C, B, L, R)
            inj = nothing
            z4 = 0.0 .* pe[:, :, :, 1:1]
            for (_, sl, f) in roles
                X = track(f(pe[:, :, :, [s == 0 ? 1 : s for s in sl]]))
                inj = addinj(inj, slotsum(X, sl, R, z4))
            end
            S.r0 .+ (inj === nothing ? 0.0 .* pe : inj)
        end
        P = sort(cat(Ps...; dims=3); dims=3)
        pe = extend4(P)
        Rv = Any[]
        roots!(Rv, track(r1(pe)), pe, fill(2, R))
        P = isempty(Rv) ? P : sort(cat(P, Rv...; dims=3); dims=3)
        addgroup!(P, pe -> bal_val.(Δt, k.cshed, k.cdump, r1(pe)) .- bal_val.(Δt, k.cshed, k.cdump, S.r0))
    end
    return pts, jmp, s0, over, nz0, s0abs
end

"Suffix sums Σ_{i ≥ m} x_i along dimension 3, accumulated from the end (no cancellation against far terms)."
rcumsum(x) = (n = size(x, 3); cumsum(x[:, :, n:-1:1]; dims=3)[:, :, n:-1:1])

"""
Potential βΔC at the sorted points q (C, B, N) from the jumps J at them and the slope s0 just right
of 0 (0 is one of the points): segment slopes from 0 outwards, values integrated from c(0) = 0
outwards; returns the extended points and values for `sample_core`. A segment slope within the
round-off of its sum (16 ε × the absolute slopes and jumps summed: `s0abs` for s0, `J0abs` for the
folded entry at 0) is 0: a flat stretch stays exactly flat, as in the dense form.
"""
function jump_values(q, J, s0, β, s0abs=0.0 .* s0, J0abs=0.0 .* s0)
    N = size(q, 3)
    ε16 = 16 * eps(Float64)
    pos = q .> 0.0
    JR = ifelse.(pos, J, 0.0)
    JL = ifelse.(pos, 0.0, J)
    cR = cumsum(JR; dims=3)                               # Σ_{i ≤ k, q_i > 0} J_i (from 0 outwards)
    sL = rcumsum(JL)                                      # Σ_{i ≥ k, q_i ≤ 0} J_i (from 0 outwards)
    aR = cumsum(absd.(JR); dims=3)
    aL = rcumsum(absd.(JL))
    qa, qb = q[:, :, 1:N-1], q[:, :, 2:N]
    rseg = qa .>= 0.0                                     # segments right of 0 (the others end at ≤ 0)
    sl = ifelse.(rseg, s0 .+ cR[:, :, 1:N-1], s0 .- sL[:, :, 2:N])
    tol = ε16 .* (s0abs .+ absd.(s0) .+ ifelse.(rseg, aR[:, :, 1:N-1], aL[:, :, 2:N] .+ J0abs))
    sl = ifelse.(absd.(sl) .<= tol, 0.0, sl)
    w = sl .* (qb .- qa)
    z = 0.0 .* q[:, :, 1:1]
    right = cat(z, cumsum(ifelse.(rseg, w, 0.0); dims=3); dims=3)        # Σ_{i < m, q_i ≥ 0} w_i
    left = cat(rcumsum(ifelse.(qb .<= 0.0, w, 0.0)), z; dims=3)          # Σ_{i ≥ m, q_{i+1} ≤ 0} w_i
    c = ifelse.(pos, right, .-left)
    gl = s0 .- sL[:, :, 1:1]                              # slope on the left ray (increasing t)
    gr = s0 .+ cR[:, :, N:N]
    base = ε16 .* (s0abs .+ absd.(s0))
    gl = ifelse.(absd.(gl) .<= base .+ ε16 .* (aL[:, :, 1:1] .+ J0abs), 0.0, gl)
    gr = ifelse.(absd.(gr) .<= base .+ ε16 .* aR[:, :, N:N], 0.0, gr)
    lo, hi = q[:, :, 1:1], q[:, :, N:N]
    plo, phi = lo .- (1.0 .+ absd.(lo)), hi .+ (1.0 .+ absd.(hi))
    clo = c[:, :, 1:1] .+ gl .* (plo .- lo)
    chi = c[:, :, N:N] .+ gr .* (phi .- hi)
    return cat(plo, q, phi; dims=3), β .* cat(clo, c, chi; dims=3)
end

"""
The sorted global points and jumps, compacted: every entry at exactly 0 is folded into one entry
(0, Σ its jumps), the others sorted; with a cap Nc > 0 only the first Nc entries are kept; the
folded ones are moved onto the largest point (zero-length segments). Returns (q, J, count, J0abs):
the count of real candidates (entries ≠ 0, and real candidates at 0; line-invariant; count ≥ Nc
means the kept entries may miss points: guard) and Σ |jumps| folded at 0 (round-off scale).
"""
function jump_points(pts, jmp, nz0, cap::Int)
    qa, Ja = cat(pts...; dims=3), cat(jmp...; dims=3)
    z = qa .== 0.0
    cnt = rs(sum(ifelse.(z, 0.0, 1.0); dims=3), size(qa, 1), size(qa, 2), 1) .+ nz0
    J0 = sum(ifelse.(z, Ja, 0.0); dims=3)
    J0abs = sum(ifelse.(z, absd.(Ja), 0.0); dims=3)
    q, J = sortpair(cat(0.0 .* J0, ifelse.(z, Inf, qa); dims=3), cat(J0, ifelse.(z, 0.0, Ja); dims=3))
    (cap <= 0 || cap >= size(q, 3)) || ((q, J) = (q[:, :, 1:cap], J[:, :, 1:cap]))
    fin = isfinite.(q)
    qmax = maximum(ifelse.(fin, q, -Inf); dims=3)          # ≥ 0: the zero entry is finite
    return ifelse.(fin, q, qmax), ifelse.(fin, J, 0.0), cnt, J0abs
end

"""
    sample_jump(tb, k, u1, u2; K, cap) -> (t, over)

Exact draw from ∝ exp(−βΔC(t)) in the slope-jump form (same points, uniforms and guard as
`sample_dense`): t (C, B, 1) and the guard (C, B, 1). `cap` > 0 keeps that many sorted entries;
a site with more real candidates stays at t = 0 (the count is a property of the line, so the
guard keeps the kernel π-invariant).
"""
function sample_jump(tb::TermBatch, k::Consts, u1, u2; K::Val=Val(4), cap::Int=0)
    pts, jmp, s0, over, nz0, s0abs = jump_parts(tb, k; K=K)
    q, J, cnt, J0abs = jump_points(pts, jmp, nz0, cap)
    cap > 0 && (over = over .| (cnt .> cap - 1))
    pe, c = jump_values(q, J, s0, k.β, s0abs, J0abs)
    t = sample_core(k, pe, c, u1, u2)
    return ifelse.(over, 0.0, t), over
end

"""
    jump_size(tb, k; K) -> (entries, elements, counts)

Shape of the slope-jump form of a term batch: the global entries per site (before the cap), the
largest number of elements per site of an evaluated tensor (columns × local points, slots × slot
points, the entries) and the real-candidate count of every site (C, B) for choosing the cap.
"""
function jump_size(tb::TermBatch, k::Consts; K::Val=Val(4))
    sz = Ref(0)
    pts, jmp, _, _, nz0 = jump_parts(tb, k; K=K, sz=sz)
    N = sum(x -> size(x, 3), pts; init=0) + 1
    _, _, cnt = jump_points(pts, jmp, nz0, 0)
    return N, max(sz[], N), cnt[:, :, 1]
end
