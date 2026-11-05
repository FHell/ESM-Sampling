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

"""
Largest axis sorted by a compare-exchange network instead of a sort op (`sort3`). XLA's sort over
a short axis runs at low occupancy on the GPU; min/max pairs fuse into the elementwise producers.
"""
const SMALLSORT = 8

"""
X sorted along dimension 3: a compare-exchange network (insertion order, n(n−1)/2 min/max pairs)
for at most `SMALLSORT` entries, else the sort op. Same values as `sort(X; dims=3)` (the points are
never NaN; ±0 may be ordered differently, which no later step distinguishes).
"""
function sort3(X)
    n = size(X, 3)
    n > SMALLSORT && return sort(X; dims=3)
    n == 1 && return X
    xs = ndims(X) == 3 ? Any[X[:, :, i:i] for i in 1:n] : Any[X[:, :, i:i, :] for i in 1:n]
    for i in 2:n, j in i-1:-1:1
        a, b = xs[j], xs[j+1]
        xs[j], xs[j+1] = min.(a, b), max.(a, b)
    end
    return cat(xs...; dims=3)
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
    roles = Any[]                                       # (group, slots, injection at points)
    gP, gJ = Any[], Any[]                               # local points and cost jumps of every column group
    # real candidates at 0 of a group (V: indicators), every column point counted once
    addz!(V, _=nothing) = (nz0 = nz0 .+ rs(sum(sum(catn(collect(V), 3); dims=3); dims=4), C, B, 1))
    copies(slotsets...) = nothing
    track(x) = (sz === nothing || (sz[] = max(sz[], size(x, 3) * size(x, 4))); x)
    function addgroup!(P, f)
        pe = extend4(P)
        v = track(f(pe))
        J, s = coljumps(pe, v)
        push!(gP, P); push!(gJ, J)
        s0 = s0 .+ rs(sum(s; dims=4), C, B, 1)
        s0abs = s0abs .+ rs(sum(absd.(s); dims=4), C, B, 1)
        return length(gP)
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
        P = sort3(cat(P...; dims=3))
        y(t) = argloc(g.y0, g.a, g.ak, Qc, gc, t)
        addgroup!(P, pe -> gen_c.(Δt, cv, y(pe), g.κ, g.w, g.mc, g.clo, g.chi, g.plo, g.hv) .-
                           gen_c.(Δt, cv, g.y0, g.κ, g.w, g.mc, g.clo, g.chi, g.plo, g.hv))
        push!(roles, (length(gP), g.slot, pe -> .-(gen_p.(y(pe), g.κ) .- gen_p.(g.y0, g.κ)),
                      pe -> absd.(gen_p.(y(pe), g.κ)) .+ absd.(gen_p.(g.y0, g.κ))))
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
        P = sort3(cat(P...; dims=3))
        av(j, t) = argloc(args[j][1], args[j][2], args[j][3], kc[j]..., t)
        pe = extend4(P)
        yc, ucl, uc1 = clamp01.(av(1, pe)), clamp01.(av(2, pe)), clamp01.(av(3, pe))
        Rv = Any[]
        roots!(Rv, zsnap(q.on .* (yc .- ucl), UC_TOL), pe, q.cy .+ q.cu)
        roots!(Rv, zsnap(q.on .* (q.gmin .* ucl .- yc), UC_TOL), pe, q.cy .+ q.cu)
        roots!(Rv, zsnap(q.on .* (uc1 .- ucl), UC_TOL), pe, q.cu .+ q.cu1)
        P = isempty(Rv) ? P : sort3(cat(P, Rv...; dims=3))
        addgroup!(P, pe -> q.on .* (uc_val.(Δt, cv, clamp01.(av(1, pe)), av(2, pe), av(3, pe), q.κ, q.w, q.gmin, q.su, q.nl) .-
                                    uc_val.(Δt, cv, clamp01.(q.y0), q.u0, q.u10, q.κ, q.w, q.gmin, q.su, q.nl)))
    end
    if S.link !== nothing && S.link.a !== nothing
        l = S.link
        P, V = Any[zcols(l.y0)], Any[]
        for c in (-1.0, 1.0, 0.0); cand!(P, V, c .- l.y0, l.a); end
        addz!(V, copies(l.slot0, l.slot1))
        P = sort3(cat(P...; dims=3))
        yl(t) = l.y0 .+ l.a .* t
        addgroup!(P, pe -> link_c.(Δt, cv, yl(pe), l.cap) .- link_c.(Δt, cv, l.y0, l.cap))
        push!(roles, (length(gP), l.slot0, pe -> link_j0.(yl(pe), l.cap, l.η) .- link_j0.(l.y0, l.cap, l.η),
                      pe -> absd.(link_j0.(yl(pe), l.cap, l.η)) .+ absd.(link_j0.(l.y0, l.cap, l.η))))
        push!(roles, (length(gP), l.slot1, pe -> link_j1.(yl(pe), l.cap, l.η) .- link_j1.(l.y0, l.cap, l.η),
                      pe -> absd.(link_j1.(yl(pe), l.cap, l.η)) .+ absd.(link_j1.(l.y0, l.cap, l.η))))
    end
    if S.line !== nothing && S.line.df !== nothing
        ln = S.line
        U4 = rs(k.ψ.U, 1, 1, length(k.ψ.U), 1)
        P, V = Any[zcols(ln.f0)], Any[]
        cand!(P, V, .-ln.f0, ln.df)
        cand!(P, V, U4 .* ln.s .- ln.f0, ln.df)
        cand!(P, V, .-U4 .* ln.s .- ln.f0, ln.df)
        addz!(V, copies(ln.near, ln.far))
        P = sort3(cat(P...; dims=3))
        fl(t) = ln.f0 .+ ln.df .* t
        u0 = absd.(ln.f0) ./ ln.s
        dloss(f) = ln.α .* ln.s .^ 2 .* (hinge(k.Lq, absd.(f) ./ ln.s) .- hinge(k.Lq, u0))
        addgroup!(P, pe -> (cv * Δt) .* linewc(ln, ln.s .* (hinge(k.ψ, absd.(fl(pe)) ./ ln.s) .- hinge(k.ψ, u0))))
        fa(pe) = (f = fl(pe); absd.(f) .+ absd.(ln.f0) .+ 0.5 .* absd.(dloss(f)))
        push!(roles, (length(gP), ln.near, pe -> (f = fl(pe); (f .- ln.f0) .+ 0.5 .* dloss(f)), fa))
        push!(roles, (length(gP), ln.far, pe -> (f = fl(pe); .-(f .- ln.f0) .+ 0.5 .* dloss(f)), fa))
    end
    if S.soc !== nothing && (S.soc.ae !== nothing || S.soc.ae1 !== nothing)
        so = S.soc
        P, V = Any[zcols(so.e0)], Any[]
        for (x0, a) in ((so.e0, so.ae), (so.e10, so.ae1))
            a === nothing && continue
            cand!(P, V, 0.0 .- x0, a); cand!(P, V, 1.0 .- x0, a)
        end
        addz!(V, copies(so.slot))
        P = sort3(cat(P...; dims=3))
        ye(t) = so.e0 .+ (so.ae === nothing ? 0.0 .* t : so.ae .* t)
        ye1(t) = so.e10 .+ (so.ae1 === nothing ? 0.0 .* t : so.ae1 .* t)
        dd(t) = soc_d.(Δt, ye(t), ye1(t), so.h, so.qt)
        pe = extend4(P)
        d1 = dd(pe)
        Rv = Any[]
        tol = soc_tol(Δt, so.h, so.qt, so.ηout)
        for zq in (zsnap(d1, tol), zsnap(ifelse.(so.ηin .> 0.0, d1 .- so.ηin, 1.0), tol), zsnap(.-d1 .* so.ηout .- 1.0, tol))
            roots!(Rv, zq, pe, so.cnt)
        end
        P = isempty(Rv) ? P : sort3(cat(P, Rv...; dims=3))
        addgroup!(P, pe -> soc_c.(Δt, cv, ye(pe), ye1(pe), so.k, so.w, so.h, so.ηin, so.ηout, so.qt, so.mc) .-
                           soc_c.(Δt, cv, so.e0, so.e10, so.k, so.w, so.h, so.ηin, so.ηout, so.qt, so.mc))
        d0 = soc_d.(Δt, so.e0, so.e10, so.h, so.qt)
        push!(roles, (length(gP), so.slot, pe -> soc_j.(dd(pe), so.k, so.ηin, so.ηout) .- soc_j.(d0, so.k, so.ηin, so.ηout),
                      pe -> absd.(soc_j.(dd(pe), so.k, so.ηin, so.ηout)) .+ absd.(soc_j.(d0, so.k, so.ηin, so.ηout))))
    end
    R = S.nr
    if R > 0 && !isempty(roles)
        # The balance cost Δt (c_shed r₊ + c_dump r₋) of a slot bends where its residual r bends (at
        # the points of the columns that inject into it) and where r crosses 0. Its slope jump at a
        # column point p is Δt w(r(p)) × (that column's injection jump at p), w = c_shed for r ≥ 0,
        # −c_dump below, so it is added to the column's own entry; the crossings are extra entries
        # with the jump Δt (w after − w before) r'. The convention r = 0 ↦ surplus side is the same in
        # both, so the jumps add up to the cost's slopes also when r vanishes at a point.
        # 1. the slot lists (point 0 and the points of every injecting column, static layout) and
        #    the residual there (one evaluation of every column at the points of its slot)
        Ps = Any[rs(z, C, B, 1, 1) .+ 0.0 .* S.r0]
        offs, np = Int[], 1
        for (gid, sl, _, _) in roles
            G = slotgather(gP[gid], sl, R)
            push!(offs, np)
            G === nothing || (push!(Ps, G); np += size(G, 3))
        end
        r1(pe; absv=false) = begin                       # residual of every slot at its points pe (C, B, L, R); absv: its round-off scale
            inj = nothing                                # |r0| + Σ(|injection| + |injection at 0|), as `eval_terms(…; absv = true)`
            z4 = 0.0 .* pe[:, :, :, 1:1]
            for (_, sl, f, fa) in roles
                X = track((absv ? fa : f)(pe[:, :, :, [s == 0 ? 1 : s for s in sl]]))
                inj = addinj(inj, slotsum(X, sl, R, z4))
            end
            (absv ? absd.(S.r0) : S.r0) .+ (inj === nothing ? 0.0 .* pe : inj)
        end
        Pu = cat(Ps...; dims=3)                           # (C, B, np, R), unsorted
        rv = track(r1(Pu))
        rvf = rs(rv, C, B, np * R)
        # 2. the balance part of the jump at every column point, and the slope of r right of 0
        rp0 = nothing
        z1 = 0.0 .* zc
        for (ρ, (gid, sl, f, _)) in enumerate(roles)
            P = gP[gid]
            m, n = size(P, 3), size(P, 4)
            pos = zeros(Int, n)
            for j in 1:R
                for (kk, c) in enumerate(findall(==(j), sl)); pos[c] = kk; end
            end
            idx = [sl[c] == 0 ? 1 : offs[ρ] + i + m * (pos[c] - 1) + np * (sl[c] - 1) for i in 1:m, c in 1:n]
            rc = rs(rvf[:, :, vec(idx)], C, B, m, n)            # residual of the column's slot at its points
            pe = extend4(P)
            Ji, si0 = coljumps(pe, track(f(pe)))               # injection jumps and slope right of 0
            on = reshape(Float64.(sl .> 0), 1, 1, 1, n)
            gJ[gid] = gJ[gid] .+ (Δt .* on) .* ifelse.(rc .>= 0.0, k.cshed, -k.cdump) .* Ji
            rp0 = addinj(rp0, slotsum(si0, sl, R, z1))
        end
        rp0 === nothing && (rp0 = 0.0 .* S.r0)
        up = (S.r0 .> 0.0) .| ((S.r0 .== 0.0) .& (rp0 .> 0.0))  # side of r just right of 0
        sb = Δt .* ifelse.(up, k.cshed, -k.cdump) .* rp0
        s0 = s0 .+ rs(sum(sb; dims=4), C, B, 1)
        s0abs = s0abs .+ rs(sum(absd.(sb); dims=4), C, B, 1)
        # 3. the crossings of 0: the slot lists sorted with r as payload (entry axis first), r on the rays
        Pq, rq = sortpair1(permutedims(Pu, (3, 1, 2, 4)), permutedims(rv, (3, 1, 2, 4)))
        lo, hi = Pq[1:1, :, :, :], Pq[np:np, :, :, :]
        pext = cat(lo .- (1.0 .+ absd.(lo)), hi .+ (1.0 .+ absd.(hi)); dims=1)     # (2, C, B, R)
        pext4 = permutedims(pext, (2, 3, 1, 4))
        rext = permutedims(r1(pext4), (3, 1, 2, 4))
        aext = permutedims(r1(pext4; absv=true), (3, 1, 2, 4))          # (2, C, B, R): the round-off scale of r on the rays
        rts, cn, jr = slot_cross(cat(rext[1:1, :, :, :], rq, rext[2:2, :, :, :]; dims=1),
                                 cat(pext[1:1, :, :, :], Pq, pext[2:2, :, :, :]; dims=1), NR, Δt, k.cshed, k.cdump;
                                 raytol=(RAY_ULP * eps(Float64)) .* aext)
        over = over .| (rs(maximum(cn; dims=(1, 4)), C, B, 1) .> NR)
        push!(gP, permutedims(rts, (2, 3, 1, 4))); push!(gJ, permutedims(jr, (2, 3, 1, 4)))
    end
    for (P, J) in zip(gP, gJ)
        push!(pts, flat3(P)); push!(jmp, flat3(J))
    end
    return pts, jmp, s0, over, nz0, s0abs
end

"""
    slot_cross(r, pe, NR, Δt, cs, cd) -> (roots, count, jumps)

Crossings of 0 of the piecewise-linear residuals r over the sorted extended points pe (both
(n, C, B, R), entry axis first; linear between the points and on the rays): the NR smallest
(NR, C, B, R; 0 where fewer), their number (1, C, B, R) and the slope jumps of the balance cost
there, Δt (w after − w before) r' (0 where none). A crossing is a change of the side r ≥ 0 between
the ends of a segment (on a ray: between its end and infinity), so it is also found where r vanishes
at a point; a crossing on a ray farther than `RAYMAX` is dropped (as in `rootsK`).
"""
function slot_cross(z, pe, NR::Int, Δt, cs, cd; raytol=nothing)
    n = size(pe, 1)
    za, zb = z[1:n-1, :, :, :], z[2:n, :, :, :]
    pa, pb = pe[1:n-1, :, :, :], pe[2:n, :, :, :]
    L = pb .- pa
    s = ifelse.(L .> 0.0, (zb .- za) ./ ifelse.(L .> 0.0, L, 1.0), 0.0)
    ua, ub = za .>= 0.0, zb .>= 0.0
    seg = reshape(collect(1:n-1), n - 1, 1, 1, 1)
    isl, isr = seg .== 1, seg .== n - 1
    if raytol !== nothing                                            # a ray on which r changes by no more than its round-off (2, C, B, R) is flat
        dz = absd.(zb .- za)
        s = ifelse.((isl .& (dz .<= raytol[1:1, :, :, :])) .| (isr .& (dz .<= raytol[2:2, :, :, :])), 0.0, s)
    end
    ulo, uhi = s .< 0.0, s .> 0.0                                # side at −∞ and +∞ of a ray
    sn = ifelse.(s .== 0.0, 1.0, s)
    x = ifelse.(isl, pb .- zb ./ sn, ifelse.(isr, pa .- za ./ sn, pa .+ L .* sdiv.(za, za .- zb)))
    ok = ifelse.(isl, (s .!= 0.0) .& (ulo .!= ub), ifelse.(isr, (s .!= 0.0) .& (ua .!= uhi), ua .!= ub))
    ok = ok .& ((.!(isl .| isr)) .| (absd.(x) .< RAYMAX))
    bef = ifelse.(isl, ulo, ua)
    aft = ifelse.(isr, uhi, ub)
    jr = Δt .* (ifelse.(aft, cs, -cd) .- ifelse.(bef, cs, -cd)) .* s
    xs = pin(ifelse.(ok, x, Inf))
    cnt = sum(ifelse.(isfinite.(xs), 1.0, 0.0); dims=1)
    out, jo = Any[], Any[]
    prev = cnt .* 0.0 .- Inf
    for _ in 1:NR
        xi = minimum(ifelse.(xs .> prev, xs, Inf); dims=1)
        fin = isfinite.(xi)
        push!(out, ifelse.(fin, xi, 0.0))
        push!(jo, ifelse.(fin, sum(ifelse.(xs .== xi, jr, 0.0); dims=1), 0.0))
        prev = ifelse.(fin, xi, prev)
    end
    return cat(out...; dims=1), cnt, cat(jo...; dims=1)
end

"Suffix sums Σ_{i ≥ m} x_i along dimension 3, accumulated from the end (no cancellation against far terms)."
rcumsum(x) = (n = size(x, 3); cumsum(x[:, :, n:-1:1]; dims=3)[:, :, n:-1:1])

# ---------------------------------------------------------------------------
# global part with the entry axis first: (N, C, B) arrays, the entry axis the fastest in memory, so
# the payload sort and the scans need no transposes
# ---------------------------------------------------------------------------

"Block length of the two-level scans of traced arrays (`cumsum1`)."
const SCANBLK = 32

"""
Prefix sums along dimension 1. Reactant lowers `cumsum` to a reduce-window over the whole axis (n²
work); a traced axis longer than `SCANBLK` is scanned in two levels: within blocks of `SCANBLK`,
then the exclusive scan of the block totals, added (work n·(b + n/b)). Partial sums keep their
terms (a prefix of exact zeros stays exactly zero), only the order of the additions differs.
"""
cumsum1(x) = cumsum(x; dims=1)
function cumsum1(x::Reactant.TracedRArray)
    n = size(x, 1)
    b = SCANBLK
    n <= 2b && return cumsum(x; dims=1)
    nb = cld(n, b)
    rest = size(x)[2:end]
    xp = nb * b == n ? x : cat(x, zeros(eltype(x), nb * b - n, rest...) .+ 0.0 .* x[1:1, ntuple(_ -> Colon(), length(rest))...]; dims=1)
    X = rs(xp, b, nb, rest...)
    W = cumsum(X; dims=1)
    tot = W[b:b, ntuple(_ -> Colon(), length(rest) + 1)...]
    inc = nb <= 2b ? cumsum(tot; dims=2) : rs(cumsum1(rs(tot, nb, rest...)), 1, nb, rest...)
    cc = ntuple(_ -> Colon(), length(rest))
    off = cat(0.0 .* tot[:, 1:1, cc...], inc[:, 1:nb-1, cc...]; dims=2)     # exclusive by a shift (no cancellation)
    Y = rs(W .+ off, nb * b, rest...)
    return nb * b == n ? Y : Y[1:n, ntuple(_ -> Colon(), length(rest))...]
end
"Suffix sums along dimension 1, accumulated from the end."
rcumsum1(x) = (n = size(x, 1); c = ntuple(_ -> Colon(), ndims(x) - 1); cumsum1(x[n:-1:1, c...])[n:-1:1, c...])

"(C, B, m) → (m, C, B) (entry axis first)."
efirst(X) = permutedims(X, (3, 1, 2))
"(1, C, B) → (C, B, 1) (a reshape: the moved axis has length 1)."
elast(x) = rs(x, size(x, 2), size(x, 3), 1)
"(C, B, 1) → (1, C, B)."
eone(x) = rs(x, 1, size(x, 1), size(x, 2))

"Sort the points p (N, C, B) with the payload v along dimension 1 (stable, `isless`)."
function sortpair1(p, v)
    perm = sortperm(p; dims=1)
    return p[perm], v[perm]
end
function sortpair1(p::Reactant.TracedRArray, v::Reactant.TracedRArray)
    a, b = Reactant.Ops.sort(p, v; comparator=(x, y, _, _) -> isless(x, y), dimension=1, is_stable=true)
    return a, b
end

"""
Potential βΔC at the sorted points q (N, C, B) from the jumps J at them and the slope s0 just right
of 0 (0 is one of the points; s0, s0abs, J0abs (1, C, B)): segment slopes from 0 outwards, values
integrated from c(0) = 0 outwards; returns the extended points and values (N + 2, C, B) for
`sample_core1`. A segment slope within the round-off of its sum (16 ε × the absolute slopes and
jumps summed: `s0abs` for s0, `J0abs` for the folded entry at 0) is 0: a flat stretch stays exactly
flat, as in the dense form.
"""
function jump_values1(q, J, s0, β, s0abs=0.0 .* s0, J0abs=0.0 .* s0)
    N = size(q, 1)
    ε16 = 16 * eps(Float64)
    pos = q .> 0.0
    JR = ifelse.(pos, J, 0.0)
    JL = ifelse.(pos, 0.0, J)
    cR = cumsum1(JR)                                      # Σ_{i ≤ k, q_i > 0} J_i (from 0 outwards)
    sL = rcumsum1(JL)                                     # Σ_{i ≥ k, q_i ≤ 0} J_i (from 0 outwards)
    aR = cumsum1(absd.(JR))
    aL = rcumsum1(absd.(JL))
    qa, qb = q[1:N-1, :, :], q[2:N, :, :]
    rseg = qa .>= 0.0                                     # segments right of 0 (the others end at ≤ 0)
    sl = ifelse.(rseg, s0 .+ cR[1:N-1, :, :], s0 .- sL[2:N, :, :])
    tol = ε16 .* (s0abs .+ absd.(s0) .+ ifelse.(rseg, aR[1:N-1, :, :], aL[2:N, :, :] .+ J0abs))
    sl = ifelse.(absd.(sl) .<= tol, 0.0, sl)
    w = sl .* (qb .- qa)
    z = 0.0 .* q[1:1, :, :]
    right = cat(z, cumsum1(ifelse.(rseg, w, 0.0)); dims=1)          # Σ_{i < m, q_i ≥ 0} w_i
    left = cat(rcumsum1(ifelse.(qb .<= 0.0, w, 0.0)), z; dims=1)    # Σ_{i ≥ m, q_{i+1} ≤ 0} w_i
    c = ifelse.(pos, right, .-left)
    gl = s0 .- sL[1:1, :, :]                              # slope on the left ray (increasing t)
    gr = s0 .+ cR[N:N, :, :]
    base = ε16 .* (s0abs .+ absd.(s0))
    gl = ifelse.(absd.(gl) .<= base .+ ε16 .* (aL[1:1, :, :] .+ J0abs), 0.0, gl)
    gr = ifelse.(absd.(gr) .<= base .+ ε16 .* aR[N:N, :, :], 0.0, gr)
    lo, hi = q[1:1, :, :], q[N:N, :, :]
    plo, phi = lo .- (1.0 .+ absd.(lo)), hi .+ (1.0 .+ absd.(hi))
    clo = c[1:1, :, :] .+ gl .* (plo .- lo)
    chi = c[N:N, :, :] .+ gr .* (phi .- hi)
    return cat(plo, q, phi; dims=1), β .* cat(clo, c, chi; dims=1)
end

"""
The sorted global points and jumps (N, C, B), compacted: every entry at exactly 0 is folded into one
entry (0, Σ its jumps), the others sorted; with a cap Nc > 0 only the first Nc entries are kept; the
folded ones are moved onto the largest point (zero-length segments). Returns (q, J, count, J0abs),
count and J0abs (1, C, B): the count of real candidates (entries ≠ 0, and real candidates at 0;
line-invariant; count ≥ Nc means the kept entries may miss points: guard) and Σ |jumps| folded at 0
(round-off scale). `pts`, `jmp`: vectors of (C, B, m) pieces; nz0 (C, B, 1).
"""
function jump_points1(pts, jmp, nz0, cap::Int)
    qa, Ja = catn([efirst(x) for x in pts], 1), catn([efirst(x) for x in jmp], 1)
    z = qa .== 0.0
    cnt = sum(ifelse.(z, 0.0, 1.0); dims=1) .+ eone(nz0)
    J0 = sum(ifelse.(z, Ja, 0.0); dims=1)
    J0abs = sum(ifelse.(z, absd.(Ja), 0.0); dims=1)
    q, J = sortpair1(cat(0.0 .* J0, ifelse.(z, Inf, qa); dims=1), cat(J0, ifelse.(z, 0.0, Ja); dims=1))
    (cap <= 0 || cap >= size(q, 1)) || ((q, J) = (q[1:cap, :, :], J[1:cap, :, :]))
    fin = isfinite.(q)
    qmax = maximum(ifelse.(fin, q, -Inf); dims=1)          # ≥ 0: the zero entry is finite
    return ifelse.(fin, q, qmax), ifelse.(fin, J, 0.0), cnt, J0abs
end

"""
Inverse-CDF draw from the potential βΔC given at the sorted extended points pe (n, C, B), values c
(linear between the points and on the two end segments, which stand for the rays); u1, u2 (1, C, B).
The same arithmetic as `sample_core` (entry axis first). Returns t (1, C, B).
"""
function sample_core1(k::Consts, pe, c, u1, u2)
    n = size(pe, 1)
    ca, cb = c[1:n-1, :, :], c[2:n, :, :]
    pa, pb = pe[1:n-1, :, :], pe[2:n, :, :]
    L = pb .- pa
    Ls = ifelse.(L .> 0.0, L, 1.0)
    g = (cb .- ca) ./ Ls
    seg = reshape(collect(1:n-1), n - 1, 1, 1)
    isl, isr = seg .== 1, seg .== n - 1
    gl = -g[1:1, :, :]
    gr = g[n-1:n-1, :, :]
    risel = c[1:1, :, :] .- c[2:2, :, :]
    riser = c[n:n, :, :] .- c[n-1:n-1, :, :]
    tol(a, b) = 1e-12 .* (absd.(a) .+ absd.(b))
    improper = (risel .<= tol(c[1:1, :, :], c[2:2, :, :])) .| (riser .<= tol(c[n:n, :, :], c[n-1:n-1, :, :])) .|
               (gl .<= 0.0) .| (gr .<= 0.0)
    lmf = ifelse.(L .> 0.0, .-min.(ca, cb) .+ logseg_d.(L, absd.(cb .- ca)), -Inf)
    lml = .-cb .- log.(ifelse.(gl .> 0.0, gl, 1.0))
    lmr = .-ca .- log.(ifelse.(gr .> 0.0, gr, 1.0))
    lm = ifelse.(isl, lml, ifelse.(isr, lmr, lmf))
    lmx = maximum(lm; dims=1)
    w = exp.(lm .- lmx)
    cum = cumsum1(w)
    tgt = u1 .* cum[n-1:n-1, :, :]
    j = sum(ifelse.(cum .< tgt, 1.0, 0.0); dims=1) .+ 1.0
    j = min.(j, Float64(n - 1))
    sel = seg .== j
    pick(X) = sum(ifelse.(sel, X, 0.0); dims=1)
    jl, jr = pick(ifelse.(isl, 1.0, 0.0)) .> 0.0, pick(ifelse.(isr, 1.0, 0.0)) .> 0.0
    ja, jb, jL, jg = pick(pa), pick(pb), pick(ifelse.(L .> 0.0, L, 0.0)), pick(ifelse.(L .> 0.0, g, 0.0))
    e = -log1p.(-u2)
    tl = jb .- e ./ ifelse.(gl .> 0.0, gl, 1.0)
    tr = ja .+ e ./ ifelse.(gr .> 0.0, gr, 1.0)
    tf = ifelse.((jg .< 0.0) .& (absd.(jg .* jL) .>= 1e-10), jb .+ trexp_end.(u2, jg, jL, k.cpumap), ja .+ trexp_d.(u2, jg, jL, k.cpumap))
    t = ifelse.(jl, tl, ifelse.(jr, tr, tf))
    return ifelse.(improper .| .!isfinite.(t), 0.0, t)
end

"`jump_values1` on (C, B, N) arrays (diagnostics and tests)."
function jump_values(q, J, s0, β, s0abs=0.0 .* s0, J0abs=0.0 .* s0)
    pe, c = jump_values1(efirst(q), efirst(J), eone(s0), β, eone(s0abs), eone(J0abs))
    return permutedims(pe, (2, 3, 1)), permutedims(c, (2, 3, 1))
end
"`jump_points1` with (C, B, N) results (diagnostics and tests)."
function jump_points(pts, jmp, nz0, cap::Int)
    q, J, cnt, J0abs = jump_points1(pts, jmp, nz0, cap)
    return permutedims(q, (2, 3, 1)), permutedims(J, (2, 3, 1)), elast(cnt), elast(J0abs)
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
    q, J, cnt, J0abs = jump_points1(pts, jmp, nz0, cap)
    cap > 0 && (over = over .| (elast(cnt) .> cap - 1))
    pe, c = jump_values1(q, J, eone(s0), k.β, eone(s0abs), J0abs)
    t = elast(sample_core1(k, pe, c, eone(u1), eone(u2)))
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
