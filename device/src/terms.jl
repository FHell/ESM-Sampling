# Term batch and exact padded dense-form line sampler (docs/reactant_design.md §3.2, §3.4 item 1).
#
# A round has B sites per chain. Every site is a line move t ↦ state + t·direction; its cost
# change ΔC(t) is a sum of piecewise-linear terms of a few linear arguments:
#
#   gen   box term of a dispatchable unit: y = y0 + a t, cost Δt·C_vint(κ clamp(y)) + c_viol Δt w |y − clamp(y)|,
#         injection −κ clamp(y) into one balance slot (merged-vintage kink as a primary breakpoint)
#   uc    fleet unit commitment of one unit at one step: (y, c, c₁) each linear, `uc_cost` (walls, no-load,
#         start-up); secondary quantities yc − cc, g_min cc − yc, cc₁ − cc
#   link  signed lossy link: u = u0 + a t, box penalty, injections `link_inj` into two slots
#   line  f = f0 + df t (oriented so that the moved node is the near end), overload c_viol Δt s ψ(|f|/s),
#         losses α s² L(|f|/s) half at each end; the hinge knots ±U_j s and f = 0 are primary breakpoints
#   soc   SoC term of one storage unit at one step: (e, e₁) each linear, `soc_cost`; secondaries
#         d, d − η_in, −d η_out − 1
#   bal   balance slot (node, step): r = r0 + Σ injection changes, cost Δt (c_shed r₊ + c_dump r₋);
#         secondary r
#
# Every field is an array broadcastable to (C, B, 1); `slot` fields are static Julia integers
# (shared by all sites of a family). Padded terms have zero slopes (they add nothing), padded
# sites have all slopes zero (improper line: t = 0).
#
# Dense form (exact, no truncation):
#   1. primary points (clamp points, kinks, line knots) and 0, sorted per site;
#   2. the secondary quantities at those points (plus one point beyond each end): each is linear
#      between consecutive primaries; their sign changes are found per segment. Every secondary
#      quantity of the plain families has at most two sign changes along the line (balance slots
#      of θ moves are convex in t because the losses are convex; all others are monotone), so the
#      smallest and the largest root per quantity are all of them (checked by `root_count`);
#      a balance-slot residual with SoC injections bends at their kinks (roots of the SoC
#      quantities), so its roots are searched on the grid refined by the other roots;
#   3. the merged sorted points, the potential βΔC at all of them (term differences in FP64),
#      the segment log-masses (`logseg`), the two rays, inverse CDF and the truncated-exponential
#      offset. An improper ray (outward slope ≤ 0) gives t = 0, as on the CPU.

"Hinge with type-parametric scalars (a captured struct's numbers become traced numbers in a device loop)."
struct HingeT{F,V}
    v0::F
    s0::F
    U::V
    d::V
end
HingeT(h::Hinge) = HingeT(h.v0, h.s0, copy(h.U), copy(h.d))
@inline function hinge(h::HingeT, u)
    g = h.v0 .+ h.s0 .* u
    for j in 1:length(h.d)
        g = g .+ h.d[j:j] .* max.(u .- h.U[j:j], 0.0)
    end
    return g
end

struct Consts{F,H}
    Δt::F
    cv::F
    cshed::F
    cdump::F
    κ0::F
    β::F
    κmin::F
    cpumap::F            # 1: the CPU's u → t map of the truncated-exponential offset (comparisons), 0: monotone map
    ψ::H                 # HingeT of the overload (in units of s)
    Lq::H                # HingeT of the loss interpolant
end

# term fields: Vectors of arrays (element types differ under tracing, hence no type parameter).
# Arguments of gen and UC terms may be kinked (exact T2 compensation, phase 3):
#   x(t) = x0 + a t + ak φ_kid(t),   φ_k(t) = (Q_k + g_k t)₊ − (Q_k)₊   (kid = 0: linear)
struct GenTerms
    y0::Vector; a::Vector; κ::Vector; w::Vector; mc::Vector; clo::Vector; chi::Vector; plo::Vector; hv::Vector; yk::Vector
    slot::Vector{Int}
    ak::Vector; kid::Vector{Int}
end
GenTerms(y0, a, κ, w, mc, clo, chi, plo, hv, yk, slot) =
    GenTerms(y0, a, κ, w, mc, clo, chi, plo, hv, yk, slot, Any[0.0 for _ in y0], zeros(Int, length(y0)))
struct UCTerms
    y0::Vector; ay::Vector; u0::Vector; au::Vector; u10::Vector; au1::Vector; κ::Vector; w::Vector
    gmin::Vector; su::Vector; nl::Vector; on::Vector
    aky::Vector; kidy::Vector{Int}; aku::Vector; kidu::Vector{Int}; aku1::Vector; kidu1::Vector{Int}
end
UCTerms(y0, ay, u0, au, u10, au1, κ, w, gmin, su, nl, on) =
    UCTerms(y0, ay, u0, au, u10, au1, κ, w, gmin, su, nl, on, (Any[0.0 for _ in y0], zeros(Int, length(y0)))...,
            (Any[0.0 for _ in y0], zeros(Int, length(y0)))..., (Any[0.0 for _ in y0], zeros(Int, length(y0)))...)
struct LinkTerms
    y0::Vector; a::Vector; cap::Vector; η::Vector
    slot0::Vector{Int}; slot1::Vector{Int}
end
struct LineTermsD
    f0::Vector; df::Vector; s::Vector; α::Vector
    near::Vector{Int}; far::Vector{Int}
end
struct SocTerms
    e0::Vector; ae::Vector; e10::Vector; ae1::Vector; k::Vector; w::Vector; h::Vector; ηin::Vector; ηout::Vector
    qt::Vector; mc::Vector
    slot::Vector{Int}
end
"""
Kinks of a T2 move (one per compensated node and step): requirement Q, its change g per unit t,
and the Jacobian data P (output of the compensating units), e = P − Q₊, nm = (number of units − 1);
`on` = 0 for an inactive (padded or uncompensated) kink (g = 0 there, so φ ≡ 0).
"""
struct Kinks
    Q::Vector; g::Vector; P::Vector; e::Vector; nm::Vector; on::Vector
end
struct TermBatch{G,U,K,L,S,J,T}
    gen::G
    uc::U
    link::K
    line::L
    soc::S
    r0::Vector           # balance-slot residuals (C, B, 1)
    kinks::J             # nothing or Kinks
    st::T                # the stacked form (term axis), built from the above (`stack_terms`)
end
TermBatch(gen, uc, link, line, soc, r0) = TermBatch(gen, uc, link, line, soc, r0, nothing)
TermBatch(gen, uc, link, line, soc, r0, kinks) =
    TermBatch(gen, uc, link, line, soc, r0, kinks, stack_terms(gen, uc, link, line, soc, r0, kinks))

@inline clamp01(y) = clamp(y, 0.0, 1.0)
"""
|x| as max(x, −x). Reactant 0.2.304 miscompiles abs(a·b) of two traced operands as a·b (the abs
is dropped); max(x, −x) is lowered correctly. Every absolute value of the device kernels uses it.
"""
@inline absd(x) = max(x, -x)
"a / b, or 0 where b = 0 (padded breakpoints are put at t = 0, a point that is always present)."
@inline sdiv(a, b) = ifelse(b == 0.0, zero(a), a / ifelse(b == 0.0, one(b), b))   # zero(a): type-stable for traced a and scalar b

# ---------------------------------------------------------------------------
# term values
# ---------------------------------------------------------------------------

@inline function gen_c(Δt, cv, y, κ, w, mc, clo, chi, plo, hv)
    yc = clamp01(y)
    p = κ * yc
    cvint = ifelse(hv > 0.0, clo * min(p, plo) + chi * max(p - plo, 0.0), mc * p)
    return Δt * cvint + cv * Δt * w * absd(y - yc)
end
@inline gen_p(y, κ) = κ * clamp01(y)

@inline function uc_val(Δt, cv, yc, u, u1, κ, w, gmin, su, nl)
    uc, uc1 = clamp01(u), clamp01(u1)
    return Δt * nl * κ * uc + su * κ * max(uc1 - uc, 0.0) +
           cv * Δt * w * (max(yc - uc, 0.0) + max(gmin * uc - yc, 0.0) + absd(u - uc))
end

@inline link_c(Δt, cv, y, cap) = cv * Δt * cap * absd(y - clamp(y, -1.0, 1.0))
@inline link_j0(y, cap, η) = (x = cap * clamp(y, -1.0, 1.0); x + (1 - η) * max(-x, 0.0))
@inline link_j1(y, cap, η) = (x = cap * clamp(y, -1.0, 1.0); -x + (1 - η) * max(x, 0.0))

@inline soc_d(Δt, ye, ye1, h, qt) = h * (clamp01(ye1) - clamp01(ye)) / Δt - qt
@inline soc_pc(d, ηin) = ifelse(d >= 0.0, ifelse(ηin > 0.0, d / ifelse(ηin > 0.0, ηin, 1.0), 0.0), 0.0)
@inline soc_mc(d, ηout) = ifelse(d >= 0.0, 0.0, min(-d * ηout, 1.0))
@inline function soc_c(Δt, cv, ye, ye1, k, w, h, ηin, ηout, qt, mc)
    d = soc_d(Δt, ye, ye1, h, qt)
    pos = d >= 0.0
    pc, mcv = soc_pc(d, ηin), soc_mc(d, ηout)
    return Δt * mc * k * (pc + mcv) + cv * w * h * absd(ye - clamp01(ye)) + cv * Δt * w * max(pc - 1.0, 0.0) +
           ifelse(!pos & (qt <= 0.0), cv * Δt * w * max(-d * ηout - 1.0, 0.0), 0.0) +
           ifelse(pos & (ηin <= 0.0), cv * Δt * w * d, 0.0)
end
@inline soc_j(d, k, ηin, ηout) = k * (soc_pc(d, ηin) - soc_mc(d, ηout))

@inline bal_val(Δt, cshed, cdump, r) = Δt * (cshed * max(r, 0.0) + cdump * max(-r, 0.0))

# ---------------------------------------------------------------------------
# stacked form: every term type's fields as one (C, B, 1, n) array (term axis 4), evaluated on
# (C, B, K, n) and reduced over n, so the traced program does not grow with the term count.
# Static per-column metadata (known at trace time): balance slots, kink ids, and whether an
# argument moves (its slope is not the literal 0.0 of the builder).
# ---------------------------------------------------------------------------

"""
Concatenate the arrays of the vector xs along dimension d. Traced arrays go to one
`Reactant.Ops.concatenate` (a vector argument): the varargs `cat(xs...)` is inferred by Reactant's
interpreter at a cost that grows about quadratically with the operand count (minutes at a few
hundred operands). Plain arrays use `cat`.
"""
function catn(xs::AbstractVector, d::Int)
    traced = true
    for x in xs
        x isa Reactant.TracedRArray || (traced = false)
    end
    (traced && !isempty(xs)) || return cat(xs...; dims=d)
    nd = d                                                     # `cat` semantics: trailing singleton dimensions
    for x in xs
        nd = max(nd, ndims(x))
    end
    ys = [ndims(x) == nd ? x : rs(x, size(x)..., ntuple(_ -> 1, nd - ndims(x))...) for x in xs]   # rs: materialized
    length(ys) == 1 && return ys[1]
    sz(y) = ntuple(i -> i == d ? 0 : size(y, i), nd)
    for y in ys
        sz(y) == sz(ys[1]) || return cat(xs...; dims=d)        # sizes differ off the axis: Base's error path
    end
    return Reactant.Ops.concatenate(ys, d)
end

"Whether a builder value is the static zero (the literal 0.0): such a slope does not move."
isstatic0(x) = x === 0.0 || x === 0       # a plain Julia zero, by egality (traced scalars and Reactant's overlays never match)
"Stack a vector of (C|1, B|1, 1) fields (or scalars) into (C, B, 1, n); nothing if all are static zeros."
function stk(v, z)
    allz = true                          # an explicit loop: Reactant overlays `all` over a Vector{Any} (traced Bool)
    for x in v
        isstatic0(x) || (allz = false)
    end
    return allz ? nothing : catn([x .+ z for x in v], 4)
end
mvflags(v) = Bool[!isstatic0(x) for x in v]
"Moving-term count of an argument x0 + a t + ak φ_kid: 0, 1 (monotone), or 2 (both parts)."
mvcount(a, ak, kid) = Int.(mvflags(a)) .+ Int.((kid .> 0) .& mvflags(ak))

function stack_terms(gen, uc, link, line, soc, r0, kk)
    cand = Any[]
    gen === nothing || append!(cand, gen.y0)
    uc === nothing || append!(cand, uc.y0)
    soc === nothing || append!(cand, soc.e0)
    line === nothing || append!(cand, line.f0)
    link === nothing || append!(cand, link.y0)
    append!(cand, r0)
    tmpl = nothing
    for x in cand                                             # a state gather: (C, B, 1)
        (tmpl === nothing && x isa AbstractArray) && (tmpl = x)
    end
    z = tmpl .* 0.0                                           # (C, B, 1) zeros
    S(v) = stk(v, z)
    G = gen === nothing ? nothing :
        (y0=S(gen.y0), a=S(gen.a), κ=S(gen.κ), w=S(gen.w), mc=S(gen.mc), clo=S(gen.clo), chi=S(gen.chi), plo=S(gen.plo),
         hv=S(gen.hv), yk=S(gen.yk), ak=S(gen.ak), kid=copy(gen.kid), slot=copy(gen.slot),
         mva=mvflags(gen.a), mvk=(gen.kid .> 0) .& mvflags(gen.ak), cnt=mvcount(gen.a, gen.ak, gen.kid))
    U = uc === nothing ? nothing :
        (y0=S(uc.y0), ay=S(uc.ay), u0=S(uc.u0), au=S(uc.au), u10=S(uc.u10), au1=S(uc.au1), κ=S(uc.κ), w=S(uc.w),
         gmin=S(uc.gmin), su=S(uc.su), nl=S(uc.nl), on=S(uc.on), aky=S(uc.aky), aku=S(uc.aku), aku1=S(uc.aku1),
         kidy=copy(uc.kidy), kidu=copy(uc.kidu), kidu1=copy(uc.kidu1),
         cy=mvcount(uc.ay, uc.aky, uc.kidy), cu=mvcount(uc.au, uc.aku, uc.kidu), cu1=mvcount(uc.au1, uc.aku1, uc.kidu1))
    Lk = link === nothing ? nothing :
        (y0=S(link.y0), a=S(link.a), cap=S(link.cap), η=S(link.η), slot0=copy(link.slot0), slot1=copy(link.slot1), mv=mvflags(link.a))
    Ln = line === nothing ? nothing :
        (f0=S(line.f0), df=S(line.df), s=S(line.s), α=S(line.α), near=copy(line.near), far=copy(line.far), mv=mvflags(line.df))
    So = soc === nothing ? nothing :
        (e0=S(soc.e0), ae=S(soc.ae), e10=S(soc.e10), ae1=S(soc.ae1), k=S(soc.k), w=S(soc.w), h=S(soc.h), ηin=S(soc.ηin),
         ηout=S(soc.ηout), qt=S(soc.qt), mc=S(soc.mc), slot=copy(soc.slot),
         mve=mvflags(soc.ae), mve1=mvflags(soc.ae1), cnt=Int.(mvflags(soc.ae)) .+ Int.(mvflags(soc.ae1)))
    K = kk === nothing ? nothing :
        (Q=S(kk.Q), g=S(kk.g), P=S(kk.P), e=S(kk.e), nm=S(kk.nm), on=S(kk.on), n=length(kk.Q))
    R = isempty(r0) ? nothing : S(r0)
    return (z=z, gen=G, uc=U, link=Lk, line=Ln, soc=So, kk=K, r0=R, nr=length(r0))
end

"Columns `cols` (a static index vector) of a stacked array (all columns: the array itself)."
cols4(X, cols) = X === nothing ? nothing : (cols == 1:size(X, 4) ? X : X[:, :, :, cols])
"Materialized (C, B, n) view of a (C, B, m, n') array."
flat3(X) = rs(X, size(X, 1), size(X, 2), size(X, 3) * size(X, 4))

"x0 + a t + ak φ_kid(t) on the points t4 (C, B, K, 1); φz (C, B, K, nk + 1) with a zero last column."
function argval(x0, a, ak, kid, t4, φz)
    v = a === nothing ? x0 .+ 0.0 .* t4 : x0 .+ a .* t4
    if ak !== nothing && any(>(0), kid)
        nk = size(φz, 4) - 1
        v = v .+ ak .* φz[:, :, :, [q == 0 ? nk + 1 : q for q in kid]]
    end
    return v
end

"""
Sums of the term columns per balance slot: X (C, B, K, n) → (C, B, K, R). A static table of the
columns of each slot, padded with an appended zero column; one gather and one sum (no 0/1 matmul,
which would turn an Inf of a padded term into NaN).
"""
function slotsum(X, slots::Vector{Int}, R::Int, z4)
    n = size(X, 4)
    cols = [findall(==(j), slots) for j in 1:R]
    mx = maximum(length.(cols); init=0)
    mx == 0 && return nothing
    if mx == 1 && all(length.(cols) .== 1)
        idx = [c[1] for c in cols]
        return idx == 1:n ? X : X[:, :, :, idx]
    end
    Xz = cat(X, z4; dims=4)
    idx = reduce(vcat, [vcat(c, fill(n + 1, mx - length(c))) for c in cols])
    G = Xz[:, :, :, idx]
    C, B, K = size(X, 1), size(X, 2), size(X, 3)
    return rs(sum(rs(G, C, B, K, mx, R); dims=4), C, B, K, R)
end
addinj(acc, x) = x === nothing ? acc : (acc === nothing ? x : acc .+ x)

"""
    eval_terms(tb, k, t; sec) -> (βΔC (C, B, K), Z, dr (C, B, K, R))

Potential βΔC(t) at the points t (C, B, K), the secondary quantities Z (a vector of
(array (C, B, K, m), roots kept) groups; `sec = false` skips them) and the balance-slot changes.
Roots kept: 1 for a quantity of one monotone moving argument, :fam (the family's K) otherwise;
quantities of static arguments only are left out.
"""
function eval_terms(tb::TermBatch, k::Consts, t; sec::Bool=true)
    S = tb.st
    Δt, cv = k.Δt, k.cv
    C, B, K = size(t, 1), size(t, 2), size(t, 3)
    t4 = rs(t, C, B, K, 1)
    z4 = 0.0 .* t4
    kk = S.kk
    φz = kk === nothing ? z4 : cat(max.(kk.Q .+ kk.g .* t4, 0.0) .- max.(kk.Q, 0.0), z4; dims=4)
    c = z4
    Z = Any[]
    inj = nothing
    R = S.nr
    red(x) = sum(x; dims=4)
    if S.gen !== nothing
        g = S.gen
        y = argval(g.y0, g.a, g.ak, g.kid, t4, φz)
        c = c .+ red(gen_c.(Δt, cv, y, g.κ, g.w, g.mc, g.clo, g.chi, g.plo, g.hv) .-
                     gen_c.(Δt, cv, g.y0, g.κ, g.w, g.mc, g.clo, g.chi, g.plo, g.hv))
        R > 0 && (inj = addinj(inj, slotsum(.-(gen_p.(y, g.κ) .- gen_p.(g.y0, g.κ)), g.slot, R, z4)))
    end
    if S.uc !== nothing && S.uc.on !== nothing
        q = S.uc
        yc = clamp01.(argval(q.y0, q.ay, q.aky, q.kidy, t4, φz))
        u = argval(q.u0, q.au, q.aku, q.kidu, t4, φz)
        u1 = argval(q.u10, q.au1, q.aku1, q.kidu1, t4, φz)
        c = c .+ red(q.on .* (uc_val.(Δt, cv, yc, u, u1, q.κ, q.w, q.gmin, q.su, q.nl) .-
                              uc_val.(Δt, cv, clamp01.(q.y0), q.u0, q.u10, q.κ, q.w, q.gmin, q.su, q.nl)))
        if sec
            ucl, uc1 = clamp01.(u), clamp01.(u1)
            for (zq, cnt) in ((q.on .* (yc .- ucl), q.cy .+ q.cu), (q.on .* (q.gmin .* ucl .- yc), q.cy .+ q.cu),
                              (q.on .* (uc1 .- ucl), q.cu .+ q.cu1))
                c1 = findall(cnt .== 1); cK = findall(cnt .>= 2)
                isempty(c1) || push!(Z, (cols4(zq, c1), 1))
                isempty(cK) || push!(Z, (cols4(zq, cK), :fam))
            end
        end
    end
    if S.link !== nothing
        l = S.link
        y = l.y0 .+ (l.a === nothing ? 0.0 : l.a) .* t4
        c = c .+ red(link_c.(Δt, cv, y, l.cap) .- link_c.(Δt, cv, l.y0, l.cap))
        inj = addinj(inj, slotsum(link_j0.(y, l.cap, l.η) .- link_j0.(l.y0, l.cap, l.η), l.slot0, R, z4))
        inj = addinj(inj, slotsum(link_j1.(y, l.cap, l.η) .- link_j1.(l.y0, l.cap, l.η), l.slot1, R, z4))
    end
    if S.line !== nothing
        ln = S.line
        f = ln.f0 .+ (ln.df === nothing ? 0.0 : ln.df) .* t4
        s = ln.s
        u1 = absd.(f) ./ s
        u0 = absd.(ln.f0) ./ s
        dov = s .* (hinge(k.ψ, u1) .- hinge(k.ψ, u0))
        dloss = ln.α .* s .^ 2 .* (hinge(k.Lq, u1) .- hinge(k.Lq, u0))
        c = c .+ (cv * Δt) .* red(linewc(ln, dov))
        df = f .- ln.f0
        inj = addinj(inj, slotsum(df .+ 0.5 .* dloss, ln.near, R, z4))
        inj = addinj(inj, slotsum(.-df .+ 0.5 .* dloss, ln.far, R, z4))
    end
    if S.soc !== nothing
        so = S.soc
        ye = so.e0 .+ (so.ae === nothing ? 0.0 .* t4 : so.ae .* t4)
        ye1 = so.e10 .+ (so.ae1 === nothing ? 0.0 .* t4 : so.ae1 .* t4)
        c = c .+ red(soc_c.(Δt, cv, ye, ye1, so.k, so.w, so.h, so.ηin, so.ηout, so.qt, so.mc) .-
                     soc_c.(Δt, cv, so.e0, so.e10, so.k, so.w, so.h, so.ηin, so.ηout, so.qt, so.mc))
        d1 = soc_d.(Δt, ye, ye1, so.h, so.qt)
        d0 = soc_d.(Δt, so.e0, so.e10, so.h, so.qt)
        inj = addinj(inj, slotsum(soc_j.(d1, so.k, so.ηin, so.ηout) .- soc_j.(d0, so.k, so.ηin, so.ηout), so.slot, R, z4))
        if sec
            for zq in (d1, ifelse.(so.ηin .> 0.0, d1 .- so.ηin, 1.0), .-d1 .* so.ηout .- 1.0)
                c1 = findall(so.cnt .== 1); cK = findall(so.cnt .>= 2)
                isempty(c1) || push!(Z, (cols4(zq, c1), 1))
                isempty(cK) || push!(Z, (cols4(zq, cK), :fam))
            end
        end
    end
    if R > 0
        inj = inj === nothing ? 0.0 .* S.r0 .+ z4 : inj
        r1 = S.r0 .+ inj
        c = c .+ red(bal_val.(Δt, k.cshed, k.cdump, r1) .- bal_val.(Δt, k.cshed, k.cdump, S.r0))
        sec && push!(Z, (r1, :fam))
    end
    return k.β .* rs(c, C, B, K), Z, inj
end

"Primary breakpoints of every term: a vector of (C, B, m) arrays; padded ones are at 0."
function primaries(tb::TermBatch, k::Consts)
    S = tb.st
    P = Any[]
    kk = S.kk
    kk === nothing || push!(P, flat3(sdiv.(.-kk.Q, kk.g)))     # the kinks t* = −Q/g
    # clamp values cs of an argument: unkinked moving columns, and both pieces of kinked columns
    function argp!(x0, a, ak, kid, cs, mask=nothing)
        lin = findall(kid .== 0)
        kin = findall(kid .> 0)
        for (cols, kinked) in ((lin, false), (kin, true))
            isempty(cols) && continue
            X0 = cols4(x0, cols)
            A = a === nothing ? nothing : cols4(a, cols)
            M = mask === nothing ? nothing : cols4(mask, cols)
            put(x) = push!(P, flat3(M === nothing ? x : ifelse.(M .> 0.0, x, 0.0)))
            if !kinked
                A === nothing && continue
                for c in cs; put(sdiv.(cv4(c, cols) .- X0, A)); end
            else
                AK = cols4(ak, cols)
                Qk = kk.Q[:, :, :, kid[cols]]; gk = kk.g[:, :, :, kid[cols]]
                Qp = max.(Qk, 0.0)
                A0 = A === nothing ? 0.0 : A
                for c in cs
                    put(sdiv.(cv4(c, cols) .- X0 .- AK .* (Qk .- Qp), A0 .+ AK .* gk))   # piece Q + g t > 0
                    A === nothing || put(sdiv.(cv4(c, cols) .+ (.-X0 .+ AK .* Qp), A))   # piece Q + g t ≤ 0
                end
            end
        end
    end
    cv4(c, cols) = c isa Number ? c : cols4(c, cols)
    if S.gen !== nothing
        g = S.gen
        sel = findall(g.mva .| g.mvk)
        if !isempty(sel)
            gs = X -> X === nothing ? nothing : cols4(X, sel)
            argp!(gs(g.y0), gs(g.a), gs(g.ak), g.kid[sel], (0.0, 1.0))
            argp!(gs(g.y0), gs(g.a), gs(g.ak), g.kid[sel], (gs(g.yk),), gs(g.hv))    # vintage kink
        end
    end
    if S.uc !== nothing
        q = S.uc
        for (x0, a, ak, kid, cnt) in ((q.y0, q.ay, q.aky, q.kidy, q.cy), (q.u0, q.au, q.aku, q.kidu, q.cu),
                                      (q.u10, q.au1, q.aku1, q.kidu1, q.cu1))
            sel = findall(cnt .> 0)
            isempty(sel) && continue
            us = X -> X === nothing ? nothing : cols4(X, sel)
            argp!(us(x0), us(a), us(ak), kid[sel], (0.0, 1.0))
        end
    end
    if S.link !== nothing && S.link.a !== nothing
        l = S.link
        for c in (-1.0, 1.0, 0.0); push!(P, flat3(sdiv.(c .- l.y0, l.a))); end
    end
    if S.line !== nothing && S.line.df !== nothing
        ln = S.line
        push!(P, flat3(sdiv.(.-ln.f0, ln.df)))
        U4 = rs(k.ψ.U, 1, 1, length(k.ψ.U), 1)
        push!(P, flat3(sdiv.(U4 .* ln.s .- ln.f0, ln.df)), flat3(sdiv.(.-U4 .* ln.s .- ln.f0, ln.df)))
    end
    if S.soc !== nothing
        so = S.soc
        for (x0, a, mv) in ((so.e0, so.ae, so.mve), (so.e10, so.ae1, so.mve1))
            sel = findall(mv)
            (a === nothing || isempty(sel)) && continue
            X0, A = cols4(x0, sel), cols4(a, sel)
            push!(P, flat3(sdiv.(0.0 .- X0, A)), flat3(sdiv.(1.0 .- X0, A)))
        end
    end
    return P
end

"φ_k(t) for every kink (empty without kinks)."
kinkφ(tb::TermBatch, t) = tb.kinks === nothing ? Any[] :
    Any[max.(tb.kinks.Q[j] .+ tb.kinks.g[j] .* t, 0.0) .- max.(tb.kinks.Q[j], 0.0) for j in eachindex(tb.kinks.Q)]

"""
log Jacobian Σ_k (m_k − 1) log|((Q_k + g_k t)₊ + e_k)/P_k| of a T2 move at t (C, B, 1); 0 without kinks.
"""
function logjac(tb::TermBatch, t)
    kk = tb.st.kk
    kk === nothing && return t .* 0.0
    C, B = size(t, 1), size(t, 2)
    t4 = rs(t, C, B, 1, 1)
    v = (max.(kk.Q .+ kk.g .* t4, 0.0) .+ kk.e) ./ ifelse.(kk.on .> 0.0, kk.P, 1.0)
    return rs(sum(ifelse.(kk.on .> 0.0, kk.nm .* log.(absd.(v)), 0.0); dims=4), C, B, 1)
end

"""
Jacobian independence step of T2 moves (`local_sweep!`): t ← 0 if u3 ≥ exp(min(0, log J(t)));
the CPU draws u3 only for t ≠ 0 and a non-empty Jacobian, which is the same decision.
"""
jac_accept(tb::TermBatch, t, u3) = tb.kinks === nothing ? t :
    ifelse.((t .!= 0.0) .& (u3 .>= exp.(min.(0.0, logjac(tb, t)))), 0.0, t)

"Points p (C, B, K) extended by one point beyond each end (the rays are linear there)."
function extend(p)
    K = size(p, 3)
    lo, hi = p[:, :, 1:1], p[:, :, K:K]
    return cat(lo .- (1.0 .+ absd.(lo)), p, hi .+ (1.0 .+ absd.(hi)); dims=3)
end

const RAYMAX = 1e12      # roots on a ray farther than this from 0 are dropped (potential there ≫ 10³ nats)

"log ∫₀ᴸ e^{−g s} ds for x = g L (L > 0; all branches finite)."
@inline function logseg_d(L, x)
    Ls = ifelse(L > 0.0, L, 1.0)
    ax = absd(x)
    small = ax < 1e-10
    axs = ifelse(small, 1.0, ax)
    big = log(Ls) + log(-expm1(-axs)) - log(axs) + ifelse(x < 0.0, ax, 0.0)
    return ifelse(small, log(Ls) - x / 2, big)
end

"""
Offset s ∈ [0, L] with density ∝ e^{−g s}: the inverse CDF at u (monotone in u). For g < 0 it is
the mirrored draw with 1 − u (the CPU's `trexp_offset` mirrors with u: the same distribution, but
that map jumps at the |gL| < 1e-10 switch, so round-off could flip a draw between u L and (1 − u) L).
`cpumap` = 1 selects the CPU's map (for comparisons with given uniforms against `sample_line!`).
"""
@inline function trexp_d(u, g, L, cpumap=0.0)
    x = g * L
    small = absd(x) < 1e-10
    gs = ifelse(g == 0.0, 1.0, g)
    xp = max(x, 0.0)
    xn = min(x, 0.0)
    um = ifelse(cpumap > 0.5, u, 1.0 - u)               # cpumap: the CPU's mirror with u
    sp = -log1p(-u * (-expm1(-xp))) / gs
    sn = L + log1p(-um * (-expm1(xn))) / (-gs)
    return ifelse(small, u * L, ifelse(g > 0.0, sp, sn))
end

"""
    sample_dense(tb, k, u1, u2; K=Val(2)) -> t (C, B, 1)

Exact draw from ∝ exp(−βΔC(t)) for every site (uniforms u1, u2 of shape (C, B, 1)).

K roots are kept per secondary quantity. A site whose line has more sign changes than that in some
secondary quantity stays at t = 0: the count is a property of the whole line (the same from every
point on it), so this selection keeps the kernel π-invariant (exact guard). The plain families have
≤ 2 by construction; compensated and cell families use K = 4 (the guard rate is reported by the tests).
"""
function sample_dense(tb::TermBatch, k::Consts, u1, u2; K::Val=Val(2))
    pts, over = dense_points(tb, k, u1 .* 0.0; K=K)
    t = sample_points(tb, k, pts, u1, u2)
    return ifelse.(over, 0.0, t)
end

"""
    dense_size(tb, k; K) -> (points, columns)

Static shape of the dense evaluation of a term batch: the points per site (0, the primaries, the
kept roots and the two ray points of `extend`) and the term columns. Every column is evaluated at
every point, so a round holds tensors of C·B·points·columns elements.
"""
function dense_size(tb::TermBatch, k::Consts; K::Val{NR}=Val(2)) where {NR}
    S = tb.st
    np = sum(x -> size(x, 3), primaries(tb, k); init=0)
    _, Z, _ = eval_terms(tb, k, S.z)
    nroot = sum(((zq, nrq),) -> size(zq, 4) * (nrq === :fam ? NR : min(nrq, NR)), Z; init=0)
    ncols(x) = x isa NamedTuple ? maximum((size(v, 4) for v in values(x) if v isa AbstractArray && ndims(v) == 4); init=0) : 0
    return 1 + np + nroot + 2, sum(ncols, (S.gen, S.uc, S.link, S.line, S.soc, S.kk))
end

"All kinks of βΔC per site, sorted (C, B, K): primaries, 0 and the roots of the secondary quantities; and the guard."
function dense_points(tb::TermBatch, k::Consts, z0; K::Val{NR}=Val(2)) where {NR}
    P = primaries(tb, k)
    p0 = sort3(cat(z0, P...; dims=3))
    pe = extend(p0)
    _, Z, _ = eval_terms(tb, k, pe)
    C, B, n = size(pe, 1), size(pe, 2), size(pe, 3)
    pe4 = rs(pe, C, B, n, 1)
    R = Any[]
    over = z0 .> 0.0
    # a balance-slot residual bends at the SoC injection kinks (d = 0, −d η_out = 1), which are
    # roots of the SoC quantities, not primaries: with SoC terms its roots are searched on the grid
    # refined by the other roots (the other quantities are linear between primaries)
    twostage = tb.st.nr > 0 && tb.st.soc !== nothing
    nz = twostage ? length(Z) - 1 : length(Z)
    for (zq, nrq) in Z[1:nz]
        nr = nrq === :fam ? NR : min(nrq, NR)
        rts, cnt = rootsK(zq, pe4, nr)
        push!(R, flat3(rts))
        over = over .| (rs(maximum(cnt; dims=4), C, B, 1) .> nr)
    end
    if twostage
        p1 = isempty(R) ? p0 : sort3(cat(p0, R...; dims=3))
        pe1 = extend(p1)
        _, Z1, _ = eval_terms(tb, k, pe1)
        rts, cnt = rootsK(Z1[end][1], rs(pe1, C, B, size(pe1, 3), 1), NR)
        over = over .| (rs(maximum(cnt; dims=4), C, B, 1) .> NR)
        return sort3(cat(p1, flat3(rts); dims=3)), over
    end
    return (isempty(R) ? p0 : sort3(cat(p0, R...; dims=3))), over
end

"""
The NR smallest sign changes of every column of z (C, B, n, m) over the extended points pe4
(C, B, n, 1): roots (C, B, NR, m) (0 where fewer) and their number (C, B, 1, m).
"""
function rootsK(z, pe4, NR::Int)
    n = size(pe4, 3)
    za, zb = z[:, :, 1:n-1, :], z[:, :, 2:n, :]
    pa, pb = pe4[:, :, 1:n-1, :], pe4[:, :, 2:n, :]
    den = za .- zb
    x = pa .+ (pb .- pa) .* sdiv.(za, den)
    seg = reshape(collect(1:n-1), 1, 1, n - 1, 1)
    isl, isr = seg .== 1, seg .== n - 1
    ok = ifelse.(isl, (den .!= 0.0) .& (zb .!= 0.0) .& (x .< pb),
                 ifelse.(isr, (den .!= 0.0) .& (za .!= 0.0) .& (x .> pa), za .* zb .< 0.0))
    ok = ok .& (absd.(x) .< RAYMAX)
    # NR > 1: materialized once; XLA would otherwise recompute z (all terms of the slot) inside each
    # of the NR + 1 reductions below. ok ⇔ xs finite (|x| < RAYMAX), so the count reads xs.
    xs = NR > 1 ? pin(ifelse.(ok, x, Inf)) : ifelse.(ok, x, Inf)
    cnt = sum(ifelse.(isfinite.(xs), 1.0, 0.0); dims=3)
    out = Any[]
    prev = cnt .* 0.0 .- Inf
    for _ in 1:NR
        xi = minimum(ifelse.(xs .> prev, xs, Inf); dims=3)
        push!(out, ifelse.(isfinite.(xi), xi, 0.0))
        prev = ifelse.(isfinite.(xi), xi, prev)
    end
    return (NR == 1 ? out[1] : cat(out...; dims=3)), cnt
end

"Inverse-CDF draw over the sorted points `pts` (every kink of βΔC is one of them)."
function sample_points(tb::TermBatch, k::Consts, pts, u1, u2)
    pe = extend(pts)
    c, _, _ = eval_terms(tb, k, pe; sec=false)
    return sample_core(k, pe, c, u1, u2)
end

"""
Inverse-CDF draw from the potential βΔC given at the sorted extended points pe (C, B, n), values
c (linear between the points and on the two end segments, which stand for the rays).
"""
function sample_core(k::Consts, pe, c, u1, u2)
    n = size(pe, 3)
    ca, cb = c[:, :, 1:n-1], c[:, :, 2:n]
    pa, pb = pe[:, :, 1:n-1], pe[:, :, 2:n]
    L = pb .- pa
    Ls = ifelse.(L .> 0.0, L, 1.0)
    g = (cb .- ca) ./ Ls                                  # slope inside the segment (increasing t)
    seg = reshape(collect(1:n-1), 1, 1, n - 1)
    isl, isr = seg .== 1, seg .== n - 1
    gl = -g[:, :, 1:1]                                    # outward slope of the left ray
    gr = g[:, :, n-1:n-1]
    # improper ray: the potential does not rise beyond the last point (a rise within the
    # round-off of the potential values counts as flat; δ ≥ 1, so a real slope g rises by ≥ g)
    risel = c[:, :, 1:1] .- c[:, :, 2:2]
    riser = c[:, :, n:n] .- c[:, :, n-1:n-1]
    tol(a, b) = 1e-12 .* (absd.(a) .+ absd.(b))
    improper = (risel .<= tol(c[:, :, 1:1], c[:, :, 2:2])) .| (riser .<= tol(c[:, :, n:n], c[:, :, n-1:n-1])) .|
               (gl .<= 0.0) .| (gr .<= 0.0)
    # ∫ e^{−c} over the segment, anchored at its lower end (−c_a + logseg(L, c_b − c_a) cancels when
    # c_a is huge, e.g. a far point at 1e20 nats)
    lmf = ifelse.(L .> 0.0, .-min.(ca, cb) .+ logseg_d.(L, absd.(cb .- ca)), -Inf)
    lml = .-cb .- log.(ifelse.(gl .> 0.0, gl, 1.0))
    lmr = .-ca .- log.(ifelse.(gr .> 0.0, gr, 1.0))
    lm = ifelse.(isl, lml, ifelse.(isr, lmr, lmf))
    lmx = maximum(lm; dims=3)
    w = exp.(lm .- lmx)
    cum = cumsum(w; dims=3)
    tgt = u1 .* cum[:, :, n-1:n-1]
    j = sum(ifelse.(cum .< tgt, 1.0, 0.0); dims=3) .+ 1.0
    j = min.(j, Float64(n - 1))
    sel = seg .== j
    pick(X) = sum(ifelse.(sel, X, 0.0); dims=3)
    jl, jr = pick(ifelse.(isl, 1.0, 0.0)) .> 0.0, pick(ifelse.(isr, 1.0, 0.0)) .> 0.0
    ja, jb, jL, jg = pick(pa), pick(pb), pick(ifelse.(L .> 0.0, L, 0.0)), pick(ifelse.(L .> 0.0, g, 0.0))
    e = -log1p.(-u2)                                      # Exp(1)
    tl = jb .- e ./ ifelse.(gl .> 0.0, gl, 1.0)
    tr = ja .+ e ./ ifelse.(gr .> 0.0, gr, 1.0)
    # in-segment offset; for g < 0 the mass sits at the right end: anchor there (b − s′) instead of
    # a + (L − s′), which cancels for long segments (L ≫ s′)
    tf = ifelse.((jg .< 0.0) .& (absd.(jg .* jL) .>= 1e-10), jb .+ trexp_end.(u2, jg, jL, k.cpumap), ja .+ trexp_d.(u2, jg, jL, k.cpumap))
    t = ifelse.(jl, tl, ifelse.(jr, tr, tf))
    return ifelse.(improper .| .!isfinite.(t), 0.0, t)
end

"For g < 0 (|gL| ≥ 1e-10): the offset measured from the right end, s − L ≤ 0 (no cancellation for L ≫ s − L)."
@inline function trexp_end(u, g, L, cpumap=0.0)
    gs = ifelse(g == 0.0, -1.0, g)
    xn = min(g * L, 0.0)
    um = ifelse(cpumap > 0.5, u, 1.0 - u)
    return log1p(-um * (-expm1(xn))) / (-gs)
end

"""
Balance-slot changes at the drawn t (for the commit): `dr[j]` is slot j's (C, B, 1) change, `dr4(dr)`
the stacked (C, B, 1, R) array. Kept stacked: R slices concatenated again would be a splat of R
arguments, which takes Reactant minutes to trace at R ≈ T (`:slong`, `:sxlong`).
"""
struct SlotChanges{A}
    X::A                                                       # (C, B, 1, R)
end
Base.getindex(d::SlotChanges, j::Int) = rs(d.X[:, :, :, j:j], size(d.X, 1), size(d.X, 2), 1)
Base.length(d::SlotChanges) = size(d.X, 4)
Base.isempty(d::SlotChanges) = size(d.X, 4) == 0
function slot_changes(tb::TermBatch, k::Consts, t)
    inj = eval_terms(tb, k, t; sec=false)[3]
    inj === nothing && return Any[]
    return SlotChanges(inj)
end

"Diagnostics: the largest number of sign changes of any secondary quantity per site (C, B, 1)."
function root_count(tb::TermBatch, k::Consts)
    P = primaries(tb, k)
    p0 = sort3(cat(P[1][:, :, 1:1] .* 0.0, P...; dims=3))
    pe = extend(p0)
    _, Z, _ = eval_terms(tb, k, pe)
    C, B, n = size(pe, 1), size(pe, 2), size(pe, 3)
    pe4 = rs(pe, C, B, n, 1)
    m = p0[:, :, 1:1] .* 0.0
    for (zq, _) in Z
        m = max.(m, rs(maximum(rootsK(zq, pe4, 1)[2]; dims=4), C, B, 1))
    end
    return m
end
