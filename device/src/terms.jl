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
struct TermBatch{G,U,K,L,S,J}
    gen::G
    uc::U
    link::K
    line::L
    soc::S
    r0::Vector           # balance-slot residuals (C, B, 1)
    kinks::J             # nothing or Kinks
end
TermBatch(gen, uc, link, line, soc, r0) = TermBatch(gen, uc, link, line, soc, r0, nothing)

@inline clamp01(y) = clamp(y, 0.0, 1.0)
"""
|x| as max(x, −x). Reactant 0.2.304 miscompiles abs(a·b) of two traced operands as a·b (the abs
is dropped); max(x, −x) is lowered correctly. Every absolute value of the device kernels uses it.
"""
@inline absd(x) = max(x, -x)
"a / b, or 0 where b = 0 (padded breakpoints are put at t = 0, a point that is always present)."
@inline sdiv(a, b) = ifelse(b == 0.0, 0.0, a / ifelse(b == 0.0, 1.0, b))

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

"""
    eval_terms(tb, k, t) -> (βΔC, Z, dr)

Potential βΔC(t) (C, B, K) of every site at the points t (C, B, K), the secondary quantities Z
(vector of (C, B, K)) and the balance-slot changes dr (vector of (C, B, K)).
"""
function eval_terms(tb::TermBatch, k::Consts, t)
    Δt, cv = k.Δt, k.cv
    z = t .* 0.0
    c = z
    Z = Any[]
    nr = length(tb.r0)
    inj = Any[z for _ in 1:nr]
    φ = kinkφ(tb, t)
    arg(x0, a, ak, kid) = kid == 0 ? x0 .+ a .* t : x0 .+ a .* t .+ ak .* φ[kid]
    if tb.gen !== nothing
        g = tb.gen
        for i in eachindex(g.y0)
            y = arg(g.y0[i], g.a[i], g.ak[i], g.kid[i])
            c1 = gen_c.(Δt, cv, y, g.κ[i], g.w[i], g.mc[i], g.clo[i], g.chi[i], g.plo[i], g.hv[i])
            c0 = gen_c.(Δt, cv, g.y0[i], g.κ[i], g.w[i], g.mc[i], g.clo[i], g.chi[i], g.plo[i], g.hv[i])
            c = c .+ (c1 .- c0)
            j = g.slot[i]
            inj[j] = inj[j] .- (gen_p.(y, g.κ[i]) .- gen_p.(g.y0[i], g.κ[i]))
        end
    end
    if tb.uc !== nothing
        q = tb.uc
        for i in eachindex(q.y0)
            yc = clamp01.(arg(q.y0[i], q.ay[i], q.aky[i], q.kidy[i]))
            u = arg(q.u0[i], q.au[i], q.aku[i], q.kidu[i])
            u1 = arg(q.u10[i], q.au1[i], q.aku1[i], q.kidu1[i])
            c1 = uc_val.(Δt, cv, yc, u, u1, q.κ[i], q.w[i], q.gmin[i], q.su[i], q.nl[i])
            c0 = uc_val.(Δt, cv, clamp01.(q.y0[i]), q.u0[i], q.u10[i], q.κ[i], q.w[i], q.gmin[i], q.su[i], q.nl[i])
            c = c .+ q.on[i] .* (c1 .- c0)
            uc, uc1 = clamp01.(u), clamp01.(u1)
            push!(Z, q.on[i] .* (yc .- uc), q.on[i] .* (q.gmin[i] .* uc .- yc), q.on[i] .* (uc1 .- uc))
        end
    end
    if tb.link !== nothing
        l = tb.link
        for i in eachindex(l.y0)
            y = l.y0[i] .+ l.a[i] .* t
            c = c .+ (link_c.(Δt, cv, y, l.cap[i]) .- link_c.(Δt, cv, l.y0[i], l.cap[i]))
            inj[l.slot0[i]] = inj[l.slot0[i]] .+ (link_j0.(y, l.cap[i], l.η[i]) .- link_j0.(l.y0[i], l.cap[i], l.η[i]))
            inj[l.slot1[i]] = inj[l.slot1[i]] .+ (link_j1.(y, l.cap[i], l.η[i]) .- link_j1.(l.y0[i], l.cap[i], l.η[i]))
        end
    end
    if tb.line !== nothing
        ln = tb.line
        for i in eachindex(ln.f0)
            f = ln.f0[i] .+ ln.df[i] .* t
            s = ln.s[i]
            u1 = absd.(f) ./ s
            u0 = absd.(ln.f0[i]) ./ s
            dov = s .* (hinge(k.ψ, u1) .- hinge(k.ψ, u0))
            dloss = ln.α[i] .* s .^ 2 .* (hinge(k.Lq, u1) .- hinge(k.Lq, u0))
            c = c .+ (cv * Δt) .* dov
            df = f .- ln.f0[i]
            inj[ln.near[i]] = inj[ln.near[i]] .+ df .+ 0.5 .* dloss
            inj[ln.far[i]] = inj[ln.far[i]] .- df .+ 0.5 .* dloss
        end
    end
    if tb.soc !== nothing
        s = tb.soc
        for i in eachindex(s.e0)
            ye, ye1 = s.e0[i] .+ s.ae[i] .* t, s.e10[i] .+ s.ae1[i] .* t
            c1 = soc_c.(Δt, cv, ye, ye1, s.k[i], s.w[i], s.h[i], s.ηin[i], s.ηout[i], s.qt[i], s.mc[i])
            c0 = soc_c.(Δt, cv, s.e0[i], s.e10[i], s.k[i], s.w[i], s.h[i], s.ηin[i], s.ηout[i], s.qt[i], s.mc[i])
            c = c .+ (c1 .- c0)
            d1 = soc_d.(Δt, ye, ye1, s.h[i], s.qt[i])
            d0 = soc_d.(Δt, s.e0[i], s.e10[i], s.h[i], s.qt[i])
            inj[s.slot[i]] = inj[s.slot[i]] .+ (soc_j.(d1, s.k[i], s.ηin[i], s.ηout[i]) .- soc_j.(d0, s.k[i], s.ηin[i], s.ηout[i]))
            push!(Z, d1, ifelse.(s.ηin[i] .> 0.0, d1 .- s.ηin[i], 1.0), .-d1 .* s.ηout[i] .- 1.0)
        end
    end
    for j in 1:nr
        r1 = tb.r0[j] .+ inj[j]
        c = c .+ (bal_val.(Δt, k.cshed, k.cdump, r1) .- bal_val.(Δt, k.cshed, k.cdump, tb.r0[j]))
        push!(Z, r1)
    end
    return k.β .* c, Z, inj
end

"Primary breakpoints of every term (vector of (C, B, 1)); padded ones are 0."
function primaries(tb::TermBatch, k::Consts)
    P = Any[]
    kk = tb.kinks
    if kk !== nothing
        for j in eachindex(kk.Q)
            push!(P, sdiv.(.-kk.Q[j], kk.g[j]))                       # the kink t* = −Q/g
        end
    end
    # clamp points c of an argument; kinked: on both sides of its kink (pieces of the argument)
    function argprims!(x0, a, ak, kid, cs, m=1.0)
        put!(x) = push!(P, m === 1.0 ? x : ifelse.(m .> 0.0, x, 0.0))
        if kid == 0
            for c in cs; put!(sdiv.(c .- x0, a)); end
        else
            Qk, gk = kk.Q[kid], kk.g[kid]
            Qp = max.(Qk, 0.0)
            for c in cs
                put!(sdiv.(c .- x0 .- ak .* (Qk .- Qp), a .+ ak .* gk))   # piece Q + g t > 0
                put!(sdiv.(c .- x0 .+ ak .* Qp, a))                      # piece Q + g t ≤ 0
            end
        end
    end
    if tb.gen !== nothing
        g = tb.gen
        for i in eachindex(g.y0)
            argprims!(g.y0[i], g.a[i], g.ak[i], g.kid[i], (0.0, 1.0))
            argprims!(g.y0[i], g.a[i], g.ak[i], g.kid[i], (g.yk[i],), g.hv[i])   # vintage kink
        end
    end
    if tb.uc !== nothing
        q = tb.uc
        for i in eachindex(q.y0), (x0, a, ak, kid) in ((q.y0[i], q.ay[i], q.aky[i], q.kidy[i]), (q.u0[i], q.au[i], q.aku[i], q.kidu[i]),
                                                     (q.u10[i], q.au1[i], q.aku1[i], q.kidu1[i]))
            argprims!(x0, a, ak, kid, (0.0, 1.0))
        end
    end
    if tb.link !== nothing
        l = tb.link
        for i in eachindex(l.y0)
            push!(P, sdiv.(-1.0 .- l.y0[i], l.a[i]), sdiv.(1.0 .- l.y0[i], l.a[i]), sdiv.(0.0 .- l.y0[i], l.a[i]))
        end
    end
    if tb.line !== nothing
        ln = tb.line
        nU = length(k.ψ.U)
        for i in eachindex(ln.f0)
            push!(P, sdiv.(0.0 .- ln.f0[i], ln.df[i]))
            for j in 1:nU
                Uj = k.ψ.U[j:j]
                push!(P, sdiv.(Uj .* ln.s[i] .- ln.f0[i], ln.df[i]), sdiv.(.-Uj .* ln.s[i] .- ln.f0[i], ln.df[i]))
            end
        end
    end
    if tb.soc !== nothing
        s = tb.soc
        for i in eachindex(s.e0), (x0, a) in ((s.e0[i], s.ae[i]), (s.e10[i], s.ae1[i]))
            push!(P, sdiv.(0.0 .- x0, a), sdiv.(1.0 .- x0, a))
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
    s = t .* 0.0
    kk = tb.kinks
    kk === nothing && return s
    for j in eachindex(kk.Q)
        v = (max.(kk.Q[j] .+ kk.g[j] .* t, 0.0) .+ kk.e[j]) ./ ifelse.(kk.on[j] .> 0.0, kk.P[j], 1.0)
        s = s .+ ifelse.(kk.on[j] .> 0.0, kk.nm[j] .* log.(absd.(v)), 0.0)
    end
    return s
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

"""
Sign changes of the secondary quantity z (values at the extended sorted points pe): the smallest
and the largest root (vector of two (C, B, 1)); 0 where there is none. Also returns the number
of roots found (for the ≤ 2 check).
"""
function roots2(z, pe)
    n = size(pe, 3)
    za, zb = z[:, :, 1:n-1], z[:, :, 2:n]
    pa, pb = pe[:, :, 1:n-1], pe[:, :, 2:n]
    den = za .- zb
    x = pa .+ (pb .- pa) .* sdiv.(za, den)
    seg = reshape(collect(1:n-1), 1, 1, n - 1)
    isl, isr = seg .== 1, seg .== n - 1
    ok = ifelse.(isl, (den .!= 0.0) .& (zb .!= 0.0) .& (x .< pb),
                 ifelse.(isr, (den .!= 0.0) .& (za .!= 0.0) .& (x .> pa), za .* zb .< 0.0))
    ok = ok .& (absd.(x) .< RAYMAX)
    rmin = minimum(ifelse.(ok, x, Inf); dims=3)
    rmax = maximum(ifelse.(ok, x, -Inf); dims=3)
    cnt = sum(ifelse.(ok, 1.0, 0.0); dims=3)
    return ifelse.(cnt .> 0.0, rmin, 0.0), ifelse.(cnt .> 0.0, rmax, 0.0), cnt
end

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

"All kinks of βΔC per site, sorted (C, B, K): primaries, 0 and the roots of the secondary quantities; and the guard."
function dense_points(tb::TermBatch, k::Consts, z0; K::Val{NR}=Val(2)) where {NR}
    P = primaries(tb, k)
    p0 = sort(cat(z0, P...; dims=3); dims=3)
    pe = extend(p0)
    _, Z, _ = eval_terms(tb, k, pe)
    R = Any[]
    over = z0 .> 0.0
    for zq in Z
        rs, cnt = rootsK(zq, pe, NR)
        append!(R, rs)
        over = over .| (cnt .> NR)
    end
    return (isempty(R) ? p0 : sort(cat(p0, R...; dims=3); dims=3)), over
end

"The NR smallest sign changes of z over the extended points (0 where fewer), and their number."
function rootsK(z, pe, NR::Int)
    n = size(pe, 3)
    za, zb = z[:, :, 1:n-1], z[:, :, 2:n]
    pa, pb = pe[:, :, 1:n-1], pe[:, :, 2:n]
    den = za .- zb
    x = pa .+ (pb .- pa) .* sdiv.(za, den)
    seg = reshape(collect(1:n-1), 1, 1, n - 1)
    isl, isr = seg .== 1, seg .== n - 1
    ok = ifelse.(isl, (den .!= 0.0) .& (zb .!= 0.0) .& (x .< pb),
                 ifelse.(isr, (den .!= 0.0) .& (za .!= 0.0) .& (x .> pa), za .* zb .< 0.0))
    ok = ok .& (absd.(x) .< RAYMAX)
    cnt = sum(ifelse.(ok, 1.0, 0.0); dims=3)
    xs = ifelse.(ok, x, Inf)
    out = Any[]
    prev = cnt .* 0.0 .- Inf
    for _ in 1:NR
        xi = minimum(ifelse.(xs .> prev, xs, Inf); dims=3)
        push!(out, ifelse.(isfinite.(xi), xi, 0.0))
        prev = ifelse.(isfinite.(xi), xi, prev)
    end
    return out, cnt
end

"Inverse-CDF draw over the sorted points `pts` (every kink of βΔC is one of them)."
function sample_points(tb::TermBatch, k::Consts, pts, u1, u2)
    pe = extend(pts)
    n = size(pe, 3)
    c, _, _ = eval_terms(tb, k, pe)
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

"Balance-slot changes at the drawn t (C, B, 1) (for the commit)."
slot_changes(tb::TermBatch, k::Consts, t) = eval_terms(tb, k, t)[3]

"Diagnostics: the largest number of sign changes of any secondary quantity per site (must be ≤ 2)."
function root_count(tb::TermBatch, k::Consts)
    P = primaries(tb, k)
    p0 = sort(cat((P[1] .* 0.0), P...; dims=3); dims=3)
    pe = extend(p0)
    _, Z, _ = eval_terms(tb, k, pe)
    m = p0[:, :, 1:1] .* 0.0
    for zq in Z
        m = max.(m, roots2(zq, pe)[3])
    end
    return m
end
