# Fused per-site kernel of the dense line draw of the plain SoC family (exploratory prototype;
# docs/fused_kernel_exploration.md §6).
#
# The dense form (`sample_dense`, device/src/terms.jl) evaluates every term column at every point
# of a site: the primaries, the roots of the secondary quantities (two stages: the SoC quantities,
# then the balance residual on the refined grid), the extended points and the inverse CDF
# (`dense_points`, `sample_points`, `sample_core`). For `:soc` a site has 2 SoC columns and 2
# balance slots: 1 + 4 primaries + 6 + 4 roots = 15 points, 17 extended. In XLA every stage is a
# handful of small fused launches over (C, B, points, columns) tensors.
#
# Here the whole draw of a site is one thread: the points in per-thread scratch (`@private`), the
# term values recomputed where needed, no shared memory, no barriers. The same per-site code
# (`soc_site!`) runs on Julia arrays as the reference (`soc_draw_ref`). The arithmetic is that of the
# dense form (the values at the points are evaluated directly, nothing accumulated), so the kernel
# and the XLA draw differ only by the GPU's libm and FMA contraction.
#
# Interface: `sample_soc_fused(tb, k, u1, u2; K, nt)` = `line_draw(Val(:soc), tb, k, u1, u2)`.

using KernelAbstractions
const KA = KernelAbstractions
using Reactant
using ESMDevice
const D = ESMDevice

"Insertion sort of x[1:n] in place (fixed trip counts: no data-dependent loop bounds)."
@inline function sortn!(x, n)
    @inbounds for i in 2:n
        for j in i:-1:2
            a = x[j-1]; b = x[j]
            sw = b < a
            x[j-1] = sw ? b : a
            x[j] = sw ? a : b
        end
    end
end

"Extended point i of the sorted list x[1:n] (i in 1:n+2): the rays' points beyond each end (`extend`)."
@inline function pext(x, n, i)
    @inbounds begin
        lo = x[1]; hi = x[n]
        return i == 1 ? lo - (1.0 + D.absd(lo)) : (i == n + 2 ? hi + (1.0 + D.absd(hi)) : x[i-1])
    end
end

"""
The `rootsK` rule for one quantity over the n + 1 segments of the extended points of x[1:n]: the NR
smallest distinct sign changes (0 where fewer) written to x[nw+1 … nw+NR] (nw ≥ n: the roots of the
other quantities of the same stage sit between), and their count. `zf(t)` is the quantity.
"""
@inline function roots_into!(x, n, nw, zf, ::Val{NR}) where {NR}
    cnt = 0
    xs1 = Inf                                   # the smallest ok root; NR = 2: also the next larger
    xs2 = Inf
    pa = pext(x, n, 1); za = zf(pa)
    @inbounds for i in 1:n+1
        pb = pext(x, n, i + 1); zb = zf(pb)
        den = za - zb
        xr = pa + (pb - pa) * D.sdiv(za, den)
        ok = i == 1 ? (den != 0.0) & (zb != 0.0) & (xr < pb) :
             i == n + 1 ? (den != 0.0) & (za != 0.0) & (xr > pa) : za * zb < 0.0
        ok &= D.absd(xr) < D.RAYMAX
        cnt += ok ? 1 : 0
        xo = ok ? xr : Inf
        xs1 = min(xs1, xo)
        pa = pb; za = zb
    end
    if NR >= 2
        pa = pext(x, n, 1); za = zf(pa)
        @inbounds for i in 1:n+1
            pb = pext(x, n, i + 1); zb = zf(pb)
            den = za - zb
            xr = pa + (pb - pa) * D.sdiv(za, den)
            ok = i == 1 ? (den != 0.0) & (zb != 0.0) & (xr < pb) :
                 i == n + 1 ? (den != 0.0) & (za != 0.0) & (xr > pa) : za * zb < 0.0
            ok &= D.absd(xr) < D.RAYMAX
            xo = ok ? xr : Inf
            xs2 = min(xs2, xo > xs1 ? xo : Inf)
            pa = pb; za = zb
        end
    end
    @inbounds begin
        x[nw+1] = isfinite(xs1) ? xs1 : 0.0
        NR >= 2 && (x[nw+2] = isfinite(xs2) ? xs2 : 0.0)
    end
    return cnt
end

"""
    soc_site!(x, cs, lm, e0, ae, e10, ae1, kk, ww, hh, ηi, ηo, qt, mc, r0, u1, u2, Δt, cv, β, cshed, cdump, cpumap,
              Val(NR), Val(MVE), Val(MVE1), Val(SLOT)) -> (t, over)

The dense draw of one site of NC SoC columns (tuples of the column values) and R balance slots (r0 a
tuple): `dense_points` (two stages), `sample_points`, `sample_core`. x, cs, lm: scratch of lengths
≥ NP + 2, NP + 2, NP + 1 (NP the points: 1 + 2·moving arguments + 3·Σ roots kept + R·NR).
"""
@inline function soc_site!(x, cs, lm, e0::NTuple{NC}, ae, e10, ae1, kk, ww, hh, ηi, ηo, qt, mc, r0::NTuple{R}, u1, u2,
                           Δt, cv, β, cshed, cdump, cpumap, ::Val{NR}, ::Val{MVE}, ::Val{MVE1}, ::Val{SLOT}) where {NC,R,NR,MVE,MVE1,SLOT}
    # the primaries (clamp points of the moving arguments) and 0
    n = 1
    @inbounds x[1] = 0.0
    for j in 1:NC
        if MVE[j]
            @inbounds begin
                x[n+1] = D.sdiv(0.0 - e0[j], ae[j]); x[n+2] = D.sdiv(1.0 - e0[j], ae[j])
            end
            n += 2
        end
    end
    for j in 1:NC
        if MVE1[j]
            @inbounds begin
                x[n+1] = D.sdiv(0.0 - e10[j], ae1[j]); x[n+2] = D.sdiv(1.0 - e10[j], ae1[j])
            end
            n += 2
        end
    end
    sortn!(x, n)
    # stage 1: the roots of the SoC quantities (d, d − ηin or 1, −d ηout − 1) of every column with a
    # moving argument: 1 root for one moving argument, NR for two
    over = false
    d1(j, t) = D.soc_d(Δt, e0[j] + ae[j] * t, e10[j] + ae1[j] * t, hh[j], qt[j])
    n1 = n
    for q in 1:3
        for j in 1:NC
            cntj = (MVE[j] ? 1 : 0) + (MVE1[j] ? 1 : 0)
            cntj == 0 && continue
            zf = t -> (q == 1 ? d1(j, t) : q == 2 ? (ηi[j] > 0.0 ? d1(j, t) - ηi[j] : 1.0) : -d1(j, t) * ηo[j] - 1.0)
            if cntj == 1
                c = roots_into!(x, n, n1, zf, Val(1)); over |= c > 1; n1 += 1
            else
                c = roots_into!(x, n, n1, zf, Val(NR)); over |= c > NR; n1 += NR
            end
        end
    end
    n = n1
    sortn!(x, n)
    # stage 2: the balance residual r0 + Σ injections on the refined grid, NR roots per slot
    inj(s, t) = begin
        local ia = 0.0                           # its own name: a closure assigns an enclosing variable of the same name (boxed)
        for j in 1:NC
            if SLOT[j] == s
                dd1 = d1(j, t)
                dd0 = D.soc_d(Δt, e0[j], e10[j], hh[j], qt[j])
                ia += D.soc_j(dd1, kk[j], ηi[j], ηo[j]) - D.soc_j(dd0, kk[j], ηi[j], ηo[j])
            end
        end
        ia
    end
    n2 = n
    for s in 1:R
        zf = t -> r0[s] + inj(s, t)
        c = roots_into!(x, n, n2, zf, Val(NR)); over |= c > NR; n2 += NR
    end
    n = n2
    sortn!(x, n)
    # the potential βΔC at the extended points → cs[1:n+2], the points → x shifted into cs? no: pext
    ne = n + 2
    @inbounds for i in 1:ne
        t = pext(x, n, i)
        sc = 0.0
        for j in 1:NC
            sc += D.soc_c(Δt, cv, e0[j] + ae[j] * t, e10[j] + ae1[j] * t, kk[j], ww[j], hh[j], ηi[j], ηo[j], qt[j], mc[j]) -
                  D.soc_c(Δt, cv, e0[j], e10[j], kk[j], ww[j], hh[j], ηi[j], ηo[j], qt[j], mc[j])
        end
        sb = 0.0
        for s in 1:R
            sb += D.bal_val(Δt, cshed, cdump, r0[s] + inj(s, t)) - D.bal_val(Δt, cshed, cdump, r0[s])
        end
        cs[i] = β * (sc + sb)
    end
    # `sample_core` over the ne points
    @inbounds begin
        p1 = pext(x, n, 1); p2 = pext(x, n, 2); pn1 = pext(x, n, ne - 1); pn = pext(x, n, ne)
        L1 = p2 - p1; Ls1 = L1 > 0.0 ? L1 : 1.0
        g1 = (cs[2] - cs[1]) / Ls1
        Ln = pn - pn1; Lsn = Ln > 0.0 ? Ln : 1.0
        gn = (cs[ne] - cs[ne-1]) / Lsn
        gl = -g1; gr = gn
        risel = cs[1] - cs[2]; riser = cs[ne] - cs[ne-1]
        improper = (risel <= 1e-12 * (D.absd(cs[1]) + D.absd(cs[2]))) | (riser <= 1e-12 * (D.absd(cs[ne]) + D.absd(cs[ne-1]))) |
                   (gl <= 0.0) | (gr <= 0.0)
        lmx = -Inf
        for j in 1:ne-1
            pa = pext(x, n, j); pb = pext(x, n, j + 1)
            Lj = pb - pa
            ca, cb = cs[j], cs[j+1]
            v = j == 1 ? -cb - log(gl > 0.0 ? gl : 1.0) :
                j == ne - 1 ? -ca - log(gr > 0.0 ? gr : 1.0) :
                (Lj > 0.0 ? -min(ca, cb) + D.logseg_d(Lj, D.absd(cb - ca)) : -Inf)
            lm[j] = v
            lmx = max(lmx, v)
        end
        acc = 0.0
        for j in 1:ne-1
            acc += exp(lm[j] - lmx)
            lm[j] = acc
        end
        tgt = u1 * lm[ne-1]
        jsel = 1
        for j in 1:ne-1
            lm[j] < tgt && (jsel += 1)
        end
        jsel = min(jsel, ne - 1)
        ja = pext(x, n, jsel); jb = pext(x, n, jsel + 1)
        Lj = jb - ja
        jL = Lj > 0.0 ? Lj : 0.0
        jg = Lj > 0.0 ? (cs[jsel+1] - cs[jsel]) / Lj : 0.0
        e = -log1p(-u2)
        tl = jb - e / (gl > 0.0 ? gl : 1.0)
        tr = ja + e / (gr > 0.0 ? gr : 1.0)
        tf = (jg < 0.0) & (D.absd(jg * jL) >= 1e-10) ? jb + D.trexp_end(u2, jg, jL, cpumap) : ja + D.trexp_d(u2, jg, jL, cpumap)
        t = jsel == 1 ? tl : (jsel == ne - 1 ? tr : tf)
        t = (improper | !isfinite(t)) ? 0.0 : t
        return (over ? 0.0 : t), over
    end
end

# ---------------------------------------------------------------------------
# the kernel: one thread per site
# ---------------------------------------------------------------------------

@inline getv(x::Number) = x
@inline getv(x) = x[]

"Column j of site g in a flattened (C·B, NC) field (nothing: a static zero)."
@inline fld(X, g, j, CB) = X === nothing ? 0.0 : (@inbounds X[g + (j - 1) * CB])

@kernel function soc_draw_kernel!(tout, oout, e0, ae, e10, ae1, kk, ww, hh, ηi, ηo, qt, mc, r0, u1, u2,
                                  Δt, cv, β, cshed, cdump, cpumap,
                                  ::Val{CB}, ::Val{NC}, ::Val{R}, ::Val{NR}, ::Val{NP}, ::Val{MVE}, ::Val{MVE1}, ::Val{SLOT}) where {CB,NC,R,NR,NP,MVE,MVE1,SLOT}
    g = @index(Global, Linear)
    x = @private Float64 (NP + 2,)
    cs = @private Float64 (NP + 2,)
    lm = @private Float64 (NP + 1,)
    if g <= CB
        col(X) = ntuple(j -> fld(X, g, j, CB), Val(NC))
        rt = ntuple(s -> fld(r0, g, s, CB), Val(R))
        t, over = soc_site!(x, cs, lm, col(e0), col(ae), col(e10), col(ae1), col(kk), col(ww), col(hh), col(ηi), col(ηo), col(qt), col(mc),
                            rt, (@inbounds u1[g]), (@inbounds u2[g]), getv(Δt), getv(cv), getv(β), getv(cshed), getv(cdump), getv(cpumap),
                            Val(NR), Val(MVE), Val(MVE1), Val(SLOT))
        @inbounds begin
            tout[g] = t
            oout[g] = over ? 1.0 : 0.0
        end
    end
end

# ---------------------------------------------------------------------------
# the Julia reference and the wrapper
# ---------------------------------------------------------------------------

"The kernel's algorithm on Julia arrays: the fields (C·B, NC) matrices (or nothing), r0 (C·B, R), u1, u2 (C·B,)."
function soc_draw_ref(e0, ae, e10, ae1, kk, ww, hh, ηi, ηo, qt, mc, r0, u1, u2, Δt, cv, β, cshed, cdump, cpumap,
                      NC::Int, R::Int, NR::Int, NP::Int, mve, mve1, slot; pts=nothing)
    CB = length(u1)
    t = zeros(CB); ov = zeros(CB)
    x = zeros(NP + 2); cs = zeros(NP + 2); lm = zeros(NP + 1)
    col(X, g) = ntuple(j -> fld(X, g, j, CB), NC)
    for g in 1:CB
        tt, over = soc_site!(x, cs, lm, col(e0, g), col(ae, g), col(e10, g), col(ae1, g), col(kk, g), col(ww, g), col(hh, g), col(ηi, g),
                             col(ηo, g), col(qt, g), col(mc, g), ntuple(s -> fld(r0, g, s, CB), R), u1[g], u2[g],
                             Δt, cv, β, cshed, cdump, cpumap, Val(NR), Val(mve), Val(mve1), Val(slot))
        t[g] = tt; ov[g] = over ? 1.0 : 0.0
        pts === nothing || (pts[:, g] .= x[1:NP])              # diagnostics: the site's final sorted points
    end
    return t, ov
end

"""
    sample_soc_fused(tb, k, u1, u2; K, nt) -> (t, over)

`line_draw(Val(:soc), tb, k, u1, u2)` with the whole dense draw in the one-thread-per-site kernel
(traced arrays: launched into the program; Julia arrays: the reference). t (C, B, 1), over Bool.
"""
function sample_soc_fused(tb, k, u1, u2; K::Val{NR}=Val(2), nt::Int=128) where {NR}
    So = tb.st.soc
    So === nothing && error("sample_soc_fused: no SoC terms")
    tb.st.gen === nothing && tb.st.uc === nothing && tb.st.link === nothing && tb.st.line === nothing && tb.st.kk === nothing ||
        error("sample_soc_fused: only SoC terms and balance slots are fused")
    C, B = size(u1, 1), size(u1, 2)
    CB = C * B
    NC = length(So.slot); R = tb.st.nr
    mve, mve1, slot = Tuple(So.mve), Tuple(So.mve1), Tuple(So.slot)
    np = 2 * (count(mve) + count(mve1))
    nroot = 3 * sum(c -> c == 0 ? 0 : (c == 1 ? 1 : NR), So.cnt; init=0) + R * NR
    NP = 1 + np + nroot
    f2(X, m) = X === nothing ? nothing : D.rs(X, CB * m)
    fields = (f2(So.e0, NC), f2(So.ae, NC), f2(So.e10, NC), f2(So.ae1, NC), f2(So.k, NC), f2(So.w, NC), f2(So.h, NC), f2(So.ηin, NC),
              f2(So.ηout, NC), f2(So.qt, NC), f2(So.mc, NC), f2(tb.st.r0, R), D.rs(u1, CB), D.rs(u2, CB))
    consts = (k.Δt, k.cv, k.β, k.cshed, k.cdump, k.cpumap)
    if So.e0 isa Reactant.TracedRArray
        backend = KA.get_backend(So.e0)
        t = KA.allocate(backend, Float64, (CB,))
        o = KA.allocate(backend, Float64, (CB,))
        kern = soc_draw_kernel!(backend, nt)
        kern(t, o, fields..., consts..., Val(CB), Val(NC), Val(R), Val(NR), Val(NP), Val(mve), Val(mve1), Val(slot); ndrange=CB)
        return D.rs(t, C, B, 1), D.rs(o, C, B, 1) .> 0.0
    end
    t, o = soc_draw_ref(fields..., consts..., NC, R, NR, NP, mve, mve1, slot)
    return reshape(t, C, B, 1), reshape(o, C, B, 1) .> 0.0
end

"`do_round` of the SoC family with the line draw taken by `sample_soc_fused`."
function do_round_soc(st, cp, sp, k, tab, r, u; nt::Int=128)
    f = Val(:soc)
    tb, aux = D.build(f, st, cp, sp, tab, r, k)
    t, _ = sample_soc_fused(tb, k, u[:, :, 1:1], u[:, :, 2:2]; K=D.nroots(f), nt=nt)
    t = D.pin(D.jac_accept(tb, t, u[:, :, 3:3]))
    dr = D.slot_changes(tb, k, t)
    return D.commit_tb(f, st, tab, r, t, dr, aux, tb), t
end
