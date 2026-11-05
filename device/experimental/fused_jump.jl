# Fused per-site kernel of the slope-jump line draw (exploratory prototype; docs/fused_kernel_exploration.md).
#
# The slope-jump form of the exact line sampler (device/src/jump.jl, docs/system/device.md §3)
# has a per-column part (`jump_parts`: local points, cost jumps, balance-slot crossings; one
# XLA fusion per term type) and a per-site "global" part: the (point, jump) entries of a site are
# folded at 0, sorted with the jumps as payload, capped, the segment slopes and the potential
# accumulated from 0 outwards, and the inverse-CDF draw taken (`jump_points1`, `jump_values1`,
# `sample_core1`). In the XLA program that global part is one payload sort over (N, C, B) plus
# about ten two-level scans over the entry axis, each a materialized (N, C, B) tensor.
#
# Here the global part is one KernelAbstractions kernel: one thread block per site (chain, site
# index), the entries in shared memory, a bitonic sort with a stable key, then the tail (the scans
# and the draw) over the ≤ cap + 2 entries, block-parallel (shared-memory scans, elementwise
# passes, block reductions; thread 1 only for the final draw) or, as mode 3, serial on thread 1.
# The serial per-site code (`fused_tail!`) runs on Julia arrays as the reference
# (`fused_draw_ref`), and the kernel is launched inside the Reactant-traced program through
# Reactant's KernelAbstractions extension (CUDA.jl must be loaded; the kernel is compiled by
# GPUCompiler and embedded as an `enzymexla.kernel_call`).
#
# Interface: `sample_jump_fused(tb, k, u1, u2; K, cap, nt, mode)` = `sample_jump(tb, k, u1, u2; K, cap)`
# to round-off (the partial sums are accumulated in another association order than the two-level
# scan `cumsum1`: block scans in the kernel, serially in the reference).

using KernelAbstractions
const KA = KernelAbstractions
using Reactant
using ESMDevice
const D = ESMDevice

# ---------------------------------------------------------------------------
# the serial per-site tail: `jump_values1` + `sample_core1` on the sorted, capped entries
# ---------------------------------------------------------------------------

"""
    fused_tail!(q, J, P, V, L, X, M, s0, s0abs, J0abs, u1, u2, β, cpumap) -> t

The draw of one site from its M sorted finite entries q[1:M] (points) and J[1:M] (slope jumps);
P, V, L, X are scratch vectors of length ≥ M + 2. J is overwritten. The arithmetic of
`jump_values1` and `sample_core1` with every partial sum accumulated serially from the ends (the
same terms, another association order than the two-level scan `cumsum1`).
"""
@inline function fused_tail!(q, J, P, V, L, X, M::Int, s0, s0abs, J0abs, u1, u2, β, cpumap)
    ε16 = 16 * eps(Float64)
    # prefix sums of the jumps right of 0 (cR → P, their absolute values aR → V)
    acc = 0.0; aacc = 0.0
    @inbounds for m in 1:M
        JR = q[m] > 0.0 ? J[m] : 0.0
        acc += JR; aacc += D.absd(JR)
        P[m] = acc; V[m] = aacc
    end
    # suffix sums of the jumps at and left of 0 (sL → L, aL → X)
    acc = 0.0; aacc = 0.0
    @inbounds for m in M:-1:1
        JL = q[m] > 0.0 ? 0.0 : J[m]
        acc += JL; aacc += D.absd(JL)
        L[m] = acc; X[m] = aacc
    end
    base = ε16 * (s0abs + D.absd(s0))
    gl = s0 - L[1]
    gr = s0 + P[M]
    gl = D.absd(gl) <= base + ε16 * (X[1] + J0abs) ? 0.0 : gl
    gr = D.absd(gr) <= base + ε16 * V[M] ? 0.0 : gr
    # segment slopes from 0 outwards and the segment masses w (→ J)
    @inbounds for m in 1:M-1
        rseg = q[m] >= 0.0
        sl = rseg ? s0 + P[m] : s0 - L[m+1]
        tol = ε16 * (s0abs + D.absd(s0) + (rseg ? V[m] : X[m+1] + J0abs))
        sl = D.absd(sl) <= tol ? 0.0 : sl
        J[m] = sl * (q[m+1] - q[m])
    end
    # the potential: right[i] = Σ_{m < i, q_m ≥ 0} w_m (→ P), left[i] = Σ_{m ≥ i, q_{m+1} ≤ 0} w_m (→ L)
    acc = 0.0
    @inbounds for i in 1:M
        P[i] = acc
        i < M && q[i] >= 0.0 && (acc += J[i])
    end
    acc = 0.0
    @inbounds for i in M:-1:1
        L[i] = acc
        i > 1 && q[i] <= 0.0 && (acc += J[i-1])
    end
    @inbounds for i in 1:M
        V[i] = q[i] > 0.0 ? P[i] : -L[i]
    end
    # extended points and values (the rays), scaled by β: pe → P[1:M+2], c → V[1:M+2]
    @inbounds begin
        lo, hi = q[1], q[M]
        plo = lo - (1.0 + D.absd(lo)); phi = hi + (1.0 + D.absd(hi))
        clo = V[1] + gl * (plo - lo); chi = V[M] + gr * (phi - hi)
        for i in M:-1:1
            P[i+1] = q[i]; V[i+1] = β * V[i]
        end
        P[1] = plo; P[M+2] = phi; V[1] = β * clo; V[M+2] = β * chi
    end
    # inverse CDF over the n = M + 2 points (`sample_core1`): segment log-masses → L, cumulative → X
    n = M + 2
    @inbounds begin
        L1 = P[2] - P[1]; Ls1 = L1 > 0.0 ? L1 : 1.0
        g1 = (V[2] - V[1]) / Ls1
        Ln = P[n] - P[n-1]; Lsn = Ln > 0.0 ? Ln : 1.0
        gn = (V[n] - V[n-1]) / Lsn
        glr = -g1; grr = gn
        risel = V[1] - V[2]; riser = V[n] - V[n-1]
        improper = (risel <= 1e-12 * (D.absd(V[1]) + D.absd(V[2]))) | (riser <= 1e-12 * (D.absd(V[n]) + D.absd(V[n-1]))) |
                   (glr <= 0.0) | (grr <= 0.0)
        lmx = -Inf
        for j in 1:n-1
            Lj = P[j+1] - P[j]
            ca, cb = V[j], V[j+1]
            lm = j == 1 ? -cb - log(glr > 0.0 ? glr : 1.0) :
                 j == n - 1 ? -ca - log(grr > 0.0 ? grr : 1.0) :
                 (Lj > 0.0 ? -min(ca, cb) + D.logseg_d(Lj, D.absd(cb - ca)) : -Inf)
            L[j] = lm
            lmx = max(lmx, lm)
        end
        acc = 0.0
        for j in 1:n-1
            acc += exp(L[j] - lmx)
            X[j] = acc
        end
        tgt = u1 * X[n-1]
        jsel = 1
        for j in 1:n-1
            X[j] < tgt && (jsel += 1)
        end
        jsel = min(jsel, n - 1)
        ja, jb = P[jsel], P[jsel+1]
        Lj = jb - ja
        jL = Lj > 0.0 ? Lj : 0.0
        jg = Lj > 0.0 ? (V[jsel+1] - V[jsel]) / Lj : 0.0
        e = -log1p(-u2)
        tl = jb - e / (glr > 0.0 ? glr : 1.0)
        tr = ja + e / (grr > 0.0 ? grr : 1.0)
        tf = (jg < 0.0) & (D.absd(jg * jL) >= 1e-10) ? jb + D.trexp_end(u2, jg, jL, cpumap) : ja + D.trexp_d(u2, jg, jL, cpumap)
        t = jsel == 1 ? tl : (jsel == n - 1 ? tr : tf)
        return (improper | !isfinite(t)) ? 0.0 : t
    end
end

# ---------------------------------------------------------------------------
# the kernel: one thread block per site
# ---------------------------------------------------------------------------

"Lexicographic order of the sort keys (point, original index): all keys are distinct."
@inline keylt(qa, ka, qb, kb) = (qa < qb) | ((qa == qb) & (ka < kb))

# Block scans in shared memory (Hillis–Steele, in place, n ≤ 2·NT elements, thread li owns li and
# li + NT): inclusive prefix sums Σ_{m ≤ i} (`pscan!`), or a prefix pair and a suffix pair at once
# (`scan2!`: A, B prefix, Cc, Dd suffix Σ_{m ≥ i}). Every step reads, barriers, writes, barriers;
# the loops are uniform over the block. The partial sums keep their terms (a prefix of exact zeros
# stays exactly zero); the association order differs from the serial tail and from `cumsum1`.
@inline function pscan!(A, n, li, ::Val{NT}) where {NT}
    d = 1
    while d < n
        i1 = li; i2 = li + NT
        a1 = (i1 <= n) & (i1 > d); a2 = (i2 <= n) & (i2 > d)
        @inbounds begin
            v1 = a1 ? A[i1 - d] : 0.0
            v2 = a2 ? A[i2 - d] : 0.0
        end
        @synchronize
        @inbounds begin
            a1 && (A[i1] += v1)
            a2 && (A[i2] += v2)
        end
        @synchronize
        d <<= 1
    end
end

@inline function scan2!(A, B, Cc, Dd, n, li, ::Val{NT}) where {NT}
    d = 1
    while d < n
        i1 = li; i2 = li + NT
        a1 = (i1 <= n) & (i1 > d); a2 = (i2 <= n) & (i2 > d)        # prefix: i > d
        c1 = i1 + d <= n; c2 = i2 + d <= n                          # suffix: i + d ≤ n
        @inbounds begin
            va1 = a1 ? A[i1 - d] : 0.0; vb1 = a1 ? B[i1 - d] : 0.0
            va2 = a2 ? A[i2 - d] : 0.0; vb2 = a2 ? B[i2 - d] : 0.0
            vc1 = c1 ? Cc[i1 + d] : 0.0; vd1 = c1 ? Dd[i1 + d] : 0.0
            vc2 = c2 ? Cc[i2 + d] : 0.0; vd2 = c2 ? Dd[i2 + d] : 0.0
        end
        @synchronize
        @inbounds begin
            a1 && (A[i1] += va1; B[i1] += vb1)
            a2 && (A[i2] += va2; B[i2] += vb2)
            c1 && (Cc[i1] += vc1; Dd[i1] += vd1)
            c2 && (Cc[i2] += vc2; Dd[i2] += vd2)
        end
        @synchronize
        d <<= 1
    end
end

@inline getval(x::Number) = x
@inline getval(x) = x[]

"""
One block per site: the N entries (qa, Ja) of site g are loaded into shared memory with their
original index as a tie-breaker (entries at exactly 0 become +Inf, their jumps are folded into one
entry at 0 with index 0, as `jump_points1`), bitonically sorted over NP ≥ N + 1 slots, capped to
M = CAP (0: all) entries, the Inf entries moved onto the largest finite point with jump 0; then
thread 1 runs `fused_tail!`. Outputs t and the guard (1.0 where the real-candidate count reaches
the cap, or the root guard was set).
"""
@kernel function fused_draw_kernel!(tout, oout, qa, Ja, nz0, s0, s0abs, overin, u1, u2, β, cpumap,
                                    ::Val{N}, ::Val{NP}, ::Val{M}, ::Val{CAP}, ::Val{NT}, ::Val{MODE}) where {N,NP,M,CAP,NT,MODE}
    g = @index(Group, Linear)
    li = @index(Local, Linear)
    qs = @localmem Float64 (NP,)
    ks = @localmem Int32 (NP,)
    J2 = @localmem Float64 (M + 2,)
    P = @localmem Float64 (M + 2,)
    V = @localmem Float64 (M + 2,)
    L = @localmem Float64 (M + 2,)
    X = @localmem Float64 (M + 2,)
    r1 = @localmem Float64 (NT,)
    r2 = @localmem Float64 (NT,)
    r3 = @localmem Float64 (NT,)
    sc = @localmem Float64 (16,)
    off = (g - 1) * N
    # load: slot 1 = the folded zero entry, slots 2..N+1 = the entries, the rest padding
    cnt = 0.0; J0 = 0.0; J0a = 0.0
    i = li
    @inbounds while i <= NP
        if i == 1
            qs[1] = 0.0; ks[1] = Int32(0)
        elseif i <= N + 1
            v = qa[off + i - 1]
            jv = Ja[off + i - 1]
            z = v == 0.0
            qs[i] = z ? Inf : v
            ks[i] = Int32(i - 1)
            cnt += z ? 0.0 : 1.0
            J0 += z ? jv : 0.0
            J0a += z ? D.absd(jv) : 0.0
        else
            qs[i] = Inf; ks[i] = Int32(i - 1)
        end
        i += NT
    end
    @inbounds begin
        r1[li] = cnt; r2[li] = J0; r3[li] = J0a
    end
    @synchronize
    # bitonic sort of (qs, ks) ascending over NP slots (MODE 2: timing without the sort)
    k = MODE >= 2 ? NP + 1 : 2
    while k <= NP
        j = k >> 1
        while j >= 1
            i = li - 1
            @inbounds while i < NP
                p = i ⊻ j
                if p > i
                    aq = qs[i+1]; ak = ks[i+1]; bq = qs[p+1]; bk = ks[p+1]
                    lt = keylt(bq, bk, aq, ak)                      # b before a
                    asc = (i & k) == 0
                    if asc == lt
                        qs[i+1] = bq; ks[i+1] = bk; qs[p+1] = aq; ks[p+1] = ak
                    end
                end
                i += NT
            end
            @synchronize
            j >>= 1
        end
        k <<= 1
    end
    # the site's scalars (thread 1): count, J0, J0abs, and the largest finite point
    if li == 1
        c = 0.0; a = 0.0; b = 0.0
        @inbounds for m in 1:NT
            c += r1[m]; a += r2[m]; b += r3[m]
        end
        qmax = -Inf
        @inbounds for m in 1:M
            isfinite(qs[m]) && (qmax = max(qmax, qs[m]))
        end
        @inbounds begin
            sc[1] = c + nz0[g]; sc[2] = a; sc[3] = b; sc[4] = qmax
        end
    end
    @synchronize
    # the capped lists: jumps by the sorted index, Inf entries onto qmax with jump 0
    i = li
    @inbounds while i <= M
        qv = qs[i]
        idx = ks[i]
        jv = idx == Int32(0) ? sc[2] : (isfinite(qv) ? Ja[off + idx] : 0.0)
        J2[i] = isfinite(qv) ? jv : 0.0
        isfinite(qv) || (qs[i] = sc[4])
        i += NT
    end
    @synchronize
    if MODE == 0
        # ---- the block-parallel tail: `fused_tail!` with the scans over the block
        ε16 = 16 * eps(Float64)
        n = M + 2
        @inbounds begin
            s0g = s0[g]; s0a = s0abs[g]; J0a = sc[3]; βv = getval(β)
        end
        # A: the jumps right of 0 and at or left of 0, and their absolute values; prefix sums cR → P,
        #    aR → V and suffix sums sL → L, aL → X
        i = li
        @inbounds while i <= M
            qv = qs[i]; jv = J2[i]
            JR = qv > 0.0 ? jv : 0.0
            JL = qv > 0.0 ? 0.0 : jv
            P[i] = JR; V[i] = D.absd(JR); L[i] = JL; X[i] = D.absd(JL)
            i += NT
        end
        @synchronize
        scan2!(P, V, L, X, M, li, Val(NT))
        # the ray slopes (thread 1 → sc[5], sc[6]; read after the next barrier)
        if li == 1
            @inbounds begin
                base = ε16 * (s0a + D.absd(s0g))
                gl = s0g - L[1]
                gr = s0g + P[M]
                sc[5] = D.absd(gl) <= base + ε16 * (X[1] + J0a) ? 0.0 : gl
                sc[6] = D.absd(gr) <= base + ε16 * V[M] ? 0.0 : gr
            end
        end
        # B: segment slopes from 0 outwards and the segment masses w → J2[1:M-1]
        i = li
        @inbounds while i <= M - 1
            qa_ = qs[i]; qb_ = qs[i+1]
            rseg = qa_ >= 0.0
            sl = rseg ? s0g + P[i] : s0g - L[i+1]
            tol = ε16 * (s0a + D.absd(s0g) + (rseg ? V[i] : X[i+1] + J0a))
            sl = D.absd(sl) <= tol ? 0.0 : sl
            J2[i] = sl * (qb_ - qa_)
            i += NT
        end
        @synchronize
        # C: right[i] = Σ_{m < i, q_m ≥ 0} w_m (a shifted prefix → P), left[i] = Σ_{m ≥ i, q_{m+1} ≤ 0} w_m (a suffix → L)
        i = li
        @inbounds while i <= M
            P[i] = (i >= 2) && (qs[i-1] >= 0.0) ? J2[i-1] : 0.0
            L[i] = (i <= M - 1) && (qs[i+1] <= 0.0) ? J2[i] : 0.0
            i += NT
        end
        @synchronize
        scan2!(P, V, L, X, M, li, Val(NT))                   # V, X are scratch here (their sums unused)
        # D: the values β·c at the points → V[2:M+1]; the rays (thread 1): V[1], V[M+2], the points plo, phi → sc[7], sc[8]
        i = li
        @inbounds while i <= M
            V[i+1] = βv * (qs[i] > 0.0 ? P[i] : -L[i])
            i += NT
        end
        if li == 1
            @inbounds begin
                lo = qs[1]; hi = qs[M]
                plo = lo - (1.0 + D.absd(lo)); phi = hi + (1.0 + D.absd(hi))
                c1 = lo > 0.0 ? P[1] : -L[1]
                cM = hi > 0.0 ? P[M] : -L[M]
                V[1] = βv * (c1 + sc[5] * (plo - lo))
                V[n] = βv * (cM + sc[6] * (phi - hi))
                sc[7] = plo; sc[8] = phi
            end
        end
        @synchronize
        # E: the inverse CDF (`sample_core1`): segment log-masses → L[1:n-1], their maximum (r1 → sc[9])
        @inbounds begin
            plo = sc[7]; phi = sc[8]
            L1 = qs[1] - plo; Ls1 = L1 > 0.0 ? L1 : 1.0
            g1 = (V[2] - V[1]) / Ls1
            Ln = phi - qs[M]; Lsn = Ln > 0.0 ? Ln : 1.0
            gn = (V[n] - V[n-1]) / Lsn
        end
        glr = -g1; grr = gn
        lmx = -Inf
        j = li
        @inbounds while j <= n - 1
            pa = j == 1 ? plo : qs[j-1]
            pb = j == n - 1 ? phi : qs[j]
            Lj = pb - pa
            ca = V[j]; cb = V[j+1]
            lm = j == 1 ? -cb - log(glr > 0.0 ? glr : 1.0) :
                 j == n - 1 ? -ca - log(grr > 0.0 ? grr : 1.0) :
                 (Lj > 0.0 ? -min(ca, cb) + D.logseg_d(Lj, D.absd(cb - ca)) : -Inf)
            L[j] = lm
            lmx = max(lmx, lm)
            j += NT
        end
        @inbounds r1[li] = lmx
        @synchronize
        if li == 1
            mxv = -Inf
            @inbounds for m in 1:NT
                mxv = max(mxv, r1[m])
            end
            @inbounds sc[9] = mxv
        end
        @synchronize
        # the masses relative to the maximum → X, their cumulative sum, the selected segment by a count
        @inbounds lmx = sc[9]
        j = li
        @inbounds while j <= n - 1
            X[j] = exp(L[j] - lmx)
            j += NT
        end
        @synchronize
        pscan!(X, n - 1, li, Val(NT))
        @inbounds tgt = u1[g] * X[n-1]
        cnt = 0.0
        j = li
        @inbounds while j <= n - 1
            cnt += X[j] < tgt ? 1.0 : 0.0
            j += NT
        end
        @inbounds r1[li] = cnt
        @synchronize
        if li == 1
            @inbounds begin
                csum = 0.0
                for m in 1:NT
                    csum += r1[m]
                end
                jsel = min(unsafe_trunc(Int, csum) + 1, n - 1)
                ja = jsel == 1 ? plo : qs[jsel-1]
                jb = jsel == n - 1 ? phi : qs[jsel]
                Lj = jb - ja
                jL = Lj > 0.0 ? Lj : 0.0
                jg = Lj > 0.0 ? (V[jsel+1] - V[jsel]) / Lj : 0.0
                u2g = u2[g]
                e = -log1p(-u2g)
                tl = jb - e / (glr > 0.0 ? glr : 1.0)
                tr = ja + e / (grr > 0.0 ? grr : 1.0)
                tf = (jg < 0.0) & (D.absd(jg * jL) >= 1e-10) ? jb + D.trexp_end(u2g, jg, jL, getval(cpumap)) : ja + D.trexp_d(u2g, jg, jL, getval(cpumap))
                t = jsel == 1 ? tl : (jsel == n - 1 ? tr : tf)
                risel = V[1] - V[2]; riser = V[n] - V[n-1]
                improper = (risel <= 1e-12 * (D.absd(V[1]) + D.absd(V[2]))) | (riser <= 1e-12 * (D.absd(V[n]) + D.absd(V[n-1]))) |
                           (glr <= 0.0) | (grr <= 0.0)
                t = (improper | !isfinite(t)) ? 0.0 : t
                over = overin[g] > 0.0
                CAP > 0 && (over |= sc[1] > CAP - 1)
                tout[g] = over ? 0.0 : t
                oout[g] = over ? 1.0 : 0.0
            end
        end
    else
        if li == 1
            @inbounds begin
                over = overin[g] > 0.0
                CAP > 0 && (over |= sc[1] > CAP - 1)
                # MODE 3: the serial tail by thread 1; MODE 1, 2: timing without the tail (t = the largest point, o = the count)
                t = MODE == 3 ? fused_tail!(qs, J2, P, V, L, X, M, s0[g], s0abs[g], sc[3], u1[g], u2[g], getval(β), getval(cpumap)) : sc[4]
                tout[g] = over ? 0.0 : t
                oout[g] = MODE == 3 ? (over ? 1.0 : 0.0) : sc[1]
            end
        end
    end
end

# ---------------------------------------------------------------------------
# the Julia reference of the kernel (same per-site code, a stable sort by (point, index))
# ---------------------------------------------------------------------------

"""
    fused_draw_ref(qa, Ja, nz0, s0, s0abs, overin, u1, u2, β, cpumap, cap) -> (t, over)

The kernel's algorithm on Julia arrays: qa, Ja (N, S) entries of S sites, the rest (S,) vectors.
"""
function fused_draw_ref(qa::AbstractMatrix, Ja::AbstractMatrix, nz0, s0, s0abs, overin, u1, u2, β, cpumap, cap::Int)
    N, S = size(qa)
    M = (cap > 0 && cap < N + 1) ? cap : N + 1
    t = zeros(S); ov = zeros(S)
    q = zeros(N + 1); J = zeros(N + 1); key = zeros(N + 1); idx = zeros(Int, N + 1)
    q2 = zeros(M); J2 = zeros(M)
    P, V, L, X = zeros(M + 2), zeros(M + 2), zeros(M + 2), zeros(M + 2)
    for g in 1:S
        cnt = 0.0; J0 = 0.0; J0a = 0.0
        q[1] = 0.0; idx[1] = 0
        for i in 1:N
            v = qa[i, g]; jv = Ja[i, g]
            z = v == 0.0
            q[i+1] = z ? Inf : v
            idx[i+1] = i
            cnt += z ? 0.0 : 1.0
            J0 += z ? jv : 0.0
            J0a += z ? abs(jv) : 0.0
        end
        perm = sortperm(q; alg=MergeSort)                       # stable: equal points keep their index order
        qmax = -Inf
        for m in 1:M
            qv = q[perm[m]]
            isfinite(qv) && (qmax = max(qmax, qv))
        end
        for m in 1:M
            qv = q[perm[m]]; id = idx[perm[m]]
            jv = id == 0 ? J0 : (isfinite(qv) ? Ja[id, g] : 0.0)
            q2[m] = isfinite(qv) ? qv : qmax
            J2[m] = isfinite(qv) ? jv : 0.0
        end
        over = overin[g] > 0.0
        cap > 0 && (over |= (cnt + nz0[g]) > cap - 1)
        tt = fused_tail!(q2, J2, P, V, L, X, M, s0[g], s0abs[g], J0a, u1[g], u2[g], β, cpumap)
        t[g] = over ? 0.0 : tt
        ov[g] = over ? 1.0 : 0.0
    end
    return t, ov
end

# ---------------------------------------------------------------------------
# the wrapper: `sample_jump` with the global part replaced
# ---------------------------------------------------------------------------

"Entries of a site as (N, C, B) arrays (the layout of `jump_points1`): points and jumps."
function fused_entries(tb, k; K)
    pts, jmp, s0, over, nz0, s0abs = D.jump_parts(tb, k; K=K)
    qa = D.catn([D.efirst(x) for x in pts], 1)
    Ja = D.catn([D.efirst(x) for x in jmp], 1)
    return qa, Ja, s0, over, nz0, s0abs
end

"Shared-memory bytes of the kernel per block (static shared memory is limited to 48 KB)."
fused_shmem(N, M, NT) = (NP = nextpow(2, N + 1); NP * 12 + 5 * (M + 2) * 8 + 3 * NT * 8 + 128)

"""
    sample_jump_fused(tb, k, u1, u2; K, cap, nt, mode) -> (t, over)

`sample_jump` with the global part in the fused kernel (traced arrays: launched into the program;
Julia arrays: the reference). t (C, B, 1), over (C, B, 1) Bool. `mode` (traced only): 0 the
block-parallel tail, 3 the serial tail by thread 1 (`fused_tail!`, the reference's code); for
timings 1 skips the tail, 2 the sort as well (the outputs are then not a draw). The parallel tail
needs cap + 2 ≤ 2·nt.
"""
function sample_jump_fused(tb, k, u1, u2; K::Val=Val(4), cap::Int=0, nt::Int=256, mode::Int=0)
    qa, Ja, s0, over, nz0, s0abs = fused_entries(tb, k; K=K)
    N, C, B = size(qa)
    M = (cap > 0 && cap < N + 1) ? cap : N + 1
    f2(x) = D.rs(x, C * B)                                      # (C, B, 1) → (C·B,)
    if qa isa Reactant.TracedRArray
        NP = nextpow(2, N + 1)
        fused_shmem(N, M, nt) <= 48 * 1024 || error("fused kernel: $(fused_shmem(N, M, nt)) bytes of shared memory per block (N = $N, M = $M)")
        (mode != 0 || M + 2 <= 2 * nt) || error("fused kernel: the parallel tail needs cap + 2 = $(M + 2) ≤ 2·nt = $(2 * nt)")
        backend = KA.get_backend(qa)
        t = KA.allocate(backend, Float64, (C * B,))
        o = KA.allocate(backend, Float64, (C * B,))
        kern = fused_draw_kernel!(backend, nt)
        kern(t, o, D.rs(qa, N * C * B), D.rs(Ja, N * C * B), f2(nz0), f2(s0), f2(s0abs), f2(ifelse.(over, 1.0, 0.0)), f2(u1), f2(u2),
             k.β, k.cpumap, Val(N), Val(NP), Val(M), Val(cap), Val(nt), Val(mode); ndrange=nt * C * B)
        return D.rs(t, C, B, 1), D.rs(o, C, B, 1) .> 0.0
    end
    t, o = fused_draw_ref(reshape(qa, N, C * B), reshape(Ja, N, C * B), f2(nz0), f2(s0), f2(s0abs), f2(ifelse.(over, 1.0, 0.0)),
                          f2(u1), f2(u2), k.β, k.cpumap, cap)
    return reshape(t, C, B, 1), reshape(o, C, B, 1) .> 0.0
end

"""
    do_round_fused(f, st, cp, sp, k, tab, r, u; nt) -> (state, t)

`do_round` with the line draw of a slope-jump family taken by `sample_jump_fused`.
"""
function do_round_fused(f::Val, st, cp, sp, k, tab, r, u; nt::Int=256)
    tb, aux = D.build(f, st, cp, sp, tab, r, k)
    t, _ = sample_jump_fused(tb, k, u[:, :, 1:1], u[:, :, 2:2]; K=D.nroots(f), cap=D.jumpcap(f), nt=nt)
    t = D.pin(D.jac_accept(tb, t, u[:, :, 3:3]))
    dr = D.slot_changes(tb, k, t)
    return D.commit_tb(f, st, tab, r, t, dr, aux, tb), t
end
