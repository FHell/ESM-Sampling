# Padded, batched exact line sampler (prototype for docs/reactant_performance.md).
#
# One "round" = B independent line moves (e.g. one move family at every step of a
# colour class, and/or at an independent set of nodes, and/or over chains). Every
# move has E padded entries (unit utilizations y, box [0, 1], cost slope ci inside,
# violation slope cv outside, injection κ into one of C padded balance cells) and
# the balance terms n_c φ(S_c / n_c), φ(x) = c_shed x₊ + c_dump (−x)₊. Along the
# line y_j(t) = y0_j + t a_j, so the cost is piecewise linear with primary
# breakpoints at the clamp points and secondary breakpoints where an S_c changes
# sign. The conditional ∝ exp(−β ΔC(t)) is sampled exactly:
#   1. primary points (2E clamp points and 0), sorted per move (fixed size);
#   2. Lin(t) and S_c(t) at all points (+ one point beyond each end for the rays);
#   3. per segment (2 rays + 2E finite) the roots of the C cell sums, sorted;
#   4. sub-segment log-masses (exact, rays included), log-sum-exp, cumulative sum,
#      inverse sampling, truncated-exponential offset;
#   5. commit: scatter y += t a and S_c += ΔS_c(t).
# The same array code runs on CPU Arrays and, traced by Reactant, on the GPU.

@inline clamp01(y) = min(max(y, zero(y)), one(y))

@inline function logseg(L, x)
    # log ∫₀ᴸ e^{−g s} ds for x = g L, finite L > 0 (rays are handled by the caller);
    # all branches are evaluated (ifelse), so every argument is kept in its domain
    T = typeof(x)
    ok = (L > 0) & isfinite(L)
    Ls = ifelse(ok, L, one(T))
    ax = abs(x)
    small = ax < T(1e-6)
    axs = ifelse(small, one(T), ax)
    big = log(Ls) + log(-expm1(-axs)) - log(axs) + ifelse(x < 0, ax, zero(T))
    return ifelse(small, log(Ls) - x / 2, big)
end

@inline function trexp(u, g, L)
    # s ∈ [0, L] with density ∝ e^{−g s}; L = Inf for rays (g > 0)
    T = typeof(g)
    x = g * ifelse(isinf(L), one(T), L)
    small = (abs(x) < T(1e-6)) & !isinf(L)
    gs = ifelse(abs(g) < T(1e-300), one(T), g)
    xp = ifelse(isinf(L), T(700), min(max(x, zero(T)), T(700)))
    xn = min(x, zero(T))
    sp = -log1p(-u * (-expm1(-xp))) / gs
    Lf = ifelse(isinf(L), zero(T), L)
    sn = Lf + log1p(-u * (-expm1(xn))) / (-gs)
    return ifelse(small, u * Lf, ifelse(g > 0, sp, sn))
end

"One batched round. Arrays: a, ci, cv, κ, idx, slot :: E×B; cidx, nc :: C×B; u1, u2 :: 1×B."
function line_round(Y, S, idx, slot, a, ci, cv, κ, cidx, nc, u1, u2, cshed, cdump, β)
    E, B = size(a)
    C = size(cidx, 1)
    T = eltype(a)
    y0 = reshape(Y[reshape(idx, E * B)], E, B)
    S0 = reshape(S[reshape(cidx, C * B)], C, B)
    az = a .== 0
    tlo = ifelse.(az, zero(T), (zero(T) .- y0) ./ ifelse.(az, one(T), a))
    thi = ifelse.(az, zero(T), (one(T) .- y0) ./ ifelse.(az, one(T), a))
    pts = sort(vcat(zeros(T, 1, B), tlo, thi); dims=1)                 # (2E+1)×B
    K1 = 2E + 1
    P = vcat(pts[1:1, :] .- one(T), pts, pts[K1:K1, :] .+ one(T))       # NP = 2E+3
    NP = K1 + 2
    # values at the points
    yt = reshape(y0, E, 1, B) .+ reshape(P, 1, NP, B) .* reshape(a, E, 1, B)
    yc = clamp01.(yt)
    lin = sum(reshape(ci, E, 1, B) .* yc .+ reshape(cv, E, 1, B) .* abs.(yt .- yc); dims=1)   # 1×NP×B
    dinj = reshape(κ, E, 1, B) .* (yc .- reshape(clamp01.(y0), E, 1, B))                       # E×NP×B
    oh = T.(reshape(slot, E, 1, 1, B) .== reshape(collect(1:C), 1, C, 1, 1))                  # E×C×1×B
    St = reshape(S0, C, 1, B) .+ reshape(sum(oh .* reshape(dinj, E, 1, NP, B); dims=1), C, NP, B)  # C×NP×B
    V = vcat(lin, St)                                                   # (1+C)×NP×B
    # segments: left ray (anchor 2, other 1), finite k = 2..K1 (anchor k, other k+1), right ray (anchor K1+1, other K1+2)
    NS = K1 + 1
    Ai = vcat(2, collect(2:K1+1))
    Oi = vcat(1, collect(3:K1+2))
    VA = V[:, Ai, :]
    VO = V[:, Oi, :]
    tA = P[Ai, :]                                                        # NS×B
    Lf = P[Oi, :] .- tA
    isray = reshape(vcat(true, falses(K1 - 1), true), NS, 1)
    dir = reshape(vcat(-one(T), ones(T, NS - 1)), NS, 1)
    Lseg = ifelse.(isray, T(Inf), Lf)
    den = ifelse.(isray, one(T), ifelse.(Lf .> 0, Lf, one(T)))
    sl = (VO .- VA) ./ reshape(den, 1, NS, B)                            # (1+C)×NS×B outward slopes
    lA, lsl = VA[1:1, :, :], sl[1:1, :, :]
    SA, Ssl = VA[2:end, :, :], sl[2:end, :, :]                           # C×NS×B
    Lb = reshape(Lseg, 1, NS, B)
    r = -SA ./ ifelse.(Ssl .== 0, one(T), Ssl)
    r = ifelse.((Ssl .!= 0) .& (r .> 0) .& (r .< Lb), r, Lb)
    sp = sort(vcat(zeros(T, 1, NS, B), r, Lb); dims=1)                   # (C+2)×NS×B sub-points
    s0 = sp[1:C+1, :, :]
    s1 = sp[2:C+2, :, :]
    valid = isfinite.(s0) .& (s1 .> s0)
    s0m = ifelse.(isfinite.(s0), s0, zero(T))
    mid = ifelse.(isinf.(s1), s0m .+ one(T), (s0m .+ ifelse.(isinf.(s1), zero(T), s1)) ./ 2)
    ncr = reshape(nc, C, 1, 1, B)
    # cost at s0 and slope inside each sub-segment
    Sat(s) = reshape(SA, C, 1, NS, B) .+ reshape(Ssl, C, 1, NS, B) .* reshape(s, 1, C + 1, NS, B)
    Ss0 = Sat(s0m)
    Smid = Sat(mid)
    φ(x) = ifelse(x > 0, cshed * x, -cdump * x)
    cost0 = reshape(lA, 1, NS, B) .+ reshape(lsl, 1, NS, B) .* s0m .+
            reshape(sum(ncr .* φ.(Ss0 ./ ncr); dims=1), C + 1, NS, B)
    gsub = reshape(lsl, 1, NS, B) .+ reshape(sum(reshape(Ssl, C, 1, NS, B) .* ifelse.(Smid .> 0, cshed, -cdump); dims=1), C + 1, NS, B)
    c00 = sum(ci .* clamp01.(y0) .+ cv .* abs.(y0 .- clamp01.(y0)); dims=1) .+ sum(nc .* φ.(S0 ./ nc); dims=1)  # 1×B
    βc = β .* (cost0 .- reshape(c00, 1, 1, B))
    βg = β .* gsub
    len = s1 .- s0m
    lenf = ifelse.(isfinite.(len), len, one(T))
    lm = ifelse.(valid, -βc .+ logseg.(lenf, βg .* lenf), T(-Inf))
    lray = ifelse.(βg .> 0, -βc .- log.(abs.(βg) .+ T(1e-300)), T(Inf))
    lm = ifelse.(valid .& isinf.(len), lray, lm)
    M = (C + 1) * NS
    lm2 = reshape(lm, M, B)
    improper = sum(T.(lm2 .== T(Inf)); dims=1) .> 0
    lmf = ifelse.(lm2 .== T(Inf), zero(T), lm2)
    mx = maximum(lmf; dims=1)
    w = exp.(lmf .- mx)
    cum = cumsum(w; dims=1)
    tgt = u1 .* cum[M:M, :]
    j = sum(Int.(cum .< tgt); dims=1) .+ 1
    sel = T.(reshape(collect(1:M), M, 1) .== min.(j, M))
    pick(X) = sum(sel .* reshape(X, M, B); dims=1)
    tAb = reshape(repeat(reshape(tA, 1, NS, B), C + 1, 1, 1), M, B)
    dirb = reshape(repeat(reshape(dir, 1, NS, 1), C + 1, 1, B), M, B)
    gj = pick(βg)
    Lj = pick(ifelse.(isinf.(len), zero(T), len))
    rayj = pick(T.(isinf.(len))) .> 0
    Lj = ifelse.(rayj, T(Inf), Lj)
    s = pick(s0m) .+ trexp.(u2, gj, Lj)
    t = pick(tAb) .+ pick(dirb) .* s
    t = ifelse.(improper .| .!isfinite.(t), zero(T), t)                  # 1×B
    # commit
    y1 = y0 .+ t .* a
    dS = reshape(sum(reshape(T.(reshape(slot, E, 1, B) .== reshape(collect(1:C), 1, C, 1)), E, C, B) .*
                     reshape(κ .* (clamp01.(y1) .- clamp01.(y0)), E, 1, B); dims=1), C, B)
    Y2 = setindex_vec(Y, reshape(idx, E * B), reshape(y1, E * B))
    S2 = setindex_vec(S, reshape(cidx, C * B), reshape(S0 .+ dS, C * B))
    return Y2, S2, t
end

setindex_vec(Y, idx, vals) = (Z = copy(Y); Z[idx .+ 0] = vals .+ zero(eltype(vals)); Z)
