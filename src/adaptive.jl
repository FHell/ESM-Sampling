# Adaptive empirical modes with diminishing adaptation (docs/archive/flow_hierarchy.md §13).
#
# The chain learns the directions along which its own state fluctuates and uses
# them as additional, overcomplete move directions:
#   spatial   the angle fluctuation covariance C_θ = E[(θ_τ − θ̄_τ)(θ_τ − θ̄_τ)ᵀ]
#             (running mean θ̄ per node and step); its leading eigenvectors are
#             transport directions (`amode` with exact T2 compensation,
#             `amode0` uncompensated), projected per step onto the face of the
#             predicted binding lines (frames from (k, x));
#   temporal  for every dyadic block length L, the covariance of intra-block
#             fluctuation profiles of the dispatchable generation and of the
#             state of charge; leading eigenvectors are shapes for storage
#             (`ashape`, unit or cell–class aggregate, T2 compensation) and for
#             balance-neutral dispatch pairs at one node (`adshape`).
# Diminishing adaptation (Roberts & Rosenthal 2007): accumulators are running
# means with weight 1/n, and the modes are recomputed only at sweeps 2^j (and
# never after `freeze`), so the change of the kernel between updates vanishes as
# the estimates converge. For every fixed set of modes each move is an exact line
# draw along a direction that depends only on the history and on (k, x), so
# each kernel is π-invariant; the fixed (non-adaptive) families always stay in
# the mixture.

mutable struct AdaptiveModes
    ℓ::Int
    K::Int
    lengths::Vector{Int}
    # spatial
    θmean::Matrix{Float64}          # running mean per (node, step)
    Cθ::Matrix{Float64}
    nθ::Int
    Dθ::Matrix{Float64}             # N × K learned transport directions (⟂ component constants)
    # temporal: per L, (r×r second moment, count) for :gen and :soc profiles
    Ygen_mean::Vector{Matrix{Float64}}   # running mean of node values of generator fields
    Ysoc_mean::Vector{Matrix{Float64}}
    Ct::Dict{Tuple{Symbol,Int,Int},Matrix{Float64}}
    nt::Dict{Tuple{Symbol,Int,Int},Int}
    shapes::Dict{Tuple{Symbol,Int,Int},Matrix{Float64}}
    nobs::Int
    updates::Vector{Int}            # sweeps at which the modes were recomputed
    freeze::Int                     # no updates after this many observations (typemax: never freeze)
    # learned linear compensation: per node, generator units (fixed order) and, per proxy class
    # (1 deficit, 2 surplus), running moments of (Q, p_i) and the response coefficients β
    runits::Vector{Vector{NTuple{2,Int}}}
    rcnt::Matrix{Float64}           # N × 2
    rmQ::Matrix{Float64}
    rvQ::Matrix{Float64}
    rmP::Vector{Matrix{Float64}}    # per node: m × 2
    rcPQ::Vector{Matrix{Float64}}
    β::Vector{Matrix{Float64}}
    H::Any                          # sparse Haar angle basis for localized learned moves
    pieces::Vector{Vector{Tuple{Int,Float64}}}
end

function AdaptiveModes(P::Problem, ℓ::Int; K::Int=8, lengths=[2, 4, 8, 16, 32], freeze::Int=typemax(Int))
    E = P.E
    lv = P.lv[ℓ+1]
    N, T = E.sys.N, lv.T
    genf = [g.vf for g in E.gens if g.vf > 0]
    socf = [s.fe for s in E.stors]
    LM = LocalModel(P, ℓ)
    runits = [[(f, i) for (f, i) in LM.units[n] if LM.role[f][1] == ROLE_GEN && LM.complete[f]] for n in 1:N]
    AdaptiveModes(ℓ, K, filter(<(T ÷ 2), lengths), zeros(N, T), zeros(N, N), 0, zeros(N, 0),
                  [zeros(nsupport(E.vspecs[f].B), T) for f in genf], [zeros(nsupport(E.vspecs[f].B), T) for f in socf],
                  Dict{Tuple{Symbol,Int,Int},Matrix{Float64}}(), Dict{Tuple{Symbol,Int,Int},Int}(),
                  Dict{Tuple{Symbol,Int,Int},Matrix{Float64}}(), 0, Int[], freeze,
                  runits, zeros(N, 2), zeros(N, 2), zeros(N, 2), [zeros(length(u), 2) for u in runits],
                  [zeros(length(u), 2) for u in runits],
                  [fill(1.0 / max(length(u), 1), length(u), 2) for u in runits],   # initial: equal split
                  haar_theta(E), Vector{Tuple{Int,Float64}}[])
end

"Accumulate one state (its local caches) into the empirical moments (Welford around the running means)."
function observe!(AM::AdaptiveModes, P::Problem, LM::LocalModel, LC::LocalCost)
    AM.nobs >= AM.freeze && return AM
    E = P.E
    AM.nobs += 1
    w = 1.0 / AM.nobs
    θ = LC.Y[E.fθ]
    # gauge: remove the component means
    θc = copy(θ)
    for nodes in LM.GS.nodes
        isempty(nodes) && continue
        θc[nodes, :] .-= mean(θc[nodes, :]; dims=1)
    end
    dev = θc .- AM.θmean
    AM.θmean .+= w .* dev
    if AM.nobs > 1
        D2 = θc .- AM.θmean
        AM.Cθ .= (1 - w) .* AM.Cθ .+ w .* (dev * D2') ./ size(θ, 2)
        AM.nθ += 1
    end
    genf = [g.vf for g in E.gens if g.vf > 0]
    socf = [s.fe for s in E.stors]
    h = P.lv[AM.ℓ+1].h
    for (kind, fields, means) in ((:gen, genf, AM.Ygen_mean), (:soc, socf, AM.Ysoc_mean))
        for (q, f) in enumerate(fields)
            Y = LC.Y[f]
            d = Y .- means[q]
            means[q] .+= w .* d
            AM.nobs > 1 || continue
            for L in AM.lengths, cl in 1:4                 # day-aligned blocks, per season class
                key = (kind, L, cl)
                C = get!(() -> zeros(L + (kind == :soc ? 1 : 0), L + (kind == :soc ? 1 : 0)), AM.Ct, key)
                nb = size(Y, 2) ÷ L
                acc = zeros(size(C))
                cnt = 0
                for b in 0:nb-1, i in axes(Y, 1)
                    season(b * L + 1, h) == cl || continue
                    if kind == :gen
                        seg = d[i, b*L+1:(b+1)*L]
                        seg = seg .- mean(seg)
                    else
                        b1 = min((b + 1) * L + 1, size(Y, 2))
                        seg = d[i, b*L+1:b1]
                        length(seg) == L + 1 || continue
                        seg = seg .- [(1 - o / L) * seg[1] + (o / L) * seg[end] for o in 0:L]
                    end
                    acc .+= seg * seg'
                    cnt += 1
                end
                cnt == 0 && continue
                n = get(AM.nt, key, 0) + 1
                AM.nt[key] = n
                C .= (1 - 1 / n) .* C .+ (1 / n) .* (acc ./ cnt)
            end
        end
    end
    return AM
end

"""
Learned linear compensation at node m and step τ for a requirement change g per
unit t: unit i changes its output by β_{m,c,i}·g (c = proxy class of (m, τ) from
the frames: deficit or surplus), the residual takes (1 − Σβ)·g. The coefficients
are fixed between updates and independent of v, so the direction is constant
along the line and no Jacobian enters.
"""
function compensate_learned!(ws, P, LM, LC, AM::AdaptiveModes, FF, m, g, τ; κmin=1.0)
    units = AM.runits[m]
    isempty(units) && return false
    c = FF.surplus[m, τ] ? 2 : 1
    β = view(AM.β[m], :, c)
    any_ = false
    for (q, (f, i)) in enumerate(units)
        κ = κunit(P, LM, LC, f, i, τ)
        (abs(κ) < κmin || β[q] == 0.0) && continue
        prep_node!(ws, P, LM, f, i, β[q] * g / κ, τ)
        any_ = true
    end
    return any_
end

"Accumulate the node-level (requirement, output) moments per proxy class (Welford, 1/n)."
function observe_response!(AM::AdaptiveModes, P::Problem, LM::LocalModel, LC::LocalCost, FF)
    E = P.E
    T = LM.T
    for n in eachindex(AM.runits)
        units = AM.runits[n]
        isempty(units) && continue
        for τ in 1:T
            κ = [κunit(P, LM, LC, f, i, τ) for (f, i) in units]
            p = κ .* clamp.([LC.Y[f][i, τ] for (f, i) in units], 0.0, 1.0)
            Q = LC.r[n, τ] + sum(p)
            c = FF.surplus[n, τ] ? 2 : 1
            AM.rcnt[n, c] += 1
            w = 1.0 / AM.rcnt[n, c]
            dQ = Q - AM.rmQ[n, c]
            AM.rmQ[n, c] += w * dQ
            dP = p .- AM.rmP[n][:, c]
            AM.rmP[n][:, c] .+= w .* dP
            AM.rvQ[n, c] = (1 - w) * AM.rvQ[n, c] + w * dQ * (Q - AM.rmQ[n, c])
            AM.rcPQ[n][:, c] .= (1 - w) .* AM.rcPQ[n][:, c] .+ w .* dP .* (Q - AM.rmQ[n, c])
        end
    end
    return AM
end

"Recompute the modes if the observation count is a power of two (geometric schedule)."
function maybe_update!(AM::AdaptiveModes, P::Problem, LM::LocalModel)
    n = AM.nobs
    (n >= 4 && ispow2(n) && n <= AM.freeze && !(n in AM.updates)) || return false
    E = P.E
    if AM.nθ >= 2
        F = eigen(Symmetric(AM.Cθ))
        D = reverse(F.vectors; dims=2)[:, 1:min(AM.K, size(F.vectors, 2))]
        # orthogonal to the component constants
        for nodes in LM.GS.nodes
            isempty(nodes) && continue
            D[nodes, :] .-= mean(D[nodes, :]; dims=1)
        end
        for k in 1:size(D, 2)
            nk = norm(D[:, k]); nk > 0 && (D[:, k] ./= nk)
        end
        AM.Dθ = D
        AM.pieces = localize(D, AM.H)
    end
    for n in eachindex(AM.runits), c in 1:2
        isempty(AM.runits[n]) && continue
        if AM.rcnt[n, c] > 10 && AM.rvQ[n, c] > 1e-6
            AM.β[n][:, c] .= AM.rcPQ[n][:, c] ./ AM.rvQ[n, c]
        end
    end
    for (key, C) in AM.Ct
        F = eigen(Symmetric(C))
        Φ = reverse(F.vectors; dims=2)[:, 1:min(2, size(C, 1) - 1)]
        if key[1] == :soc
            Φ[1, :] .= 0; Φ[end, :] .= 0
        end
        for k in 1:size(Φ, 2)
            m = maximum(abs.(Φ[:, k])); m > 0 && (Φ[:, k] ./= m)
        end
        AM.shapes[key] = Φ
    end
    push!(AM.updates, n)
    return true
end

"Project a node direction onto the null space of the predicted binding loading rows at τ."
function project_face(d::AbstractVector, FF::FlowFrames, sys, τ::Int)
    B = FF.binding[τ]
    isempty(B) && return collect(d)
    WB = zeros(length(B), sys.N)
    for (r, e) in enumerate(B)
        w = sys.line_b[e] / sys.line_s[e]
        WB[r, sys.line_bus0[e]] += w; WB[r, sys.line_bus1[e]] -= w
    end
    A = WB * WB'
    A += 1e-10 * max(tr(A), 1e-12) / size(A, 1) * I
    return d .- WB' * (Symmetric(A) \ (WB * d))
end

"Learned transport direction k at τ (projected), with exact T2 compensation or uncompensated."
function prep_amode!(rng, ws, P, LM, LC, FF, AM::AdaptiveModes, τ, jac; comp::Bool=true, learn=nothing)
    E = P.E
    LM.complete[E.fθ] || return false
    size(AM.Dθ, 2) == 0 && return false
    k = rand(rng, 1:size(AM.Dθ, 2))
    d = FF === nothing ? AM.Dθ[:, k] : project_face(view(AM.Dθ, :, k), FF, E.sys, τ)
    Ut = E.vspecs[E.fθ].B.Ut
    lv = P.lv[LM.ℓ+1]
    dz = zeros(lv.m[E.fθ])
    for n in eachindex(d)
        d[n] == 0.0 && continue
        for q in nzrange(Ut, n)
            j = rowvals(Ut)[q]
            j <= length(dz) && (dz[j] += nonzeros(Ut)[q] * d[n])
        end
    end
    return prep_tdir!(ws, P, LM, LC, τ, d, dz, jac; comp=comp, learn=learn)
end

"""
    adaptive_block_sweep!(rng, P, LM, AM, s; nmoves, LC, pools) -> (s, stats)

Block moves with the learned temporal shapes: storage shapes (`ashape`, unit or
cell–class aggregate, T2 compensation; via `prep_storage_ms!`) and balance-neutral
dispatch-pair shapes at one node (`adshape`). Blocks of each learned length,
random offset, parity classes in parallel.
"""
function adaptive_block_sweep!(rng, P::Problem, LM::LocalModel, AM::AdaptiveModes, s::State; nmoves::Int=8,
                               LC=nothing, pools=nothing, κmin=1.0, learn=nothing)
    E = P.E
    T = LM.T
    LCc = LC === nothing ? LocalCost(P, LM, s) : LC     # single assignment: not boxed in the threaded closure
    Lmax = maximum(AM.lengths; init=1)
    pl::Vector{BlockPool} = pools === nothing ? [BlockPool(P, Lmax + 2) for _ in 1:Threads.maxthreadid()] : pools
    ensure_pools!(pl, P, Lmax + 2)
    units_st = [(si, i) for (si, st) in enumerate(E.stors) for i in eachindex(st.bus) if LM.complete[st.fe]]
    cellof = E.flow === nothing ? E.parts[2] : E.flow.cell
    groups = Dict{Tuple{Int,Int},Vector{NTuple{2,Int}}}()
    for (si, i) in units_st
        push!(get!(groups, (cellof[E.stors[si].bus[i]], si), NTuple{2,Int}[]), (si, i))
    end
    gkeys = collect(keys(groups))
    gen_nodes = [n for n in 1:E.sys.N if count(u -> LM.role[u[1]][1] == ROLE_GEN && LM.complete[u[1]], LM.units[n]) >= 2]
    stats = zeros(2, 2)
    h = P.lv[LM.ℓ+1].h
    for L in shuffle(rng, AM.lengths)
        haskey(AM.shapes, (:soc, L, 1)) || haskey(AM.shapes, (:gen, L, 1)) || continue
        o = rand(rng) < 0.5 ? 0 : rand(rng, 0:L-1)         # day-aligned half of the time (state-independent)
        starts = collect(1+o:L:T-L)
        length(starts) < 2 && continue
        for cls in shuffle(rng, [starts[1:2:end], starts[2:2:end]])
            seeds = rand(rng, UInt64, length(cls))
            loc = [zeros(2, 2) for _ in cls]
            Threads.@threads :static for q in eachindex(cls)
                r = Xoshiro(seeds[q])
                pool = pl[Threads.threadid()]
                bm = blockmove(pool.wss, pool.wss[1], P, LM, LCc)
                secprev = pool.wss[1].secprev
                jac = empty!(pool.wss[1].jac)
                us1 = NTuple{2,Int}[(0, 0)]                     # single-unit selection (reused)
                b0 = cls[q]
                cl = season(b0, h)
                Φs = get(AM.shapes, (:soc, L, cl), nothing)
                Φg = get(AM.shapes, (:gen, L, cl), nothing)
                for _ in 1:nmoves
                    if Φs !== nothing && !isempty(units_st) && (Φg === nothing || isempty(gen_nodes) || rand(r) < 0.5)
                        us = rand(r) < 0.5 ? groups[rand(r, gkeys)] : (us1[1] = rand(r, units_st); us1)
                        φ = view(Φs, :, rand(r, 1:size(Φs, 2)))
                        prep_storage_ms!(bm, P, LM, LCc, us, b0, L, φ, jac; learn=(learn !== nothing && rand(r) < 0.5) ? learn : nothing) || continue
                        fam = 1
                    else
                        (Φg === nothing || isempty(gen_nodes)) && continue
                        n = rand(r, gen_nodes)
                        gu = LM.genunits[n]
                        u1, u2 = rand(r, gu), rand(r, gu)
                        u1 == u2 && continue
                        φ = view(Φg, :, rand(r, 1:size(Φg, 2)))
                        bm.n = L
                        resize!(bm.τs, L)
                        empty!(jac); empty!(bm.bps)
                        nonempty = false
                        for j in 1:L
                            τ = b0 + j - 1
                            bm.τs[j] = τ
                            ws = bm.wss[j]
                            clear!(ws)
                            κ1, κ2 = κunit(P, LM, LCc, u1..., τ), κunit(P, LM, LCc, u2..., τ)
                            if abs(κ1) >= κmin && abs(κ2) >= κmin && φ[j] != 0.0
                                prep_node!(ws, P, LM, u1[1], u1[2], φ[j] / κ1, τ)
                                prep_node!(ws, P, LM, u2[1], u2[2], -φ[j] / κ2, τ)
                            end
                            finalize!(ws, P, LM, LCc, τ)
                            append!(bm.bps, ws.bps)
                            nonempty |= !isempty(ws.ent)
                        end
                        nonempty || continue
                        fam = 2
                    end
                    t = sample_line!(r, bm, P.βv, secprev)
                    (t != 0.0 && !isempty(jac) && rand(r) >= exp(min(0.0, logjac(jac, t, bm)))) && (t = 0.0)
                    commit_block!(bm, s.v, t)
                    loc[q][fam, 1] += 1
                    loc[q][fam, 2] += t != 0.0
                end
                for q2 in 1:bm.n
                    clear!(bm.wss[q2])
                end
            end
            stats .+= sum(loc)
        end
    end
    return evaluate!(P, s), stats
end

"""
    adaptive_sweep!(rng, P, LM, FF, AM, s; wsets, pools, extra) -> (s, stats_local, stats_blocks)

One sweep of the adaptive flow sampler at a complete level with (k, x) fixed:
the local sweep with the plain, flow and learned (`amode`, `amode0`) families,
the storage multiscale sweep, the long-range storage sweep (probability `slong.p`) and the
learned block shapes, then observation of
the state and a mode update on the geometric schedule (diminishing adaptation).
"""
function adaptive_sweep!(rng, P::Problem, LM::LocalModel, FF::FlowFrames, AM::AdaptiveModes, s::State;
                         wsets=nothing, pools=nothing, extra=4.0, learned::Bool=true,
                         slong::NamedTuple=(p=0.0, nmoves=32, hmin=24.0, nx=0, sx=0.0))
    refresh!(FF, P, s)
    fams, mult = flow_families(P.E)
    fams = vcat(fams, [(:amode, 0.03), (:acut, 0.1), (:amode0, 0.03)])
    learned && append!(fams, [(:tmodeL, 0.03), (:tcutL, 0.1), (:tlineL, 0.05), (:tpotL, 0.05), (:amodeL, 0.02), (:acutL, 0.08)])
    learned && push!(fams, (:tsocL, 0.05))
    wb = sum(last.(fams[1:findfirst(f -> f[1] == :tpot, fams)-1]))
    mult = sum(last.(fams)) / wb
    s, st1 = local_sweep!(rng, P, LM, s; wsets=wsets, extra=extra * mult, families=fams, frames=FF, adapt=AM,
                          sxpair=get(slong, :sx, 0.0))
    lrn = learned && length(AM.updates) > 0 ? (AM=AM, FF=FF) : nothing
    rand(rng) < 0.5 && ((s, _) = storage_ms_sweep!(rng, P, LM, s; pools=pools,
                                                   learn=(lrn !== nothing && rand(rng) < 0.5) ? lrn : nothing))
    rand(rng) < slong.p && ((s, _) = storage_long_sweep!(rng, P, LM, s; pools=pools, nmoves=slong.nmoves, hmin=slong.hmin,
                                                         nxmoves=get(slong, :nx, 0),
                                                         learn=(lrn !== nothing && rand(rng) < 0.5) ? lrn : nothing))
    s, st2 = adaptive_block_sweep!(rng, P, LM, AM, s; pools=pools, learn=lrn)
    LC = LocalCost(P, LM, s)
    observe!(AM, P, LM, LC)
    AM.nobs <= AM.freeze && observe_response!(AM, P, LM, LC, FF)
    maybe_update!(AM, P, LM)
    return s, st1, st2
end

# ---------------------------------------------------------------------------
# transported peak-set capacity moves (α = 1; docs/archive/setup_overview.md §10)
# ---------------------------------------------------------------------------
#
# At α = 1 a dispatchable capacity is pinned to its peak dispatch within ≈ 1/(β_v c),
# so moves of k at fixed dispatch do not mix capacities. The peak-set move changes k
# together with the dispatch at the node's peak steps S (top `q` of the proxy
# requirement load − renewable output, which depends on (x, k_ren) only and is
# unchanged by the move). With c = clamp to the capacity box (the capacity the
# operation sees):
#   k → k + δ;  on S: p → p + A (c(k+δ) − c(k)) (ω' = (ω c(k) + c(k+δ) − c(k))/c(k+δ));
#   off S: p fixed (ω' = ω c(k)/c(k+δ)); inside the box c(k) = k, beyond it only the raw
#   capacity moves; at S either the residual absorbs (shed falls, `absorb`) or the other
#   units at the node give way with the learned coefficients (`learned`; translations).
# The maps form a one-parameter group (translations in (k, p_S − A c(k), p_off)), so δ is
# drawn from the generalized Gibbs conditional ∝ π(g_δ x)·(c(k)/c(k + δ))^T, where π
# includes capex, the bias and the reference measure (the α = 1 target). Slice sampling
# in u = log((k + δ)/k), truncated to k + δ ≥ κmin (the selection rule k ≥ κmin is then
# invariant).

"Utilization map of the peak-set move on a step in (`on`) or out of the peak set, for clamped capacities kc0 → kc1."
@inline kpeak_map(x::Float64, on::Bool, kc0::Float64, kc1::Float64) =
    kc0 == kc1 ? x : on ? (x * kc0 + (kc1 - kc0)) / kc1 : x * kc0 / kc1

"Peak steps of node n: the top fraction q of the proxy requirement (load − renewable output)."
function peak_set(P::Problem, LM::LocalModel, LC::LocalCost, Xn, n::Int; q::Float64=0.05)
    E = P.E
    T = LM.T
    Q = max.(Xn[1][n, :], 0.0)
    for (gi, g) in enumerate(E.gens)
        g.vf == 0 || continue                      # renewables as load: output A·k
        for (i, bn) in enumerate(g.bus)
            bn == n && (Q .-= LC.A[gi][i, :] .* LC.kgen[gi][i])
        end
    end
    m = max(1, round(Int, q * T))
    return sort(partialsortperm(Q, 1:m; rev=true))
end

"""
    kpeak_sweep!(rng, P, LM, s; LC, BC, AM, q, w, κmin) -> (s, nmoved)

One peak-set move per investable dispatchable generator unit (random order) at a
complete level, variant `absorb` or `learned` with probability 1/2 each (learned
only once AM has been updated). At α = 1 the bias change comes from the incremental
bias cache `BC` (built from `s` if not given). For tests: `select(1, gi, i)` restricts
the units, and `rec` (a vector) receives per move δ, lf(δ) − lf(0) and its log Jacobian.
"""
function kpeak_sweep!(rng, P::Problem, LM::LocalModel, s::State; LC=nothing, BC=nothing, AM=nothing, q::Float64=0.05,
                      w::Float64=0.1, κmin::Float64=1.0, mh::Bool=false, σk=nothing, nprop::Int=1, order::Symbol=:shuffle,
                      select=nothing, rec=nothing)
    E = P.E
    ℓ = LM.ℓ
    lk, lv = P.lk[ℓ+1], P.lv[ℓ+1]
    T = LM.T
    Δt, cv, κ0 = LM.Δt, E.c_viol, E.κ0
    LC = LC === nothing ? LocalCost(P, LM, s) : LC
    BC = (P.use_bias && BC === nothing) ? BiasCache(P, s) : BC
    Xn = weather_nodes(P.W, ℓ, s.xs[end])
    Kraw = knodes(P, ℓ, s.k)
    units = [(gi, i) for (gi, g) in enumerate(E.gens) if g.kf > 0 && g.vf > 0 && LM.complete[g.vf] &&
                                                         (g.ucf == 0 || LM.complete[g.ucf]) &&
                                                         iscomplete(lk, E.kspecs, g.kf) for i in eachindex(g.bus)]
    nmoved = 0
    for (ju, (gi, i)) in enumerate(order == :fixed ? units : shuffle(rng, units))
        g = E.gens[gi]
        kf = g.kf
        sp = E.kspecs[kf]
        k0 = Kraw[kf][i]
        sel = select === nothing || select(1, gi, i)
        # Metropolis form (the device kernel): proposals δ' = δ + σ_j z with a fixed per-unit scale σk[j],
        # absorb variant only; the draws of a unit are taken before any skip
        draws = mh ? [(randn(rng), rand(rng)) for _ in 1:nprop] : nothing
        (sel && k0 >= κmin) || continue
        n = g.bus[i]
        S = peak_set(P, LM, LC, Xn, n; q=q)
        inS = falses(T); inS[S] .= true
        learned = !mh && AM !== nothing && !isempty(AM.updates) && rand(rng) < 0.5
        # compensators: the other generator units at n with their learned weights (deficit class), renormalized
        comps = Tuple{Int,Int,Float64}[]
        if learned
            us = AM.runits[n]
            βs = AM.β[n][:, 1]
            tot = 0.0
            for (q_, (f, j)) in enumerate(us)
                (f == g.vf && j == i) && continue
                βs[q_] > 0 || continue
                push!(comps, (f, j, βs[q_])); tot += βs[q_]
            end
            tot > 0 ? (comps = [(f, j, b / tot) for (f, j, b) in comps]) : (learned = false; empty!(comps))
        end
        c = LM.cell[n]; nc = LM.ncell[c]
        Ylast = LC.Y[g.vf]
        cap = f_capex(E, kf)[i]
        # reference measure: the unit's term n_u log(k_c + κ0) of `log_measure` (physical measure, dispatchable)
        nmeas = (E.measure == :physical && g.wf == 0) ? (g.ucf > 0 ? 2 : 1) * T : 0
        lm0 = nmeas > 0 ? nmeas * log(clamp(k0, sp.lo[i], sp.hi[i]) + κ0) : 0.0
        Ut = E.kspecs[kf].B.Ut
        function setk!(δ)
            for qq in nzrange(Ut, i)
                s.k[lk.off[kf]+rowvals(Ut)[qq]] += δ * nonzeros(Ut)[qq]
            end
        end
        # operational cost change and the per-step injection changes of the move
        function opdelta(δ)
            k1 = k0 + δ
            kc0, kc1 = clamp(k0, sp.lo[i], sp.hi[i]), clamp(k1, sp.lo[i], sp.hi[i])
            Δ = 0.0
            for τ in 1:T
                A = LC.A[gi][i, τ]
                y0 = Ylast[i, τ]
                y1 = kpeak_map(y0, inS[τ], kc0, kc1)
                c0, p0 = gen_cost(g, i, A, kc0, kc0 + κ0, Δt, cv, y0)
                c1, p1 = gen_cost(g, i, A, kc1, kc1 + κ0, Δt, cv, y1)
                if g.ucf > 0                               # commitment moves like the output (same affine map)
                    τ1 = mod(τ, T) + 1
                    Uu = LC.Y[g.ucf]
                    um(x, τ_) = kpeak_map(x, inS[τ_], kc0, kc1)
                    c0 += uc_cost(g, i, A, kc0, kc0 + κ0, Δt, cv, clamp(y0, 0.0, 1.0), Uu[i, τ], Uu[i, τ1])
                    c1 += uc_cost(g, i, A, kc1, kc1 + κ0, Δt, cv, clamp(y1, 0.0, 1.0), um(Uu[i, τ], τ), um(Uu[i, τ1], τ1))
                end
                d = p1 - p0                                # extra injection
                Δ += c1 - c0
                if inS[τ] && learned
                    for (f, j, wj) in comps                # other units give way: −wj·A·δ
                        rl, gj = LM.role[f]
                        gg = E.gens[gj]
                        κj = κunit(P, LM, LC, f, j, τ)
                        abs(κj) < 1e-9 && continue
                        yj = LC.Y[f][j, τ]
                        yj1 = yj - wj * A * (kc1 - kc0) / κj
                        kj = LC.kgen[gj][j]
                        cj0, pj0 = gen_cost(gg, j, LC.A[gj][j, τ], kj, kj + κ0, Δt, cv, yj)
                        cj1, pj1 = gen_cost(gg, j, LC.A[gj][j, τ], kj, kj + κ0, Δt, cv, yj1)
                        if gg.ucf > 0                      # its commitment stays: the walls see the output
                            uj, uj1 = LC.Y[gg.ucf][j, τ], LC.Y[gg.ucf][j, mod(τ, T)+1]
                            Aj = LC.A[gj][j, τ]
                            cj0 += uc_cost(gg, j, Aj, kj, kj + κ0, Δt, cv, clamp(yj, 0.0, 1.0), uj, uj1)
                            cj1 += uc_cost(gg, j, Aj, kj, kj + κ0, Δt, cv, clamp(yj1, 0.0, 1.0), uj, uj1)
                        end
                        Δ += cj1 - cj0
                        d += pj1 - pj0
                    end
                end
                S0 = LC.S[c, τ]
                Δ += Δt * nc * (φbal(E, (S0 - d) / nc) - φbal(E, S0 / nc))
            end
            Δcap = cap * δ + E.γk * cap * (abs(k1 - kc1) - abs(k0 - kc0))
            return Δ, Δcap
        end
        # log Jacobian of the group in δ: (1 + uc) T log(c(k0)/c(k0 + δ))
        function kpeak_logjac(δ)
            kc0, kc1 = clamp(k0, sp.lo[i], sp.hi[i]), clamp(k0 + δ, sp.lo[i], sp.hi[i])
            return kc0 == kc1 ? 0.0 : (g.ucf > 0 ? 2 : 1) * T * log(kc0 / kc1)
        end
        function lf(δ)
            k0 + δ < κmin && return -Inf
            Δop, Δcap = opdelta(δ)
            val = -P.βv * Δop - P.βk * Δcap + kpeak_logjac(δ)
            nmeas > 0 && (val += nmeas * log(clamp(k0 + δ, sp.lo[i], sp.hi[i]) + κ0) - lm0)
            BC === nothing || (val -= bias_delta_k(BC, P.bias, E, lk, 1, gi, i, k0 + δ))
            return val
        end
        if mh
            δ, lδ = 0.0, 0.0
            for (z, uu) in draws
                δp = δ + σk[ju] * z
                k0 + δp < κmin && continue
                lp = lf(δp)
                log(uu) < lp - lδ && (δ = δp; lδ = lp)
            end
        else
            # slice sampling in u = log((k0 + δ)/k0) (Jacobian e^u) with a width that does not
            # depend on the state: a width w·k0 in δ would differ between the points of the line,
            # and the stepping-out rule would then not be reversible
            lfu(u) = lf(k0 * expm1(u)) + u
            δ = k0 * expm1(slice_scalar(rng, lfu, 0.0, w, log(κmin / k0)))
        end
        rec === nothing || push!(rec, (; kind=1, gi, i, δ, lf=lf(δ) - lf(0.0), logjac=kpeak_logjac(δ)))
        δ == 0.0 && continue
        nmoved += 1
        # commit: capacity, utilizations, compensators, residuals and cell sums, bias cache
        k1 = k0 + δ
        kq0, kq1 = clamp(k0, sp.lo[i], sp.hi[i]), clamp(k1, sp.lo[i], sp.hi[i])
        setk!(δ)
        BC === nothing || bias_delta_k(BC, P.bias, E, lk, 1, gi, i, k1; commit=true)
        Kraw[kf][i] = k1
        LC.kgen[gi][i] = clamp(k1, sp.lo[i], sp.hi[i])
        m = lv.m[g.vf]
        Utv = E.vspecs[g.vf].B.Ut
        if g.ucf > 0                                       # commitment: the same affine map as the output
            mu = lv.m[g.ucf]
            Utu = E.vspecs[g.ucf].B.Ut
            for τ in 1:T
                u0 = LC.Y[g.ucf][i, τ]
                u1 = kpeak_map(u0, inS[τ], kq0, kq1)
                LC.Y[g.ucf][i, τ] = u1
                o = lv.off[g.ucf] + (τ - 1) * mu
                for qq in nzrange(Utu, i)
                    j = rowvals(Utu)[qq]
                    j <= mu && (s.v[o+j] += (u1 - u0) * nonzeros(Utu)[qq])
                end
            end
        end
        for τ in 1:T
            y0 = Ylast[i, τ]
            y1 = kpeak_map(y0, inS[τ], kq0, kq1)
            A = LC.A[gi][i, τ]
            p0 = A * kq0 * clamp(y0, 0.0, 1.0); p1 = A * LC.kgen[gi][i] * clamp(y1, 0.0, 1.0)
            dinj = p1 - p0
            Ylast[i, τ] = y1
            o = lv.off[g.vf] + (τ - 1) * m
            for qq in nzrange(Utv, i)
                j = rowvals(Utv)[qq]
                j <= m && (s.v[o+j] += (y1 - y0) * nonzeros(Utv)[qq])
            end
            if inS[τ] && learned
                for (f, j, wj) in comps
                    gj = LM.role[f][2]
                    κj = κunit(P, LM, LC, f, j, τ)
                    abs(κj) < 1e-9 && continue
                    yj = LC.Y[f][j, τ]
                    yj1 = yj - wj * A * (kq1 - kq0) / κj
                    kj = LC.kgen[gj][j]
                    dinj += LC.A[gj][j, τ] * kj * (clamp(yj1, 0.0, 1.0) - clamp(yj, 0.0, 1.0))
                    LC.Y[f][j, τ] = yj1
                    mf = lv.m[f]; of = lv.off[f] + (τ - 1) * mf
                    Utf = E.vspecs[f].B.Ut
                    for qq in nzrange(Utf, j)
                        jj = rowvals(Utf)[qq]
                        jj <= mf && (s.v[of+jj] += (yj1 - yj) * nonzeros(Utf)[qq])
                    end
                end
            end
            LC.r[n, τ] -= dinj
            LC.S[c, τ] -= dinj
        end
    end
    return evaluate!(P, s), nmoved
end
