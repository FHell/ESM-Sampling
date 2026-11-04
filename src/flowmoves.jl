# Flow-based hierarchy, run-time part (docs/archive/flow_hierarchy.md §3, §6): active-set
# frames per step and the transport and storage moves of the fine level.

# ---------------------------------------------------------------------------
# run-time active-set frames (§3)
# ---------------------------------------------------------------------------

"""
Per-step frames of one level: predicted binding lines (from the proxy QP at the
current weather and capacities) and the transport modes projected onto the
null space of their loading rows, as node directions and as angle coefficients.
Built and refreshed from (x, k) only.
"""
mutable struct FlowFrames
    ℓ::Int
    mask::BitMatrix                  # nl × T predicted binding lines
    binding::Vector{Vector{Int}}     # the same as index lists (CPU moves)
    D::Array{Float64,3}              # N × K × T projected transport modes
    DZ::Array{Float64,3}             # m_θ × K × T angle-coefficient changes
    surplus::BitMatrix               # N × T proxy surplus (renewables exceed load + share of balancing) per node and step
    key::UInt
    H::Any                           # sparse graph-Haar angle basis (localized moves)
    pieces::Vector{Vector{Tuple{Int,Float64}}}   # dominant Haar atoms of each proxy mode
end

function frame_injections(P::Problem, ℓ::Int, s::State)
    E = P.E
    sys = E.sys
    Xn = weather_nodes(P.W, ℓ, s.xs[end])
    kc, _ = node_capacities(E, P.lk[ℓ+1], s.k)
    T = P.lv[ℓ+1].T
    p = zeros(sys.N, T)
    p[E.wspecs[1].B.support, :] .-= max.(Xn[1], 0.0)
    for g in E.gens
        g.wf > 0 || continue
        kv = g.kf > 0 ? kc[g.kf] : g.kfix
        A = clamp.(Xn[g.wf], 0.0, 1.0)
        for t in 1:T, i in eachindex(g.bus)
            p[g.bus[i], t] += A[i, t] * kv[i]
        end
    end
    d = vec(mean(max.(Xn[1], 0.0); dims=2))
    w = zeros(sys.N); w[E.wspecs[1].B.support] .= d ./ sum(d)
    p .-= w .* sum(p; dims=1)
    return p
end

function FlowFrames(P::Problem, ℓ::Int, s::State)
    H = haar_theta(P.E)
    FF = FlowFrames(ℓ, falses(0, 0), Vector{Int}[], zeros(0, 0, 0), zeros(0, 0, 0), falses(0, 0), UInt(0),
                    H, localize(P.E.flow.modes, H))
    refresh!(FF, P, s; force=true)
    return FF
end

framekey(s::State) = hash((s.k, s.xs[end]))

"""
Masked projection of the modes onto the null space of the binding loading rows,
batched over steps (array-first): D_τ = D − Wᵀ Λ_τ with
(M_τ W Wᵀ M_τ + (I − M_τ)) Λ_τ = M_τ W D, M_τ = diag(mask_τ).
"""
function project_modes(Dm::Matrix{Float64}, W::Matrix{Float64}, mask::AbstractMatrix{Bool})
    N, K = size(Dm)
    nl, T = size(mask)
    WD = W * Dm
    WW = W * W'
    out = zeros(N, K, T)
    for τ in 1:T
        B = findall(view(mask, :, τ))
        if isempty(B)
            out[:, :, τ] .= Dm
            continue
        end
        WB = W[B, :]
        Aτ = WB * WB'                                # binding subsystem only (well scaled)
        Λ = pinv(Aτ) * (WB * Dm)                     # pinv: binding rows may be dependent (cycles)
        out[:, :, τ] .= Dm .- WB' * Λ
    end
    return out
end

function refresh!(FF::FlowFrames, P::Problem, s::State; force::Bool=false)
    key = framekey(s)
    (!force && key == FF.key) && return FF
    E = P.E
    sys = E.sys
    FB = E.flow
    lv = P.lv[FF.ℓ+1]
    p = frame_injections(P, FF.ℓ, s)
    FF.surplus = p .> 0
    Q = CongestionADMM(sys)
    _, _, mask = project_congestion(Q, p)
    W = zeros(length(sys.line_s), sys.N)
    for e in eachindex(sys.line_s)
        w = sys.line_b[e] / sys.line_s[e]
        W[e, sys.line_bus0[e]] += w; W[e, sys.line_bus1[e]] -= w
    end
    FF.mask = mask
    FF.binding = [findall(view(mask, :, τ)) for τ in 1:size(mask, 2)]
    FF.D = project_modes(FB.modes, W, mask)
    Uθ = Matrix(cols(E.vspecs[E.fθ].B, lv.m[E.fθ]))
    N, K, T = size(FF.D)
    FF.DZ = reshape(Uθ' * reshape(FF.D, N, K * T), size(Uθ, 2), K, T)
    FF.key = key
    return FF
end

# ---------------------------------------------------------------------------
# transport moves at the complete level (§6): tmode, tline
# ---------------------------------------------------------------------------

"""
Transport direction d (node angles) at step τ with angle-coefficient change dz,
plus pro-rata re-dispatch at every node whose requirement changes (ΔQ = L_b d).
"""
function prep_tdir!(ws, P, LM, LC, τ, d::AbstractVector, dz::AbstractVector, jac; κmin=1.0, comp::Bool=true, learn=nothing)
    E = P.E
    sys = E.sys
    lv = P.lv[LM.ℓ+1]
    fθ = E.fθ
    empty!(jac)
    for i in eachindex(d)
        abs(d[i]) > 1e-14 && add_entry!(ws, fθ, i, d[i])
    end
    o = lv.off[fθ] + (τ - 1) * lv.m[fθ]
    for j in eachindex(dz)
        abs(dz[j]) > 1e-14 && addz!(ws, o + j, dz[j])
    end
    g = ws.gbuf                              # zero between uses
    for e in eachindex(sys.line_s)
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        df = sys.line_b[e] * (d[a] - d[b])
        g[a] += df; g[b] -= df
    end
    gmax = maximum(abs, g; init=0.0)
    if gmax != 0.0 && comp                   # uncompensated: the residuals absorb the transport change
        for n in 1:sys.N
            abs(g[n]) > 1e-9 * gmax && prorata!(ws, jac, P, LM, LC, n, g[n], τ; κmin=κmin, learn=learn)
        end
    end
    fill!(g, 0.0)
    return gmax != 0.0
end

"Transport mode k (projected onto the face of the predicted binding set) at step τ."
function prep_tmode!(rng, ws, P, LM, LC, FF::FlowFrames, τ, jac; comp::Bool=true, learn=nothing)
    LM.complete[P.E.fθ] || return false
    K = size(FF.D, 2)
    K == 0 && return false
    k = rand(rng, 1:K)
    return prep_tdir!(ws, P, LM, LC, τ, view(FF.D, :, k, τ), view(FF.DZ, :, k, τ), jac; comp=comp, learn=learn)
end

"Normal of a predicted binding line at τ: injection pair at its ends with re-routed flows, pro rata at both ends."
function prep_tline!(rng, ws, P, LM, LC, FF::FlowFrames, τ, jac; κmin=1.0, comp::Bool=true, learn=nothing)
    E = P.E
    sys = E.sys
    LM.complete[E.fθ] || return false
    B = FF.binding[τ]
    isempty(B) && return false
    e = rand(rng, B)
    a, b = sys.line_bus0[e], sys.line_bus1[e]
    d = ws.nbuf
    reroute!(d, LM.GS, a, b, ws.rb, ws.ry)
    empty!(jac)
    for i in LM.GS.nodes[LM.comp[a]]
        abs(d[i]) > 1e-14 && prep_node!(ws, P, LM, E.fθ, i, d[i], τ)
    end
    comp || return true                      # uncompensated: export from a to b out of the residuals
    prorata!(ws, jac, P, LM, LC, a, 1.0, τ; κmin=κmin, learn=learn)
    prorata!(ws, jac, P, LM, LC, b, -1.0, τ; κmin=κmin, learn=learn)
    return true
end

# ---------------------------------------------------------------------------
# storage multiscale moves (§5, §6): splateau, sshape (SoC form)
# ---------------------------------------------------------------------------

"""
Prepare a storage multiscale move on the block of instants b0+1 … b0+L for the
storage units `us` (one unit, or all units of one class in one cell; equal
utilization change = weights E_s/ΣE). `shape === nothing`: plateau (energy moved
from step b0 to step b0+L); otherwise the shape φ (length L+1, zero ends) inside
the block. Every step whose power changes is compensated pro rata.
"""
function prep_storage_ms!(bm::BlockMove, P, LM, LC, us::Vector{NTuple{2,Int}}, b0::Int, L::Int, shape, jac; κmin=1.0, learn=nothing)
    E = P.E
    n = shape === nothing ? L + 1 : L
    bm.n = n
    resize!(bm.τs, n)
    for q in 1:n
        bm.τs[q] = b0 + q - 1
        clear!(bm.wss[q])
    end
    empty!(jac)
    Δe(j) = shape === nothing ? 1.0 : shape[j+1]           # utilization change at instant b0+j
    for (si, i) in us
        s = E.stors[si]
        LM.complete[s.fe] || return false
        for j in 1:(shape === nothing ? L : L - 1)
            a = Δe(j)
            a == 0.0 && continue
            prep_node!(bm.wss[j+1], P, LM, s.fe, i, a, b0 + j)
        end
    end
    # requirement changes per (step, node), then pro-rata compensation
    for q in 1:n
        j = q - 1                                            # step b0+j: power from e(b0+j) → e(b0+j+1)
        wq = bm.wss[q]
        gq, gn = wq.gbuf, wq.gnodes                          # per-node requirement change (zero between uses)
        empty!(gn)
        for (si, i) in us
            s = E.stors[si]
            κs = LC.kst[si][i] * s.hours[i] / LM.Δt
            κs < κmin && continue
            e0 = j == 0 ? 0.0 : Δe(j)
            e1 = (shape === nothing ? (j + 1 <= L ? 1.0 : 0.0) : (j + 1 <= L - 1 ? Δe(j + 1) : 0.0))
            dg = κs * (e1 - e0)
            dg == 0.0 && continue
            node = s.bus[i]
            (gq[node] == 0.0 && !(node in gn)) && push!(gn, node)
            gq[node] += dg
        end
        for node in gn
            prorata!(wq, jac, P, LM, LC, node, gq[node], bm.τs[q]; κmin=κmin, learn=learn)
        end
        for node in gn
            gq[node] = 0.0
        end
        empty!(gn)
    end
    empty!(bm.bps)
    nonempty = false
    for q in 1:n
        finalize!(bm.wss[q], P, LM, LC, bm.τs[q])
        append!(bm.bps, bm.wss[q].bps)
        nonempty |= !isempty(bm.wss[q].ent)
    end
    for q in 1:n-1
        link_soc!(bm.wss[q], bm.wss[q+1], LM)
    end
    return nonempty
end

"""
    storage_ms_sweep!(rng, P, LM, s; lengths, nmoves, p_agg, p_shape, LC, pools) -> (s, stats)

Storage multiscale sweep at a complete level in the SoC form: for each dyadic
block length (random order, random offset, no wrap), `nmoves` moves per block,
plateau or shape (probability `p_shape`), for one unit or a cell–class aggregate
(probability `p_agg`). Blocks of one parity are updated in parallel. Selection
uses only (k, x) and the precomputed shapes.
"""
function storage_ms_sweep!(rng, P::Problem, LM::LocalModel, s::State; lengths=[2, 4, 8, 16, 32, 64, 128],
                           nmoves::Int=8, p_agg::Float64=0.5, p_shape::Float64=0.5, LC=nothing, pools=nothing, learn=nothing)
    E = P.E
    FB = E.flow
    T = LM.T
    h = P.lv[LM.ℓ+1].h
    LCc = LC === nothing ? LocalCost(P, LM, s) : LC     # single assignment: not boxed in the threaded closure
    lengths = filter(L -> L < T ÷ 2, lengths)
    Lmax = maximum(lengths; init=1)
    pl::Vector{BlockPool} = pools === nothing ? [BlockPool(P, Lmax + 2) for _ in 1:Threads.maxthreadid()] : pools
    ensure_pools!(pl, P, Lmax + 2)
    units = [(si, i) for (si, st) in enumerate(E.stors) for i in eachindex(st.bus) if LM.complete[st.fe]]
    isempty(units) && return evaluate!(P, s), zeros(2, 2)
    cellof = FB === nothing ? E.parts[2] : FB.cell
    groups = Dict{Tuple{Int,Int},Vector{NTuple{2,Int}}}()
    for (si, i) in units
        push!(get!(groups, (cellof[E.stors[si].bus[i]], si), NTuple{2,Int}[]), (si, i))
    end
    gkeys = collect(keys(groups))
    # shapes per (L, cell), computed before the parallel phase
    shapes = Dict{Tuple{Int,Int},Matrix{Float64}}()
    if FB !== nothing
        for L in lengths, c in unique(cellof)
            shapes[(L, c)] = storage_shapes(FB, E.sys, h, L, c)
        end
    end
    stats = zeros(2, 2)
    for L in shuffle(rng, lengths)
        o = rand(rng, 0:L-1)
        starts = collect(1+o:L:T-L)
        nb = length(starts)
        nb < 2 && continue
        classes = [starts[1:2:end], starts[2:2:end]]
        for cls in shuffle(rng, classes)
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
                for _ in 1:nmoves
                    us = rand(r) < p_agg ? groups[rand(r, gkeys)] : (us1[1] = rand(r, units); us1)
                    shp = nothing
                    fam = 1
                    if rand(r) < p_shape && FB !== nothing
                        Φ = shapes[(L, cellof[E.stors[us[1][1]].bus[us[1][2]]])]
                        size(Φ, 2) == 0 && continue
                        shp = view(Φ, :, rand(r, 1:size(Φ, 2)))
                        fam = 2
                    end
                    prep_storage_ms!(bm, P, LM, LCc, us, b0, L, shp, jac; learn=learn) || continue
                    t = sample_line!(r, bm, P.βv, secprev)
                    (t != 0.0 && rand(r) >= exp(min(0.0, logjac(jac, t, bm)))) && (t = 0.0)
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
Families of the flow design: the plain local families at their usual rates (the
residual-changing moves that relaxation needs) plus the flow families on top.
Returns (families, extra multiplier that keeps the plain rates unchanged).
"""
function flow_families(E::EnergyModel)
    base = plain_families(E)
    flow = [(:tpot, 0.1), (:tmode, 0.05), (:tcut, 0.15), (:tline, 0.05), (:tmode0, 0.05), (:tline0, 0.05)]
    push!(flow, (:tsoc, 0.1))
    wb, wf = sum(last.(base)), sum(last.(flow))
    return vcat(base, flow), (wb + wf) / wb
end

"""
    flow_sweep!(rng, P, LM, FF, s; wsets, pools, extra, p_storage) -> (s, stats_local, stats_storage)

One sweep of the flow design at a complete level with (k, x) fixed: the local
sweep with the plain and the flow families (frames refreshed if (k, x)
changed), then with probability `p_storage` the storage multiscale sweep.
"""
function flow_sweep!(rng, P::Problem, LM::LocalModel, FF::FlowFrames, s::State; wsets=nothing, pools=nothing, extra=4.0,
                     p_storage::Float64=0.5)
    refresh!(FF, P, s)
    fams, mult = flow_families(P.E)
    s, st1 = local_sweep!(rng, P, LM, s; wsets=wsets, extra=extra * mult, families=fams, frames=FF)
    st2 = zeros(2, 2)
    rand(rng) < p_storage && ((s, st2) = storage_ms_sweep!(rng, P, LM, s; pools=pools))
    return s, st1, st2
end

# ---------------------------------------------------------------------------
# localized transport moves (docs/archive/flow_hierarchy.md §16)
# ---------------------------------------------------------------------------
#
# A transport mode d (dense node angles) is expanded in the sparse graph-Haar angle
# basis H (component → country → node, without the component constants): d = H c.
# A coarse Haar atom is constant inside each of the two cells it contrasts, so its
# flows change only on lines leaving those cells and the compensation is needed
# only at their boundary nodes. A localized move picks one of the dominant atoms of
# a mode (those carrying `frac` of ‖c‖², at most `jmax`) with probability ∝ c_j²
# and moves along it with compensation (T2 exact or learned). The selection depends
# only on the fixed (or frozen learned) modes.

"Sparse graph-Haar angle basis over all nodes (no component constants)."
function haar_theta(E::EnergyModel)
    sys = E.sys
    N = sys.N
    Zc = component_indicators(N, sys.line_bus0, sys.line_bus1)
    comp = [argmax(abs.(Zc[i, :])) for i in 1:N]
    gcounts = [length(unique(zip(E.parts[1:q]...))) for q in eachindex(E.parts)]
    return haar_field_basis([comp, E.parts[2], collect(1:N)], gcounts, collect(1:N); drop_top=true)
end

"Dominant Haar atoms of each column of D: per mode a vector of (atom index, weight c_j²)."
function localize(D::AbstractMatrix, H::ModeBasis; frac::Float64=0.9, jmax::Int=8)
    C = Matrix(H.U' * D)
    out = Vector{Vector{Tuple{Int,Float64}}}()
    for k in 1:size(D, 2)
        c2 = C[:, k] .^ 2
        tot = sum(c2)
        p = sortperm(c2; rev=true)
        pieces = Tuple{Int,Float64}[]
        acc = 0.0
        for j in p
            (acc >= frac * tot || length(pieces) >= jmax) && break
            push!(pieces, (j, c2[j]))
            acc += c2[j]
        end
        push!(out, pieces)
    end
    return out
end

"Move along a sparse angle direction (node entries) with compensation; coefficients via the rows of the angle basis."
function prep_tsparse!(ws, P, LM, LC, τ, nodes::AbstractVector{Int}, vals::AbstractVector{Float64}, jac;
                       κmin=1.0, comp::Bool=true, learn=nothing)
    E = P.E
    sys = E.sys
    fθ = E.fθ
    empty!(jac)
    for (n, a) in zip(nodes, vals)
        abs(a) > 1e-14 && prep_node!(ws, P, LM, fθ, n, a, τ)
    end
    comp || return true
    d = ws.nbuf                              # node angle changes of the direction (zero elsewhere)
    for n in nodes; d[n] = 0.0; end
    for n in nodes, e in LM.nodelines[n]
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        d[a] = 0.0; d[b] = 0.0
    end
    for (n, a) in zip(nodes, vals); d[n] = a; end
    g, gn = ws.gbuf, ws.gnodes               # requirement changes (zero between uses), first-touch order
    empty!(gn)
    seen, sl = ws.lseen, ws.lseenl
    empty!(sl)
    for n in nodes, e in LM.nodelines[n]
        seen[e] && continue
        seen[e] = true; push!(sl, e)
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        df = sys.line_b[e] * (d[a] - d[b])
        abs(df) <= 1e-14 && continue
        (g[a] == 0.0 && !(a in gn)) && push!(gn, a)
        g[a] += df
        (g[b] == 0.0 && !(b in gn)) && push!(gn, b)
        g[b] -= df
    end
    for e in sl; seen[e] = false; end
    empty!(sl)
    for m in gn
        abs(g[m]) > 1e-12 && prorata!(ws, jac, P, LM, LC, m, g[m], τ; κmin=κmin, learn=learn)
    end
    for m in gn; g[m] = 0.0; end
    empty!(gn)
    return true
end

"Localized transport move: one dominant Haar atom of a mode (pieces from `localize`)."
function prep_tcut!(rng, ws, P, LM, LC, H::ModeBasis, pieces, τ, jac; comp::Bool=true, learn=nothing)
    LM.complete[P.E.fθ] || return false
    isempty(pieces) && return false
    pk = pieces[rand(rng, eachindex(pieces))]
    isempty(pk) && return false
    tot = 0.0                                # atom ∝ c_j² (inverse cdf, same comparisons as with cumsum)
    for (_, w) in pk; tot += w; end
    u = rand(rng) * tot
    j = pk[end][1]
    acc = 0.0
    for (jj, w) in pk
        acc += w
        if acc >= u
            j = jj
            break
        end
    end
    rg = nzrange(H.U, j)
    return prep_tsparse!(ws, P, LM, LC, τ, view(rowvals(H.U), rg), view(nonzeros(H.U), rg), jac; comp=comp, learn=learn)
end
