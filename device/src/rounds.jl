# Plain families, schedule and the device loop (docs/reactant_design.md §3.3, §4; phase 2).
#
# Families (node values at the complete level; every site is one exact line draw):
#   :gen    coordinate of one dispatchable unit at (node, step); with fleet unit commitment the
#           commitment follows the output (direction (1, 1), headroom kept, as `ws.follow` on the CPU)
#   :uc     commitment coordinate of one committable unit at fixed output
#   :pair   two dispatchable units at one node, direction (1/κ₁, −1/κ₂); masked (t = 0) where
#           κ = A k < κ_min for one of them (per chain, as `prep_random!`)
#   :soc    state of charge of one storage unit at instant τ (SoC terms τ−1 and τ, two balance slots)
#   :link   one link at one step
#   :theta  angle of one node at one step (NONE: the residuals absorb the flow change)
#
# Rounds (conflict-free, §4.1): :gen, :pair — one unit slot (or pair) per node × a time class
# (the commitment couples τ−1 and τ); :uc — all committable units × a time class; :soc — one storage
# slot per node × a time class; :link — an edge colour of the links on the buses × all steps;
# :theta — a distance-2 colour class × all steps. Time classes: odd/even steps, three classes
# when T is odd. Large rounds are split into chunks of at most `bmax` sites (still conflict-free).
#
# State in the device loop: flattened (C, T·X) matrices, column τ + (x − 1) T. Gathers are
# `X[:, idx]`; the commit is a scatter-add (`scadd`), so padded sites (zero changes) may point
# at any column.

using Random

const FAMILIES = (:gen, :uc, :pair, :soc, :link, :theta)

"""
Structure-specialised plain families of device runs. `:gen0`/`:pair0` move units without fleet
unit commitment: no UC terms (the UC terms of such a unit are identically zero) and no coupling of
τ − 1 and τ, so one round takes all steps. With `:gen0` (`:pair0`) in the list, `:gen` (`:pair`)
moves only the committable units (pairs with at least one). Same moves and targets as `FAMILIES`.
"""
const FAMILIES_FAST = (:gen0, :gen, :uc, :pair0, :pair, :soc, :link, :theta3, :theta5, :theta)

# Angle families by node degree: :theta3 (degree ≤ 3) and :theta5 (4–5) are padded to their own
# maximum degree; with them in the list, :theta keeps the nodes of degree ≥ 6. The cost of a θ site
# grows about quadratically with the padded degree (points × terms).
const THETA_BUCKETS = (theta3=(0, 3), theta5=(4, 5))
istheta(f::Symbol) = f == :theta || haskey(THETA_BUCKETS, f)
"Degree range of the nodes of θ family f under the family list `families`."
function theta_range(f::Symbol, families)
    haskey(THETA_BUCKETS, f) && return THETA_BUCKETS[f]
    hi = maximum((THETA_BUCKETS[g][2] for g in families if haskey(THETA_BUCKETS, g)); init=0)
    return (hi + 1, typemax(Int))
end

# ---------------------------------------------------------------------------
# static per-unit parameters and per-chain parameters
# ---------------------------------------------------------------------------

"Static unit, link and line parameters shared by all chains (vectors)."
struct StaticParams{V}
    gmc::V; clo::V; chi::V; hv::V; ucq::V; gmin::V; su::V; nl::V; gav::V
    sh::V; sηin::V; sηout::V; sqt::V; smc::V
    kcap::V; η::V
    lb::V; ls::V; lα::V
    β1::V; β2::V          # LINEAR compensation: learned response coefficient of each unit (deficit, surplus class)
end

function static_params(M::DeviceModel)
    oldfirst = M.gmc0 .<= M.gmc
    clo = ifelse.(oldfirst, M.gmc0, M.gmc)
    chi = ifelse.(oldfirst, M.gmc, M.gmc0)
    hv = Float64.((M.gk0 .> 0) .& (M.gmc0 .!= M.gmc))
    nz(v) = isempty(v) ? zeros(1) : copy(v)              # (empty unit classes: a dummy entry, never moved)
    # initial β as the CPU's AdaptiveModes: equal split over the units of the node
    gn = [findfirst(!=(0.0), M.G[q, :]) for q in 1:size(M.G, 1)]
    β0 = [1.0 / count(==(gn[q]), gn) for q in eachindex(gn)]
    return StaticParams(nz(M.gmc), nz(clo), nz(chi), nz(hv), nz(M.guc), nz(M.ggmin), nz(M.gsu), nz(M.gnl), nz(M.gav),
                        nz(M.sh), nz(M.sηin), nz(M.sηout), nz(M.sqt), nz(M.smc), nz(M.kcap), nz(M.η),
                        nz(M.lb), nz(M.ls), nz(M.lα), nz(β0), nz(copy(β0)))
end

"Per-chain unit parameters (they depend on the capacities only): κ = A k, w = k + κ0, vintage split."
struct ChainParams{A}
    κg::A; wg::A; plo::A; yk::A
    ks::A; ws::A
    cls::A                # (C, T·N): proxy class of (node, step), 1 = surplus (LINEAR compensation; from x only)
end

function chain_params(M::DeviceModel, S::DeviceState; cls=nothing)
    kg, _ = capacities(M, Array(S.kg), M.glo, M.ghi, M.gkfix, M.ginv, M.gcap)
    ks, _ = capacities(M, Array(S.ks), M.slo, M.shi, M.skfix, M.sinv, M.scap)
    C = size(kg, 1)
    A = reshape(M.gav, 1, :)
    κg = A .* kg
    q = min.(reshape(M.gk0, 1, :), kg)
    oldfirst = reshape(M.gmc0 .<= M.gmc, 1, :)
    qlo = ifelse.(oldfirst, q, kg .- q)
    yk = ifelse.(kg .> 0, qlo ./ ifelse.(kg .> 0, kg, 1.0), 0.0)        # `vint_kink`
    pad(X) = size(X, 2) == 0 ? zeros(C, 1) : X
    cl = cls === nothing ? zeros(C, M.T * M.N) : Float64.(cls)
    return ChainParams(pad(κg), pad(kg .+ M.κ0), pad(A .* qlo), pad(yk), pad(ks), pad(ks .+ M.κ0), cl)
end

consts(M::DeviceModel, β; κmin=1.0, cpumap::Bool=false) =
    Consts(M.Δt, M.c_viol, M.c_shed, M.c_dump, M.κ0, Float64(β), Float64(κmin), Float64(cpumap), HingeT(M.ψ), HingeT(M.Lq))

# ---------------------------------------------------------------------------
# flattened loop state
# ---------------------------------------------------------------------------

"Loop state: node values and the residual cache, each (C, T·X)."
struct FlatState{A}
    θ::A; ω::A; uc::A; e::A; u::A; r::A
end

flat2(X) = (C = size(X, 1); n = size(X, 2) * size(X, 3); n == 0 ? zeros(C, size(X, 2)) : reshape(copy(X), C, n))

"Flattened loop state of a device state (r = rebuild_r)."
function flat_state(M::DeviceModel, S::DeviceState)
    r = rebuild_r(M, S)
    return FlatState(flat2(S.θ), flat2(S.ω), flat2(S.uc), flat2(S.e), flat2(S.u), flat2(r))
end

"Write the loop state back into a (host) device state; returns the residual cache (C, T, N)."
function unflat!(S::DeviceState, F::FlatState)
    C, T = size(S.θ, 1), size(S.θ, 2)
    put!(X, Y) = (size(X, 3) > 0 && (X .= reshape(Array(Y), C, T, size(X, 3))); X)
    put!(S.θ, F.θ); put!(S.ω, F.ω); put!(S.uc, F.uc); put!(S.e, F.e); put!(S.u, F.u)
    return reshape(Array(F.r), C, T, :)
end

# ---------------------------------------------------------------------------
# host-side graph tables, time classes, colourings
# ---------------------------------------------------------------------------

"Topology of the device model (host)."
struct Topo
    N::Int; T::Int
    gnode::Vector{Int}; snode::Vector{Int}
    lbus0::Vector{Int}; lbus1::Vector{Int}          # links
    ebus0::Vector{Int}; ebus1::Vector{Int}          # lines
    nodegens::Vector{Vector{Int}}; nodestor::Vector{Vector{Int}}
    nodelines::Vector{Vector{Int}}; nbrs::Vector{Vector{Int}}
end

function topo(M::DeviceModel)
    N, T = M.N, M.T
    rowarg(A) = [findfirst(!=(0.0), A[i, :]) for i in 1:size(A, 1)]
    gnode, snode = rowarg(M.G), rowarg(M.S)
    lb0, lb1 = rowarg(M.K0), rowarg(M.K1)
    nl = size(M.Dl, 2)
    eb0 = [findfirst(==(1.0), M.Dl[:, e]) for e in 1:nl]
    eb1 = [findfirst(==(-1.0), M.Dl[:, e]) for e in 1:nl]
    nodegens = [findall(==(n), gnode) for n in 1:N]
    nodestor = [findall(==(n), snode) for n in 1:N]
    nodelines = [[e for e in 1:nl if eb0[e] == n || eb1[e] == n] for n in 1:N]
    nbrs = [sort(unique([eb0[e] == n ? eb1[e] : eb0[e] for e in nodelines[n]])) for n in 1:N]
    return Topo(N, T, gnode, snode, lb0, lb1, eb0, eb1, nodegens, nodestor, nodelines, nbrs)
end

"Time classes: odd and even steps, three classes when T is odd (as `local_sweep!`)."
tclasses(T) = T == 1 ? [[1]] : isodd(T) ? [collect(1:2:T-1), collect(2:2:T-1), [T]] : [collect(1:2:T), collect(2:2:T)]

"Greedy distance-2 colouring of the node graph (θ rounds)."
function colour_d2(tp::Topo)
    col = zeros(Int, tp.N)
    for n in sortperm(length.(tp.nbrs); rev=true)
        used = Set{Int}()
        for m in tp.nbrs[n]
            push!(used, col[m])
            for m2 in tp.nbrs[m]; push!(used, col[m2]); end
        end
        c = 1
        while c in used; c += 1; end
        col[n] = c
    end
    return col
end

"Greedy edge colouring of the links on their buses."
function colour_links(tp::Topo)
    nl = length(tp.lbus0)
    col = zeros(Int, nl)
    for l in 1:nl
        used = Set(col[m] for m in 1:l-1 if !isempty(intersect((tp.lbus0[l], tp.lbus1[l]), (tp.lbus0[m], tp.lbus1[m]))))
        c = 1
        while c in used; c += 1; end
        col[l] = c
    end
    return col
end

chunks(v, b) = [v[i:min(i + b - 1, end)] for i in 1:b:length(v)]

"The family list of the schedule being built (split families such as `:spair`/`:spair0` read it)."
const SCHED_FAMILIES = Ref{Tuple}(())

"""
    schedule(tp; families, bmax) -> Dict(family => rounds)

Rounds of every family; a round is a vector of sites (tuples, see the family list above).
"""
function schedule(tp::Topo, M::DeviceModel; families=FAMILIES, bmax=Dict{Symbol,Int}(), bdefault=1 << 15, rng=MersenneTwister(1))
    T = tp.T
    cls = tclasses(T)
    any(length(tp.nbrs[n]) != length(tp.nodelines[n]) for n in 1:tp.N) &&
        error("parallel lines: the θ family needs one balance slot per line end (not implemented)")
    S = Dict{Symbol,Vector{Vector{Tuple}}}()
    G = maximum(length.(tp.nodegens); init=0)
    Sm = maximum(length.(tp.nodestor); init=0)
    SCHED_FAMILIES[] = Tuple(families)                         # family_rounds of split families read it (:spair0)
    for fam in families
        rounds = Vector{Vector{Tuple}}()
        hasuc(q) = M.guc[q] > 0
        if fam == :gen0 || (fam == :gen && :gen0 in families)       # split by UC (FAMILIES_FAST)
            ng = [filter(q -> hasuc(q) == (fam == :gen), tp.nodegens[n]) for n in 1:tp.N]
            for j in 1:maximum(length.(ng); init=0), c in (fam == :gen0 ? (1:T,) : cls)
                push!(rounds, [(ng[n][j], τ) for τ in c for n in 1:tp.N if length(ng[n]) >= j])
            end
        elseif fam == :gen
            for j in 1:G, c in cls
                push!(rounds, [(tp.nodegens[n][j], τ) for τ in c for n in 1:tp.N if length(tp.nodegens[n]) >= j])
            end
        elseif fam == :uc
            ucu = [q for q in eachindex(tp.gnode) if M.guc[q] > 0]
            isempty(ucu) || for c in cls
                push!(rounds, [(q, τ) for τ in c for q in ucu])
            end
        elseif fam == :pair || fam == :pair0
            prs = [[(g[a], g[b]) for a in eachindex(g) for b in a+1:length(g)] for g in tp.nodegens]
            if fam == :pair0 || :pair0 in families                     # split by UC (FAMILIES_FAST)
                prs = [filter(pq -> (hasuc(pq[1]) || hasuc(pq[2])) == (fam == :pair), p) for p in prs]
            end
            np = maximum(length.(prs); init=0)
            for p in 1:np, c in (fam == :pair0 ? (1:T,) : cls)
                push!(rounds, [(prs[n][p]..., τ) for τ in c for n in 1:tp.N if length(prs[n]) >= p])
            end
        elseif fam == :soc
            for j in 1:Sm, c in cls
                push!(rounds, [(tp.nodestor[n][j], τ) for τ in c for n in 1:tp.N if length(tp.nodestor[n]) >= j])
            end
        elseif fam == :link
            lc = colour_links(tp)
            for c in 1:maximum(lc; init=0)
                push!(rounds, [(l, τ) for τ in 1:T for l in findall(==(c), lc)])
            end
        elseif istheta(fam)
            nc = colour_d2(tp)
            dlo, dhi = theta_range(fam, families)
            indeg(n) = dlo <= length(tp.nodelines[n]) <= dhi
            for c in 1:maximum(nc; init=0)
                push!(rounds, [(n, τ) for τ in 1:T for n in findall(==(c), nc) if !isempty(tp.nodelines[n]) && indeg(n)])
            end
        else
            rounds = family_rounds(Val(fam), tp, M, cls, rng)      # phase-3 families (comp.jl)
        end
        b = get(bmax, fam, bdefault)
        S[fam] = [ch for r in rounds if !isempty(r) for ch in chunks(r, b)]
    end
    return S
end

# ---------------------------------------------------------------------------
# index tables (host: 0 = padding; device: padding → column 1)
# ---------------------------------------------------------------------------

colτ(T, τ, x) = mod1(τ, T) + (x - 1) * T

"Index columns of one site of family `fam` (named; vectors for the per-line fields of :theta)."
function site_cols(fam::Symbol, tp::Topo, site, D::Int)
    T = tp.T
    if fam == :gen0
        q, τ = site
        return (q=q, cw=colτ(T, τ, q), cr=colτ(T, τ, tp.gnode[q]))
    elseif fam == :pair0
        q1, q2, τ = site
        return (q1=q1, q2=q2, cwa=colτ(T, τ, q1), cwb=colτ(T, τ, q2), cr=colτ(T, τ, tp.gnode[q1]))
    elseif fam == :gen || fam == :uc
        q, τ = site
        nt = (q=q, cw=colτ(T, τ, q), cwm=colτ(T, τ - 1, q), cw1=colτ(T, τ + 1, q))
        return fam == :gen ? merge(nt, (cr=colτ(T, τ, tp.gnode[q]),)) : nt
    elseif fam == :pair
        q1, q2, τ = site
        return (q1=q1, q2=q2, cwa=colτ(T, τ, q1), cwma=colτ(T, τ - 1, q1), cw1a=colτ(T, τ + 1, q1),
                cwb=colτ(T, τ, q2), cwmb=colτ(T, τ - 1, q2), cw1b=colτ(T, τ + 1, q2), cr=colτ(T, τ, tp.gnode[q1]))
    elseif fam == :soc
        s, τ = site
        n = tp.snode[s]
        return (s=s, ce=colτ(T, τ, s), cem=colτ(T, τ - 1, s), ce1=colτ(T, τ + 1, s), crm=colτ(T, τ - 1, n), cr=colτ(T, τ, n))
    elseif fam == :link
        l, τ = site
        return (l=l, cu=colτ(T, τ, l), cra=colτ(T, τ, tp.lbus0[l]), crb=colτ(T, τ, tp.lbus1[l]))
    elseif istheta(fam)
        n, τ = site
        ls = tp.nodelines[n]
        far = [tp.ebus0[e] == n ? tp.ebus1[e] : tp.ebus0[e] for e in ls]
        pad(v) = vcat(v, zeros(Int, D - length(v)))
        return (cθ=colτ(T, τ, n), cr=colτ(T, τ, n), ln=pad(ls), cθf=pad([colτ(T, τ, m) for m in far]),
                crf=pad([colτ(T, τ, m) for m in far]))
    end
    return site_cols(Val(fam), tp, site, D)                    # phase-3 families (comp.jl)
end

"Padded line count of family f: the largest degree of its θ sites (all θ tables use the global maximum otherwise)."
family_degree(f::Symbol, tp::Topo, rounds) =
    istheta(f) ? maximum((length(tp.nodelines[site[1]]) for r in rounds for site in r); init=1) : maximum(length.(tp.nodelines); init=1)

"""
Host index table of a family: NamedTuple of (B, R) Int arrays (θ per-line fields (B, D, R)),
0 = padding, plus `on` (B, R) ∈ {0, 1}.
"""
function host_table(fam::Symbol, tp::Topo, rounds; D=maximum(length.(tp.nodelines); init=1))
    R = length(rounds)
    B = maximum(length.(rounds); init=1)
    proto = site_cols(fam, tp, rounds[1][1], D)
    tab = Dict{Symbol,Array{Int}}()
    for (nm, v) in pairs(proto)
        tab[nm] = v isa Vector ? zeros(Int, B, length(v), R) : zeros(Int, B, R)
    end
    on = zeros(B, R)
    for (r, sites) in enumerate(rounds), (b, site) in enumerate(sites)
        sc = site_cols(fam, tp, site, D)
        for (nm, v) in pairs(sc)
            v isa Vector ? (tab[nm][b, :, r] .= v) : (tab[nm][b, r] = v)
        end
        on[b, r] = 1.0
    end
    return merge(NamedTuple{Tuple(keys(proto))}(Tuple(tab[k] for k in keys(proto))), (on=on,))
end

"Device table: padding → index 1 (zero changes there); for :theta the per-line mask `lon`."
function device_table(fam::Symbol, ht)
    nt = NamedTuple{keys(ht)}(Tuple(k == :on ? copy(v) : max.(v, 1) for (k, v) in pairs(ht)))
    istheta(fam) && (nt = merge(nt, (lon=Float64.(ht.ln .> 0),)))
    return merge(nt, device_extra(Val(fam), ht))
end
device_extra(::Val, ht) = NamedTuple()

# read/write footprints by table field: (field, array, writes?)
const FOOT = Dict(
    :gen => [(:cw, :ω, true), (:cw, :uc, true), (:cwm, :ω, false), (:cwm, :uc, false), (:cw1, :uc, false), (:cr, :r, true)],
    :uc => [(:cw, :uc, true), (:cw, :ω, false), (:cwm, :ω, false), (:cwm, :uc, false), (:cw1, :uc, false)],
    :pair => [(:cwa, :ω, true), (:cwa, :uc, true), (:cwma, :ω, false), (:cwma, :uc, false), (:cw1a, :uc, false),
              (:cwb, :ω, true), (:cwb, :uc, true), (:cwmb, :ω, false), (:cwmb, :uc, false), (:cw1b, :uc, false), (:cr, :r, true)],
    :gen0 => [(:cw, :ω, true), (:cr, :r, true)],
    :pair0 => [(:cwa, :ω, true), (:cwb, :ω, true), (:cr, :r, true)],
    :soc => [(:ce, :e, true), (:cem, :e, false), (:ce1, :e, false), (:crm, :r, true), (:cr, :r, true)],
    :link => [(:cu, :u, true), (:cra, :r, true), (:crb, :r, true)],
    :theta => [(:cθ, :θ, true), (:cθf, :θ, false), (:cr, :r, true), (:crf, :r, true)])
for f in keys(THETA_BUCKETS); FOOT[f] = FOOT[:theta]; end

"""
    check_conflicts(fam, ht) -> (ok, nconflicts)

From the host index table: within every round, no item written by one site is read or written by
another site, and no site writes one balance slot twice. The tables checked are the ones the
device kernels gather and scatter with.
"""
function check_conflicts(fam::Symbol, ht)
    B, R = size(ht.on)
    bad = 0
    for r in 1:R
        writer = Dict{Tuple{Symbol,Int},Int}()
        readers = Dict{Tuple{Symbol,Int},Set{Int}}()
        for b in 1:B
            ht.on[b, r] > 0 || continue
            seen = Set{Tuple{Symbol,Int}}()
            for (fld, arr, w) in FOOT[fam]
                X = getfield(ht, fld)
                cols = ndims(X) == 3 ? X[b, :, r] : [X[b, r]]
                for c in cols
                    c == 0 && continue
                    key = (arr, c)
                    if w
                        (arr == :r && key in seen) && (bad += 1)     # one slot twice in one site
                        push!(seen, key)
                        if haskey(writer, key) && writer[key] != b
                            bad += 1
                        end
                        writer[key] = b
                    end
                    push!(get!(readers, key, Set{Int}()), b)
                end
            end
        end
        for (key, b) in writer
            any(!=(b), readers[key]) && (bad += 1)
        end
    end
    return bad == 0, bad
end

# ---------------------------------------------------------------------------
# gathers and scatter-add (Julia arrays and traced arrays)
# ---------------------------------------------------------------------------

"Materialized reshape (a lazy reshape of a traced array is not kept)."
rs(X, dims...) = reshape(X, dims...)
rs(X::Reactant.TracedRArray, dims...) = Reactant.TracedUtils.materialize_traced_array(reshape(X, dims...))
"X[:, idx] as (C, B, 1)."
g3(X, idx) = (Y = X[:, idx]; rs(Y, size(Y, 1), size(Y, 2), 1))
"Static vector v[idx] as (1, B, 1)."
s3(v, idx) = (y = v[idx]; rs(y, 1, length(y), 1))
"(1|C, B) mask as (1, B, 1)."
m3(v) = rs(v, 1, length(v), 1)
flat(X) = rs(X, size(X, 1), size(X, 2))
"""
x where the mask m > 0, else exactly 0. Padded sites and slots need it: XLA contracts
κ·clamp(y₀ + 0) − κ·clamp(y₀) into fma(κ, c, −κc), the rounding error of κc, not 0, and
thousands of padded sites would accumulate that at their shared column.
"""
msk(x, m) = ifelse.(m .> 0.0, x, 0.0)

"X[:, idx[b]] += V[:, b] (scatter-add; repeated indices add)."
function scadd(X::AbstractMatrix, idx::AbstractVector, V::AbstractMatrix)
    Y = copy(X)
    @inbounds for (b, j) in enumerate(idx)
        for c in axes(Y, 1)
            Y[c, j] += V[c, b]
        end
    end
    return Y
end

function scadd(X::Reactant.TracedRArray{T,2}, idx::Reactant.TracedRArray{<:Integer,1}, V::Reactant.TracedRArray{T,2}) where {T}
    ii = Reactant.TracedUtils.materialize_traced_array(reshape(Int64.(idx), length(idx), 1))
    return Reactant.Ops.scatter((a, b) -> a + b, [X], ii, [V];
        update_window_dims=Int64[1], inserted_window_dims=Int64[2], input_batching_dims=Int64[],
        scatter_indices_batching_dims=Int64[], scatter_dims_to_operand_dims=Int64[2], index_vector_dim=Int64(2))[1]
end
scadd(X::Reactant.TracedRArray{T,2}, idx::Reactant.TracedRArray{<:Integer,1}, V::AbstractMatrix) where {T} =
    scadd(X, idx, Reactant.TracedUtils.materialize_traced_array(V .+ zero(T) .* X[:, 1:1]))

# ---------------------------------------------------------------------------
# family term batches and commits
# ---------------------------------------------------------------------------

"Term batch of round r of family `fam`; returns (tb, commit closure data)."
function build(::Val{:gen}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r)
    q, cw, cwm, cw1, cr = tab.q[:, r], tab.cw[:, r], tab.cwm[:, r], tab.cw1[:, r], tab.cr[:, r]
    a = m3(tab.on[:, r])
    ucq = s3(sp.ucq, q)
    au = a .* ucq
    y0, u0, u10, um, ym = g3(st.ω, cw), g3(st.uc, cw), g3(st.uc, cw1), g3(st.uc, cwm), g3(st.ω, cwm)
    κ, w = g3(cp.κg, q), g3(cp.wg, q)
    z = 0.0                    # static zero slope (no primaries, no root search)
    gmin, su, nl = s3(sp.gmin, q), s3(sp.su, q), s3(sp.nl, q)
    gen = GenTerms([y0], [a], [κ], [w], [s3(sp.gmc, q)], [s3(sp.clo, q)], [s3(sp.chi, q)], [g3(cp.plo, q)],
                   [s3(sp.hv, q)], [g3(cp.yk, q)], [1])
    uc = UCTerms([y0, ym], [a, z], [u0, um], [au, z], [u10, u0], [z, au], [κ, κ], [w, w],
                 [gmin, gmin], [su, su], [nl, nl], [ucq, ucq])
    tb = TermBatch(gen, uc, nothing, nothing, nothing, [g3(st.r, cr)])
    return tb, (a, au)
end

function commit(::Val{:gen}, st::FlatState, tab, r, t, dr, aux)
    a, au = aux
    cw, cr = tab.cw[:, r], tab.cr[:, r]
    return FlatState(st.θ, scadd(st.ω, cw, flat(t .* a)), scadd(st.uc, cw, flat(t .* au)), st.e, st.u,
                     scadd(st.r, cr, flat(msk(dr[1], a))))
end

"Generator terms of unit q at the given columns (no UC terms)."
gen_terms(st, cp, sp, q, cw, a) =
    (g3(st.ω, cw), a, g3(cp.κg, q), g3(cp.wg, q), s3(sp.gmc, q), s3(sp.clo, q), s3(sp.chi, q), g3(cp.plo, q),
     s3(sp.hv, q), g3(cp.yk, q))

function build(::Val{:gen0}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r)
    q, cw, cr = tab.q[:, r], tab.cw[:, r], tab.cr[:, r]
    a = m3(tab.on[:, r])
    gen = GenTerms((Any[x] for x in gen_terms(st, cp, sp, q, cw, a))..., [1])
    return TermBatch(gen, nothing, nothing, nothing, nothing, [g3(st.r, cr)]), a
end

commit(::Val{:gen0}, st::FlatState, tab, r, t, dr, a) =
    FlatState(st.θ, scadd(st.ω, tab.cw[:, r], flat(t .* a)), st.uc, st.e, st.u, scadd(st.r, tab.cr[:, r], flat(msk(dr[1], a))))

function build(::Val{:pair0}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts)
    q1, q2, cr = tab.q1[:, r], tab.q2[:, r], tab.cr[:, r]
    on = m3(tab.on[:, r])
    κ1, κ2 = g3(cp.κg, q1), g3(cp.κg, q2)
    ok = (on .> 0.0) .& (absd.(κ1) .>= k.κmin) .& (absd.(κ2) .>= k.κmin)        # κ_min selection rule (per chain)
    a1 = ifelse.(ok, 1.0 ./ ifelse.(ok, κ1, 1.0), 0.0)
    a2 = ifelse.(ok, -1.0 ./ ifelse.(ok, κ2, 1.0), 0.0)
    G = (gen_terms(st, cp, sp, q1, tab.cwa[:, r], a1), gen_terms(st, cp, sp, q2, tab.cwb[:, r], a2))
    gen = GenTerms((Any[x[j] for x in G] for j in 1:10)..., [1, 1])
    return TermBatch(gen, nothing, nothing, nothing, nothing, [g3(st.r, cr)]), (a1, a2)
end

function commit(::Val{:pair0}, st::FlatState, tab, r, t, dr, aux)
    a1, a2 = aux
    ω = scadd(scadd(st.ω, tab.cwa[:, r], flat(t .* a1)), tab.cwb[:, r], flat(t .* a2))
    return FlatState(st.θ, ω, st.uc, st.e, st.u, scadd(st.r, tab.cr[:, r], flat(msk(dr[1], absd.(a1)))))
end

function build(::Val{:uc}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r)
    q, cw, cwm, cw1 = tab.q[:, r], tab.cw[:, r], tab.cwm[:, r], tab.cw1[:, r]
    a = m3(tab.on[:, r])
    ucq = s3(sp.ucq, q)
    au = a .* ucq
    z = 0.0                    # static zero slope (no primaries, no root search)
    y0, u0, u10, um, ym = g3(st.ω, cw), g3(st.uc, cw), g3(st.uc, cw1), g3(st.uc, cwm), g3(st.ω, cwm)
    κ, w = g3(cp.κg, q), g3(cp.wg, q)
    gmin, su, nl = s3(sp.gmin, q), s3(sp.su, q), s3(sp.nl, q)
    uc = UCTerms([y0, ym], [z, z], [u0, um], [au, z], [u10, u0], [z, au], [κ, κ], [w, w],
                 [gmin, gmin], [su, su], [nl, nl], [ucq, ucq])
    tb = TermBatch(nothing, uc, nothing, nothing, nothing, Any[])
    return tb, au
end

commit(::Val{:uc}, st::FlatState, tab, r, t, dr, au) =
    FlatState(st.θ, st.ω, scadd(st.uc, tab.cw[:, r], flat(t .* au)), st.e, st.u, st.r)

function build(::Val{:pair}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts)
    q1, q2, cr = tab.q1[:, r], tab.q2[:, r], tab.cr[:, r]
    on = m3(tab.on[:, r])
    κ1, κ2 = g3(cp.κg, q1), g3(cp.κg, q2)
    ok = (on .> 0.0) .& (absd.(κ1) .>= k.κmin) .& (absd.(κ2) .>= k.κmin)        # κ_min selection rule (per chain)
    a1 = ifelse.(ok, 1.0 ./ ifelse.(ok, κ1, 1.0), 0.0)
    a2 = ifelse.(ok, -1.0 ./ ifelse.(ok, κ2, 1.0), 0.0)
    G = Any[]
    U = Any[]
    for (q, cw, cwm, cw1, κ, a) in ((q1, tab.cwa[:, r], tab.cwma[:, r], tab.cw1a[:, r], κ1, a1),
                                    (q2, tab.cwb[:, r], tab.cwmb[:, r], tab.cw1b[:, r], κ2, a2))
        ucq = s3(sp.ucq, q)
        au = a .* ucq
        z = 0.0                    # static zero slope (no primaries, no root search)
        y0, u0, u10, um, ym = g3(st.ω, cw), g3(st.uc, cw), g3(st.uc, cw1), g3(st.uc, cwm), g3(st.ω, cwm)
        w = g3(cp.wg, q)
        push!(G, (y0, a, κ, w, s3(sp.gmc, q), s3(sp.clo, q), s3(sp.chi, q), g3(cp.plo, q), s3(sp.hv, q), g3(cp.yk, q)))
        gm, su, nl = s3(sp.gmin, q), s3(sp.su, q), s3(sp.nl, q)
        push!(U, (y0, a, u0, au, u10, z, κ, w, gm, su, nl, ucq), (ym, z, um, z, u0, au, κ, w, gm, su, nl, ucq))
    end
    col(X, j) = [x[j] for x in X]
    gen = GenTerms((col(G, j) for j in 1:10)..., [1, 1])
    uc = UCTerms((col(U, j) for j in 1:12)...)
    tb = TermBatch(gen, uc, nothing, nothing, nothing, [g3(st.r, cr)])
    return tb, (a1, a2, a1 .* s3(sp.ucq, q1), a2 .* s3(sp.ucq, q2))
end

function commit(::Val{:pair}, st::FlatState, tab, r, t, dr, aux)
    a1, a2, au1, au2 = aux
    cwa, cwb = tab.cwa[:, r], tab.cwb[:, r]
    ω = scadd(scadd(st.ω, cwa, flat(t .* a1)), cwb, flat(t .* a2))
    uc = scadd(scadd(st.uc, cwa, flat(t .* au1)), cwb, flat(t .* au2))
    return FlatState(st.θ, ω, uc, st.e, st.u, scadd(st.r, tab.cr[:, r], flat(msk(dr[1], absd.(a1)))))
end

function build(::Val{:soc}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r)
    s, ce, cem, ce1, crm, cr = tab.s[:, r], tab.ce[:, r], tab.cem[:, r], tab.ce1[:, r], tab.crm[:, r], tab.cr[:, r]
    a = m3(tab.on[:, r])
    z = 0.0                    # static zero slope (no primaries, no root search)
    e0, em, e1 = g3(st.e, ce), g3(st.e, cem), g3(st.e, ce1)
    k, w = g3(cp.ks, s), g3(cp.ws, s)
    h, ηi, ηo, qt, mc = s3(sp.sh, s), s3(sp.sηin, s), s3(sp.sηout, s), s3(sp.sqt, s), s3(sp.smc, s)
    soc = SocTerms([em, e0], [z, a], [e0, e1], [a, z], [k, k], [w, w], [h, h], [ηi, ηi], [ηo, ηo], [qt, qt], [mc, mc], [1, 2])
    tb = TermBatch(nothing, nothing, nothing, nothing, soc, [g3(st.r, crm), g3(st.r, cr)])
    return tb, a
end

commit(::Val{:soc}, st::FlatState, tab, r, t, dr, a) =
    FlatState(st.θ, st.ω, st.uc, scadd(st.e, tab.ce[:, r], flat(t .* a)), st.u,
              scadd(scadd(st.r, tab.crm[:, r], flat(msk(dr[1], a))), tab.cr[:, r], flat(msk(dr[2], a))))

function build(::Val{:link}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r)
    l, cu, cra, crb = tab.l[:, r], tab.cu[:, r], tab.cra[:, r], tab.crb[:, r]
    a = m3(tab.on[:, r])
    lk = LinkTerms([g3(st.u, cu)], [a], [s3(sp.kcap, l)], [s3(sp.η, l)], [1], [2])
    tb = TermBatch(nothing, nothing, lk, nothing, nothing, [g3(st.r, cra), g3(st.r, crb)])
    return tb, a
end

commit(::Val{:link}, st::FlatState, tab, r, t, dr, a) =
    FlatState(st.θ, st.ω, st.uc, st.e, scadd(st.u, tab.cu[:, r], flat(t .* a)),
              scadd(scadd(st.r, tab.cra[:, r], flat(msk(dr[1], a))), tab.crb[:, r], flat(msk(dr[2], a))))

function build(::Val{:theta}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r)
    cθ, cr = tab.cθ[:, r], tab.cr[:, r]
    a = m3(tab.on[:, r])
    θn = g3(st.θ, cθ)
    D = size(tab.ln, 2)
    f0s, dfs, ss, αs, r0 = Any[], Any[], Any[], Any[], Any[g3(st.r, cr)]
    for i in 1:D
        ln = tab.ln[:, i, r]
        b = s3(sp.lb, ln)
        push!(f0s, b .* (θn .- g3(st.θ, tab.cθf[:, i, r])))      # oriented: the moved node is the near end
        push!(dfs, b .* m3(tab.lon[:, i, r]) .* a)
        push!(ss, s3(sp.ls, ln)); push!(αs, s3(sp.lα, ln))
        push!(r0, g3(st.r, tab.crf[:, i, r]))
    end
    lt = LineTermsD(f0s, dfs, ss, αs, fill(1, D), collect(2:D+1))
    return TermBatch(nothing, nothing, nothing, lt, nothing, r0), a
end

function commit(::Val{:theta}, st::FlatState, tab, r, t, dr, a)
    rr = scadd(st.r, tab.cr[:, r], flat(msk(dr[1], a)))
    for i in 1:size(tab.ln, 2)
        rr = scadd(rr, tab.crf[:, i, r], flat(msk(dr[1+i], a .* m3(tab.lon[:, i, r]))))
    end
    return FlatState(scadd(st.θ, tab.cθ[:, r], flat(t .* a)), st.ω, st.uc, st.e, st.u, rr)
end

build(f::Val, st, cp, sp, tab, r, k) = build(f, st, cp, sp, tab, r)

"Roots kept per secondary quantity: 2 for the plain families (proven), 4 for compensated and cell families."
nroots(::Val) = Val(4)
nroots(::Val{:gen}) = Val(2); nroots(::Val{:uc}) = Val(2); nroots(::Val{:pair}) = Val(2)
nroots(::Val{:gen0}) = Val(2); nroots(::Val{:pair0}) = Val(2)
for f in keys(THETA_BUCKETS)                       # degree buckets: the θ family with their own padding
    @eval build(::Val{$(QuoteNode(f))}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r) = build(Val(:theta), st, cp, sp, tab, r)
    @eval commit(::Val{$(QuoteNode(f))}, st::FlatState, tab, r, t, dr, a) = commit(Val(:theta), st, tab, r, t, dr, a)
    @eval nroots(::Val{$(QuoteNode(f))}) = Val(2)
end
nroots(::Val{:soc}) = Val(2); nroots(::Val{:link}) = Val(2); nroots(::Val{:theta}) = Val(2)

"""
One value for every consumer: XLA may recompute an elementwise producer inside each consumer
fusion, and the copies need not be bitwise equal (FMA contraction), so the drawn t of the commit
and of the residual update would differ by ulps (a cache drift of κ·ulp). The barrier pins it.
"""
pin(x) = x
pin(x::Reactant.TracedRArray) = Reactant.Ops.optimization_barrier(x)[1]

"""
One round: term batch, exact line draws, Jacobian step (T2 families), commit. u: (C, B, 3)
uniforms (u1 segment, u2 offset, u3 Jacobian). Returns (state, t).
"""
function do_round(f::Val, st::FlatState, cp, sp, k::Consts, tab, r, u)
    tb, aux = build(f, st, cp, sp, tab, r, k)
    t = sample_dense(tb, k, u[:, :, 1:1], u[:, :, 2:2]; K=nroots(f))
    t = pin(jac_accept(tb, t, u[:, :, 3:3]))
    dr = slot_changes(tb, k, t)
    return commit_tb(f, st, tab, r, t, dr, aux, tb), t
end

# ---------------------------------------------------------------------------
# the device loop
# ---------------------------------------------------------------------------

draw_u(rng, ::Val{D}) where {D} = rand(rng, Float64, D)

"""
    run_rounds(st, cp, sp, k, tabs, nsweeps, rng, U; fams) -> st

`nsweeps` sweeps; a sweep runs every family of `fams` (a Val of a tuple of symbols) in order, each
family's rounds in a device loop (`@trace for`). Uniforms from `rng` (Reactant's RNG when traced, a
Julia RNG on Arrays), or given: `U` a NamedTuple of (C, B, 3, R·nsweeps) arrays per family.
"""
function run_rounds(st::FlatState, cp, sp, k::Consts, tabs, nsweeps::Int, rng, U, ::Val{fams}) where {fams}
    θ, ω, uc, e, u, r = st.θ, st.ω, st.uc, st.e, st.u, st.r
    C = size(θ, 1)
    for s in 1:nsweeps
        for f in fams
            tab = tabs[f]
            B, R = size(tab.on)
            dims = Val((C, B, 3))
            Uf = U === nothing ? nothing : U[f]
            off = (s - 1) * R
            vf = Val(f)
            @trace for j in 1:R
                uj = Uf === nothing ? draw_u(rng, dims) : Uf[:, :, :, j+off]
                st2, _ = do_round(vf, FlatState(θ, ω, uc, e, u, r), cp, sp, k, tab, j, uj)
                θ, ω, uc, e, u, r = st2.θ, st2.ω, st2.uc, st2.e, st2.u, st2.r
            end
        end
    end
    return FlatState(θ, ω, uc, e, u, r)
end

"""
    run_sweeps(st, cp, sp, k, tabs, ns, rng, Val(fams)) -> st

`ns` sweeps (a traced number: no recompilation for another count) in one compiled program: an
outer device loop over sweeps around the per-family device loops; uniforms from `rng`.
"""
function run_sweeps(st::FlatState, cp, sp, k::Consts, tabs, ns, rng, fv::Val{fams}) where {fams}
    θ, ω, uc, e, u, r = st.θ, st.ω, st.uc, st.e, st.u, st.r
    @trace for s in 1:ns
        st2 = run_rounds(FlatState(θ, ω, uc, e, u, r), cp, sp, k, tabs, 1, rng, nothing, fv)
        θ, ω, uc, e, u, r = st2.θ, st2.ω, st2.uc, st2.e, st2.u, st2.r
    end
    return FlatState(θ, ω, uc, e, u, r)
end

"Number of real sites (line moves) per chain and sweep, per family."
moves_per_sweep(pl) = NamedTuple{pl.fams}(Tuple(Int(sum(pl.host[f].on)) for f in pl.fams))

# ---------------------------------------------------------------------------
# setup helper
# ---------------------------------------------------------------------------

"""
Memory budget of a round, bytes: the dense tensor C·B·points·columns·8 of `sample_dense` (see
`dense_size`); the compiled round holds about six of them at once. `plan` splits the rounds of a
compensated, cell or block family whose largest round exceeds it (`budget = Inf`: no split); the
probe (one round built on host arrays) costs a few seconds of compilation per family.
"""
const DENSE_BUDGET = Ref(2.0^28)

"Everything the device loop needs for model M and device state S (host Arrays)."
function plan(M::DeviceModel, S::DeviceState, β; κmin=1.0, families=FAMILIES, bmax=Dict{Symbol,Int}(), bdefault=1 << 15,
              cpumap::Bool=false, cls=nothing, rng=MersenneTwister(1), cells=nothing, budget=DENSE_BUDGET[])
    tp = topo(M)
    cells === nothing || (CELLINFO[] = cells)            # cell structure of the partition levels (cells.jl)
    sch = schedule(tp, M; families=families, bmax=bmax, bdefault=bdefault, rng=rng)
    fams = Tuple(f for f in families if haskey(sch, f) && !isempty(sch[f]))
    sp, cp, k, st = static_params(M), chain_params(M, S; cls=cls), consts(M, β; κmin=κmin, cpumap=cpumap), flat_state(M, S)
    C = size(st.θ, 1)
    ht = Dict(f => host_table(f, tp, sch[f]; D=family_degree(f, tp, sch[f])) for f in fams)
    for f in fams                                         # memory cap: even chunks of at most b sites
        (f in FAMILIES || f in FAMILIES_FAST) && continue # plain: ≤ 40 points × 7 columns per site, far below
        tab = device_table(f, ht[f])
        np, nc = dense_size(build(Val(f), st, cp, sp, tab, 1, k)[1], k; K=nroots(Val(f)))
        bf = budget / (8.0 * C * np * nc)
        size(tab.on, 1) <= bf && continue
        b = max(1, floor(Int, bf))
        sch[f] = [ch for r in sch[f] for ch in chunks(r, cld(length(r), cld(length(r), b)))]
        ht[f] = host_table(f, tp, sch[f]; D=family_degree(f, tp, sch[f]))
    end
    hts = NamedTuple{fams}(Tuple(ht[f] for f in fams))
    dts = NamedTuple{fams}(Tuple(device_table(f, hts[f]) for f in fams))
    return (tp=tp, sched=sch, fams=fams, host=hts, dev=dts, sp=sp, cp=cp, k=k, st=st)
end
