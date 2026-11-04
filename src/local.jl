# Sparse local moves of the operational block at one level, with capacities k
# and weather x fixed (the fast block of v2 §5.1).
#
# State. Besides the mode coefficients v (the canonical coordinates of the
# level), `LocalCost` caches the node values Y_f of every v field, the nodal
# residual r = demand − supply (N × T), the line flows F, and the residual sums
# S_c of the balance cells of the level (graph-Haar cells: the projected
# residual of the level is the cell mean, so the balance cost of step τ is
# Δt Σ_c n_c φ(S_c/n_c)). A move is a direction at one step τ given both in node
# space (entries (f, i, a) of ΔY) and in coefficient space (Δv); it changes
# only the terms of step τ (and the storage dynamics term of τ−1 when a state
# of charge moves), so its cost change is evaluated in O(size of the move).
#
# Moves (all π-reversible; selection uses only (k, x), which are fixed here):
#   atom      one basis vector of a field (the coordinate update, sparse Haar column)
#   unit      one unit at one node (spatially complete fields)
#   pair      shift injection between two units; across nodes the flow is
#             re-routed, Δθ = L⁺(e_n − e_m), so every nodal residual is unchanged
#   potential the voltage angle of one node
#   cellpair  shift injection between two carriers over a whole cell of a
#             coarser partition (cell sums of the residual unchanged): the
#             overcomplete, transported coarse direction
#   cell      one carrier over a whole cell (multigrid atom of the overcomplete
#             Haar frame: cell indicators at every coarser level)
#
# Line conditionals. Along a move the cost is piecewise linear in t, so the
# conditional exp(−β_v ΔC(t)) is piecewise exponential. Primary breakpoints are
# the clamp points of every moved node value and the limit points of every
# moved flow; between two consecutive primary breakpoints the residual cell
# sums and the storage dynamics residuals are linear, so their sign changes
# (secondary breakpoints) are found exactly. The conditional is sampled exactly
# by segment masses and inversion, walking outward from t = 0 until the
# potential exceeds its running minimum by `maxnats` (40: neglected mass < e⁻⁴⁰).
# No step widths.

const ROLE_THETA, ROLE_LINK, ROLE_GEN, ROLE_SE, ROLE_UC = 1, 2, 3, 6, 7

"""
Sparse solver for flow re-routing: per AC component, the sparse Cholesky factor
of the Laplacian grounded at one node (pure-Julia triangular solves, thread-safe).
`reroute!(x, GS, n, m)` writes Δθ with L Δθ = e_n − e_m on the component of n
(defined up to the component constant, which does not change flows).
"""
struct GridSolver
    comp::Vector{Int}
    nodes::Vector{Vector{Int}}           # per component
    loc::Vector{Int}                     # local index of a node in its component's reduced system (0 = ground)
    Lf::Vector{SparseMatrixCSC{Float64,Int}}
    perm::Vector{Vector{Int}}
end

function GridSolver(sys)
    N = sys.N
    Zc = component_indicators(N, sys.line_bus0, sys.line_bus1)
    comp = [argmax(abs.(Zc[i, :])) for i in 1:N]
    nc = maximum(comp)
    nodes = [findall(==(c), comp) for c in 1:nc]
    loc = zeros(Int, N)
    Lfs, perms = SparseMatrixCSC{Float64,Int}[], Vector{Int}[]
    for c in 1:nc
        nd = nodes[c]
        for (j, n) in enumerate(nd[2:end]); loc[n] = j; end
        k = length(nd) - 1
        if k == 0
            push!(Lfs, spzeros(0, 0)); push!(perms, Int[]); continue
        end
        I_, J_, V_ = Int[], Int[], Float64[]
        for (a, b, w) in zip(sys.line_bus0, sys.line_bus1, sys.line_b)
            comp[a] == c || continue
            ia, ib = loc[a], loc[b]
            ia > 0 && (push!(I_, ia); push!(J_, ia); push!(V_, w))
            ib > 0 && (push!(I_, ib); push!(J_, ib); push!(V_, w))
            if ia > 0 && ib > 0
                push!(I_, ia); push!(J_, ib); push!(V_, -w)
                push!(I_, ib); push!(J_, ia); push!(V_, -w)
            end
        end
        A = sparse(I_, J_, V_, k, k)
        F = cholesky(Symmetric(A))
        push!(Lfs, sparse(F.L)); push!(perms, F.p)
    end
    return GridSolver(comp, nodes, loc, Lfs, perms)
end

reroute!(x::Vector{Float64}, GS::GridSolver, n::Int, m::Int) = reroute!(x, GS, n, m, Float64[], Float64[])

"Allocation-free form: `b`, `y` are scratch vectors (resized here)."
function reroute!(x::Vector{Float64}, GS::GridSolver, n::Int, m::Int, b::Vector{Float64}, y::Vector{Float64})
    c = GS.comp[n]
    nd = GS.nodes[c]
    k = length(nd) - 1
    resize!(b, k); fill!(b, 0.0)
    GS.loc[n] > 0 && (b[GS.loc[n]] += 1.0)
    GS.loc[m] > 0 && (b[GS.loc[m]] -= 1.0)
    p = GS.perm[c]
    resize!(y, k)
    @inbounds for j in 1:k
        y[j] = b[p[j]]
    end
    ldiv!(LowerTriangular(GS.Lf[c]), y)
    ldiv!(UpperTriangular(transpose(GS.Lf[c])), y)
    z = b                                  # z[p] = y (b no longer needed)
    @inbounds for j in 1:k
        z[p[j]] = y[j]
    end
    for i in nd
        x[i] = GS.loc[i] > 0 ? z[GS.loc[i]] : 0.0
    end
    return x
end

"""
Localized flow re-routing (`xloc` moves): for every node n its partners m (same AC
component, 1 ≤ hops ≤ `rp`) and, per pair, the angle change Δθ on the ball B of all
nodes within `rb` hops of n or m that solves L_BB Δθ_B = e_n − e_m with Δθ = 0
outside B (L_BB: principal submatrix of the line Laplacian, i.e. a zero Dirichlet
boundary). Every node of B keeps its residual; the small dipole leak flows into the
first layer outside B (returned per pair as the requirement changes Δr_j of those
nodes, for the compensated variant `xlocc`). If B is the whole component the exact
global re-routing is used. Depends only on the network (never on the state).
"""
function xloc_directions(sys, GS::GridSolver, nodelines; rp::Int=1, rb::Int=0)
    N = sys.N
    nbr = [Int[] for _ in 1:N]
    for (a, b) in zip(sys.line_bus0, sys.line_bus1)
        a == b && continue
        push!(nbr[a], b); push!(nbr[b], a)
    end
    dist = fill(-1, N)
    function hops(n, R)                     # nodes within R hops of n (truncated BFS)
        out = [n]; dist[n] = 0
        head = 1
        while head <= length(out)
            i = out[head]; head += 1
            dist[i] >= R && continue
            for j in nbr[i]
                dist[j] < 0 || continue
                dist[j] = dist[i] + 1
                push!(out, j)
            end
        end
        for i in out; dist[i] = -1; end
        return out
    end
    balls = [hops(n, rb) for n in 1:N]
    part = [sort!(filter(m -> m != n, hops(n, rp))) for n in 1:N]
    xnod = [Vector{Vector{Int}}() for _ in 1:N]
    xval = [Vector{Vector{Float64}}() for _ in 1:N]
    xlkn = [Vector{Vector{Int}}() for _ in 1:N]
    xlkg = [Vector{Vector{Float64}}() for _ in 1:N]
    loc = zeros(Int, N)
    x = zeros(N)
    for n in 1:N, m in part[n]
        B = sort!(unique!(vcat(balls[n], balls[m])))
        if length(B) >= length(GS.nodes[GS.comp[n]])           # whole component: exact re-routing
            reroute!(x, GS, n, m)
            nd = GS.nodes[GS.comp[n]]
            push!(xnod[n], [i for i in nd if abs(x[i]) >= 1e-15]); push!(xval[n], [x[i] for i in nd if abs(x[i]) >= 1e-15])
            push!(xlkn[n], Int[]); push!(xlkg[n], Float64[])        # no leak
            continue
        end
        for (j, i) in enumerate(B); loc[i] = j; end
        nb = length(B)
        L = zeros(nb, nb)
        for i in B, e in nodelines[i]
            a, b = sys.line_bus0[e], sys.line_bus1[e]
            o = a == i ? b : a
            w = sys.line_b[e]
            L[loc[i], loc[i]] += w
            loc[o] > 0 && (L[loc[i], loc[o]] -= w)
        end
        rhs = zeros(nb); rhs[loc[n]] += 1.0; rhs[loc[m]] -= 1.0
        θ = cholesky(Symmetric(L)) \ rhs
        keep = findall(v -> abs(v) >= 1e-15, θ)
        push!(xnod[n], B[keep]); push!(xval[n], θ[keep])
        # leak: requirement change Δr_j = −Σ_e b_e Δθ_i at the first layer j outside B
        ln, lg = Int[], Float64[]
        for (q, i) in enumerate(B), e in nodelines[i]
            a, b = sys.line_bus0[e], sys.line_bus1[e]
            o = a == i ? b : a
            loc[o] > 0 && continue
            jl = findfirst(==(o), ln)
            jl === nothing && (push!(ln, o); push!(lg, 0.0); jl = length(ln))
            lg[jl] -= sys.line_b[e] * θ[q]
        end
        p = sortperm(ln)
        push!(xlkn[n], ln[p]); push!(xlkg[n], lg[p])
        for i in B; loc[i] = 0; end
    end
    return part, xnod, xval, xlkn, xlkg
end

struct LocalModel
    ℓ::Int
    T::Int
    Δt::Float64
    role::Vector{NTuple{2,Int}}               # per v field: (role, group)
    bus::Vector{Vector{Int}}                  # per v field: node of each support entry (links: bus0)
    nodelines::Vector{Vector{Int}}
    cell::Vector{Int}                         # balance cell of every node at this level
    ncell::Vector{Int}
    units::Vector{Vector{NTuple{2,Int}}}      # per node: (field, support index) of the dispatchable generators
    GS::GridSolver                            # sparse flow re-routing
    comp::Vector{Int}
    complete::Vector{Bool}                    # per v field
    qlev::Int                                 # finest global partition level resolved at this level
    cellunits::Vector{Vector{Vector{Vector{NTuple{2,Int}}}}}   # [q][cell][field slot] → (field, index); injecting fields
    cellfields::Vector{Vector{Vector{Int}}}  # [q][cell] → injecting fields present
    qcum::Vector{Float64}                    # cumulative level probabilities of cell moves (∝ #cells^cell_power)
    genunits::Vector{Vector{NTuple{2,Int}}}  # per node: generator units with a complete utilization field
    nodestor::Vector{Vector{NTuple{2,Int}}}  # per node: storage units (group, index)
    xpart::Vector{Vector{Int}}               # xloc: partners of each node
    xnod::Vector{Vector{Vector{Int}}}        # xloc: [n][j] nodes of the local re-routing to partner j
    xval::Vector{Vector{Vector{Float64}}}    # xloc: [n][j] angle changes
    xlkn::Vector{Vector{Vector{Int}}}        # xloc: [n][j] nodes outside the ball whose requirement changes (leak)
    xlkg::Vector{Vector{Vector{Float64}}}    # xloc: [n][j] their requirement change per unit t
end

function LocalModel(P::Problem, ℓ::Int; cell_power::Float64=1.0, xr_partner::Int=1, xr_ball::Int=0)
    E = P.E
    sys = E.sys
    lv = P.lv[ℓ+1]
    N = sys.N
    nv = length(E.vspecs)
    role = fill((0, 0), nv)
    bus = [Int[] for _ in 1:nv]
    role[E.fθ] = (ROLE_THETA, 0); bus[E.fθ] = collect(1:N)
    E.flink > 0 && (role[E.flink] = (ROLE_LINK, 0); bus[E.flink] = copy(sys.link_bus0))
    for (gi, g) in enumerate(E.gens)
        g.vf > 0 || continue
        role[g.vf] = (ROLE_GEN, gi); bus[g.vf] = g.bus
        g.ucf > 0 && (role[g.ucf] = (ROLE_UC, gi); bus[g.ucf] = g.bus)
    end
    for (si, s) in enumerate(E.stors)
        role[s.fe] = (ROLE_SE, si); bus[s.fe] = s.bus
    end
    nodelines = [Int[] for _ in 1:N]
    for e in eachindex(sys.line_s)
        push!(nodelines[sys.line_bus0[e]], e); push!(nodelines[sys.line_bus1[e]], e)
    end
    Bl = E.wspecs[1].B
    @assert Bl.support == collect(1:N)
    mr = modes_at(Bl, lv.M)
    cell = mr == N ? collect(1:N) : Bl.cells[mr]
    ncell = [count(==(c), cell) for c in 1:maximum(cell)]
    units = [NTuple{2,Int}[] for _ in 1:N]
    for g in E.gens, (i, n) in enumerate(g.bus)
        g.vf > 0 || continue
        push!(units[n], (g.vf, i))
    end
    GS = GridSolver(sys)
    comp = GS.comp
    complete = [iscomplete(lv, E.vspecs, f) for f in 1:nv]
    gcounts = [length(unique(zip(E.parts[1:q]...))) for q in eachindex(E.parts)]
    qlev = something(findlast(<=(lv.M), gcounts), 1)
    injecting = [f for f in 1:nv if role[f][1] == ROLE_GEN]
    cellunits = Vector{Vector{Vector{Vector{NTuple{2,Int}}}}}()
    cellfields = Vector{Vector{Vector{Int}}}()
    for q in 1:qlev
        lab = [hash(Tuple(E.parts[p][n] for p in 1:q)) for n in 1:N]
        keys_ = unique(lab)
        cid = Dict(k => c for (c, k) in enumerate(keys_))
        cu = [[NTuple{2,Int}[] for _ in injecting] for _ in keys_]
        for (slot, f) in enumerate(injecting), (i, n) in enumerate(bus[f])
            push!(cu[cid[lab[n]]][slot], (f, i))
        end
        push!(cellunits, cu)
        push!(cellfields, [[injecting[sl] for sl in eachindex(injecting) if !isempty(cu[c][sl])] for c in eachindex(keys_)])
    end
    # cell moves: level q with fixed probability ∝ (number of cells)^cell_power (finer cells favoured)
    qw = [Float64(length(cu))^cell_power for cu in cellunits]
    qcum = cumsum(qw); qcum ./= qcum[end]
    genunits = [[(f, i) for (f, i) in units[n] if role[f][1] == ROLE_GEN && complete[f]] for n in 1:N]
    nodestor = [[(si, i) for (si, s) in enumerate(E.stors) for (i, bn) in enumerate(s.bus) if bn == n] for n in 1:N]
    xpart, xnod, xval, xlkn, xlkg = complete[E.fθ] ? xloc_directions(sys, GS, nodelines; rp=xr_partner, rb=xr_ball) :
        ([Int[] for _ in 1:N], [Vector{Int}[] for _ in 1:N], [Vector{Float64}[] for _ in 1:N],
         [Vector{Int}[] for _ in 1:N], [Vector{Float64}[] for _ in 1:N])
    return LocalModel(ℓ, lv.T, Float64(lv.h), role, bus, nodelines, cell, ncell, units, GS, comp,
                      complete, qlev, cellunits, cellfields, qcum, genunits, nodestor, xpart, xnod, xval, xlkn, xlkg)
end

"Partition level of a cell move (fixed probabilities, independent of the state)."
@inline draw_level(rng, LM::LocalModel) = something(findfirst(>=(rand(rng)), LM.qcum), length(LM.qcum))

"Cached node quantities of a level state (k, x fixed)."
struct LocalCost
    Y::Vector{Matrix{Float64}}       # node values per v field (support × T)
    A::Vector{Matrix{Float64}}       # clamped availability per generator group
    kgen::Vector{Vector{Float64}}
    kst::Vector{Vector{Float64}}
    r::Matrix{Float64}               # residual (N × T)
    F::Matrix{Float64}               # line flows
    S::Matrix{Float64}               # residual sums of the balance cells
    X::Vector{Matrix{Float64}}       # raw node weather per weather field (support × T), before the data map
end

function LocalCost(P::Problem, LM::LocalModel, s::State)
    E = P.E
    sys = E.sys
    ℓ = LM.ℓ
    lv = P.lv[ℓ+1]
    T = lv.T
    ctx = SweepCtx(P, s.k; ℓ=ℓ)
    Y = [Matrix(nodevals(s.v, lv, E.vspecs, f)) for f in eachindex(E.vspecs)]
    Xn = weather_nodes(P.W, ℓ, s.xs[end])
    A = [g.wf > 0 ? clamp.(Xn[g.wf], 0.0, 1.0) : repeat(g.avail, 1, T) for g in E.gens]
    r = zeros(sys.N, T)
    F = zeros(length(sys.line_s), T)
    LC = LocalCost(Y, A, ctx.kgen, ctx.kst, r, F, zeros(length(LM.ncell), T), [Matrix{Float64}(x) for x in Xn])
    for τ in 1:T
        rebuild_step!(LC, P, LM, Xn, τ)
    end
    return LC
end

function rebuild_step!(LC::LocalCost, P::Problem, LM::LocalModel, Xn, τ::Int)
    E = P.E
    sys = E.sys
    r = view(LC.r, :, τ)
    @inbounds for n in 1:sys.N
        r[n] = max(Xn[1][n, τ], 0.0)
    end
    θ = LC.Y[E.fθ]
    for e in eachindex(sys.line_s)
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        fl = sys.line_b[e] * (θ[a, τ] - θ[b, τ])
        LC.F[e, τ] = fl
        _, loss = line_eval(E.lines, e, sys.line_s[e], fl)
        r[a] += fl + 0.5 * loss; r[b] += -fl + 0.5 * loss
    end
    if E.flink > 0
        Yl = LC.Y[E.flink]
        for l in eachindex(sys.link_cap)
            u = sys.link_cap[l] * clamp(Yl[l, τ], -1.0, 1.0)
            j0, j1 = link_inj(E.link_eta[l], u)
            r[sys.link_bus0[l]] += j0; r[sys.link_bus1[l]] += j1
        end
    end
    for (gi, g) in enumerate(E.gens)
        for i in eachindex(g.bus)
            y = g.vf > 0 ? clamp(LC.Y[g.vf][i, τ], 0.0, 1.0) : 1.0
            r[g.bus[i]] -= LC.A[gi][i, τ] * LC.kgen[gi][i] * y
        end
    end
    T = size(LC.r, 2)
    for (si, s) in enumerate(E.stors)
        Ye = LC.Y[s.fe]
        for i in eachindex(s.bus)
            pc, mc = soc_power(s, i, Float64(P.lv[LM.ℓ+1].h), clamp(Ye[i, τ], 0.0, 1.0), clamp(Ye[i, mod(τ, T)+1], 0.0, 1.0))
            r[s.bus[i]] += LC.kst[si][i] * (pc - mc)
        end
    end
    Sτ = view(LC.S, :, τ)
    fill!(Sτ, 0.0)
    for n in 1:sys.N
        Sτ[LM.cell[n]] += r[n]
    end
end

# ---------------------------------------------------------------------------
# per-thread move workspace
# ---------------------------------------------------------------------------

# Compact per-move terms with their values at t = 0, filled at the start of a line walk
# (eval_move! mode 1) and used by every further evaluation of the walk (mode 2): the
# same arithmetic as the plain evaluation, without re-reading the caches or
# re-evaluating the t = 0 costs.
struct EntTerm                       # generator (rl = ROLE_GEN) or link (ROLE_LINK) entry
    rl::Int
    kid::Int
    b0::Int
    b1::Int
    y0::Float64
    A::Float64                       # generators: availability; links: capacity
    k::Float64
    mc::Float64
    a::Float64
    ak::Float64
    c0::Float64                      # generators: cost at 0; links: |y0 − clamp(y0)|
    p0::Float64                      # generators: output at 0; links: flow at 0
end
struct LineTerm
    b0::Int
    b1::Int
    f0::Float64
    df::Float64
    se::Float64
    c0::Float64                      # max(|f0| − s, 0)
end
struct StuTerm                       # SoC form: storage unit term of the step
    si::Int
    i::Int
    bus::Int
    ye::Float64
    ye1::Float64
    dye::Float64
    dye1::Float64
    k::Float64
    w::Float64
    c0::Float64
    j0::Float64
    ηin::Float64
    ηout::Float64
end
struct CellTerm
    c::Int
    nc::Int
    S0::Float64
    φ0::Float64
end

"Scratch of the exact line sampler (breakpoints, walk points, segment masses)."
struct LineBuf
    right::Vector{Float64}
    left::Vector{Float64}
    pr::Vector{Float64}
    cr::Vector{Float64}
    pl::Vector{Float64}
    cl::Vector{Float64}
    roots::Vector{Float64}
    pts::Vector{Float64}
    cs::Vector{Float64}
    lm::Vector{Float64}
    sec0::Vector{Float64}
end
LineBuf() = LineBuf(Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])

mutable struct MoveWS
    dy::Vector{Vector{Float64}}
    emark::Vector{Vector{Bool}}
    ent::Vector{NTuple{2,Int}}
    lmark::Vector{Bool}
    lines::Vector{Int}
    dfl::Vector{Float64}             # flow change per unit t of each touched line
    nmark::Vector{Bool}
    nodes::Vector{Int}
    dr::Vector{Float64}
    cmark::Vector{Bool}
    cells::Vector{Int}
    dS::Vector{Float64}
    smark::Vector{Vector{Bool}}
    stu::Vector{NTuple{2,Int}}       # storage units whose step-τ term changes
    zidx::Vector{Int}
    zval::Vector{Float64}
    bps::Vector{Float64}
    sec::Vector{Float64}
    secprev::Vector{Float64}
    pts::Vector{Float64}
    cs::Vector{Float64}
    # kinked entries (exact T2 compensation): Δy = dyk·φ_k(t), φ_k(t) = (Q_k + g_k t)₊ − (Q_k)₊
    dyk::Vector{Vector{Float64}}
    kid::Vector{Vector{Int}}
    kQ::Vector{Float64}
    kg::Vector{Float64}
    kφ::Vector{Float64}
    zkidx::Vector{Int}
    zkval::Vector{Float64}
    zkid::Vector{Int}
    # requirement-tracking kinks: Q_k(t) = Q_k + Δr_n(t), the actual change of the residual at node n
    # from all entries except the compensating (kinked) units (losses, lossy links, weather)
    kdyn::Vector{Bool}
    knode::Vector{Int}
    kP::Vector{Float64}
    knm::Vector{Float64}
    anydyn::Bool
    follow::Bool                     # commitment follows the output of committable units (headroom kept)
    # weather entries (fine-level weather moves): raw node weather X(t) = X0 + t dX of field wf at support position wpos
    wf::Vector{Int}
    wpos::Vector{Int}
    wX0::Vector{Float64}
    wdX::Vector{Float64}
    # scratch for allocation-free move preparation and line sampling
    lb::LineBuf                      # line-sampler scratch of the step move
    gbuf::Vector{Float64}            # per node, kept zero between uses
    gnodes::Vector{Int}
    nbuf::Vector{Float64}            # per node (dense re-routing)
    rb::Vector{Float64}
    ry::Vector{Float64}
    lseen::Vector{Bool}              # per line, kept false between uses
    lseenl::Vector{Int}
    tent::Vector{EntTerm}            # cached terms of the walk (eval_move! modes 1, 2)
    tline::Vector{LineTerm}
    tstu::Vector{StuTerm}
    tcell::Vector{CellTerm}
    trec::Bool                       # the recorded terms belong to the prepared move and the current caches
    # buffers of the moves driven by this workspace (kept across steps and blocks)
    bsec::Vector{Float64}            # BlockMove secondary quantities
    bbps::Vector{Float64}            # BlockMove breakpoints
    bτs::Vector{Int}                 # BlockMove steps
    jac::Vector{NTuple{5,Float64}}   # Jacobian terms of pro-rata moves
    mvbuf::Vector{NTuple{3,Int}}     # move list of a step (local_sweep!)
end

@inline kinkφ(Q, g, t) = max(Q + g * t, 0.0) - max(Q, 0.0)


function MoveWS(P::Problem)
    E = P.E
    N, nl = E.sys.N, length(E.sys.line_s)
    sup = [nsupport(s.B) for s in E.vspecs]
    MoveWS([zeros(n) for n in sup], [falses(n) |> collect for n in sup], NTuple{2,Int}[],
           fill(false, nl), Int[], zeros(nl), fill(false, N), Int[], zeros(N), fill(false, N), Int[], zeros(N),
           [fill(false, length(s.bus)) for s in E.stors], NTuple{2,Int}[],
           Int[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
           [zeros(n) for n in sup], [zeros(Int, n) for n in sup], Float64[], Float64[], Float64[], Int[], Float64[], Int[],
           Bool[], Int[], Float64[], Float64[], false, true, Int[], Int[], Float64[], Float64[],
           LineBuf(), zeros(N), Int[], zeros(N), Float64[], Float64[], fill(false, nl), Int[],
           EntTerm[], LineTerm[], StuTerm[], CellTerm[], false,
           Float64[], Float64[], Int[], NTuple{5,Float64}[], NTuple{3,Int}[])
end

function clear!(ws::MoveWS)
    for (f, i) in ws.ent
        ws.dy[f][i] = 0.0; ws.emark[f][i] = false
        ws.dyk[f][i] = 0.0; ws.kid[f][i] = 0
    end
    empty!(ws.ent)
    empty!(ws.kQ); empty!(ws.kg); empty!(ws.kφ); empty!(ws.zkidx); empty!(ws.zkval); empty!(ws.zkid)
    empty!(ws.kdyn); empty!(ws.knode); empty!(ws.kP); empty!(ws.knm); ws.anydyn = false
    empty!(ws.wf); empty!(ws.wpos); empty!(ws.wX0); empty!(ws.wdX)
    for e in ws.lines; ws.lmark[e] = false; end
    empty!(ws.lines)
    for n in ws.nodes; ws.nmark[n] = false; ws.dr[n] = 0.0; end
    empty!(ws.nodes)
    for c in ws.cells; ws.cmark[c] = false; ws.dS[c] = 0.0; end
    empty!(ws.cells)
    for (si, i) in ws.stu; ws.smark[si][i] = false; end
    empty!(ws.stu)
    empty!(ws.zidx); empty!(ws.zval); empty!(ws.bps)
    ws.trec = false
    return ws
end

@inline function add_entry!(ws::MoveWS, f::Int, i::Int, a::Float64)
    a == 0.0 && return
    if !ws.emark[f][i]
        ws.emark[f][i] = true
        push!(ws.ent, (f, i))
    end
    ws.dy[f][i] += a
end

@inline addz!(ws::MoveWS, j::Int, a::Float64) = (push!(ws.zidx, j); push!(ws.zval, a))

"""
New kink (requirement Q, change g per unit t) of the prepared move; returns its index.
With `dyn`, the requirement along the line is the actual one, Q + Δr_node(t) (piecewise
linear: losses, lossy links, weather), and the kink becomes a secondary breakpoint.
`P`, `nm`: output of the compensating units and their number minus one (Jacobian).
"""
function newkink!(ws::MoveWS, Q::Float64, g::Float64; node::Int=0, dyn::Bool=false, P::Float64=NaN, nm::Float64=0.0)
    push!(ws.kQ, Q); push!(ws.kg, g); push!(ws.kφ, 0.0)
    push!(ws.kdyn, dyn); push!(ws.knode, node); push!(ws.kP, P); push!(ws.knm, nm)
    dyn && (ws.anydyn = true)
    return length(ws.kQ)
end

"Kinked node-space change a·φ_k(t) of entry i of field f (generators only), with its coefficient changes."
function add_kentry!(ws::MoveWS, lv::Layout, E, f::Int, i::Int, a::Float64, k::Int, τ::Int)
    a == 0.0 && return
    if !ws.emark[f][i]
        ws.emark[f][i] = true
        push!(ws.ent, (f, i))
    end
    @assert ws.kid[f][i] == 0 || ws.kid[f][i] == k "one kink per entry"
    ws.kid[f][i] = k
    ws.dyk[f][i] += a
    Ut = E.vspecs[f].B.Ut
    m = lv.m[f]
    o = lv.off[f] + (τ - 1) * m
    for q in nzrange(Ut, i)
        j = rowvals(Ut)[q]
        j <= m || continue
        push!(ws.zkidx, o + j); push!(ws.zkval, a * nonzeros(Ut)[q]); push!(ws.zkid, k)
    end
end

"Node-value change of entry (f, i) at parameter t (linear plus kinked part); kφ must be current."
@inline entryΔ(ws::MoveWS, f, i, t) = t * ws.dy[f][i] + (ws.kid[f][i] > 0 ? ws.dyk[f][i] * ws.kφ[ws.kid[f][i]] : 0.0)

@inline function update_kinks!(ws::MoveWS, t)
    @inbounds for k in eachindex(ws.kQ)
        ws.kdyn[k] && continue
        ws.kφ[k] = kinkφ(ws.kQ[k], ws.kg[k], t)
    end
end

"Node-value change of entry (f, i) at t, with the kink evaluated on the fly (requirement-tracking kinks: as last evaluated)."
@inline function entryΔt(ws::MoveWS, f, i, t)
    k = ws.kid[f][i]
    k == 0 && return t * ws.dy[f][i]
    φ = ws.kdyn[k] ? ws.kφ[k] : kinkφ(ws.kQ[k], ws.kg[k], t)
    return t * ws.dy[f][i] + ws.dyk[f][i] * φ
end

@inline function touch_node!(ws::MoveWS, n::Int)
    ws.nmark[n] && return
    ws.nmark[n] = true
    push!(ws.nodes, n)
end

"""
SoC form: the storage term of the step before `wb` sees the state-of-charge
entries of `wb` (as its end value): register those units and their nodes in `wa`.
"""
function link_soc!(wa::MoveWS, wb::MoveWS, LM::LocalModel)
    for (f, i) in wb.ent
        LM.role[f][1] == ROLE_SE || continue
        wb.dy[f][i] == 0.0 && continue
        si = LM.role[f][2]
        wa.smark[si][i] || (wa.smark[si][i] = true; push!(wa.stu, (si, i)))
        n = LM.bus[f][i]
        touch_node!(wa, n)
        c = LM.cell[n]
        wa.cmark[c] || (wa.cmark[c] = true; push!(wa.cells, c))
    end
end

"Coefficient change of a node-space change `a` of entry i of field f at step τ (Δv = Uᵀ e_i a)."
function addz_node!(ws::MoveWS, lv::Layout, E::EnergyModel, f::Int, τ::Int, i::Int, a::Float64)
    Ut = E.vspecs[f].B.Ut
    m = lv.m[f]
    o = lv.off[f] + (τ - 1) * m
    rv, nz = rowvals(Ut), nonzeros(Ut)
    for q in nzrange(Ut, i)
        j = rv[q]
        j <= m && addz!(ws, o + j, a * nz[q])
    end
end

"Derived touch sets and primary breakpoints of the prepared move at step τ."
function finalize!(ws::MoveWS, P::Problem, LM::LocalModel, LC::LocalCost, τ::Int)
    E = P.E
    sys = E.sys
    lv = P.lv[LM.ℓ+1]
    # fleet unit commitment: the commitment of a committable unit follows its output (headroom kept)
    if ws.follow
        nent = length(ws.ent)
        for q in 1:nent
            f, i = ws.ent[q]
            rl, g = LM.role[f]
            rl == ROLE_GEN || continue
            uf = E.gens[g].ucf
            (uf > 0 && LM.complete[uf] && !ws.emark[uf][i]) || continue
            a, ak = ws.dy[f][i], ws.dyk[f][i]
            if a != 0.0
                add_entry!(ws, uf, i, a)
                addz_node!(ws, lv, E, uf, τ, i, a)
            end
            ak != 0.0 && add_kentry!(ws, lv, E, uf, i, ak, ws.kid[f][i], τ)
        end
    end
    for k in eachindex(ws.kQ)
        (!ws.kdyn[k] && ws.kg[k] != 0.0) && push!(ws.bps, -ws.kQ[k] / ws.kg[k])
    end
    for (f, i) in ws.ent
        a = ws.dy[f][i]
        ak = ws.dyk[f][i]
        abs(a) < 1e-14 && (ws.dy[f][i] = 0.0; a = 0.0)
        (a == 0.0 && ak == 0.0) && continue
        rl, g = LM.role[f]
        y0 = LC.Y[f][i, τ]
        yk = rl == ROLE_GEN ? vint_kink(E.gens[g], i, LC.kgen[g][i]) : NaN     # merit-order kink of two vintages
        if ak != 0.0                       # kinked entry: clamp points on both sides of the kink
            k = ws.kid[f][i]
            rl == ROLE_GEN && touch_node!(ws, LM.bus[f][i])
            ws.kdyn[k] && continue         # requirement-tracking kink: clamps are secondary breakpoints
            Q, gk = ws.kQ[k], ws.kg[k]
            for (off, sl) in ((0.0, a + ak * gk), (0.0, a), (-ak * Q, a), (ak * Q, a + ak * gk))
                sl == 0.0 && continue
                push!(ws.bps, (0.0 - y0 - off) / sl, (1.0 - y0 - off) / sl)
                isnan(yk) || push!(ws.bps, (yk - y0 - off) / sl)
            end
            continue
        end
        if rl == ROLE_THETA
            for e in LM.nodelines[i]
                ws.lmark[e] || (ws.lmark[e] = true; push!(ws.lines, e))
            end
            continue
        end
        lo = rl == ROLE_LINK ? -1.0 : 0.0
        push!(ws.bps, (lo - y0) / a, (1.0 - y0) / a)
        isnan(yk) || push!(ws.bps, (yk - y0) / a)
        if rl == ROLE_LINK
            E.link_eta[i] < 1.0 && push!(ws.bps, -y0 / a)          # lossy link: kink at zero flow
            touch_node!(ws, sys.link_bus0[i]); touch_node!(ws, sys.link_bus1[i])
        elseif rl == ROLE_GEN
            touch_node!(ws, LM.bus[f][i])
        elseif rl == ROLE_UC
            # no injection; the start-up and wall kinks are secondary breakpoints
        else
            ws.smark[g][i] || (ws.smark[g][i] = true; push!(ws.stu, (g, i)))
            touch_node!(ws, LM.bus[f][i])     # e_τ moves the injection at τ (and at τ−1: linked step)
        end
    end
    for q in eachindex(ws.wf)                 # weather entries: data-map kinks (load ≥ 0, availability in [0, 1])
        f, pos, X0, dX = ws.wf[q], ws.wpos[q], ws.wX0[q], ws.wdX[q]
        dX == 0.0 && continue
        push!(ws.bps, -X0 / dX)
        f > 1 && push!(ws.bps, (1.0 - X0) / dX)
        touch_node!(ws, E.wspecs[f].B.support[pos])
    end
    fθ = E.fθ
    dθ = ws.dy[fθ]
    LT = E.lines
    keep = 0
    for e in ws.lines                       # lines whose flow does not change (inside a cell-constant
        a, b = sys.line_bus0[e], sys.line_bus1[e]   # angle change) are dropped before any evaluation
        df = sys.line_b[e] * (dθ[a] - dθ[b])
        if abs(df) <= 1e-14
            ws.lmark[e] = false
            continue
        end
        keep += 1
        ws.lines[keep] = e
        ws.dfl[e] = df
        touch_node!(ws, a); touch_node!(ws, b)
        f0 = LC.F[e, τ]
        se = sys.line_s[e]
        LT.losses && push!(ws.bps, -f0 / df)             # kinks of the loss and overload terms
        for u in LT.U
            u > 0 || continue
            push!(ws.bps, (u * se - f0) / df, (-u * se - f0) / df)
        end
    end
    resize!(ws.lines, keep)
    for n in ws.nodes
        c = LM.cell[n]
        ws.cmark[c] || (ws.cmark[c] = true; push!(ws.cells, c))
    end
    return ws
end

"Cost (box penalty included) and output at utilization y of a unit with marginal cost mc."
@inline function gen_cost(mc::Real, A, k, w, Δt, cv, y)
    yc = clamp(y, 0.0, 1.0)
    p = A * k * yc
    return Δt * mc * p + cv * Δt * w * abs(y - yc), p
end

"Cost (merit order of the vintages, box penalty) and output of generator unit i at utilization y."
@inline function gen_cost(g::GenGroup, i, A, k, w, Δt, cv, y)
    hasvintage(g, i) || return gen_cost(g.mc[i], A, k, w, Δt, cv, y)    # (the form of the recorded terms)
    yc = clamp(y, 0.0, 1.0)
    return Δt * gen_vint(g, i, A, k, yc)[1] + cv * Δt * w * abs(y - yc), A * k * yc
end

@inline φbal(E::EnergyModel, x) = x > 0 ? E.c_shed * x : -E.c_dump * x

"""
Cost change ΔC(t) (€) of the prepared move at step τ; fills `ws.sec` with the
secondary quantities (storage dynamics residuals, cell sums, requirement-tracking
kinks, commitment walls) and `ws.dr`/`ws.dS`. Two passes: every entry except the
units of requirement-tracking kinks first (their node residual changes define the
kinks), then those units.
`mode`: 0 plain, 1 plain and record the terms at
t = 0 (start of a line walk), 2 evaluate from the recorded terms (same arithmetic;
valid while the caches and the prepared move are unchanged, i.e. from the walk up to
the commit; `clear!` invalidates them, and mode 2 without valid terms is plain).
Terms are recorded only for the plain model (`cacheable`): moves with the model-
completion terms (losses, soft limits, lossy links, vintages, commitment, weather
entries, requirement-tracking kinks) always use the plain evaluation.
"""
function eval_move!(ws::MoveWS, P::Problem, LM::LocalModel, LC::LocalCost, τ::Int, t::Float64;
                    prev::Union{Nothing,MoveWS}=nothing, next::Union{Nothing,MoveWS}=nothing, mode::Int=0)
    E = P.E
    mode == 2 && ws.trec && return eval_cached!(ws, P, LM, LC, τ, t)
    rec = mode == 1 && cacheable(E, ws)
    rec && (empty!(ws.tent); empty!(ws.tline); empty!(ws.tstu); empty!(ws.tcell); ws.trec = true)
    sys = E.sys
    Δt, cv, κ0 = LM.Δt, E.c_viol, E.κ0
    T = LM.T
    Δ = 0.0
    empty!(ws.sec)
    for n in ws.nodes
        ws.dr[n] = 0.0
    end
    update_kinks!(ws, t)
    @inbounds for (f, i) in ws.ent
        a = ws.dy[f][i]
        kid = ws.kid[f][i]
        (a == 0.0 && kid == 0) && continue
        rl, g = LM.role[f]
        if rl == ROLE_GEN
            (kid > 0 && ws.kdyn[kid]) && continue          # second pass
            if rec
                gg = E.gens[g]
                y0 = LC.Y[f][i, τ]
                k, A = LC.kgen[g][i], LC.A[g][i, τ]
                c0, p0 = gen_cost(gg, i, A, k, k + κ0, Δt, cv, y0)
                push!(ws.tent, EntTerm(ROLE_GEN, ws.kid[f][i], gg.bus[i], 0, y0, A, k, gg.mc[i], a, ws.dyk[f][i], c0, p0))
            end
            Δ += gen_entry!(ws, P, LM, LC, f, i, g, τ, t, next)
        elseif rl == ROLE_LINK
            y0 = LC.Y[f][i, τ]; y1 = y0 + t * a
            cap = sys.link_cap[i]
            yc0, yc1 = clamp(y0, -1.0, 1.0), clamp(y1, -1.0, 1.0)
            Δ += cv * Δt * cap * (abs(y1 - yc1) - abs(y0 - yc0))
            η = E.link_eta[i]
            a0, b0 = link_inj(η, cap * yc0)
            a1, b1 = link_inj(η, cap * yc1)
            ws.dr[sys.link_bus0[i]] += a1 - a0
            ws.dr[sys.link_bus1[i]] += b1 - b0
            rec && push!(ws.tent, EntTerm(ROLE_LINK, 0, sys.link_bus0[i], sys.link_bus1[i], y0, cap, 0.0, 0.0, a, 0.0, abs(y0 - yc0), cap * yc0))
        end
    end
    # weather entries: load and renewable output (renewable utilizations are not part of a weather move)
    @inbounds for q in eachindex(ws.wf)
        f, pos, X0, dX = ws.wf[q], ws.wpos[q], ws.wX0[q], ws.wdX[q]
        X1 = X0 + t * dX
        if f == 1
            ws.dr[E.wspecs[1].B.support[pos]] += max(X1, 0.0) - max(X0, 0.0)
        else
            A0, A1 = clamp(X0, 0.0, 1.0), clamp(X1, 0.0, 1.0)
            A0 == A1 && continue
            for (gi, g) in enumerate(E.gens)
                g.wf == f || continue
                k = LC.kgen[gi][pos]
                y = g.vf > 0 ? LC.Y[g.vf][pos, τ] : 1.0
                c0, p0 = gen_cost(g, pos, A0, k, k + κ0, Δt, cv, y)
                c1, p1 = gen_cost(g, pos, A1, k, k + κ0, Δt, cv, y)
                Δ += c1 - c0
                ws.dr[g.bus[pos]] -= p1 - p0
            end
        end
    end
    # lines (overload terms and losses, half at each end)
    LT = E.lines
    @inbounds for e in ws.lines
        df = ws.dfl[e]
        df == 0.0 && continue
        f0 = LC.F[e, τ]; f1 = f0 + t * df
        se = sys.line_s[e]
        if !(LT.losses || LT.soft)                    # plain hard limit (the recorded terms use this form)
            Δ += cv * Δt * (max(abs(f1) - se, 0.0) - max(abs(f0) - se, 0.0))
            ws.dr[sys.line_bus0[e]] += f1 - f0
            ws.dr[sys.line_bus1[e]] -= f1 - f0
            rec && push!(ws.tline, LineTerm(sys.line_bus0[e], sys.line_bus1[e], f0, df, se, max(abs(f0) - se, 0.0)))
        else
            ov0, L0 = line_eval(LT, e, se, f0)
            ov1, L1 = line_eval(LT, e, se, f1)
            Δ += cv * Δt * (ov1 - ov0)
            ws.dr[sys.line_bus0[e]] += f1 - f0 + 0.5 * (L1 - L0)
            ws.dr[sys.line_bus1[e]] += -(f1 - f0) + 0.5 * (L1 - L0)
        end
    end
    # storage, step τ
    τ1 = mod(τ, T) + 1
    @inbounds for (si, i) in ws.stu
        s = E.stors[si]
        k = LC.kst[si][i]; w = k + κ0
        Ye = LC.Y[s.fe]
        ye, ye1 = Ye[i, τ], Ye[i, τ1]
        dye1 = next === nothing ? 0.0 : next.dy[s.fe][i]       # linked and block moves: τ+1 moves too
        c0, _, j0 = soc_cost(s, i, k, w, Δt, cv, ye, ye1)
        c1, d1, j1 = soc_cost(s, i, k, w, Δt, cv, ye + t * ws.dy[s.fe][i], ye1 + t * dye1)
        Δ += c1 - c0
        ws.dr[s.bus[i]] += j1 - j0
        push!(ws.sec, d1, s.eta_in[i] > 0 ? d1 - s.eta_in[i] : 1.0, -d1 * s.eta_out[i] - 1.0)
        rec && push!(ws.tstu, StuTerm(si, i, s.bus[i], ye, ye1, ws.dy[s.fe][i], dye1, k, w, c0, j0, s.eta_in[i], s.eta_out[i]))
    end
    # requirement-tracking kinks and their compensating units
    if ws.anydyn
        @inbounds for k in eachindex(ws.kQ)
            ws.kdyn[k] || continue
            Qt = ws.kQ[k] + ws.dr[ws.knode[k]]
            ws.kφ[k] = max(Qt, 0.0) - max(ws.kQ[k], 0.0)
            push!(ws.sec, Qt)
        end
        @inbounds for (f, i) in ws.ent
            kid = ws.kid[f][i]
            (kid > 0 && ws.kdyn[kid] && LM.role[f][1] == ROLE_GEN) || continue
            Δ += gen_entry!(ws, P, LM, LC, f, i, LM.role[f][2], τ, t, next)
        end
    end
    # fleet unit commitment
    @inbounds for (f, i) in ws.ent
        LM.role[f][1] == ROLE_UC || continue
        (ws.dy[f][i] == 0.0 && ws.kid[f][i] == 0) && continue
        Δ += uc_entry!(ws, P, LM, LC, f, i, LM.role[f][2], τ, t, prev, next)
    end
    # balance on the cells of the level
    for c in ws.cells
        ws.dS[c] = 0.0
    end
    @inbounds for n in ws.nodes
        ws.dS[LM.cell[n]] += ws.dr[n]
    end
    @inbounds for c in ws.cells
        nc = LM.ncell[c]
        S0 = LC.S[c, τ]; S1 = S0 + ws.dS[c]
        Δ += Δt * nc * (φbal(E, S1 / nc) - φbal(E, S0 / nc))
        push!(ws.sec, S1)
        rec && push!(ws.tcell, CellTerm(c, nc, S0, φbal(E, S0 / nc)))
    end
    return Δ
end

"Evaluation from the recorded terms (eval_move! mode 2); same arithmetic as the plain path."
function eval_cached!(ws::MoveWS, P::Problem, LM::LocalModel, LC::LocalCost, τ::Int, t::Float64)
    E = P.E
    Δt, cv, κ0 = LM.Δt, E.c_viol, E.κ0
    Δ = 0.0
    sec = ws.sec
    empty!(sec)
    dr = ws.dr
    @inbounds for n in ws.nodes
        dr[n] = 0.0
    end
    update_kinks!(ws, t)
    kφ = ws.kφ
    @inbounds for et in ws.tent
        if et.rl == ROLE_GEN
            y1 = et.y0 + (t * et.a + (et.kid > 0 ? et.ak * kφ[et.kid] : 0.0))
            c1, p1 = gen_cost(et.mc, et.A, et.k, et.k + κ0, Δt, cv, y1)
            Δ += c1 - et.c0
            dr[et.b0] -= p1 - et.p0
        else
            y1 = et.y0 + t * et.a
            cap = et.A
            yc1 = clamp(y1, -1.0, 1.0)
            u1 = cap * yc1
            Δ += cv * Δt * cap * (abs(y1 - yc1) - et.c0)
            dr[et.b0] += u1 - et.p0
            dr[et.b1] -= u1 - et.p0
        end
    end
    @inbounds for lt in ws.tline
        f1 = lt.f0 + t * lt.df
        Δ += cv * Δt * (max(abs(f1) - lt.se, 0.0) - lt.c0)
        dr[lt.b0] += f1 - lt.f0
        dr[lt.b1] -= f1 - lt.f0
    end
    @inbounds for st in ws.tstu
        c1, d1, j1 = soc_cost(E.stors[st.si], st.i, st.k, st.w, Δt, cv, st.ye + t * st.dye, st.ye1 + t * st.dye1)
        Δ += c1 - st.c0
        dr[st.bus] += j1 - st.j0
        push!(sec, d1, st.ηin > 0 ? d1 - st.ηin : 1.0, -d1 * st.ηout - 1.0)
    end
    dS = ws.dS
    @inbounds for ct in ws.tcell
        dS[ct.c] = 0.0
    end
    cell = LM.cell
    @inbounds for n in ws.nodes
        dS[cell[n]] += dr[n]
    end
    @inbounds for ct in ws.tcell
        S1 = ct.S0 + dS[ct.c]
        Δ += Δt * ct.nc * (φbal(E, S1 / ct.nc) - ct.φ0)
        push!(sec, S1)
    end
    return Δ
end

"Whether the recorded-term evaluation applies: the plain model and a move without weather entries or tracking kinks."
cacheable(E::EnergyModel, ws::MoveWS) = !(ws.anydyn || !isempty(ws.wf) || E.lines.losses || E.lines.soft ||
                                          any(<(1.0), E.link_eta) || any(g -> g.ucf > 0 || any(>(0.0), g.k0), E.gens))

"Commitment change of unit i (field uf) at step τ+1 by the next step of the move."
@inline nextΔu(next, uf, i, t) = (next === nothing || !next.emark[uf][i]) ? 0.0 : entryΔt(next, uf, i, t)

"Cost change of a generator entry (output, vintages; the commitment walls of step τ if its commitment does not move)."
@inline function gen_entry!(ws::MoveWS, P::Problem, LM::LocalModel, LC::LocalCost, f, i, g, τ, t, next)
    E = P.E
    gg = E.gens[g]
    Δt, cv, κ0 = LM.Δt, E.c_viol, E.κ0
    y0 = LC.Y[f][i, τ]
    k, A = LC.kgen[g][i], LC.A[g][i, τ]
    y1 = y0 + entryΔ(ws, f, i, t)
    c0, p0 = gen_cost(gg, i, A, k, k + κ0, Δt, cv, y0)
    c1, p1 = gen_cost(gg, i, A, k, k + κ0, Δt, cv, y1)
    d = c1 - c0
    ws.dr[gg.bus[i]] -= p1 - p0
    kid = ws.kid[f][i]
    if kid > 0 && ws.kdyn[kid]
        push!(ws.sec, y1, y1 - 1.0)
        yk = vint_kink(gg, i, k)
        isnan(yk) || push!(ws.sec, y1 - yk)
    end
    uf = gg.ucf
    if uf > 0 && !ws.emark[uf][i]
        τ1 = mod(τ, LM.T) + 1
        u = LC.Y[uf][i, τ]
        u10 = LC.Y[uf][i, τ1]
        u1 = u10 + nextΔu(next, uf, i, t)
        yc0, yc1 = clamp(y0, 0.0, 1.0), clamp(y1, 0.0, 1.0)
        d += uc_cost(gg, i, A, k, k + κ0, Δt, cv, yc1, u, u1) - uc_cost(gg, i, A, k, k + κ0, Δt, cv, yc0, u, u10)
        uc = clamp(u, 0.0, 1.0)
        push!(ws.sec, yc1 - uc, gg.uc[1] * uc - yc1, clamp(u1, 0.0, 1.0) - uc)
    end
    return d
end

"""
Cost change of a commitment entry at step τ: the commitment terms of step τ (with the
output of τ and the start-up into τ+1), and the start-up into τ (term τ−1) unless the
previous step of the move evaluates that term itself.
"""
@inline function uc_entry!(ws::MoveWS, P::Problem, LM::LocalModel, LC::LocalCost, f, i, g, τ, t, prev, next)
    E = P.E
    gg = E.gens[g]
    Δt, cv, κ0, T = LM.Δt, E.c_viol, E.κ0, LM.T
    vf = gg.vf
    τ1, τm = mod(τ, T) + 1, mod(τ - 2, T) + 1
    k, A = LC.kgen[g][i], LC.A[g][i, τ]
    w = k + κ0
    u0 = LC.Y[f][i, τ]; u = u0 + entryΔ(ws, f, i, t)
    y0 = LC.Y[vf][i, τ]; y = y0 + (ws.emark[vf][i] ? entryΔ(ws, vf, i, t) : 0.0)
    u10 = LC.Y[f][i, τ1]; u1 = u10 + nextΔu(next, f, i, t)
    yc0, yc = clamp(y0, 0.0, 1.0), clamp(y, 0.0, 1.0)
    d = uc_cost(gg, i, A, k, w, Δt, cv, yc, u, u1) - uc_cost(gg, i, A, k, w, Δt, cv, yc0, u0, u10)
    uc = clamp(u, 0.0, 1.0)
    push!(ws.sec, yc - uc, gg.uc[1] * uc - yc, clamp(u1, 0.0, 1.0) - uc)
    kid = ws.kid[f][i]
    (kid > 0 && ws.kdyn[kid]) && push!(ws.sec, u, u - 1.0)     # its clamps (requirement-tracking kink)
    if !(prev !== nothing && (prev.emark[f][i] || prev.emark[vf][i]))
        Am = LC.A[g][i, τm]
        ym = clamp(LC.Y[vf][i, τm], 0.0, 1.0)
        um = LC.Y[f][i, τm]
        d += uc_cost(gg, i, Am, k, w, Δt, cv, ym, um, u) - uc_cost(gg, i, Am, k, w, Δt, cv, ym, um, u0)
        push!(ws.sec, uc - clamp(um, 0.0, 1.0))
    end
    return d
end

"Apply the prepared move with parameter t to the coefficients and the caches."
function commit!(ws::MoveWS, P::Problem, LM::LocalModel, LC::LocalCost, v::AbstractVector, τ::Int, t::Float64)
    t == 0.0 && return
    eval_move!(ws, P, LM, LC, τ, t; mode=2)    # fills dr, dS at t (from the walk's terms if recorded)
    ws.trec = false                            # the caches change below
    @inbounds for q in eachindex(ws.zidx)
        v[ws.zidx[q]] += t * ws.zval[q]
    end
    @inbounds for q in eachindex(ws.zkidx)
        v[ws.zkidx[q]] += ws.zkval[q] * ws.kφ[ws.zkid[q]]
    end
    @inbounds for (f, i) in ws.ent
        LC.Y[f][i, τ] += entryΔ(ws, f, i, t)
    end
    @inbounds for e in ws.lines
        LC.F[e, τ] += t * ws.dfl[e]
    end
    @inbounds for n in ws.nodes
        LC.r[n, τ] += ws.dr[n]
    end
    @inbounds for c in ws.cells
        LC.S[c, τ] += ws.dS[c]
    end
    commit_weather!(ws, P, LC, τ, t)
end

"Raw node weather and clamped availabilities after a weather move."
function commit_weather!(ws::MoveWS, P::Problem, LC::LocalCost, τ::Int, t::Float64)
    E = P.E
    @inbounds for q in eachindex(ws.wf)
        f, pos = ws.wf[q], ws.wpos[q]
        X1 = ws.wX0[q] + t * ws.wdX[q]
        LC.X[f][pos, τ] = X1
        f == 1 && continue
        for (gi, g) in enumerate(E.gens)
            g.wf == f && (LC.A[gi][pos, τ] = clamp(X1, 0.0, 1.0))
        end
    end
end

# ---------------------------------------------------------------------------
# exact sampling of a piecewise-exponential line conditional
# ---------------------------------------------------------------------------

"log ∫₀ᴸ e^{−g s} ds for x = gL (stable for all signs)."
@inline function logseg(L::Float64, x::Float64)
    abs(x) < 1e-10 && return log(L) - x / 2
    x > 0 && return log(L) + log(-expm1(-x)) - log(x)
    return log(L) - x + log(-expm1(x)) - log(-x)     # x < 0: (e^{−x} − 1)/(−x)
end

"Offset s ∈ [0, L] with density ∝ e^{−g s}."
@inline function trexp_offset(rng, g::Float64, L::Float64)
    x = g * L
    abs(x) < 1e-10 && return rand(rng) * L
    u = rand(rng)
    if g > 0
        return -log1p(-u * (-expm1(-x))) / g
    else
        return L + log1p(-u * (-expm1(x))) / (-g)    # mirror: density ∝ e^{−|g|(L−s)}
    end
end

"log(Φ(b) − Φ(a)) for a < b (standard normal), stable in both tails."
function logΦdiff(a::Float64, b::Float64)
    b <= a && return -Inf
    if a >= 0
        x, y = a / sqrt(2), b / sqrt(2)
        d = erfcx(x) - (isfinite(y) ? erfcx(y) * exp(x^2 - y^2) : 0.0)
        return log(0.5) - x^2 + log(max(d, 1e-300))
    elseif b <= 0
        return logΦdiff(-b, -a)
    else
        return log(max(1.0 - 0.5 * erfc(b / sqrt(2)) - 0.5 * erfc(-a / sqrt(2)), 1e-300))
    end
end

"Standard normal truncated to [a, b] (exact: inversion in the centre, exponential-envelope rejection in the tails)."
function trnorm(rng, a::Float64, b::Float64)
    if b <= -1.0
        return -trnorm(rng, -b, -a)
    elseif a >= 1.0                       # tangent envelope at a: Exp(a) truncated to [a, b], accept e^{−(z−a)²/2}
        while true
            z = a + trexp_offset(rng, a, isfinite(b) ? b - a : 50.0 / a)
            rand(rng) <= exp(-0.5 * (z - a)^2) && return z
        end
    else
        pa, pb = 0.5 * erfc(-a / sqrt(2)), 0.5 * erfc(-b / sqrt(2))
        u = pa + rand(rng) * (pb - pa)
        return clamp(-sqrt(2) * erfcinv(2u), a, b)
    end
end

"""
Exact draw from ∝ exp(−c(t) − κt²/2 − b t) with c piecewise linear through (pts, cs):
truncated Gaussian segments.
"""
function sample_segments_gauss(rng, pts, cs, κ::Float64, b::Float64)
    nseg = length(pts) - 1
    lm = fill(-Inf, nseg)
    sk = sqrt(κ)
    μs, As, Bs = zeros(nseg), zeros(nseg), zeros(nseg)
    for j in 1:nseg
        L = pts[j+1] - pts[j]
        L > 0 || continue
        g = (cs[j+1] - cs[j]) / L
        μ = -(g + b) / κ
        V0 = cs[j] - g * pts[j] - 0.5 * κ * μ^2
        A, B = sk * (pts[j] - μ), sk * (pts[j+1] - μ)
        μs[j], As[j], Bs[j] = μ, A, B
        lm[j] = -V0 + 0.5 * log(2π / κ) + logΦdiff(A, B)
    end
    mx = maximum(lm)
    isfinite(mx) || return 0.0
    u = rand(rng) * sum(x -> exp(x - mx), lm)
    acc = 0.0
    j = nseg
    for q in 1:nseg
        acc += exp(lm[q] - mx)
        if u <= acc
            j = q
            break
        end
    end
    z = trnorm(rng, As[j], Bs[j])
    return clamp(μs[j] + z / sk, pts[j], pts[j+1])
end

# A mover is a prepared line move: mv_eval!(mv, t) → ΔC(t), mv_sec(mv) → its secondary
# quantities at the last evaluation, mv_bps(mv) → primary breakpoints.
struct StepMove
    ws::MoveWS
    P::Problem
    LM::LocalModel
    LC::LocalCost
    τ::Int
end
mv_eval!(m::StepMove, t; mode::Int=0) = eval_move!(m.ws, m.P, m.LM, m.LC, m.τ, t; mode=mode)
mv_sec(m::StepMove) = m.ws.sec
mv_bps(m::StepMove) = m.ws.bps

"A move over consecutive steps τs (one prepared MoveWS per step; no wrap inside)."
mutable struct BlockMove
    wss::Vector{MoveWS}
    n::Int
    τs::Vector{Int}
    P::Problem
    LM::LocalModel
    LC::LocalCost
    sec::Vector{Float64}
    bps::Vector{Float64}
    lb::LineBuf
end
BlockMove(wss, n, τs, P, LM, LC, sec, bps) = BlockMove(wss, n, τs, P, LM, LC, sec, bps, LineBuf())
"Block move over `wss` that uses the persistent buffers of workspace `own` (no allocation per step or block)."
blockmove(wss::Vector{MoveWS}, own::MoveWS, P, LM, LC) =
    BlockMove(wss, 0, empty!(own.bτs), P, LM, LC, empty!(own.bsec), empty!(own.bbps), own.lb)
linebuf(m::StepMove) = m.ws.lb
linebuf(m::BlockMove) = m.lb
linebuf(m) = LineBuf()
# walk evaluations: record the t = 0 terms once (mv_fill!), then evaluate from them (mv_evalc!)
mv_fill!(m::Union{StepMove,BlockMove}) = mv_eval!(m, 0.0; mode=1)
mv_evalc!(m::Union{StepMove,BlockMove}, t) = mv_eval!(m, t; mode=2)
mv_fill!(m) = mv_eval!(m, 0.0)
mv_evalc!(m, t) = mv_eval!(m, t)
function mv_eval!(m::BlockMove, t; mode::Int=0)
    Δ = 0.0
    empty!(m.sec)
    for q in m.n:-1:1                    # later steps first: a step may read the (requirement-tracking) kinks of the next
        prev = q > 1 ? m.wss[q-1] : nothing
        next = q < m.n ? m.wss[q+1] : nothing
        Δ += eval_move!(m.wss[q], m.P, m.LM, m.LC, m.τs[q], t; prev=prev, next=next, mode=mode)
        append!(m.sec, m.wss[q].sec)
    end
    return Δ
end
mv_wss(m::BlockMove) = view(m.wss, 1:m.n)
mv_wss(m::StepMove) = (m.ws,)
mv_sec(m::BlockMove) = m.sec
mv_bps(m::BlockMove) = m.bps

"""
Walk from t = 0 in direction `sgn` over the sorted primary breakpoints `bp`,
appending (t, βΔC) points including the secondary sign changes, until the
potential (plus the optional quadratic `quad` = (κ, b): κt²/2 + bt, the Gaussian
prior of weather moves) exceeds its running minimum by `maxnats`. Returns false
if no such point is found (improper line).
"""
function walk!(pts, cs, mv, β, bp, sgn, maxnats, secprev::Vector{Float64}, roots::Vector{Float64}=Float64[];
               heap::Bool=false, sec0=nothing, quad=(0.0, 0.0))
    qκ, qb = quad
    qf(x) = 0.5 * qκ * x^2 + qb * x
    tp = 0.0
    if sec0 === nothing
        mv_fill!(mv)
        sec0 = mv_sec(mv)
    end
    resize!(secprev, length(sec0)); copyto!(secprev, sec0)
    nonlin = any(ws -> ws.anydyn, mv_wss(mv))       # requirement-tracking kinks
    secT, secA = nonlin ? (Float64[], Float64[]) : (secprev, secprev)
    cmin = 0.0
    idx = 1
    nh = length(bp)                          # heap mode: breakpoints still in the heap
    δ = 0.0
    nx = 0
    slope = 0.0
    while true
        if heap ? nh > 0 : idx <= length(bp)
            if heap                          # next breakpoint in walking order (binary heap, no full sort)
                t = heappop!(bp, nh, sgn); nh -= 1
            else
                t = bp[idx]; idx += 1
            end
            t == tp && continue
        else
            # beyond the last breakpoint: the step from the outgoing slope at tp (a probe, not the slope of
            # the last segment, which can be flat to round-off and send the step to ~1e15), capped
            cprev = isempty(cs) ? 0.0 : cs[end]
            hp = 1e-6 * max(1.0, abs(tp))
            sl = (β * mv_evalc!(mv, tp + sgn * hp) + qf(tp + sgn * hp) - cprev - qf(tp)) / hp
            if sl > 0
                δ = min(2 * maxnats / sl, 1e3 * max(abs(tp), 1e-3))
            else
                δ = δ == 0.0 ? max(abs(tp), 1e-6) : 2δ
            end
            t = tp + sgn * δ
            nx += 1
            nx > 80 && return false
        end
        c = β * mv_evalc!(mv, t)
        sec = mv_sec(mv)
        if !nonlin
            # secondary sign changes between tp and t (all secondary quantities are linear there)
            empty!(roots)
            @inbounds for j in eachindex(sec)
                a, b = secprev[j], sec[j]
                if (a < 0) != (b < 0) && a != b
                    x = tp + (t - tp) * a / (a - b)
                    (sgn * (x - tp) > 0 && sgn * (t - x) > 0) && push!(roots, x)
                end
            end
            copyto!(secprev, sec)          # sec at t (before evaluating the roots)
            if !isempty(roots)
                insort!(roots, sgn)                                   # in walking order (few roots: insertion sort)
                for x in roots
                    cx = β * mv_evalc!(mv, x)
                    push!(pts, x); push!(cs, cx)
                    cmin = min(cmin, cx + qf(x))
                end
            end
        else
            # the secondary quantities are only piecewise linear between primary breakpoints
            # (a kink of one bends others). Event by event: the one-sided slopes at the current
            # point (a probe step) extrapolate every secondary exactly up to the first kink,
            # which is the earliest extrapolated root.
            resize!(secT, length(sec)); copyto!(secT, sec)
            cur = tp
            resize!(secA, length(secprev)); copyto!(secA, secprev)
            nev = 0
            while nev < 50 * (length(secT) + 2)
                h = 1e-7 * abs(t - cur)
                h > 0 || break
                mv_eval!(mv, cur + sgn * h)
                sp = mv_sec(mv)
                length(sp) == length(secA) || break
                xbest = t
                for j in eachindex(secA)
                    a = secA[j]
                    d = (sp[j] - a) / h                         # slope along sgn
                    d == 0.0 && continue
                    τx = -a / d
                    if τx > 0 && a != 0.0
                        x = cur + sgn * τx
                        sgn * (x - cur) > 2h && sgn * (xbest - x) > 0 && (xbest = x)
                    end
                end
                xbest == t && break
                nev += 1
                cx = β * mv_eval!(mv, xbest)
                push!(pts, xbest); push!(cs, cx)
                cmin = min(cmin, cx + qf(xbest))
                copyto!(secA, mv_sec(mv))
                cur = xbest
            end
            copyto!(secprev, secT)
        end
        tb, cb = isempty(pts) ? (0.0, 0.0) : (pts[end], cs[end])
        push!(pts, t); push!(cs, c)
        slope = (c + qf(t) - cb - qf(tb)) / max(abs(t - tb), 1e-300)
        cmin = min(cmin, c + qf(t))
        c + qf(t) - cmin > maxnats && return true
        tp = t
    end
end

"""
Exact draw of t from ∝ exp(−β ΔC(t)) for the prepared move (piecewise
exponential). Returns t (0 if the line conditional is improper).
"""
function sample_line!(rng, ws::MoveWS, P::Problem, LM::LocalModel, LC::LocalCost, τ::Int, β::Float64; maxnats=40.0)
    return sample_line!(rng, StepMove(ws, P, LM, LC, τ), β, ws.secprev; maxnats=maxnats)
end

const NSORTHEAD = 16

"Insertion sort of a short vector in walking order (s > 0 ascending, else descending); no allocation."
function insort!(v::Vector{Float64}, s::Float64)
    @inbounds for i in 2:length(v)
        x = v[i]
        j = i - 1
        while j >= 1 && hlt(x, v[j], s)
            v[j+1] = v[j]
            j -= 1
        end
        v[j+1] = x
    end
    return v
end

# binary heap in place on v[1:n], ordered by walking direction s (s > 0: smallest first)
@inline hlt(a::Float64, b::Float64, s::Float64) = s > 0 ? a < b : a > b
@inline function siftdown!(v::Vector{Float64}, i::Int, n::Int, s::Float64)
    @inbounds begin
        x = v[i]
        while true
            c = 2i
            c > n && break
            (c < n && hlt(v[c+1], v[c], s)) && (c += 1)
            hlt(v[c], x, s) || break
            v[i] = v[c]
            i = c
        end
        v[i] = x
    end
    return v
end
function heapify!(v::Vector{Float64}, s::Float64)
    n = length(v)
    for i in n÷2:-1:1
        siftdown!(v, i, n, s)
    end
    return v
end
"Remove and return the first element (in walking order) of the heap v[1:n]."
@inline function heappop!(v::Vector{Float64}, n::Int, s::Float64)
    @inbounds x = v[1]
    @inbounds v[1] = v[n]
    siftdown!(v, 1, n - 1, s)
    return x
end

function sample_line!(rng, mv, β::Float64, secprev::Vector{Float64}; maxnats=40.0, quad=(0.0, 0.0))
    lb = linebuf(mv)
    right, left = lb.right, lb.left
    empty!(right); empty!(left)
    for x in mv_bps(mv)
        isfinite(x) || continue
        if x > 0.0
            push!(right, x)
        elseif x < 0.0
            push!(left, x)
        end
    end
    # the walk usually stops after a few breakpoints: short lists are sorted, long ones are
    # heapified and popped in walking order (the same sequence, O(n + p log n), no allocation)
    hr, hl = length(right) > NSORTHEAD, length(left) > NSORTHEAD
    hr ? heapify!(right, 1.0) : sort!(right)
    hl ? heapify!(left, -1.0) : sort!(left; rev=true)
    pr, cr, pl, cl = lb.pr, lb.cr, lb.pl, lb.cl
    empty!(pr); empty!(cr); empty!(pl); empty!(cl)
    mv_fill!(mv)                           # terms and secondary quantities at t = 0, once for both walks
    sec0 = lb.sec0
    resize!(sec0, length(mv_sec(mv))); copyto!(sec0, mv_sec(mv))
    walk!(pr, cr, mv, β, right, 1.0, maxnats, secprev, lb.roots; heap=hr, sec0=sec0, quad=quad) || return 0.0
    walk!(pl, cl, mv, β, left, -1.0, maxnats, secprev, lb.roots; heap=hl, sec0=sec0, quad=quad) || return 0.0
    # points in increasing order: the left walk reversed, 0, the right walk (each walk is monotone)
    pts, cs = lb.pts, lb.cs
    empty!(pts); empty!(cs)
    for j in length(pl):-1:1
        push!(pts, pl[j]); push!(cs, cl[j])
    end
    push!(pts, 0.0); push!(cs, 0.0)
    append!(pts, pr); append!(cs, cr)
    if !issorted(pts)                      # not expected; keeps the old behaviour (stable sort) if it happens
        p = sortperm(pts)
        pts .= pts[p]; cs .= cs[p]
    end
    quad[1] > 0 && return sample_segments_gauss(rng, pts, cs, quad[1], quad[2])
    nseg = length(pts) - 1
    lm = resize!(lb.lm, nseg)
    @inbounds for j in 1:nseg
        L = pts[j+1] - pts[j]
        lm[j] = L > 0 ? -min(cs[j], cs[j+1]) + logseg(L, abs(cs[j+1] - cs[j])) : -Inf   # anchored at the lower end
    end
    mx = maximum(lm)
    u = rand(rng) * sum(x -> exp(x - mx), lm)
    acc = 0.0
    j = nseg
    for q in 1:nseg
        acc += exp(lm[q] - mx)
        if u <= acc
            j = q
            break
        end
    end
    L = pts[j+1] - pts[j]
    g = (cs[j+1] - cs[j]) / L
    # a decreasing segment is sampled from its right end (the same u → t map as the mirrored offset,
    # without the cancellation a + (L − s′) on long segments)
    return (g >= 0 || abs(g * L) < 1e-10) ? pts[j] + trexp_offset(rng, g, L) : pts[j+1] - trexp_offset(rng, -g, L)
end

# ---------------------------------------------------------------------------
# move construction
# ---------------------------------------------------------------------------

"Basis vector j of field f at step τ."
function prep_atom!(ws, P, LM, f, j, τ)
    E = P.E
    lv = P.lv[LM.ℓ+1]
    U = E.vspecs[f].B.U
    rv, nz = rowvals(U), nonzeros(U)
    for q in nzrange(U, j)
        add_entry!(ws, f, rv[q], nz[q])
    end
    addz!(ws, lv.off[f] + (τ - 1) * lv.m[f] + j, 1.0)
end

"Node-space change a·e_i of field f (complete field)."
function prep_node!(ws, P, LM, f, i, a, τ)
    add_entry!(ws, f, i, a)
    addz_node!(ws, P.lv[LM.ℓ+1], P.E, f, τ, i, a)
end

"Injection per unit utilization of unit (f, i) at τ (generator Ak; storage k)."
function κunit(P::Problem, LM::LocalModel, LC::LocalCost, f::Int, i::Int, τ::Int)
    rl, g = LM.role[f]
    rl == ROLE_GEN && return LC.A[g][i, τ] * LC.kgen[g][i]
    return LC.kst[g][i]
end

"Flow re-routing for an injection change +1 at n, −1 at m: Δθ = L⁺(e_n − e_m)."
function prep_reroute!(ws, P, LM, n, m, τ)
    E = P.E
    lv = P.lv[LM.ℓ+1]
    fθ = E.fθ
    x = ws.nbuf
    reroute!(x, LM.GS, n, m, ws.rb, ws.ry)
    for i in LM.GS.nodes[LM.comp[n]]
        abs(x[i]) < 1e-15 && continue
        prep_node!(ws, P, LM, fθ, i, x[i], τ)
    end
end

"""
Prepare one random move of family `fam` at step τ (selection independent of v).
Returns false if the draw does not give a valid move.
"""
function prep_random!(rng, ws, P, LM, LC, fam::Symbol, τ; κmin=1.0)
    E = P.E
    N = E.sys.N
    if fam == :unit
        n = rand(rng, 1:N)
        isempty(LM.units[n]) && return false
        f, i = rand(rng, LM.units[n])
        LM.complete[f] || return false
        prep_node!(ws, P, LM, f, i, 1.0, τ)
    elseif fam == :pair || fam == :xpair
        n = rand(rng, 1:N)
        m = fam == :pair ? n : rand(rng, 1:N)
        (fam == :xpair && (m == n || LM.comp[m] != LM.comp[n])) && return false
        (isempty(LM.units[n]) || isempty(LM.units[m])) && return false
        u1, u2 = rand(rng, LM.units[n]), rand(rng, LM.units[m])
        u1 == u2 && return false
        (LM.complete[u1[1]] && LM.complete[u2[1]]) || return false
        (fam == :xpair && !LM.complete[E.fθ]) && return false
        κ1, κ2 = κunit(P, LM, LC, u1..., τ), κunit(P, LM, LC, u2..., τ)
        (abs(κ1) < κmin || abs(κ2) < κmin) && return false
        prep_node!(ws, P, LM, u1[1], u1[2], 1 / κ1, τ)
        prep_node!(ws, P, LM, u2[1], u2[2], -1 / κ2, τ)
        fam == :xpair && prep_reroute!(ws, P, LM, n, m, τ)
    elseif fam == :xloc
        return prep_xloc!(rng, ws, P, LM, LC, τ, nothing; κmin=κmin)
    elseif fam == :potential
        LM.complete[E.fθ] || return false
        n = rand(rng, 1:N)
        nnz_row = length(nzrange(E.vspecs[E.fθ].B.Ut, n))
        nnz_row == 0 && return false
        prep_node!(ws, P, LM, E.fθ, n, 1.0, τ)
    elseif fam == :cellpair || fam == :cell
        q = draw_level(rng, LM)
        cu, cf = LM.cellunits[q], LM.cellfields[q]
        c = rand(rng, eachindex(cu))
        fl = cf[c]
        isempty(fl) && return false
        if fam == :cell
            f = rand(rng, fl)
            cell_entries!(ws, P, LM, cu[c], f, 1.0, τ)
        else
            length(fl) < 2 && return false
            f1 = rand(rng, fl); f2 = rand(rng, fl)
            f1 == f2 && return false
            K1 = sum(κunit(P, LM, LC, f, i, τ) for (f, i) in cell_units(cu[c], f1))
            K2 = sum(κunit(P, LM, LC, f, i, τ) for (f, i) in cell_units(cu[c], f2))
            (abs(K1) < κmin || abs(K2) < κmin) && return false
            cell_entries!(ws, P, LM, cu[c], f1, 1 / K1, τ)
            cell_entries!(ws, P, LM, cu[c], f2, -1 / K2, τ)
        end
    else
        error("unknown move family $fam")
    end
    return true
end

"""
Localized cross-node pair (`xloc`, `xlocc`): node n uniform, partner m uniform among
the precomputed partners of n, one random unit at each; the flow is re-routed on
the ball around (n, m) (`xloc_directions`; fixed template, independent of v).
`xloc` (jac === nothing): the leak into the first layer outside the ball goes to the
residuals there. `xlocc`: the leak is compensated pro rata (exact T2, `prorata!`;
Jacobian terms in `jac`).
"""
function prep_xloc!(rng, ws, P, LM, LC, τ, jac; κmin=1.0)
    E = P.E
    LM.complete[E.fθ] || return false
    n = rand(rng, 1:E.sys.N)
    pn = LM.xpart[n]
    isempty(pn) && return false
    j = rand(rng, eachindex(pn))
    m = pn[j]
    (isempty(LM.units[n]) || isempty(LM.units[m])) && return false
    u1, u2 = rand(rng, LM.units[n]), rand(rng, LM.units[m])
    (LM.complete[u1[1]] && LM.complete[u2[1]]) || return false
    κ1, κ2 = κunit(P, LM, LC, u1..., τ), κunit(P, LM, LC, u2..., τ)
    (abs(κ1) < κmin || abs(κ2) < κmin) && return false
    jac === nothing || empty!(jac)
    prep_node!(ws, P, LM, u1[1], u1[2], 1 / κ1, τ)
    prep_node!(ws, P, LM, u2[1], u2[2], -1 / κ2, τ)
    xn, xv = LM.xnod[n][j], LM.xval[n][j]
    @inbounds for q in eachindex(xn)
        prep_node!(ws, P, LM, E.fθ, xn[q], xv[q], τ)
    end
    if jac !== nothing
        ln, lg = LM.xlkn[n][j], LM.xlkg[n][j]
        @inbounds for q in eachindex(ln)
            prorata!(ws, jac, P, LM, LC, ln[q], lg[q], τ; κmin=κmin)
        end
    end
    return true
end

"""
SoC form: storage–generator pair across the steps τm = τ−1 and τ at one node.
The state of charge at instant τ rises by t; the extra charging at τm is
compensated by unit u1 at τm and the reduced net withdrawal at τ by unit u2 at τ
(injection scale κ_s = k h/Δt, without efficiencies, so the direction does not
depend on v). `wa` holds step τm, `ws` step τ.
"""
function prep_spair!(rng, wa, ws, P, LM, LC, τm, τ; κmin=1.0)
    E = P.E
    n = rand(rng, 1:E.sys.N)
    st = LM.nodestor[n]
    (isempty(st) || isempty(LM.units[n])) && return false
    si, i = rand(rng, st)
    s = E.stors[si]
    LM.complete[s.fe] || return false
    κs = LC.kst[si][i] * s.hours[i] / LM.Δt
    κs < κmin && return false
    u1, u2 = rand(rng, LM.units[n]), rand(rng, LM.units[n])
    (LM.complete[u1[1]] && LM.complete[u2[1]]) || return false
    κ1, κ2 = κunit(P, LM, LC, u1..., τm), κunit(P, LM, LC, u2..., τ)
    (abs(κ1) < κmin || abs(κ2) < κmin) && return false
    prep_node!(wa, P, LM, u1[1], u1[2], κs / κ1, τm)
    prep_node!(ws, P, LM, s.fe, i, 1.0, τ)
    prep_node!(ws, P, LM, u2[1], u2[2], -κs / κ2, τ)
    return true
end

"""
Transport move with pro-rata re-dispatch (the transport-first formulation T2 of
docs/sampling_formulations.md §8, as a move in the current coordinates). The
transport direction (here: the angle of node n) changes the requirement Q_m at
n and its neighbours by g_m per unit t. At every affected node the dispatchable
generators S_m (all generator units with a utilization field) follow pro rata:
Δp_i = σ_i g_m t with σ_i = p_i/P_m, P_m = Σ_{S_m} p_i (unclamped p = κω), so the
nodal residual is unchanged inside the boxes. σ is invariant along the line, so
the move is reversible; the conditional along t carries the Jacobian
Π_m (P_m(t)/P_m(0))^{|S_m|−1}, applied by an independence Metropolis step on top
of the exact piecewise-exponential draw. Returns the Jacobian terms (P_m, g_m, |S_m|−1)
in `jac`, or false.
"""
function prep_tpot!(rng, ws, P, LM, LC, τ, jac; κmin=1.0, n=0, comp::Bool=true, learn=nothing)
    E = P.E
    sys = E.sys
    LM.complete[E.fθ] || return false
    n = n == 0 ? rand(rng, 1:sys.N) : n
    length(nzrange(E.vspecs[E.fθ].B.Ut, n)) == 0 && return false
    empty!(jac)
    prep_node!(ws, P, LM, E.fθ, n, 1.0, τ)
    comp || return true
    # requirement change per unit t (r[a] += f_e, r[b] −= f_e for a line a→b)
    g, gn = ws.gbuf, ws.gnodes
    empty!(gn)
    for e in LM.nodelines[n]
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        df = sys.line_b[e] * ((a == n) - (b == n))
        g[a] += df; g[b] -= df
        push!(gn, a); push!(gn, b)
    end
    sort!(gn); unique!(gn)                   # ascending node order, as the dense loop over 1:N
    for m in gn
        g[m] != 0.0 && prorata!(ws, jac, P, LM, LC, m, g[m], τ; κmin=κmin, learn=learn)
    end
    for m in gn
        g[m] = 0.0
    end
    empty!(gn)
    return true
end

"""
Exact T2 compensation at node m and step τ for a requirement change g per unit t.
Generators S_m with p_i = κ_i ω_i (unclamped), P = Σ p_i, requirement
Q = r_m + Σ κ_i clamp(ω_i) (it changes exactly by g t along the move). Coordinates
held fixed: σ_i = p_i/P and the excess e = P − Q₊ (any sign; −e is the shed while
Q > 0). Along the line P(t) = (Q + g t)₊ + e: dispatch follows the requirement
pro rata while it is positive, and once the node is in surplus (Q ≤ 0) dispatch
stays at e and the transport change goes into the residual (dump). The entries
are kinked at t* = −Q/g. Jacobian term (P, e, Q, g, |S| − 1): |P(t)/P|^{|S|−1}.
"""
function prorata!(ws, jac, P, LM, LC, m, g, τ; κmin=1.0, learn=nothing, dyn=nothing, skipweather::Bool=false)
    g == 0.0 && return false
    learn === nothing || return compensate_learned!(ws, P, LM, LC, learn.AM, learn.FF, m, g, τ; κmin=κmin)
    E = P.E
    units = LM.genunits[m]
    Pm, Qs, nu = 0.0, 0.0, 0                   # P = Σ κ ω, Σ κ clamp(ω) over the units with |κ| ≥ κmin
    @inbounds for (f, i) in units
        (skipweather && E.gens[LM.role[f][2]].wf > 0) && continue
        κ = κunit(P, LM, LC, f, i, τ)
        abs(κ) >= κmin || continue
        ω = LC.Y[f][i, τ]
        Pm += κ * ω
        Qs += κ * clamp(ω, 0.0, 1.0)
        nu += 1
    end
    nu == 0 && return false
    abs(Pm) < 1e-6 && return false
    Q = LC.r[m, τ] + Qs
    e = Pm - max(Q, 0.0)
    dyn = dyn === nothing ? tracking_kinks(E) : dyn
    nm = Float64(nu - 1)
    k = newkink!(ws, Q, g; node=m, dyn=dyn, P=Pm, nm=nm)
    lv = P.lv[LM.ℓ+1]
    for (f, i) in units
        (skipweather && E.gens[LM.role[f][2]].wf > 0) && continue
        κ = κunit(P, LM, LC, f, i, τ)
        abs(κ) >= κmin || continue
        σ = κ * LC.Y[f][i, τ] / Pm
        σ == 0.0 && continue
        add_kentry!(ws, lv, E, f, i, σ / κ, k, τ)
    end
    push!(jac, (Pm, e, Q, dyn ? NaN : g, nm))
    return true
end

"""
SoC-form transport in time with pro-rata re-dispatch: the state of charge of one
storage unit at instant τ rises by t (energy moved from step τ−1 to step τ); the
generators at its node follow pro rata at τ−1 (requirement +κ_s t) and at τ
(−κ_s t), κ_s = k h/Δt (efficiencies are left to the cost, so the direction does
not depend on v). `wa` holds step τ−1, `ws` step τ.
"""
function prep_tsoc!(rng, wa, ws, P, LM, LC, τm, τ, jac; κmin=1.0, learn=nothing)
    E = P.E
    n = rand(rng, 1:E.sys.N)
    st = LM.nodestor[n]
    isempty(st) && return false
    si, i = rand(rng, st)
    s = E.stors[si]
    LM.complete[s.fe] || return false
    κs = LC.kst[si][i] * s.hours[i] / LM.Δt
    κs < κmin && return false
    empty!(jac)
    prep_node!(ws, P, LM, s.fe, i, 1.0, τ)
    prorata!(wa, jac, P, LM, LC, n, κs, τm; learn=learn)
    prorata!(ws, jac, P, LM, LC, n, -κs, τ; learn=learn)
    return true
end

"log Jacobian of linear pro-rata moves (weather shear): Σ (m−1) log|1 + g t/P|."
logjac_lin(jac, t) = sum((nm * log(abs(1 + gm * t / Pm)) for (Pm, gm, nm) in jac); init=0.0)

"log Jacobian Σ (|S|−1) log|P(t)/P(0)|, P(t) = (Q + g t)₊ + e, of an exactly compensated transport move."
logjac(jac, t) = sum((nm * log(abs((max(Q + g * t, 0.0) + e) / Pm)) for (Pm, e, Q, g, nm) in jac); init=0.0)

"""
The same for a prepared mover `mv`; with requirement-tracking kinks (g = NaN in `jac`)
P(t) = P + φ_k(t) needs the actual requirement at t, so the move is evaluated there.
"""
function logjac(jac, t, mv)
    any(x -> isnan(x[4]), jac) || return logjac(jac, t)
    mv_eval!(mv, t)
    val = 0.0
    for ws in mv_wss(mv), k in eachindex(ws.kQ)
        isnan(ws.kP[k]) && continue
        val += ws.knm[k] * log(abs((ws.kP[k] + ws.kφ[k]) / ws.kP[k]))
    end
    return val
end

"Whether T2 kinks must track the actual requirement (line losses or lossy links make it piecewise linear)."
tracking_kinks(E::EnergyModel) = E.lines.losses || any(<(1.0), E.link_eta)

cell_units(cu, f) = first(sl for sl in cu if !isempty(sl) && sl[1][1] == f)

"Cell-constant change a on the units of field f in a cell (in the span of the level)."
function cell_entries!(ws, P, LM, cu, f, a, τ)
    lv = P.lv[LM.ℓ+1]
    for (_, i) in cell_units(cu, f)
        add_entry!(ws, f, i, a)
        addz_node!(ws, lv, P.E, f, τ, i, a)
    end
end

# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

"""
Plain local families and their rates. The cross-node pairs (total rate 0.15) are
mostly localized: `xlocc` (partner among the graph neighbours, local re-routing,
leak compensated pro rata) and `xloc` (leak into the residuals); the dense global
re-routing `xpair` stays at a low rate (its cost grows like N per move).
"""
function plain_families(E::EnergyModel)
    return [(:unit, 0.3), (:pair, 0.25), (:xpair, 0.03), (:xlocc, 0.08), (:xloc, 0.04), (:potential, 0.1), (:cellpair, 0.1), (:cell, 0.1),
            (:spair, 0.3)]
end

"""
    local_sweep!(rng, P, LM, s; extra, families, wsets) -> (s, stats)

One sweep of the operational block at the level of `s` with (k, x) fixed: for
every time step all basis atoms of all v fields plus `extra` random moves per
node drawn from `families` (pairs of (family, weight)), in uniformly random
order, each sampled exactly along its line. Steps of one parity class are
updated in parallel (a move at τ reads τ±1 only). Returns the updated state
(evaluated) and move counts per family.
"""
function local_sweep!(rng, P::Problem, LM::LocalModel, s::State;
                      extra::Float64=4.0, families=nothing,
                      atoms::Bool=true, wsets=nothing, LC=nothing, check::Bool=false, frames=nothing, adapt=nothing)
    E = P.E
    lv = P.lv[LM.ℓ+1]
    T = lv.T
    fl = families === nothing ? plain_families(E) : families   # (no reassignment: keeps `fams` concretely typed)
    LCc = LC === nothing ? LocalCost(P, LM, s) : LC     # single assignment: not boxed in the threaded closure
    ws0 = wsets === nothing ? [MoveWS(P) for _ in 1:Threads.maxthreadid()] : wsets
    wsl::Vector{MoveWS} = length(ws0) < 2 * Threads.maxthreadid() ? vcat(ws0, [MoveWS(P) for _ in 1:Threads.maxthreadid()]) : ws0
    nth = Threads.maxthreadid()
    β = P.βv
    nextra = round(Int, extra * E.sys.N)
    fams::Vector{Symbol} = first.(fl)
    wts = cumsum(last.(fl)); wts ./= wts[end]
    classes = isodd(T) ? [collect(1:2:T-1), collect(2:2:T-1), [T]] : [collect(1:2:T), collect(2:2:T)]
    nf = length(fams) + 1
    stats = zeros(nf, 2)
    for cls in shuffle(rng, classes)
        seeds = rand(rng, UInt64, length(cls))
        loc = [zeros(nf, 2) for _ in cls]
        Threads.@threads :static for q in eachindex(cls)
            r = Xoshiro(seeds[q])
            ws = wsl[Threads.threadid()]
            wa = wsl[nth+Threads.threadid()]                # the step before τ is linked (state of charge)
            wa.follow = false                                 # (commitment moves only at τ: parallel classes stay independent)
            τ = cls[q]
            τm = mod(τ - 2, T) + 1
            bm = blockmove([wa, ws], ws, P, LM, LCc)
            bm.n = 2; push!(bm.τs, τm, τ)
            jac = ws.jac
            moves = empty!(ws.mvbuf)
            if atoms
                for f in eachindex(E.vspecs), j in 1:lv.m[f]
                    push!(moves, (0, f, j))
                end
            end
            for _ in 1:nextra
                u = rand(r)
                push!(moves, (findfirst(>=(u), wts), 0, 0))
            end
            shuffle!(r, moves)
            for (kind, f, j) in moves
                clear!(ws)
                clear!(wa)
                if kind == 0
                    prep_atom!(ws, P, LM, f, j, τ)
                    fi = nf
                elseif fams[kind] == :spair
                    prep_spair!(r, wa, ws, P, LM, LCc, τm, τ) || continue
                    fi = kind
                elseif fams[kind] == :tpot
                    prep_tpot!(r, ws, P, LM, LCc, τ, jac) || continue
                    fi = kind
                elseif fams[kind] == :xlocc
                    prep_xloc!(r, ws, P, LM, LCc, τ, jac) || continue
                    fi = kind
                elseif fams[kind] == :tsoc
                    prep_tsoc!(r, wa, ws, P, LM, LCc, τm, τ, jac) || continue
                    fi = kind
                elseif fams[kind] in (:amode, :amode0, :amodeL)
                    adapt === nothing && continue
                    lrn = fams[kind] == :amodeL ? (AM=adapt, FF=frames) : nothing
                    prep_amode!(r, ws, P, LM, LCc, frames, adapt, τ, jac; comp=fams[kind] != :amode0, learn=lrn) || continue
                    fi = kind
                elseif fams[kind] in (:tcut, :tcutL, :acut, :acutL)
                    frames === nothing && continue
                    lrn = fams[kind] in (:tcutL, :acutL) ? (adapt === nothing ? nothing : (AM=adapt, FF=frames)) : nothing
                    (fams[kind] in (:tcutL, :acutL) && lrn === nothing) && continue
                    if fams[kind] in (:tcut, :tcutL)
                        prep_tcut!(r, ws, P, LM, LCc, frames.H, frames.pieces, τ, jac; learn=lrn) || continue
                    else
                        (adapt === nothing || isempty(adapt.pieces)) && continue
                        prep_tcut!(r, ws, P, LM, LCc, adapt.H, adapt.pieces, τ, jac; learn=lrn) || continue
                    end
                    fi = kind
                elseif fams[kind] in (:tmodeL, :tlineL, :tpotL)
                    (adapt === nothing || frames === nothing) && continue
                    lrn = (AM=adapt, FF=frames)
                    ok = fams[kind] == :tmodeL ? prep_tmode!(r, ws, P, LM, LCc, frames, τ, jac; learn=lrn) :
                         fams[kind] == :tlineL ? prep_tline!(r, ws, P, LM, LCc, frames, τ, jac; learn=lrn) :
                                                 prep_tpot!(r, ws, P, LM, LCc, τ, jac; learn=lrn)
                    ok || continue
                    fi = kind
                elseif fams[kind] == :tsocL
                    (adapt !== nothing && frames !== nothing) || continue
                    prep_tsoc!(r, wa, ws, P, LM, LCc, τm, τ, jac; learn=(AM=adapt, FF=frames)) || continue
                    fi = kind
                elseif fams[kind] in (:tmode, :tline, :tmode0, :tline0)
                    frames === nothing && continue
                    cmp = fams[kind] in (:tmode, :tline)
                    ok = fams[kind] in (:tmode, :tmode0) ? prep_tmode!(r, ws, P, LM, LCc, frames, τ, jac; comp=cmp) :
                                                           prep_tline!(r, ws, P, LM, LCc, frames, τ, jac; comp=cmp)
                    ok || continue
                    fi = kind
                else
                    prep_random!(r, ws, P, LM, LCc, fams[kind], τ) || continue
                    fi = kind
                end
                finalize!(ws, P, LM, LCc, τ)
                isempty(ws.ent) && continue
                finalize!(wa, P, LM, LCc, τm)
                link_soc!(wa, ws, LM)
                empty!(bm.bps); append!(bm.bps, wa.bps); append!(bm.bps, ws.bps)
                t = sample_line!(r, bm, β, ws.secprev)
                if kind > 0 && fams[kind] in (:tpot, :tsoc, :tmode, :tline, :amode, :tcut, :acut, :xlocc) && t != 0.0 && !isempty(jac) &&
                   rand(r) >= exp(min(0.0, logjac(jac, t, bm)))
                    t = 0.0                                  # Jacobian correction rejected
                end
                commit_block!(bm, s.v, t)
                loc[q][fi, 1] += 1
                loc[q][fi, 2] += t != 0.0
            end
            clear!(ws)
            clear!(wa)
        end
        stats .+= sum(loc)
    end
    if check
        LC2 = LocalCost(P, LM, s)
        errY = maximum(maximum(abs.(LCc.Y[f] .- LC2.Y[f]); init=0.0) for f in eachindex(LCc.Y) if f != E.fθ)
        errF = maximum(abs.(LCc.F .- LC2.F); init=0.0)
        errS = maximum(abs.(LCc.S .- LC2.S); init=0.0)
        @info "local cache drift" errY errF errS
    end
    return evaluate!(P, s), stats
end

"Total level cost from the caches (for checks): Σ_τ cost_τ in €."
function cache_cost(P::Problem, LM::LocalModel, LC::LocalCost)
    E = P.E
    sys = E.sys
    Δt, cv, κ0, T = LM.Δt, E.c_viol, E.κ0, LM.T
    c = 0.0
    for τ in 1:T
        τ1 = mod(τ, T) + 1
        for e in eachindex(sys.line_s)
            c += cv * Δt * line_eval(E.lines, e, sys.line_s[e], LC.F[e, τ])[1]
        end
        if E.flink > 0
            for l in eachindex(sys.link_cap)
                y = LC.Y[E.flink][l, τ]
                c += cv * Δt * sys.link_cap[l] * abs(y - clamp(y, -1.0, 1.0))
            end
        end
        for (gi, g) in enumerate(E.gens), i in eachindex(g.bus)
            k = LC.kgen[gi][i]
            y = g.vf > 0 ? LC.Y[g.vf][i, τ] : 1.0
            c += gen_cost(g, i, LC.A[gi][i, τ], k, k + κ0, Δt, cv, y)[1]
            g.ucf > 0 && (c += uc_cost(g, i, LC.A[gi][i, τ], k, k + κ0, Δt, cv, clamp(y, 0.0, 1.0),
                                       LC.Y[g.ucf][i, τ], LC.Y[g.ucf][i, τ1]))
        end
        for (si, s) in enumerate(E.stors), i in eachindex(s.bus)
            k = LC.kst[si][i]
            c += soc_cost(s, i, k, k + κ0, Δt, cv, LC.Y[s.fe][i, τ], LC.Y[s.fe][i, τ1])[1]
        end
        for cl in eachindex(LM.ncell)
            nc = LM.ncell[cl]
            c += Δt * nc * φbal(E, LC.S[cl, τ] / nc)
        end
    end
    return c
end

# ---------------------------------------------------------------------------
# investment moves in physical coordinates (v2 §5.1: k-moves with the k^{−n} Jacobian)
# ---------------------------------------------------------------------------
#
# For one investable unit (node capacity k of a generator or storage unit at a
# spatially complete level) the scaling group (k, y_1..y_n) ↦ (e^u k, e^{−u} y)
# acts on the raw capacity and all n utilizations of that unit (T for a
# generator, 2T with commitment, T for storage: the state of charge). Inside the
# boxes it keeps every injection and state of charge unchanged, so only capex,
# penalty weights and the clamps change. u is drawn from the generalized Gibbs
# conditional ∝ π(g_u x) e^{(1−n)u} (Liu & Sabatti 2000) by slice sampling,
# truncated to k e^u ≥ κmin so that the selection rule (k ≥ κmin) is invariant.

"Raw node capacities of every k field at the level."
knodes(P::Problem, ℓ::Int, k) = [vec(nodevals(k, P.lk[ℓ+1], P.E.kspecs, f)) for f in eachindex(P.E.kspecs)]

"""
Change of (β_k capex + β_v C) for scaling unit (kind, group, i) by e^u, and the
per-step injection changes written into `dp` (MW, + = more injection).
"""
function kscale_delta!(dp, P::Problem, LM::LocalModel, LC::LocalCost, Kraw, kind::Int, gi::Int, i::Int, u::Float64)
    E = P.E
    Δt, cv, κ0, T = LM.Δt, E.c_viol, E.κ0, LM.T
    if kind == 1
        g = E.gens[gi]; kf = g.kf; Y = g.vf > 0 ? LC.Y[g.vf] : nothing; bus = g.bus[i]
    else
        st = E.stors[gi]; kf = st.kf; bus = st.bus[i]
    end
    sp = E.kspecs[kf]
    k0 = Kraw[kf][i]; k1 = k0 * exp(u)
    kc0, kc1 = clamp(k0, sp.lo[i], sp.hi[i]), clamp(k1, sp.lo[i], sp.hi[i])
    cap = f_capex(E, kf)[i]
    Δcap = cap * (k1 - k0) + E.γk * cap * (abs(k1 - kc1) - abs(k0 - kc0))
    sc = exp(-u)
    Δop = 0.0
    c = LM.cell[bus]; nc = LM.ncell[c]
    for τ in 1:T
        if kind == 1
            y0 = Y === nothing ? 1.0 : Y[i, τ]
            y1 = Y === nothing ? 1.0 : y0 * sc
            A = LC.A[gi][i, τ]
            c0, p0 = gen_cost(g, i, A, kc0, kc0 + κ0, Δt, cv, y0)
            c1, p1 = gen_cost(g, i, A, kc1, kc1 + κ0, Δt, cv, y1)
            if g.ucf > 0                       # the commitment scales with the output (committed MW kept)
                τ1 = mod(τ, T) + 1
                u, u1 = LC.Y[g.ucf][i, τ], LC.Y[g.ucf][i, τ1]
                c0 += uc_cost(g, i, A, kc0, kc0 + κ0, Δt, cv, clamp(y0, 0.0, 1.0), u, u1)
                c1 += uc_cost(g, i, A, kc1, kc1 + κ0, Δt, cv, clamp(y1, 0.0, 1.0), u * sc, u1 * sc)
            end
            d = p1 - p0
        else
            τ1 = mod(τ, T) + 1
            ye, ye1 = LC.Y[st.fe][i, τ], LC.Y[st.fe][i, τ1]
            c0, _, j0 = soc_cost(st, i, kc0, kc0 + κ0, Δt, cv, ye, ye1)
            c1, _, j1 = soc_cost(st, i, kc1, kc1 + κ0, Δt, cv, ye * sc, ye1 * sc)
            d = -(j1 - j0)
        end
        dp[τ] = d
        S0 = LC.S[c, τ]
        Δop += c1 - c0 + Δt * nc * (φbal(E, (S0 - d) / nc) - φbal(E, S0 / nc))
    end
    return P.βk * Δcap + P.βv * Δop
end

"Scalar slice sampler (stepping out, shrinkage) on u ∈ [lo, ∞) for log density lf."
function slice_scalar(rng, lf, u0::Float64, w::Float64, lo::Float64; maxsteps=12)
    ly = lf(u0) - randexp(rng)
    L = u0 - w * rand(rng); R = L + w
    J = floor(Int, maxsteps * rand(rng)); K = maxsteps - 1 - J
    while J > 0 && L > lo && lf(L) > ly
        L -= w; J -= 1
    end
    L = max(L, lo)
    while K > 0 && lf(R) > ly
        R += w; K -= 1
    end
    for _ in 1:100
        u = L + rand(rng) * (R - L)
        lf(u) > ly && return u
        u < u0 ? (L = u) : (R = u)
    end
    return u0
end

"""
    kscale_sweep!(rng, P, LM, s; LC, BC, κmin, w) -> (s, nmoved)

One scaling move per investable unit (random order) at a spatially complete
level, with the level's weather fixed. Needs the k fields and the units' v
fields complete at the level. At α = 1 the bias change comes from the incremental
bias cache `BC` (built from `s` if not given), which follows the committed moves.
"""
function kscale_sweep!(rng, P::Problem, LM::LocalModel, s::State; LC=nothing, BC=nothing, κmin::Float64=1.0, w::Float64=0.3,
                       mh::Bool=false, σ::Float64=0.3, nprop::Int=1, order::Symbol=:shuffle, kinds::Symbol=:all)
    E = P.E
    ℓ = LM.ℓ
    lk, lv = P.lk[ℓ+1], P.lv[ℓ+1]
    T = LM.T
    LC = LC === nothing ? LocalCost(P, LM, s) : LC
    BC = (P.use_bias && BC === nothing) ? BiasCache(P, s) : BC
    Kraw = knodes(P, ℓ, s.k)
    units = NTuple{3,Int}[]
    for (gi, g) in enumerate(E.gens)
        g.kf > 0 && iscomplete(lk, E.kspecs, g.kf) && (g.vf == 0 || LM.complete[g.vf]) && (g.ucf == 0 || LM.complete[g.ucf]) || continue
        (kinds == :dispatch && g.vf == 0) && continue        # unit subsets, for comparisons with the device kernel
        (kinds == :renewables && g.vf != 0) && continue
        append!(units, [(1, gi, i) for i in eachindex(g.bus)])
    end
    for (si, st) in enumerate(E.stors)
        kinds == :renewables && break
        st.kf > 0 && iscomplete(lk, E.kspecs, st.kf) && LM.complete[st.fe] || continue
        append!(units, [(2, si, i) for i in eachindex(st.bus)])
    end
    dp = zeros(T)
    nmoved = 0
    for (kind, gi, i) in (order == :fixed ? units : shuffle(rng, units))
        kf = kind == 1 ? E.gens[gi].kf : E.stors[gi].kf
        k0 = Kraw[kf][i]
        # Metropolis form: the draws of a unit (z₁, u₁, z₂, u₂, …) are taken before any skip, so that a
        # pre-generated stream per (unit, proposal) lines up with the device kernel
        draws = mh ? [(randn(rng), rand(rng)) for _ in 1:nprop] : nothing
        k0 >= κmin || continue
        n = kind == 1 ? (E.gens[gi].vf > 0 ? T : 0) * (E.gens[gi].ucf > 0 ? 2 : 1) : T
        function setk!(u)
            Δ = k0 * (exp(u) - 1)
            Ut = E.kspecs[kf].B.Ut
            for q in nzrange(Ut, i)
                s.k[lk.off[kf]+rowvals(Ut)[q]] += Δ * nonzeros(Ut)[q]
            end
        end
        sp_ = E.kspecs[kf]
        phys = E.measure == :physical && (kind == 2 || E.gens[gi].wf == 0)
        lm0 = phys ? n * log(clamp(k0, sp_.lo[i], sp_.hi[i]) + E.κ0) : 0.0
        function lf(u)
            val = -kscale_delta!(dp, P, LM, LC, Kraw, kind, gi, i, u) + (1 - n) * u
            phys && (val += n * log(clamp(k0 * exp(u), sp_.lo[i], sp_.hi[i]) + E.κ0) - lm0)
            BC === nothing || (val -= bias_delta_k(BC, P.bias, E, lk, kind, gi, i, k0 * exp(u)))
            return val
        end
        if mh
            # Metropolis with fixed-scale proposals u = σ z (the device form, docs/reactant_design.md §3.6):
            # `nprop` proposals; draws from `rng` in the order z, then the uniform of the test
            u = 0.0
            for (z, uu) in draws
                up = u + σ * z
                up < log(κmin / k0) && continue
                log(uu) < lf(up) - lf(u) && (u = up)
            end
        else
            u = slice_scalar(rng, lf, 0.0, w, log(κmin / k0))
        end
        u == 0.0 && continue
        nmoved += 1
        # commit: capacities, utilizations, residuals, cell sums, bias cache
        kscale_delta!(dp, P, LM, LC, Kraw, kind, gi, i, u)
        setk!(u)
        k1 = k0 * exp(u)
        BC === nothing || bias_delta_k(BC, P.bias, E, lk, kind, gi, i, k1; commit=true)
        Kraw[kf][i] = k1
        sp = E.kspecs[kf]
        kc1 = clamp(k1, sp.lo[i], sp.hi[i])
        sc = exp(-u)
        bus = kind == 1 ? E.gens[gi].bus[i] : E.stors[gi].bus[i]
        c = LM.cell[bus]
        if kind == 1
            g = E.gens[gi]
            LC.kgen[gi][i] = kc1
            fields = g.vf > 0 ? [g.vf] : Int[]
            g.ucf > 0 && push!(fields, g.ucf)
        else
            st = E.stors[gi]
            LC.kst[gi][i] = kc1
            fields = [st.fe]
        end
        for f in fields
            m = lv.m[f]
            Ut = E.vspecs[f].B.Ut
            for τ in 1:T
                y0 = LC.Y[f][i, τ]
                Δy = y0 * (sc - 1)
                LC.Y[f][i, τ] = y0 * sc
                o = lv.off[f] + (τ - 1) * m
                for q in nzrange(Ut, i)
                    s.v[o+rowvals(Ut)[q]] += Δy * nonzeros(Ut)[q]
                end
            end
        end
        for τ in 1:T
            LC.r[bus, τ] -= dp[τ]
            LC.S[c, τ] -= dp[τ]
        end
    end
    return evaluate!(P, s), nmoved
end

# ---------------------------------------------------------------------------
# the kernels of a chain: fine-level step and level-0 step
# ---------------------------------------------------------------------------

"Per-level local models (built lazily) and per-thread workspaces."
struct LocalKernel
    models::Dict{Int,LocalModel}
    wsets::Vector{MoveWS}
    extra::Float64
    lk::ReentrantLock
    frames::Dict{Int,Any}            # flow design: FlowFrames per level
    pools::Vector{Any}
    mode::Symbol                     # fine-level kernel: :local, :flow or :adaptive (:auto → flow if a flow basis exists)
    adapt::Dict{Int,Any}             # AdaptiveModes per level
    weather::Float64                 # fine-level weather moves: fraction of the weather coordinates per step (0: off)
    wpools::Vector{Any}
    lmkw::NamedTuple                 # LocalModel options (cell_power, xr_partner, xr_ball)
    kop::Bool                        # capacity moves with their operation (kop.jl) after kscale and kpeak
end
LocalKernel(P::Problem; extra=4.0, mode::Symbol=:auto, weather::Float64=0.0, lmkw::NamedTuple=NamedTuple(), kop::Bool=false) =
    LocalKernel(Dict{Int,LocalModel}(), [MoveWS(P) for _ in 1:Threads.maxthreadid()], extra, ReentrantLock(), Dict{Int,Any}(), Any[],
                mode == :auto ? (P.E.flow === nothing ? :local : :flow) : mode, Dict{Int,Any}(), weather, Any[], lmkw, kop)

"Fine-level weather moves of the local kernel (if enabled and the level is spatially complete)."
function maybe_weather!(rng, P::Problem, K::LocalKernel, LM::LocalModel, s::State; BC=nothing)
    (K.weather > 0 && s.ℓ >= 1 && LM.complete[P.E.fθ]) || return s
    s, _ = fine_weather_sweep!(rng, P, LM, s; frac=K.weather, frac0=K.weather, pools=K.wpools, BC=BC)
    return s
end

local_model(K::LocalKernel, P::Problem, ℓ::Int) = lock(() -> get!(() -> LocalModel(P, ℓ; K.lmkw...), K.models, ℓ), K.lk)

"""
One π_ℓ-invariant transition of the operation and (where complete) the
capacities at the level of `s`: the operation sweep (local sweep, then with
probability 1/2 the storage multiscale sweep; or the flow/adaptive sweep) and the
capacity moves (`kscale`, `kpeak`, and `kop` if enabled) in random order, then the fine
weather moves if enabled. At α = 1 one
bias cache serves the capacity and weather moves (the operation sweep changes neither
k nor x).
"""
function fine_local_step(rng, P::Problem, s::State, S)
    K = S.loc
    LM = local_model(K, P, s.ℓ)
    s = copy(s)
    BC = BiasCache(P, s)
    kopmoves! = x -> K.kop ? kop_sweep!(rng, P, LM, x; BC=BC)[1] : x      # capacity with its operation (kop.jl)
    if K.mode in (:flow, :adaptive) && P.E.flow !== nothing && LM.complete[P.E.fθ]   # flow design (docs/flow_hierarchy.md)
        FF = lock(() -> get!(() -> FlowFrames(P, s.ℓ, s), K.frames, s.ℓ), K.lk)
        isempty(K.pools) && append!(K.pools, [BlockPool(P, 140) for _ in 1:Threads.maxthreadid()])
        sweep! = if K.mode == :adaptive
            AM = lock(() -> get!(() -> AdaptiveModes(P, s.ℓ), K.adapt, s.ℓ), K.lk)
            x -> adaptive_sweep!(rng, P, LM, FF, AM, x; wsets=K.wsets, pools=K.pools, extra=K.extra)[1]
        else
            x -> flow_sweep!(rng, P, LM, FF, x; wsets=K.wsets, pools=K.pools, extra=K.extra)[1]
        end
        AMk = K.mode == :adaptive ? get(K.adapt, s.ℓ, nothing) : nothing
        kmoves! = x -> kopmoves!(kpeak_sweep!(rng, P, LM, kscale_sweep!(rng, P, LM, x; BC=BC)[1]; AM=AMk, BC=BC)[1])
        if rand(rng) < 0.5
            s = sweep!(s)
            s = kmoves!(s)
        else
            s = kmoves!(s)
            s = sweep!(s)
        end
        return maybe_weather!(rng, P, K, LM, s; BC=BC)
    end
    kmoves! = x -> kopmoves!(kpeak_sweep!(rng, P, LM, kscale_sweep!(rng, P, LM, x; BC=BC)[1]; BC=BC)[1])
    # the plain sweep, then (SoC form, complete level) with probability 1/2 the storage multiscale
    # sweep as in the flow kernel: without it the seasonal state of charge does not relax
    # (docs/mixing_diagnosis.md §6.2); it needs no flow basis
    function op!(x)
        x, _ = local_sweep!(rng, P, LM, x; wsets=K.wsets, extra=K.extra)
        if rand(rng) < 0.5 && LM.complete[P.E.fθ]
            isempty(K.pools) && append!(K.pools, [BlockPool(P, 140) for _ in 1:Threads.maxthreadid()])
            x, _ = storage_ms_sweep!(rng, P, LM, x; pools=K.pools)
        end
        return x
    end
    if rand(rng) < 0.5
        s = op!(s)
        s = kmoves!(s)
    else
        s = kmoves!(s)
        s = op!(s)
    end
    return maybe_weather!(rng, P, K, LM, s; BC=BC)
end

"""
Level-0 transition of the bias warmup chain: the moves {local sweep of v, weather
colour classes (coordinate slice on z0 at fixed v), weather with pro-rata shear
(`weather_shear_sweep!`), k slice block, FFBS, carrier scaling} in uniformly random
order (a mixture over orderings of π₀-invariant updates).
"""
function local_base_step(rng, P::Problem, s::State, S)
    K = S.loc
    G = S.gibbs
    LM = local_model(K, P, 0)
    s = copy(s)
    moves = shuffle(rng, vcat([0], collect(1:NCOLOURS), [-1, -2, -3, -4]))
    for mv in moves
        if mv == -4
            s.xs = [s.z0]
            evaluate!(P, s)
            s, _, _ = weather_shear_sweep!(rng, P, G, s)
        elseif mv == 0
            s.xs = [s.z0]
            evaluate!(P, s)
            s, _ = local_sweep!(rng, P, LM, s; wsets=K.wsets, extra=K.extra)
        elseif mv > 0
            ctx = SweepCtx(P, s.k)
            update_colour!(rng, P, G, ctx, s.v, s.z0, s.m, mv; vmoves=false)
        elseif mv == -1
            update_k!(rng, P, G, s)
        elseif mv == -2
            s.m = ffbs(rng, regime_loglik(P.W, s.z0), P.W.Ptr, P.W.π0)
        else
            s.xs = [s.z0]
            evaluate!(P, s)
            s = scale_step(rng, P, s, S)
        end
    end
    s.xs = [s.z0]
    G.sweeps += 1
    if G.adapt && G.sweeps % 5 == 0
        nv = length(P.E.vspecs)
        for f in eachindex(G.nmove)
            f <= nv && continue
            G.nmove[f] > 0 || continue
            G.wz[f-nv] = max(3 * G.smove[f] / G.nmove[f], 1e-8)
        end
        fill!(G.nmove, 0.0); fill!(G.smove, 0.0)
    end
    return evaluate!(P, s)
end

# ---------------------------------------------------------------------------
# block moves: workspaces and commit (storage multiscale, learned shapes, weather)
# ---------------------------------------------------------------------------

"Per-thread pool of step workspaces for block moves."
struct BlockPool
    wss::Vector{MoveWS}
end
BlockPool(P::Problem, L::Int) = BlockPool([MoveWS(P) for _ in 1:L])

"Grow every pool to at least n step workspaces (before a parallel phase)."
function ensure_pools!(pools, P::Problem, n::Int)
    for pool in pools
        while length(pool.wss) < n
            push!(pool.wss, MoveWS(P))
        end
    end
    return pools
end

function commit_block!(bm::BlockMove, v::AbstractVector, t::Float64)
    t == 0.0 && return
    mv_eval!(bm, t; mode=2)              # fills dr, dS of every step at t (from the walk's terms if recorded)
    for q in 1:bm.n
        bm.wss[q].trec = false           # the caches change below
    end
    LC = bm.LC
    for q in 1:bm.n
        ws = bm.wss[q]; τ = bm.τs[q]
        @inbounds for j in eachindex(ws.zidx)
            v[ws.zidx[j]] += t * ws.zval[j]
        end
        @inbounds for j in eachindex(ws.zkidx)
            v[ws.zkidx[j]] += ws.zkval[j] * ws.kφ[ws.zkid[j]]
        end
        @inbounds for (f, i) in ws.ent
            LC.Y[f][i, τ] += entryΔ(ws, f, i, t)
        end
        @inbounds for e in ws.lines
            LC.F[e, τ] += t * ws.dfl[e]
        end
        @inbounds for n in ws.nodes
            LC.r[n, τ] += ws.dr[n]
        end
        @inbounds for c in ws.cells
            LC.S[c, τ] += ws.dS[c]
        end
        commit_weather!(ws, bm.P, LC, τ, t)
    end
end

