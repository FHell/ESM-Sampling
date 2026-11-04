# Phase 3, part 3: storage multiscale block moves (CPU `prep_storage_ms!`, `storage_ms_sweep!`:
# splateau and sshape, SoC form), one storage unit per site.
#
# A site moves the state of charge of one unit at the instants b0+j, j = 0 … L, by w_j·t with a
# profile w (w_0 = 0): the plateau w = (0, 1, …, 1) moves energy from step b0 to step b0+L; a shape
# φ (length L+1, zero ends) moves it inside the block. The power at step b0+j changes by
# κ_s (w_{j+1} − w_j) (w_{L+1} = 0); every such step is compensated at the unit's node, pro rata
# with a kink and the Jacobian step (T2, models without requirement tracking) or with the learned
# coefficients (LINEAR). As on the CPU's block pool, the commitment of the compensating units
# follows at every step of the block.
#
# Terms of a site: L+1 SoC terms (steps b0 … b0+L), L+1 balance slots, (L+1)·G generator terms and
# (L+2)·G commitment terms (steps b0−1 … b0+L, each with the arguments of both of its steps).
# Rounds: blocks of one parity (start offset o, stride L, every other block) × one storage slot per
# node; families are bucketed by L (`:sblk2`, `:sblk4`, … and `:sblkL2`, … for LINEAR), each with
# its own profile table (plateau and shapes as rows).

const BLOCK_LENGTHS = (2, 4, 8, 16, 32)
const PROFILES = Dict{Int,Matrix{Float64}}()       # L → (L+1, P) profile table; column 1 the plateau
"L → season class of every profile column (0: any season; 1–4: DJF, MAM, JJA, SON of the block start)."
const PROFILE_SEASON = Dict{Int,Vector{Int}}()

"""
Profiles of length L: the plateau, then the shape columns `shapes` ((L+1) × S, zero ends), with the
season class of each shape (`seasons`, 0: any). A site draws its profile among the columns of its
block's season and those of any season (`block_rounds`).
"""
function set_profiles!(L::Int, shapes=zeros(L + 1, 0); seasons=zeros(Int, size(shapes, 2)))
    plat = vcat(0.0, ones(L))
    PROFILES[L] = hcat(plat, shapes)
    PROFILE_SEASON[L] = vcat(0, seasons)
    return PROFILES[L]
end

"""
Learned storage shapes (CPU `AdaptiveModes.shapes[(:soc, L, season)]`, `adaptive_block_sweep!`) as
profile columns of block length L: per season class `nper` columns (the learned shapes, or the
plateau while none is learned), after the plateau and the fixed shapes `fixed`. The table keeps its
shape when the shapes are relearned, so the device tables rebuilt after relearning (`device_table`) upload without recompilation.
"""
function set_learned_profiles!(L::Int, AM=nothing; fixed=zeros(L + 1, 0), nper::Int=2)
    cols, seas = Any[fixed], Int[zeros(Int, size(fixed, 2))...]
    plat = vcat(0.0, ones(L))
    for cl in 1:4
        Φ = AM === nothing ? nothing : get(AM.shapes, (:soc, L, cl), nothing)
        for j in 1:nper
            push!(cols, Φ !== nothing && j <= size(Φ, 2) ? Φ[:, j] : plat)
            push!(seas, cl)
        end
    end
    return set_profiles!(L, hcat(cols...); seasons=seas)
end
profiles(L) = get(PROFILES, L) do
    set_profiles!(L)
end

blksym(L, lin) = Symbol(lin ? :sblkL : :sblk, L)

"Rounds of block length L: offset o, stride L, blocks of one parity × storage slot j; a profile per site."
function block_rounds(L::Int, tp::Topo, rng; offset=nothing, h=nothing)
    T = tp.T
    o = offset === nothing ? rand(rng, 0:L-1) : offset
    starts = collect(1+o:L:T-L-1)                            # steps b0 … b0+L and the commitment term at b0−1
    starts = filter(b -> b >= 2, starts)
    Sm = maximum(length.(tp.nodestor); init=0)
    np = size(profiles(L), 2)
    seas = get(PROFILE_SEASON, L, zeros(Int, np))
    compat(b0) = h === nothing ? collect(1:np) : findall(s -> s == 0 || s == ML.season(b0, h), seas)
    rounds = Vector{Vector{Tuple}}()
    for j in 1:Sm, par in (1, 2)
        sites = Tuple[]
        for (bi, b0) in enumerate(starts)
            isodd(bi) == (par == 1) || continue
            for n in 1:tp.N
                length(tp.nodestor[n]) >= j || continue
                push!(sites, (tp.nodestor[n][j], b0, rand(rng, compat(b0))))
            end
        end
        push!(rounds, sites)
    end
    return rounds
end

"""
Columns of a block site of the units `s` (one unit, or a cell–class group of units at distinct nodes),
padded to U units: SoC instants, residual of every unit's node, generators per step (b0−1 … b0+L+1).
"""
function block_cols(tp::Topo, site, L::Int, U::Int=1)
    T = tp.T
    s, b0, p = site
    us = s isa Integer ? [s] : collect(s)
    G = gmax(tp)
    Z(n) = zeros(Int, n)
    ce, cr, gq, gcol, cew, cer, gw, grd = Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[]
    for u in 1:U
        if u > length(us)
            append!(ce, Z(L + 2)); append!(cr, Z(L + 1)); append!(gq, Z(G)); append!(gcol, Z((L + 3) * G))
            append!(cew, Z(L)); append!(cer, Z(2)); append!(gw, Z((L + 1) * G)); append!(grd, Z(2G))
            continue
        end
        su = us[u]
        n = tp.snode[su]
        c = [colτ(T, b0 + j, su) for j in 0:L+1]              # instants b0 … b0+L+1
        gc(j) = padv([colτ(T, b0 + j, q) for q in tp.nodegens[n]], G)
        append!(ce, c); append!(cr, [colτ(T, b0 + j, n) for j in 0:L]); append!(gq, padv(tp.nodegens[n], G))
        append!(gcol, reduce(vcat, [gc(j) for j in -1:L+1]))
        # footprint fields: written (instants b0+1 … b0+L, generator steps b0 … b0+L), read only (the rest)
        append!(cew, c[2:L+1]); append!(cer, [c[1], c[L+2]]); append!(gw, reduce(vcat, [gc(j) for j in 0:L]))
        append!(grd, vcat(gc(-1), gc(L + 1)))
    end
    return (s=padv(us, U), uon=Float64[u <= length(us) for u in 1:U], p=p, ce=ce, cr=cr, gq=gq, gcol=gcol, cc=cr,
            cew=cew, cer=cer, gw=gw, grd=grd)
end

"""
Cell–class groups of the storage units for the aggregate block moves (`:sblkg{L}`): per cell the
groups (device storage units of one class at the cell's nodes, at distinct nodes); set by the
driver or a test (`block_groups`). Round j takes the j-th group of every cell.
"""
const BLOCK_GROUPS = Ref{Any}(nothing)
block_umax() = maximum((length(g) for c in BLOCK_GROUPS[] for g in c); init=1)

"Cell–class groups of a problem (the CPU's `storage_ms_sweep!` groups: cell of the flow basis or partition level 2)."
function block_groups(P::ML.Problem, ix::DeviceIndex)
    E = P.E
    cellof = E.flow === nothing ? E.parts[2] : E.flow.cell
    groups = Dict{Tuple{Int,Int},Vector{Int}}()
    for (q, (si, i)) in enumerate(ix.sidx)
        push!(get!(groups, (cellof[E.stors[si].bus[i]], si), Int[]), q)
    end
    cells = sort(unique(first.(keys(groups))))
    return [[groups[k] for k in sort([k for k in keys(groups) if k[1] == c])] for c in cells]
end

function block_group_rounds(L::Int, tp::Topo, rng; h=nothing)
    T = tp.T
    o = rand(rng, 0:L-1)
    starts = filter(b -> b >= 2, collect(1+o:L:T-L-1))
    np = size(profiles(L), 2)
    seas = get(PROFILE_SEASON, L, zeros(Int, np))
    compat(b0) = h === nothing ? collect(1:np) : findall(s -> s == 0 || s == ML.season(b0, h), seas)
    gr = BLOCK_GROUPS[]
    rounds = Vector{Vector{Tuple}}()
    for j in 1:maximum(length.(gr); init=0), par in (1, 2)
        sites = Tuple[]
        for (bi, b0) in enumerate(starts)
            isodd(bi) == (par == 1) || continue
            for c in gr
                length(c) >= j && push!(sites, (c[j], b0, rand(rng, compat(b0))))
            end
        end
        push!(rounds, sites)
    end
    return rounds
end

blkgsym(L, lin) = Symbol(lin ? :sblkgL : :sblkg, L)
for L in BLOCK_LENGTHS, lin in (false, true), grp in (false, true)
    nm = grp ? blkgsym(L, lin) : blksym(L, lin)
    rounds_ = grp ? :(block_group_rounds($L, tp, rng; h=round(Int, M.Δt))) : :(block_rounds($L, tp, rng; h=round(Int, M.Δt)))
    cols_ = grp ? :(block_cols(tp, site, $L, block_umax())) : :(block_cols(tp, site, $L))
    @eval begin
        family_rounds(::Val{$(QuoteNode(nm))}, tp::Topo, M::DeviceModel, cls, rng) =
            ($lin || !tracking(M) || error("block T2 needs a model without requirement tracking (use :sblkL…)");
             ($grp && BLOCK_GROUPS[] === nothing) ? Vector{Vector{Tuple}}() : $rounds_)
        site_cols(::Val{$(QuoteNode(nm))}, tp::Topo, site, D::Int) = $cols_
        device_extra(::Val{$(QuoteNode(nm))}, ht) = (gon=Float64.(ht.gq .> 0), W=copy(profiles($L)))
        build(::Val{$(QuoteNode(nm))}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) =
            build_block(st, cp, sp, tab, r, k, Val($L), Val($lin), $(QuoteNode(nm)))
        commit(::Val{$(QuoteNode(nm))}, st::FlatState, tab, r, t, dr, aux) = error("block commits use commit_tb")
        commit_tb(::Val{$(QuoteNode(nm))}, st, tab, r, t, dr, aux, tb) = commit_block(st, tab, r, t, dr, aux, tb)
    end
    FOOT[nm] = [(:cew, :e, true), (:cer, :e, false), (:cr, :r, true), (:gw, :ω, true), (:gw, :uc, true),
                (:grd, :ω, false), (:grd, :uc, false)]
end

"Profile weights w_1 … w_L of the sites' profile ids (1, B, 1, L) (w_0 = w_{L+1} = 0 are static)."
function block_weights4(tab, r, L)
    pid = tab.p[:, r]
    W = tab.W[2:L+1, :]                                       # (L, P) static table (a device array)
    return M4(permutedims(W[:, pid]))
end

"""
Term batch of a block site of U storage units (one unit, or the units of one storage class in one
cell, `:sblkg…`), each at its own node, all with the same profile (equal utilization change, as the
CPU's `prep_storage_ms!` for a cell–class group). Columns unit-major: per unit its L+1 SoC terms
(steps b0 … b0+L, balance slots (u, j)), per unit and step its node's G compensation columns.
"""
function build_block(st, cp, sp, tab, r, k, ::Val{L}, ::Val{lin}, fam::Symbol=blksym(L, lin)) where {L,lin}
    on = m3(tab.on[:, r])
    s = tab.s[:, :, r]                                        # (B, U)
    U = size(s, 2)
    uon = M4(tab.uon[:, :, r])                                # (1, B, 1, U): real units of the site
    κs = G4(cp.ks, s) .* S4(sp.sh, s) ./ k.Δt                 # (C, B, 1, U)
    z = 0.0 .* g3(cp.ks, s[:, 1])
    z4 = c4(z)
    w = block_weights4(tab, r, L) .* c4(ifelse.(on .> 0.0, 1.0, 0.0))     # (C|1, B, 1, L)
    zw = 0.0 .* w[:, :, :, 1:1]
    wa1 = cat(zw, w; dims=4)                                  # w_0 … w_L
    wb1 = cat(w, zw; dims=4)                                  # w_1 … w_{L+1}
    rep(x) = cat(fill(x, U)...; dims=4)                       # the profile for every unit
    perstep(x) = x[:, :, :, repeat(1:U; inner=L + 1)]         # a per-unit field at the U(L+1) SoC columns
    um = perstep(uon)
    wa, wb = rep(wa1) .* um, rep(wb1) .* um
    G = size(tab.gq, 2) ÷ U
    ce = tab.ce[:, :, r]                                      # (B, U(L+2)): instants b0 … b0+L+1 per unit
    i0 = [(u - 1) * (L + 2) + j for u in 1:U for j in 1:L+1]
    i1 = i0 .+ 1
    soc = ssoc(z4, U * (L + 1); e0=G4(st.e, ce[:, i0]), ae=wa, e10=G4(st.e, ce[:, i1]), ae1=wb,
               k=perstep(G4(cp.ks, s)), w=perstep(G4(cp.ws, s)), h=perstep(S4(sp.sh, s)), ηin=perstep(S4(sp.sηin, s)),
               ηout=perstep(S4(sp.sηout, s)), qt=perstep(S4(sp.sqt, s)), mc=perstep(S4(sp.smc, s)), slot=collect(1:U*(L+1)),
               mve=repeat(vcat(false, fill(true, L)), U), mve1=repeat(vcat(fill(true, L), false), U))
    r0 = G4(st.r, tab.cr[:, :, r])                            # (C, B, 1, U(L+1))
    dg = ifelse.(c4(on .+ z) .> 0.0, perstep(κs) .* (wb .- wa), 0.0)   # requirement change of every (unit, step) per unit t
    q = tab.gq[:, :, r]                                       # (B, U·G): the compensation units of every unit's node
    gcol = tab.gcol[:, :, r]                                  # (B, U(L+3)G): per unit, steps b0−1 … b0+L+1
    col(u, j, jj) = (u - 1) * (L + 3) * G + (j + 1) * G + jj   # unit u's node, step b0+j, unit jj
    cs = [col(u, j, jj) for u in 1:U for j in 0:L for jj in 1:G]
    gs = gcol[:, cs]
    qs = q[:, [(u - 1) * G + jj for u in 1:U for j in 0:L for jj in 1:G]]
    gon = M4(tab.gon[:, :, r]) .> 0.0
    gons = gon[:, :, :, [(u - 1) * G + jj for u in 1:U for j in 0:L for jj in 1:G]]
    κ = G4(cp.κg, qs)
    nslot = U * (L + 1)
    sl = slotmap(G, nslot)
    if !lin
        ω = G4(st.ω, gs)
        v = gons .& (absd.(κ) .>= k.κmin)
        d = prorata4(κ, ω, v, r0, dg, nslot; qadj=soc_qadj4(k.Δt, soc.k, soc.h, soc.ηin, soc.ηout, soc.qt, soc.e0, soc.e10))
        kk = skinks(z4; Q=d.Q, g=ifelse.(d.act, dg, 0.0), P=d.P, e=d.e, nm=d.nm, on=ifelse.(d.act, 1.0, 0.0))
        x = ifelse.(d.act[:, :, :, sl] .& v, ω ./ ifelse.(d.act[:, :, :, sl], d.P[:, :, :, sl], 1.0), 0.0)
        gen = gen4(st, cp, sp, z4, qs, gs, sl; ak=x, kid=sl)
    else
        β = βof4(cp, sp, qs, G4(cp.cls, tab.cc[:, :, r])[:, :, :, sl])
        dgc = dg[:, :, :, sl]
        v = gons .& (absd.(κ) .>= k.κmin) .& (β .!= 0.0) .& (dgc .!= 0.0)
        x = ifelse.(v, β .* dgc ./ ifelse.(v, κ, 1.0), 0.0)
        kk = nothing
        gen = gen4(st, cp, sp, z4, qs, gs, sl; a=x)
    end
    uj = [c for c in 1:U*G if ucon(fam, c)]                   # committable compensation columns (unit node, unit slot)
    uc, ucq = nothing, nothing
    if !isempty(uj)
        ucq = ucflags(sp, q, uj, gon)
        nU = length(uj)
        # slopes of the committable columns at step b0+j: x at the column of (u, j, jj), c = (u−1)G + jj
        xi(j) = [((c - 1) ÷ G) * (L + 1) * G + j * G + (c - 1) % G + 1 for c in uj]
        xs(j) = x[:, :, :, xi(j)]
        ks(j) = [((c - 1) ÷ G) * (L + 1) + j + 1 for c in uj]
        gstep(j) = gcol[:, [col((c - 1) ÷ G + 1, j, (c - 1) % G + 1) for c in uj]]
        sets = [(cy=gstep(j), y=0 <= j ? xs(j) : nothing, ky=0 <= j ? ks(j) : nothing, cu=gstep(j),
                 u=0 <= j ? xs(j) .* ucq : nothing, ku=0 <= j ? ks(j) : nothing,
                 cu1=gstep(j + 1), u1=j + 1 <= L ? xs(j + 1) .* ucq : nothing, ku1=j + 1 <= L ? ks(j + 1) : nothing) for j in -1:L]
        uc = uc4s(st, cp, sp, z4, q[:, uj], ucq, sets, !lin)
    end
    tb = TermBatchS(z; gen=gen, uc=uc, soc=soc, kk=kk, r0=r0)
    return tb, (on, w, x, gs, ucq, uj, G, U, lin, um)
end

function commit_block(st, tab, r, t, dr, aux, tb)
    on, w, x, gs, ucq, uj, G, U, lin, um = aux
    L = size(w, 4)
    t4 = c4(t)
    iw = [(u - 1) * (L + 2) + j for u in 1:U for j in 2:L+1]   # instants b0+1 … b0+L of every unit (w_0 = 0)
    uw = um[:, :, :, [(u - 1) * (L + 1) + 1 for u in 1:U for _ in 1:L]]
    e = scadd4(st.e, tab.ce[:, iw, r], t4 .* cat(fill(w, U)...; dims=4) .* uw)
    nslot = U * (L + 1)
    Δ = unitΔ(x, tb, t, slotmap(G, nslot), !lin)
    ω = scadd4(st.ω, gs, Δ)
    uc = st.uc
    if ucq !== nothing
        cols = [((c - 1) ÷ G) * (L + 1) * G + j * G + (c - 1) % G + 1 for j in 0:L for c in uj]
        uc = scadd4(st.uc, gs[:, cols], Δ[:, :, :, cols] .* cat(fill(ucq, L + 1)...; dims=4))
    end
    rr = scadd4(st.r, tab.cr[:, :, r], msk(dr4(dr), c4(on .+ 0.0 .* t) .* um))
    return FlatState(st.θ, ω, uc, e, st.u, rr)
end
