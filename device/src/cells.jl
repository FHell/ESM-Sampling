# Phase 3, part 2: cell templates (docs/reactant_design.md §3.3: `cell`, `cellpair` at all coarser
# partition levels; the multigrid part of the move set).
#
#   :cell      one carrier g over a whole cell c of level q: +1 on the utilization of every unit of g
#              in the cell (CPU `prep_random!(:cell)`, `cell_entries!`)
#   :cellpair  two carriers g1 ≠ g2 over cell c: +1/K1 on the units of g1, −1/K2 on those of g2,
#              K = Σ κ over the carrier's units in the cell (|K| ≥ κ_min, per chain), so the cell
#              injection is balanced (CPU `prep_random!(:cellpair)`)
#
# Levels: the CPU draws level q with probability p_q ∝ (number of cells)^cell_power (`LM.qcum`),
# then a cell uniformly. A device round covers every cell of one level at once (cells of a level
# are disjoint, so the round is conflict-free; × time classes because the commitment follows), so
# level q gets rounds ∝ p_q / (cells of q): equal numbers of rounds per level for cell_power = 1.
# The carriers per (cell, step) are drawn on the host (state-independent).
#
# Terms of a site are indexed by the node slots of its cell (padded to the largest cell): node slot j
# carries the unit of g (and of g2) at that node and the balance slot j, so the slot map is static.

"Cell structure of the partition levels of the complete level (from the CPU LocalModel)."
struct CellInfo
    nodes::Vector{Vector{Vector{Int}}}      # [q][c] → nodes of the cell that have injecting units
    carriers::Vector{Vector{Vector{Int}}}   # [q][c] → generator groups present in the cell
    unitat::Dict{Tuple{Int,Int},Int}        # (group, node) → device unit
    ncells::Vector{Int}
    ucg::Vector{Bool}                       # per generator group: committable
end

function cell_info(P, LM, ix::DeviceIndex)
    E = P.E
    unitat = Dict{Tuple{Int,Int},Int}()
    for (q, (g, i)) in enumerate(ix.gidx)
        unitat[(g, E.gens[g].bus[i])] = q
    end
    gof(f) = findfirst(gg -> gg.vf == f, E.gens)
    nodes = Vector{Vector{Vector{Int}}}()
    carriers = Vector{Vector{Vector{Int}}}()
    for q in eachindex(LM.cellunits)
        cu = LM.cellunits[q]
        push!(nodes, [sort(unique([LM.bus[f][i] for sl in cu[c] for (f, i) in sl])) for c in eachindex(cu)])
        push!(carriers, [[gof(sl[1][1]) for sl in cu[c] if !isempty(sl) && gof(sl[1][1]) !== nothing] for c in eachindex(cu)])
    end
    return CellInfo(nodes, carriers, unitat, length.(LM.cellunits), [g.ucf > 0 for g in E.gens])
end

"Rounds per level: ∝ (cells)^(cell_power − 1), the smallest level count scaled to `base` rounds."
function level_rounds(ci::CellInfo; cell_power=1.0, base=1)
    w = Float64.(ci.ncells) .^ (cell_power - 1)
    return max.(1, round.(Int, base .* w ./ minimum(w)))
end

"The cell structure of a plan (`plan(…; cells=cell_info(…))`, in its schedule context)."
function cellinfo(tp::Topo)
    ci = tp.ctx.cells
    ci === nothing && error("cell families need the cell structure: plan(…; cells=cell_info(P, LM, ix))")
    return ci
end

"""
Rounds of the cell family `base` (:cell or :cellpair) at the levels `qs`. `uc`: nothing (any
carriers), false (carriers without commitment only), true (:cell: a committable carrier; :cellpair:
at least one of the two committable).
"""
function cell_rounds(base::Symbol, qs, tp::Topo, cls, rng; uc=nothing)
    ci = cellinfo(tp)
    rounds = Vector{Vector{Tuple}}()
    lr = level_rounds(ci)
    isuc(g) = ci.ucg[g]
    for q in qs, _ in 1:lr[q], c in cls
        sites = Tuple[]
        for τ in c, k in eachindex(ci.nodes[q])
            cs = ci.carriers[q][k]
            if base == :cell
                cs = uc === nothing ? cs : filter(g -> isuc(g) == uc, cs)
                isempty(cs) || push!(sites, (q, k, rand(rng, cs), τ))
            else
                prs = [(g1, g2) for g1 in cs for g2 in cs if g1 != g2 && (uc === nothing || (isuc(g1) | isuc(g2)) == uc)]
                isempty(prs) || push!(sites, (q, k, rand(rng, prs)..., τ))
            end
        end
        push!(rounds, sites)
    end
    return rounds
end
family_rounds(::Val{:cell}, tp::Topo, M::DeviceModel, cls, rng) = cell_rounds(:cell, eachindex(cellinfo(tp).nodes), tp, cls, rng)
family_rounds(::Val{:cellpair}, tp::Topo, M::DeviceModel, cls, rng) = cell_rounds(:cellpair, eachindex(cellinfo(tp).nodes), tp, cls, rng)

ncmax(ci::CellInfo) = maximum(length(n) for lv in ci.nodes for n in lv; init=1)
ncmax(ci::CellInfo, q::Int) = maximum(length.(ci.nodes[q]); init=1)
"Node slots of a site: the largest cell of its level in the level buckets (`cell2`, …, `bucketed`), of all levels in `:cell`."
ncsite(ci, q, bucketed::Bool) = bucketed ? ncmax(ci, q) : ncmax(ci)

"Columns of carrier g's unit at every node slot of the cell (0 where the carrier has no unit there)."
function cellcols(ci::CellInfo, T, q, k, g, τ, Nc)
    us = [get(ci.unitat, (g, n), 0) for n in ci.nodes[q][k]]
    c(τx) = padv([u == 0 ? 0 : colτ(T, τx, u) for u in us], Nc)
    return padv(us, Nc), c(τ - 1), c(τ), c(τ + 1)
end

site_cols(::Val{:cell}, tp::Topo, site, D::Int) = cell_cols(tp, site, false)
site_cols(::Val{:cellpair}, tp::Topo, site, D::Int) = cellpair_cols(tp, site, false)
function cell_cols(tp::Topo, site, bucketed::Bool)
    ci = cellinfo(tp); T = tp.T; Nc = ncsite(ci, site[1], bucketed)
    q, k, g, τ = site
    ua, cam, ca, ca1 = cellcols(ci, T, q, k, g, τ, Nc)
    return (ua=ua, cwam=cam, cwa=ca, cwa1=ca1, crn=padv([colτ(T, τ, n) for n in ci.nodes[q][k]], Nc))
end
function cellpair_cols(tp::Topo, site, bucketed::Bool)
    ci = cellinfo(tp); T = tp.T; Nc = ncsite(ci, site[1], bucketed)
    q, k, g1, g2, τ = site
    ua, cam, ca, ca1 = cellcols(ci, T, q, k, g1, τ, Nc)
    ub, cbm, cb, cb1 = cellcols(ci, T, q, k, g2, τ, Nc)
    return (ua=ua, cwam=cam, cwa=ca, cwa1=ca1, ub=ub, cwbm=cbm, cwb=cb, cwb1=cb1,
            crn=padv([colτ(T, τ, n) for n in ci.nodes[q][k]], Nc))
end
device_extra(::Val{:cell}, ht, tp) = (aon=Float64.(ht.ua .> 0), non=Float64.(ht.crn .> 0))
device_extra(::Val{:cellpair}, ht, tp) = (aon=Float64.(ht.ua .> 0), bon=Float64.(ht.ub .> 0), non=Float64.(ht.crn .> 0))

let fa = [(:cwa, :ω, true), (:cwa, :uc, true), (:cwam, :ω, false), (:cwam, :uc, false), (:cwa1, :uc, false)],
    fb = [(:cwb, :ω, true), (:cwb, :uc, true), (:cwbm, :ω, false), (:cwbm, :uc, false), (:cwb1, :uc, false)]
    FOOT[:cell] = vcat(fa, [(:crn, :r, true)])
    FOOT[:cellpair] = vcat(fa, fb, [(:crn, :r, true)])
end

"""
Generator and commitment columns (follow at τ) of carriers over the node slots: unit tables `uf`,
columns `cm`, `c0`, `c1` (τ−1, τ, τ+1), masks `onf`, slopes a (C|1, B, 1, Nc) per carrier (one entry of
the tuple `parts` per carrier); node slot j is balance slot j. Returns (gen, uc, ucq per carrier).
"""
function carriers4(st, cp, sp, tab, r, z4, parts, withuc::Bool)
    qs = reduce(hcat2, [tab[p.uf][:, :, r] for p in parts])
    cws = reduce(hcat2, [tab[p.c0][:, :, r] for p in parts])
    Nc = size(tab[parts[1].uf], 2)
    a = cat((p.a for p in parts)...; dims=4)
    gen = gen4(st, cp, sp, z4, qs, cws, repeat(collect(1:Nc), length(parts)); a=a)
    withuc || return gen, nothing, nothing
    ucq = cat((S4(sp.ucq, tab[p.uf][:, :, r]) .* M4(tab[p.onf][:, :, r]) for p in parts)...; dims=4)
    cms = reduce(hcat2, [tab[p.cm][:, :, r] for p in parts])
    c1s = reduce(hcat2, [tab[p.c1][:, :, r] for p in parts])
    au = a .* ucq
    sets = [(cy=cws, y=a, ky=nothing, cu=cws, u=au, ku=nothing, cu1=c1s, u1=nothing, ku1=nothing),
            (cy=cms, y=nothing, ky=nothing, cu=cms, u=nothing, ku=nothing, cu1=cws, u1=au, ku1=nothing)]
    return gen, uc4s(st, cp, sp, z4, qs, ucq, sets, false), ucq
end

build(::Val{:cell}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_cell(st, cp, sp, tab, r, k, Val(true))
build(::Val{:cellpair}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_cellpair(st, cp, sp, tab, r, k, Val(true))

function build_cell(st, cp, sp, tab, r, k, ::Val{withuc}) where {withuc}
    on = m3(tab.on[:, r])
    z = 0.0 .* g3(st.r, tab.crn[:, 1, r])
    z4 = c4(z)
    a = c4(on .+ z) .* M4(tab.aon[:, :, r])
    gen, uc, ucq = carriers4(st, cp, sp, tab, r, z4, ((uf=:ua, cm=:cwam, c0=:cwa, c1=:cwa1, onf=:aon, a=a),), withuc)
    tb = TermBatchS(z; gen=gen, uc=uc, r0=G4(st.r, tab.crn[:, :, r]))
    return tb, (a, ucq, (:cwa,), c4(on .+ z))
end

function build_cellpair(st, cp, sp, tab, r, k, ::Val{withuc}) where {withuc}
    on = m3(tab.on[:, r])
    z = 0.0 .* g3(st.r, tab.crn[:, 1, r])
    z4 = c4(z)
    K1 = sum(M4(tab.aon[:, :, r]) .* G4(cp.κg, tab.ua[:, :, r]); dims=4)
    K2 = sum(M4(tab.bon[:, :, r]) .* G4(cp.κg, tab.ub[:, :, r]); dims=4)
    ok = (c4(on .+ z) .> 0.0) .& (absd.(K1) .>= k.κmin) .& (absd.(K2) .>= k.κmin)
    x1 = ifelse.(ok, 1.0 ./ ifelse.(ok, K1, 1.0), 0.0)
    x2 = ifelse.(ok, -1.0 ./ ifelse.(ok, K2, 1.0), 0.0)
    a = x1 .* M4(tab.aon[:, :, r])
    b = x2 .* M4(tab.bon[:, :, r])
    gen, uc, ucq = carriers4(st, cp, sp, tab, r, z4, ((uf=:ua, cm=:cwam, c0=:cwa, c1=:cwa1, onf=:aon, a=a),
                                                    (uf=:ub, cm=:cwbm, c0=:cwb, c1=:cwb1, onf=:bon, a=b)), withuc)
    tb = TermBatchS(z; gen=gen, uc=uc, r0=G4(st.r, tab.crn[:, :, r]))
    return tb, (cat(a, b; dims=4), ucq, (:cwa, :cwb), ifelse.(ok, 1.0, 0.0))
end

function commit_cells(st, tab, r, t, dr, aux)
    a, ucq, cf, m = aux
    cws = reduce(hcat2, [tab[c][:, :, r] for c in cf])
    Δ = a .* c4(t)
    ω = scadd4(st.ω, cws, Δ)
    uc = ucq === nothing ? st.uc : scadd4(st.uc, cws, Δ .* ucq)
    rr = scadd4(st.r, tab.crn[:, :, r], msk(dr4(dr), m .* M4(tab.non[:, :, r])))
    return FlatState(st.θ, ω, uc, st.e, st.u, rr)
end

commit(::Val{:cell}, st::FlatState, tab, r, t, dr, aux) = commit_cells(st, tab, r, t, dr, aux)
commit(::Val{:cellpair}, st::FlatState, tab, r, t, dr, aux) = commit_cells(st, tab, r, t, dr, aux)

# level buckets: :cell1, :cellu1, … and :cellpair1, :cellpairu1, … (one family per partition level,
# padded to the largest cell of that level; the dense form costs O(cell size²) per site, so levels
# are not mixed). `u` families hold the committable carriers (commitment terms), the others none.
const CELL_LEVELS = 9
cellsym(q, uc=false) = Symbol(uc ? :cellu : :cell, q)
pairsym(q, uc=false) = Symbol(uc ? :cellpairu : :cellpair, q)
for q in 1:CELL_LEVELS, uc in (false, true), (nm, base) in ((cellsym(q, uc), :cell), (pairsym(q, uc), :cellpair))
    @eval begin
        family_rounds(::Val{$(QuoteNode(nm))}, tp::Topo, M::DeviceModel, cls, rng) =
            cell_rounds($(QuoteNode(base)), ($q,), tp, cls, rng; uc=$uc)
        site_cols(::Val{$(QuoteNode(nm))}, tp::Topo, site, D::Int) = $(base == :cell ? :cell_cols : :cellpair_cols)(tp, site, true)
        device_extra(::Val{$(QuoteNode(nm))}, ht, tp) = device_extra(Val($(QuoteNode(base))), ht, tp)
        build(::Val{$(QuoteNode(nm))}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) =
            $(base == :cell ? :build_cell : :build_cellpair)(st, cp, sp, tab, r, k, Val($uc))
        commit(::Val{$(QuoteNode(nm))}, st::FlatState, tab, r, t, dr, aux) = commit_cells(st, tab, r, t, dr, aux)
    end
    FOOT[nm] = FOOT[base]
end

"The level-bucketed cell families of a cell structure (only those with sites)."
function cell_families(ci::CellInfo)
    fams = Symbol[]
    for q in eachindex(ci.nodes), uc in (false, true)
        any(cs -> any(g -> ci.ucg[g] == uc, cs), ci.carriers[q]) && push!(fams, cellsym(q, uc))
    end
    for q in eachindex(ci.nodes), uc in (false, true)
        any(cs -> any(ci.ucg[g1] | ci.ucg[g2] == uc for g1 in cs for g2 in cs if g1 != g2), ci.carriers[q]) && push!(fams, pairsym(q, uc))
    end
    return Tuple(fams)
end
