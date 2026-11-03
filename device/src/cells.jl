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

const CELLINFO = Ref{Any}(nothing)      # set by `plan(…; cells=cell_info(…))` for the schedule

"""
Rounds of the cell family `base` (:cell or :cellpair) at the levels `qs`. `uc`: nothing (any
carriers), false (carriers without commitment only), true (:cell: a committable carrier; :cellpair:
at least one of the two committable).
"""
function cell_rounds(base::Symbol, qs, tp::Topo, cls, rng; uc=nothing)
    ci = CELLINFO[]
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
family_rounds(::Val{:cell}, tp::Topo, M::DeviceModel, cls, rng) = cell_rounds(:cell, eachindex(CELLINFO[].nodes), tp, cls, rng)
family_rounds(::Val{:cellpair}, tp::Topo, M::DeviceModel, cls, rng) = cell_rounds(:cellpair, eachindex(CELLINFO[].nodes), tp, cls, rng)

ncmax(ci::CellInfo) = maximum(length(n) for lv in ci.nodes for n in lv; init=1)
ncmax(ci::CellInfo, q::Int) = maximum(length.(ci.nodes[q]); init=1)
"Node slots of a site: the largest cell of its level in the level buckets (`cell2`, …), of all levels in `:cell`."
const CELL_BUCKETED = Ref(false)
ncsite(ci, q) = CELL_BUCKETED[] ? ncmax(ci, q) : ncmax(ci)

"Columns of carrier g's unit at every node slot of the cell (0 where the carrier has no unit there)."
function cellcols(ci::CellInfo, T, q, k, g, τ, Nc)
    us = [get(ci.unitat, (g, n), 0) for n in ci.nodes[q][k]]
    c(τx) = padv([u == 0 ? 0 : colτ(T, τx, u) for u in us], Nc)
    return padv(us, Nc), c(τ - 1), c(τ), c(τ + 1)
end

function site_cols(::Val{:cell}, tp::Topo, site, D::Int)
    ci = CELLINFO[]; T = tp.T; Nc = ncsite(ci, site[1])
    q, k, g, τ = site
    ua, cam, ca, ca1 = cellcols(ci, T, q, k, g, τ, Nc)
    return (ua=ua, cwam=cam, cwa=ca, cwa1=ca1, crn=padv([colτ(T, τ, n) for n in ci.nodes[q][k]], Nc))
end
function site_cols(::Val{:cellpair}, tp::Topo, site, D::Int)
    ci = CELLINFO[]; T = tp.T; Nc = ncsite(ci, site[1])
    q, k, g1, g2, τ = site
    ua, cam, ca, ca1 = cellcols(ci, T, q, k, g1, τ, Nc)
    ub, cbm, cb, cb1 = cellcols(ci, T, q, k, g2, τ, Nc)
    return (ua=ua, cwam=cam, cwa=ca, cwa1=ca1, ub=ub, cwbm=cbm, cwb=cb, cwb1=cb1,
            crn=padv([colτ(T, τ, n) for n in ci.nodes[q][k]], Nc))
end
device_extra(::Val{:cell}, ht) = (aon=Float64.(ht.ua .> 0), non=Float64.(ht.crn .> 0))
device_extra(::Val{:cellpair}, ht) = (aon=Float64.(ht.ua .> 0), bon=Float64.(ht.ub .> 0), non=Float64.(ht.crn .> 0))

let fa = [(:cwa, :ω, true), (:cwa, :uc, true), (:cwam, :ω, false), (:cwam, :uc, false), (:cwa1, :uc, false)],
    fb = [(:cwb, :ω, true), (:cwb, :uc, true), (:cwbm, :ω, false), (:cwbm, :uc, false), (:cwb1, :uc, false)]
    FOOT[:cell] = vcat(fa, [(:crn, :r, true)])
    FOOT[:cellpair] = vcat(fa, fb, [(:crn, :r, true)])
end

"Gen and commitment terms (follow at τ) of one carrier over the node slots; slopes a[j]; returns a·ucq per slot."
function carrier_terms!(gens, ucs, st, cp, sp, tab, r, uf, cm, c0, c1, onf, a, withuc=true)
    au = Any[]
    for j in 1:size(tab[uf], 2)
        q = tab[uf][:, j, r]
        on = m3(tab[onf][:, j, r])
        cw, cwm, cw1 = tab[c0][:, j, r], tab[cm][:, j, r], tab[c1][:, j, r]
        ucq = s3(sp.ucq, q) .* on
        push!(gens, gent(st, cp, sp, q, cw, j; a=a[j]))
        withuc && push!(ucs, uct(cp, sp, q; y0=g3(st.ω, cw), ay=a[j], u0=g3(st.uc, cw), au=a[j] .* ucq, u10=g3(st.uc, cw1), on=ucq),
                             uct(cp, sp, q; y0=g3(st.ω, cwm), u0=g3(st.uc, cwm), u10=g3(st.uc, cw), au1=a[j] .* ucq, on=ucq))
        push!(au, a[j] .* ucq .* (withuc ? 1.0 : 0.0))
    end
    return au
end

build(::Val{:cell}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_cell(st, cp, sp, tab, r, k, Val(true))
build(::Val{:cellpair}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_cellpair(st, cp, sp, tab, r, k, Val(true))

function build_cell(st, cp, sp, tab, r, k, ::Val{withuc}) where {withuc}
    on = m3(tab.on[:, r])
    Nc = size(tab.ua, 2)
    a = [on .* m3(tab.aon[:, j, r]) for j in 1:Nc]
    gens, ucs = Any[], Any[]
    au = carrier_terms!(gens, ucs, st, cp, sp, tab, r, :ua, :cwam, :cwa, :cwa1, :aon, a, withuc)
    tb = TermBatch(mkgen(gens), mkuc(ucs), nothing, nothing, nothing, Any[g3(st.r, tab.crn[:, j, r]) for j in 1:Nc])
    return tb, (a, au, nothing, nothing, on)
end

function build_cellpair(st, cp, sp, tab, r, k, ::Val{withuc}) where {withuc}
    on = m3(tab.on[:, r])
    Nc = size(tab.ua, 2)
    K1, K2 = on .* 0.0, on .* 0.0
    for j in 1:Nc
        K1 = K1 .+ m3(tab.aon[:, j, r]) .* g3(cp.κg, tab.ua[:, j, r])
        K2 = K2 .+ m3(tab.bon[:, j, r]) .* g3(cp.κg, tab.ub[:, j, r])
    end
    ok = (on .> 0.0) .& (absd.(K1) .>= k.κmin) .& (absd.(K2) .>= k.κmin)
    x1 = ifelse.(ok, 1.0 ./ ifelse.(ok, K1, 1.0), 0.0)
    x2 = ifelse.(ok, -1.0 ./ ifelse.(ok, K2, 1.0), 0.0)
    a = [x1 .* m3(tab.aon[:, j, r]) for j in 1:Nc]
    b = [x2 .* m3(tab.bon[:, j, r]) for j in 1:Nc]
    gens, ucs = Any[], Any[]
    au = carrier_terms!(gens, ucs, st, cp, sp, tab, r, :ua, :cwam, :cwa, :cwa1, :aon, a, withuc)
    bu = carrier_terms!(gens, ucs, st, cp, sp, tab, r, :ub, :cwbm, :cwb, :cwb1, :bon, b, withuc)
    tb = TermBatch(mkgen(gens), mkuc(ucs), nothing, nothing, nothing, Any[g3(st.r, tab.crn[:, j, r]) for j in 1:Nc])
    return tb, (a, au, b, bu, ifelse.(ok, 1.0, 0.0))
end

function commit_cells(st, tab, r, t, dr, aux)
    a, au, b, bu, m = aux
    ω, uc = st.ω, st.uc
    for j in eachindex(a)
        ω = scadd(ω, tab.cwa[:, j, r], flat(t .* a[j]))
        uc = scadd(uc, tab.cwa[:, j, r], flat(t .* au[j]))
        if b !== nothing
            ω = scadd(ω, tab.cwb[:, j, r], flat(t .* b[j]))
            uc = scadd(uc, tab.cwb[:, j, r], flat(t .* bu[j]))
        end
    end
    rr = st.r
    for j in eachindex(dr)
        rr = scadd(rr, tab.crn[:, j, r], flat(msk(dr[j], m .* m3(tab.non[:, j, r]))))
    end
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
            (CELL_BUCKETED[] = true; cell_rounds($(QuoteNode(base)), ($q,), tp, cls, rng; uc=$uc))
        site_cols(::Val{$(QuoteNode(nm))}, tp::Topo, site, D::Int) = (CELL_BUCKETED[] = true; site_cols(Val($(QuoteNode(base))), tp, site, D))
        device_extra(::Val{$(QuoteNode(nm))}, ht) = device_extra(Val($(QuoteNode(base))), ht)
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
