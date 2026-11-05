# Host sources of the state-independent site choices and their redraw between device calls.
#
# The sparse angle families, the storage pairs, the cell–class aggregate blocks and the dense angle
# families draw their sites on the host from sources (`host_sources`), which a run hands to its plan
# (`plan(…; sources)`; the plan's schedule context `tp.ctx` holds them, so two plans do not share
# them). Between two device calls the driver redraws the sites of every family whose sites are random
# (the sparse atoms and node pairs, the storage pairs, the `:slong` instants and lengths, the block
# offsets and profiles, the dispatch shapes, the `:spair` partners, the cell carriers) into tables of
# the plan's shapes (`redraw_tables`): an upload, no recompilation. The dense angle families redraw their directions in `tdir_refresh`.

"Families whose site sets vary between draws (their commitment flags cover every unit, `uc_cols_any`; `:slevel` has fixed sites)."
redrawn_sites(f::Symbol) = haskey(SPARSE_FAMILIES, f) || haskey(SX_FAMILIES, f) || f in (:slong, :slongL) ||
                           startswith(string(f), "sblk") || startswith(string(f), "dshp")
"Families whose host choices `redraw_tables` redraws: those, the `:spair` partners and the cell carriers."
redrawn(f::Symbol) = redrawn_sites(f) || f in (:spair, :spair0) || startswith(string(f), "cell")

"""
    host_sources(P, LM, ix; HA=nothing, chains=1) -> NamedTuple

The host sources of a run, for `plan(…; sources)`: `sparse` the sparse atoms from the transport
modes of the flow basis (Haar pieces, `ML.localize`) and, with host adaptation `HA`, the learned
pieces (read at every draw, so a mode update reaches the next redraw); `groups` the cell–class
groups; `sx` the storage pairs; `tdir` the directions of the dense angle families (with `HA` only:
`HA.tdir`, else `nothing`: no sites) for `chains` chains (device rows).
"""
function host_sources(P::ML.Problem, LM, ix::DeviceIndex; HA=nothing, chains::Int=1)
    E = P.E
    fixed = Any[]
    if E.flow !== nothing && size(E.flow.modes, 2) > 0
        H = ML.haar_theta(E)
        push!(fixed, (H, ML.localize(E.flow.modes, H)))
    end
    AM = HA === nothing ? nothing : HA.AM
    atoms() = AM === nothing || isempty(AM.pieces) ? fixed : vcat(fixed, Any[(AM.H, AM.pieces)])
    return (sparse=sparse_source(P, LM, ix; atoms=atoms), groups=block_groups(P, ix), sx=sx_source(P, LM, ix),
            tdir=HA === nothing ? nothing : HA.tdir, chains=chains)
end

"""
    set_sources!(P, LM, ix; HA=nothing) -> NamedTuple

`host_sources` as the module-level defaults of the sparse atoms, the cell–class groups and the
storage pairs (`SPARSE_SOURCE`, `BLOCK_GROUPS`, `SX_SOURCE`), which a plan made without `sources`
copies (tests, scripts); returns the sources. The drivers pass theirs to `plan` instead.
"""
function set_sources!(P::ML.Problem, LM, ix::DeviceIndex; HA=nothing)
    src = host_sources(P, LM, ix; HA=HA)
    SPARSE_SOURCE[] = src.sparse
    BLOCK_GROUPS[] = src.groups
    SX_SOURCE[] = src.sx
    return src
end

"""
Profile tables of a run, for `plan(…; tables)`: with host adaptation `HA` the learned storage and
dispatch shapes (`learned_profile_table`, `learned_dshape_table`: plateau and zero stand-ins until
learned, a fixed table shape), else the plateau only and zero dispatch shapes (whose sites do not move).
"""
host_tables(HA) = (profiles=Dict{Int,Any}(L => HA === nothing ? profile_table(L) : learned_profile_table(L, HA.AM) for L in BLOCK_LENGTHS),
                   dshapes=Dict{Int,Any}(L => learned_dshape_table(L, HA === nothing ? nothing : HA.AM) for L in DSHAPE_LENGTHS),
                   dpairs=nothing)

"""
    redraw_tables(pl, M, rng) -> (tables, dropped, hosts)

New sites for the families of `pl` with random sites (`redrawn`), in the plan's table shapes:
the rounds drawn again (`family_rounds` on the plan's schedule context), cut into chunks of at most the plan's B sites (as the
plan's memory chunks), at most the plan's R rounds (the rest padded with empty rounds): the first
chunk of every drawn round first, further chunks (the largest first) while rounds are left. Sites beyond the shape are
dropped (the sites of a round are shuffled first, so none is dropped systematically); `dropped`
counts them. A family whose redraw does not fit the shape keeps its
table, and so does one whose redraw puts a committed unit into a column the plan's commitment
flags (`pl.meta[f].uc`, `ucon`) leave out. Returns NamedTuples family → device table (with the
plan's static data) and family → host table (for `check_conflicts`).
"""
function redraw_tables(pl, M::DeviceModel, rng)
    tp = pl.tp
    out = Pair{Symbol,Any}[]; hosts = Pair{Symbol,Any}[]
    dropped = 0
    for f in pl.fams
        redrawn(f) || continue
        ho = pl.host[f]
        B0, R0 = size(ho.on)
        rs = family_rounds(Val(f), tp, M, tclasses(tp.T), rng)
        rounds = Vector{Vector{Tuple}}()                # each drawn round cut to B0 sites, its rest in spare rounds
        rest = Vector{Vector{Tuple}}()
        for r in rs
            isempty(r) && continue
            shuffle!(rng, r)
            ch = [collect(Tuple, c) for c in Iterators.partition(r, B0)]
            push!(rounds, ch[1]); append!(rest, ch[2:end])
        end
        isempty(rounds) && continue
        append!(rounds, sort(rest; by=length, rev=true))
        if length(rounds) > R0
            dropped += sum(length, rounds[R0+1:end])
            rounds = rounds[1:R0]
        end
        while length(rounds) < R0
            push!(rounds, Tuple[])
        end
        ht = host_table(f, tp, rounds; D=family_degree(f, tp, rounds), R=R0, B=B0)
        all(k -> size(getfield(ht, k)) == size(getfield(ho, k)), keys(ho)) || continue
        if haskey(ht, :gq) && pl.meta[f].uc !== nothing   # the commitment columns are static (`ucon`)
            dt0 = pl.dev[f]
            any(j -> !ucon(dt0, j) && any(q -> q > 0 && M.guc[q] > 0, ht.gq[:, j, :]), 1:size(ht.gq, 2)) && continue
        end
        push!(out, f => device_table(pl, f, ht)); push!(hosts, f => ht)
    end
    return (; out...), dropped, (; hosts...)
end

"Replica driver: redraw the random sites of its plan and upload them (`redraw_tables`; same shapes)."
function replica_redraw!(RD)
    dev = RD.dev
    rd, _, _ = redraw_tables(dev[:pl], RD.M, dev[:srng])
    isempty(rd) || (dev[:tabs] = merge(dev[:tabs], dev[:up](rd)))
    return nothing
end
