# Host sources of the state-independent site choices and their redraw between device calls.
#
# The sparse angle families (`SPARSE_SOURCE`), the storage pairs (`SX_SOURCE`) and the cell–class
# aggregate blocks (`BLOCK_GROUPS`) draw their sites on the host; `set_sources!` sets them for a run.
# Between two device calls the driver redraws the sites of every family whose sites are random
# (the sparse atoms and node pairs, the storage pairs, the `:slong` instants and lengths, the block
# offsets and profiles, the dispatch shapes, the `:spair` partners, the cell carriers) into tables of
# the plan's shapes (`redraw_tables`): an upload, no recompilation. The dense angle families redraw their directions in `tdir_refresh`.

"Families whose site sets vary between draws (their commitment flags cover every unit, `uc_cols_any`)."
redrawn_sites(f::Symbol) = haskey(SPARSE_FAMILIES, f) || haskey(SX_FAMILIES, f) || f in (:slong, :slongL) ||
                           startswith(string(f), "sblk") || startswith(string(f), "dshp")
"Families whose host choices `redraw_tables` redraws: those, the `:spair` partners and the cell carriers."
redrawn(f::Symbol) = redrawn_sites(f) || f in (:spair, :spair0) || startswith(string(f), "cell")

"""
    set_sources!(P, LM, ix; HA=nothing)

The host sources of a run: the sparse atoms from the transport modes of the flow basis (Haar
pieces, `ML.localize`) and, with host adaptation `HA`, the learned pieces (read at every draw, so
a mode update reaches the next redraw); the cell–class groups; the storage pairs.
"""
function set_sources!(P::ML.Problem, LM, ix::DeviceIndex; HA=nothing)
    E = P.E
    fixed = Any[]
    if E.flow !== nothing && size(E.flow.modes, 2) > 0
        H = ML.haar_theta(E)
        push!(fixed, (H, ML.localize(E.flow.modes, H)))
    end
    AM = HA === nothing ? nothing : HA.AM
    atoms() = AM === nothing || isempty(AM.pieces) ? fixed : vcat(fixed, Any[(AM.H, AM.pieces)])
    SPARSE_SOURCE[] = sparse_source(P, LM, ix; atoms=atoms)
    BLOCK_GROUPS[] = block_groups(P, ix)
    SX_SOURCE[] = sx_source(P, LM, ix)
    return nothing
end

"""
    redraw_tables(pl, M, rng) -> (tables, dropped, hosts)

New sites for the families of `pl` with random sites (`redrawn`), in the plan's table shapes:
the rounds drawn again (`family_rounds`), cut into chunks of at most the plan's B sites (as the
plan's memory chunks), at most the plan's R rounds (the rest padded with empty rounds): the first
chunk of every drawn round first, further chunks (the largest first) while rounds are left. Sites beyond the shape are
dropped (the sites of a round are shuffled first, so none is dropped systematically); `dropped`
counts them. A family whose redraw does not fit the shape keeps its
table, and so does one whose redraw puts a committed unit into a column the plan's commitment
flags (`UC_COLS`) leave out. Returns NamedTuples family → device table and family → host
table (for `check_conflicts`).
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
        if haskey(ht, :gq) && haskey(UC_COLS, f)          # the commitment columns are static (`ucon`)
            any(j -> !UC_COLS[f][j] && any(q -> q > 0 && M.guc[q] > 0, ht.gq[:, j, :]), 1:size(ht.gq, 2)) && continue
        end
        push!(out, f => device_table(f, ht)); push!(hosts, f => ht)
    end
    return (; out...), dropped, (; hosts...)
end
