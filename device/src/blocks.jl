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

"Columns of a block site: SoC instants, residual of the node, generators per step (b0−1 … b0+L+1)."
function block_cols(tp::Topo, site, L::Int)
    T = tp.T
    s, b0, p = site
    n = tp.snode[s]
    G = gmax(tp)
    qs = padv(tp.nodegens[n], G)
    ce = [colτ(T, b0 + j, s) for j in 0:L+1]                  # instants b0 … b0+L+1
    cr = [colτ(T, b0 + j, n) for j in 0:L]
    gc(j) = padv([colτ(T, b0 + j, q) for q in tp.nodegens[n]], G)
    gcol = reduce(vcat, [gc(j) for j in -1:L+1])                                  # (L+3)·G
    # footprint fields: written (instants b0+1 … b0+L, generator steps b0 … b0+L), read only (the rest)
    return (s=s, p=p, ce=ce, cr=cr, gq=qs, gcol=gcol, cc=cr, cew=ce[2:L+1], cer=[ce[1], ce[L+2]],
            gw=reduce(vcat, [gc(j) for j in 0:L]), grd=vcat(gc(-1), gc(L + 1)))
end

for L in BLOCK_LENGTHS, lin in (false, true)
    nm = blksym(L, lin)
    @eval begin
        family_rounds(::Val{$(QuoteNode(nm))}, tp::Topo, M::DeviceModel, cls, rng) =
            ($lin || !tracking(M) || error("block T2 needs a model without requirement tracking (use :sblkL…)");
             block_rounds($L, tp, rng; h=round(Int, M.Δt)))
        site_cols(::Val{$(QuoteNode(nm))}, tp::Topo, site, D::Int) = block_cols(tp, site, $L)
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

function build_block(st, cp, sp, tab, r, k, ::Val{L}, ::Val{lin}, fam::Symbol=blksym(L, lin)) where {L,lin}
    on = m3(tab.on[:, r])
    s = tab.s[:, r]
    κs = g3(cp.ks, s) .* s3(sp.sh, s) ./ k.Δt
    okm = (on .> 0.0) .& (κs .>= k.κmin)
    z = 0.0 .* κs
    z4 = c4(z)
    w = block_weights4(tab, r, L) .* c4(ifelse.(on .> 0.0, 1.0, 0.0))     # (C|1, B, 1, L)
    zw = 0.0 .* w[:, :, :, 1:1]
    wa = cat(zw, w; dims=4)                                   # w_0 … w_L
    wb = cat(w, zw; dims=4)                                   # w_1 … w_{L+1}
    G = size(tab.gq, 2)
    # SoC terms of steps b0+j (args e(b0+j), e(b0+j+1)), j = 0 … L, and balance slots
    rep(x) = (y = c4(x); cat(fill(y, L + 1)...; dims=4))
    ce = tab.ce[:, :, r]                                      # instants b0 … b0+L+1
    soc = ssoc(z4, L + 1; e0=G4(st.e, ce[:, 1:L+1]), ae=wa, e10=G4(st.e, ce[:, 2:L+2]), ae1=wb,
               k=rep(g3(cp.ks, s)), w=rep(g3(cp.ws, s)), h=rep(s3(sp.sh, s)), ηin=rep(s3(sp.sηin, s)), ηout=rep(s3(sp.sηout, s)),
               qt=rep(s3(sp.sqt, s)), mc=rep(s3(sp.smc, s)), slot=collect(1:L+1),
               mve=vcat(false, fill(true, L)), mve1=vcat(fill(true, L), false))
    r0 = G4(st.r, tab.cr[:, :, r])                            # (C, B, 1, L+1)
    dg = ifelse.(c4(okm), c4(κs) .* (wb .- wa), 0.0)          # requirement change of every step per unit t
    q = tab.gq[:, :, r]
    gcol = tab.gcol[:, :, r]                                  # (B, (L+3) G): steps b0−1 … b0+L+1, G units each
    gstep(j) = gcol[:, (j+1)*G+1:(j+2)*G]                     # units at step b0+j
    gs = gcol[:, G+1:(L+2)*G]                                 # steps b0 … b0+L
    qs = reduce(hcat2, fill(q, L + 1))
    gon = M4(tab.gon[:, :, r]) .> 0.0
    gons = cat(fill(gon, L + 1)...; dims=4)
    κ = G4(cp.κg, qs)
    sl = slotmap(G, L + 1)
    if !lin
        ω = G4(st.ω, gs)
        v = gons .& (absd.(κ) .>= k.κmin)
        d = prorata4(κ, ω, v, r0, dg, L + 1)
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
    uj = [jj for jj in 1:G if ucon(fam, jj)]
    uc, ucq = nothing, nothing
    if !isempty(uj)
        ucq = ucflags(sp, q, uj, gon)
        nU = length(uj)
        xs(j) = x[:, :, :, j*G .+ uj]                         # slopes of the committable units at step b0+j
        ks(j) = fill(j + 1, nU)
        # commitment term of step b0+j: (y_j, u_j, u_{j+1}), j = −1 … L (y_j moves for 0 ≤ j ≤ L, u_{j+1} for j+1 ≤ L)
        sets = [(cy=gstep(j)[:, uj], y=0 <= j ? xs(j) : nothing, ky=ks(j), cu=gstep(j)[:, uj], u=0 <= j ? xs(j) .* ucq : nothing, ku=ks(j),
                 cu1=gstep(j + 1)[:, uj], u1=j + 1 <= L ? xs(j + 1) .* ucq : nothing, ku1=ks(j + 1)) for j in -1:L]
        uc = uc4s(st, cp, sp, z4, q[:, uj], ucq, sets, !lin)
    end
    tb = TermBatchS(z; gen=gen, uc=uc, soc=soc, kk=kk, r0=r0)
    return tb, (on, w, x, gs, ucq, uj, G, lin)
end

function commit_block(st, tab, r, t, dr, aux, tb)
    on, w, x, gs, ucq, uj, G, lin = aux
    L = size(w, 4)
    t4 = c4(t)
    e = scadd4(st.e, tab.ce[:, 2:L+1, r], t4 .* w)            # instants b0+1 … b0+L (w_0 = 0)
    Δ = unitΔ(x, tb, t, slotmap(G, L + 1), !lin)
    ω = scadd4(st.ω, gs, Δ)
    uc = st.uc
    if ucq !== nothing
        cols = [j * G + jj for j in 0:L for jj in uj]
        uc = scadd4(st.uc, gs[:, cols], Δ[:, :, :, cols] .* cat(fill(ucq, L + 1)...; dims=4))
    end
    rr = scadd4(st.r, tab.cr[:, :, r], msk(dr4(dr), c4(on .+ 0.0 .* t)))
    return FlatState(st.θ, ω, uc, e, st.u, rr)
end
