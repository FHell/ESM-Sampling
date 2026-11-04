# Sparse angle directions with compensation: the localized transport moves (CPU `prep_tcut!`,
# `prep_tsparse!`: Haar atoms of the proxy modes `tcut` and of the learned modes `acut`) and the
# cross-node pairs with re-routed flows (`prep_xloc!`: `xloc`, `xlocc`; `prep_random!(:xpair)`:
# global re-routing), docs/system/moves/transport.md, pairs.md, learned.md.
#
# A site is one step τ with a sparse angle direction: its node slots (the nodes whose angle moves
# and the far ends of the lines that move; a cross-node pair has its two nodes first), the angle
# change d and the requirement change g per unit t of every slot (g = 0: no compensation there),
# and, for the pair families, the two generator units (+1/κ₁ at slot 1, −1/κ₂ at slot 2). Every
# line that moves is listed at both its ends (line ends: injection at the near end only, half its
# overload cost at each), so the balance slots stay static whatever the site. All of it is host
# data (`SPARSE_SOURCE`), drawn per site from (k, x)-independent choices (atoms ∝ c², partners,
# units), packed into rounds of sites with disjoint node slots per step.
#
#   family      CPU                      rule (g)            pair units   node slots
#   :tcut       tcut, acut               share-preserving    no           ≤ SPARSE_NS (small atoms)
#   :tcutw      tcut, acut               share-preserving    no           every node (wide atoms)
#   :tcutL      tcutL, acutL             linear              no           ≤ SPARSE_NS
#   :tcutLw     tcutL, acutL             linear              no           every node
#   :xloc       xloc                     none (leak)         yes          ≤ SPARSE_NS
#   :xlocc      xlocc                    share-preserving    yes          ≤ SPARSE_NS
#   :xpair      xpair                    none                yes          every node

"family → (rule, pair units, bucket); bucket :S has `SPARSE_NS` node slots, :W every node."
const SPARSE_FAMILIES = Dict(:tcut => (:T2, false, :S), :tcutw => (:T2, false, :W), :tcutL => (:L, false, :S),
                             :tcutLw => (:L, false, :W), :xloc => (:none, true, :S), :xlocc => (:T2, true, :S),
                             :xpair => (:none, true, :W))
"Node slots of the small sparse sites."
const SPARSE_NS = 16
"Sites drawn per step and round (first fit of disjoint node slots)."
const SPARSE_PACK = Ref(8)

"""
Host source of the sparse sites: (fam, τ, rng) → (nodes, d, g, pair) or `nothing`; `nodes` the node
slots (pair families: the two units' nodes first), `d`, `g` per slot, `pair` = (q₁, q₂) device units or
`nothing`. Set by the driver or a test before `plan` (`sparse_source`).
"""
const SPARSE_SOURCE = Ref{Any}(nothing)

sparse_ns(fam::Symbol, tp::Topo) = SPARSE_FAMILIES[fam][3] === :S ? min(SPARSE_NS, tp.N) : tp.N
sparse_dm(tp::Topo) = maximum(length.(tp.nodelines); init=1)

function sparse_rounds(fam::Symbol, tp::Topo, M::DeviceModel, cls, rng)
    TDIR_TOPO[] = tp
    src = SPARSE_SOURCE[]
    src === nothing && return Vector{Vector{Tuple}}()
    rule, pair, bucket = SPARSE_FAMILIES[fam]
    (rule === :T2 && tracking(M)) && error(":$fam needs a model without line losses and lossy links")
    ns = sparse_ns(fam, tp)
    rounds = Vector{Vector{Tuple}}()
    for c in cls
        sites = Tuple[]
        for τ in c
            used = Set{Int}()
            for _ in 1:(bucket === :W ? 1 : SPARSE_PACK[])
                spec = src(fam, τ, rng)
                (spec === nothing || length(spec[1]) > ns || any(in(used), spec[1])) && continue
                union!(used, spec[1])
                push!(sites, (τ, spec))
            end
        end
        push!(rounds, sites)
    end
    return rounds
end

function sparse_cols(fam::Symbol, tp::Topo, site)
    T = tp.T
    G = gmax(tp)
    Ns, Dm = sparse_ns(fam, tp), sparse_dm(tp)
    pair = SPARSE_FAMILIES[fam][2]
    τ, (nodes, d, g, pq) = site
    dof = Dict(n => d[j] for (j, n) in enumerate(nodes))
    ln, cf, dd = zeros(Int, Ns * Dm), zeros(Int, Ns * Dm), zeros(Ns * Dm)
    for (j, n) in enumerate(nodes)
        q = 0
        for e in tp.nodelines[n]
            far = tp.ebus0[e] == n ? tp.ebus1[e] : tp.ebus0[e]
            Δ = get(dof, n, 0.0) - get(dof, far, 0.0)
            Δ == 0.0 && continue                                  # a line that does not move (as the CPU: df = 0)
            q += 1
            q <= Dm || error("sparse site: more moving lines at node $n than $Dm")
            o = (j - 1) * Dm + q
            ln[o], cf[o], dd[o] = e, colτ(T, τ, far), Δ
        end
    end
    cols = [gencols(tp, n, τ, G) for n in nodes]
    gq = vcat(reduce(vcat, [c[1] for c in cols]; init=Int[]), zeros(Int, (Ns - length(nodes)) * G))
    gcm = vcat(reduce(vcat, [c[2] for c in cols]; init=Int[]), zeros(Int, (Ns - length(nodes)) * G))
    gc = vcat(reduce(vcat, [c[3] for c in cols]; init=Int[]), zeros(Int, (Ns - length(nodes)) * G))
    gc1 = vcat(reduce(vcat, [c[4] for c in cols]; init=Int[]), zeros(Int, (Ns - length(nodes)) * G))
    if pair                                                       # the pair units first (columns 1, 2)
        q1, q2 = pq
        pc(q, x) = q == 0 ? 0 : colτ(T, τ + x, q)
        gq = vcat(q1, q2, gq); gcm = vcat(pc(q1, -1), pc(q2, -1), gcm); gc = vcat(pc(q1, 0), pc(q2, 0), gc)
        gc1 = vcat(pc(q1, 1), pc(q2, 1), gc1)
    end
    padf(v) = vcat(v, zeros(Ns - length(v)))
    return (cθ=padv([colτ(T, τ, n) for n in nodes], Ns), dv=padf(d), gv=padf(g), ln=ln, cf=cf, dd=dd,
            gq=gq, gcm=gcm, gc=gc, gc1=gc1)
end

for (f, (rule, pair, _)) in SPARSE_FAMILIES
    @eval begin
        family_rounds(::Val{$(QuoteNode(f))}, tp::Topo, M::DeviceModel, cls, rng) = sparse_rounds($(QuoteNode(f)), tp, M, cls, rng)
        site_cols(::Val{$(QuoteNode(f))}, tp::Topo, site, D::Int) = sparse_cols($(QuoteNode(f)), tp, site)
        device_extra(::Val{$(QuoteNode(f))}, ht) = (gon=Float64.(ht.gq .> 0), lon=Float64.(ht.ln .> 0))
        build(::Val{$(QuoteNode(f))}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) =
            build_sparse(st, cp, sp, tab, r, k, Val($(QuoteNode(rule))), Val($pair), $(QuoteNode(f)))
        commit_tb(::Val{$(QuoteNode(f))}, st, tab, r, t, dr, aux, tb) = commit_sparse(st, tab, r, t, dr, aux, tb)
        nroots(::Val{$(QuoteNode(f))}) = Val(6)
    end
    FOOT[f] = [(:cθ, :θ, true), (:cθ, :r, true), (:cf, :θ, false), (:gc, :ω, true), (:gc, :uc, true), (:gcm, :ω, false),
               (:gcm, :uc, false), (:gc1, :uc, false)]
end

function build_sparse(st, cp, sp, tab, r, k, ::Val{rule}, ::Val{pair}, fam::Symbol) where {rule,pair}
    Ns = size(tab.cθ, 2)
    Dm = size(tab.ln, 2) ÷ Ns
    np = pair ? 2 : 0
    G = (size(tab.gq, 2) - np) ÷ Ns
    on = m3(tab.on[:, r])
    cθ = tab.cθ[:, :, r]
    z = 0.0 .* g3(st.θ, cθ[:, 1])
    z4 = c4(z)
    q = tab.gq[:, :, r]
    gc, gcm, gc1 = tab.gc[:, :, r], tab.gcm[:, :, r], tab.gc1[:, :, r]
    gon = M4(tab.gon[:, :, r]) .> 0.0
    κ = G4(cp.κg, q)
    ok = c4(on .+ z) .> 0.0
    if pair                                                       # the pair is valid where both |κ| ≥ κmin (per chain)
        ok = ok .& (absd.(κ[:, :, :, 1:1]) .>= k.κmin) .& (absd.(κ[:, :, :, 2:2]) .>= k.κmin)
    end
    a4 = ifelse.(ok, 1.0, 0.0)
    θ = G4(st.θ, cθ)
    d = M4(tab.dv[:, :, r]) .* a4
    ln = tab.ln[:, :, r]
    near = slotmap(Dm, Ns)
    bl = S4(sp.lb, ln)
    line = sline(z4; f0=bl .* (θ[:, :, :, near] .- G4(st.θ, tab.cf[:, :, r])), df=bl .* M4(tab.dd[:, :, r]) .* a4,
                 s=S4(sp.ls, ln), α=S4(sp.lα, ln), near=near, far=zeros(Int, Ns * Dm), wc=fill(0.5, Ns * Dm))
    r0 = G4(st.r, cθ)
    slc = slotmap(G, Ns)                                          # node slot of every compensation column
    slot = vcat(pair ? [1, 2] : Int[], slc)
    gm = M4(tab.gv[:, :, r]) .* a4
    ci = np .+ (1:Ns*G)                                           # compensation columns
    xa = nothing                                                  # linear slopes of the pair columns
    if pair
        sa = ifelse.(ok, 1.0 ./ ifelse.(ok, κ[:, :, :, 1:2], 1.0), 0.0) .* reshape([1.0, -1.0], 1, 1, 1, 2)
        xa = sa
    end
    xk, xl, kk = nothing, nothing, nothing
    if rule === :T2
        ω = G4(st.ω, gc[:, ci])
        v = gon[:, :, :, ci] .& (absd.(κ[:, :, :, ci]) .>= k.κmin)
        dd_ = prorata4(κ[:, :, :, ci], ω, v, r0, gm, Ns)
        kk = skinks(z4; Q=dd_.Q, g=ifelse.(dd_.act, gm, 0.0), P=dd_.P, e=dd_.e, nm=dd_.nm, on=ifelse.(dd_.act, 1.0, 0.0))
        xk = ifelse.(dd_.act[:, :, :, slc] .& v, ω ./ ifelse.(dd_.act[:, :, :, slc], dd_.P[:, :, :, slc], 1.0), 0.0)
    elseif rule === :L
        β = βof4(cp, sp, q[:, ci], G4(cp.cls, cθ)[:, :, :, slc])
        gmc = gm[:, :, :, slc]
        v = ok .& gon[:, :, :, ci] .& (absd.(κ[:, :, :, ci]) .>= k.κmin) .& (β .!= 0.0) .& (gmc .!= 0.0)
        xl = ifelse.(v, β .* gmc ./ ifelse.(v, κ[:, :, :, ci], 1.0), 0.0)
    end
    # generator columns: the pair units (linear), then the compensation units (kinked or linear)
    zc(n) = 0.0 .* z4 .* zeros(1, 1, 1, n)
    lin = pair ? (rule === :L ? cat(xa, xl; dims=4) : cat(xa, zc(Ns * G); dims=4)) : xl
    kin = rule === :T2 ? (pair ? cat(zc(2), xk; dims=4) : xk) : nothing
    xall = rule === :none ? xa : rule === :T2 ? (pair ? cat(xa, xk; dims=4) : xk) : lin
    gen = nothing
    if rule === :none                                             # only the pair columns
        pair && (gen = gen4(st, cp, sp, z4, q[:, 1:2], gc[:, 1:2], [1, 2]; a=xa))
    else
        gen = gen4(st, cp, sp, z4, q, gc, slot; a=lin, ak=kin, kid=rule === :T2 ? vcat(pair ? [0, 0] : Int[], slc) : zeros(Int, length(slot)))
        mva, mvk = vcat(fill(pair, np), fill(rule === :L, Ns * G)), vcat(fill(false, np), fill(rule === :T2, Ns * G))
        gen = merge(gen, (mva=mva, mvk=mvk, cnt=Int.(mva) .+ Int.(mvk)))
    end
    nuse = rule === :none ? np : np + Ns * G
    uj = [j for j in 1:nuse if ucon(fam, j)]
    uc, ucq = nothing, nothing
    if !isempty(uj)
        ucq = ucflags(sp, q, uj, gon)
        xu = xall[:, :, :, uj]
        ku = vcat(pair ? [0, 0] : Int[], slc)[uj]
        sets = Any[]
        for (sel, lin_) in ((findall(<=(np), uj), true), (findall(>(np), uj), rule !== :T2))
            isempty(sel) && continue
            u_, x_, q_ = uj[sel], xu[:, :, :, sel], ucq[:, :, :, sel]
            push!(sets, (cy=gc[:, u_], y=x_, ky=ku[sel], cu=gc[:, u_], u=x_ .* q_, ku=ku[sel], cu1=gc1[:, u_], u1=nothing, ku1=nothing,
                         lin=lin_, q=q[:, u_], ucq=q_),
                        (cy=gcm[:, u_], y=nothing, ky=nothing, cu=gcm[:, u_], u=nothing, ku=nothing, cu1=gc[:, u_], u1=x_ .* q_,
                         ku1=ku[sel], lin=lin_, q=q[:, u_], ucq=q_))
        end
        uc = uc_concat([uc4s(st, cp, sp, z4, s1.q, s1.ucq, [s1, s2], rule === :T2) for (s1, s2) in zip(sets[1:2:end], sets[2:2:end])])
    end
    tb = TermBatchS(z; gen=gen, uc=uc, line=line, kk=kk, r0=r0)
    return tb, (a4, d, xall, ucq, uj, G, Ns, np, rule)
end

"Concatenate stacked commitment terms along the term axis (fields `nothing` as zeros)."
function uc_concat(us)
    length(us) == 1 && return us[1]
    out = Dict{Symbol,Any}()
    for f in keys(us[1])
        vs = [u[f] for u in us]
        if f in (:kidy, :kidu, :kidu1, :cy, :cu, :cu1)
            out[f] = reduce(vcat, vs)
        elseif all(v -> v === nothing, vs)
            out[f] = nothing
        else
            ref = first(v for v in vs if v !== nothing)
            out[f] = cat((v === nothing ? 0.0 .* us[j].y0 : v for (j, v) in enumerate(vs))...; dims=4)
        end
    end
    return NamedTuple{keys(us[1])}(Tuple(out[f] for f in keys(us[1])))
end

function commit_sparse(st, tab, r, t, dr, aux, tb)
    a4, d, xall, ucq, uj, G, Ns, np, rule = aux
    cθ = tab.cθ[:, :, r]
    θ = scadd4(st.θ, cθ, d .* c4(t))
    rr = scadd4(st.r, cθ, msk(dr4(dr), a4 .+ 0.0 .* d))
    ω, uc = st.ω, st.uc
    if xall !== nothing
        gc = tab.gc[:, :, r]
        nuse = size(xall, 4)
        kidc = vcat(fill(0, np), slotmap(G, Ns))[1:nuse]
        Δ = rule === :T2 ? xall .* cat(c4(t) .* ones(1, 1, 1, np), kinkφ4(tb, t)[:, :, :, slotmap(G, Ns)]; dims=4)[:, :, :, 1:nuse] :
            xall .* c4(t)
        ω = scadd4(st.ω, gc[:, 1:nuse], Δ)
        ucq === nothing || (uc = scadd4(st.uc, gc[:, uj], Δ[:, :, :, uj] .* ucq))
    end
    return FlatState(θ, ω, uc, st.e, st.u, rr)
end

# ---------------------------------------------------------------------------
# host: the sites, as the CPU preps
# ---------------------------------------------------------------------------

"""
Requirement changes of a sparse angle direction (the loop of `prep_tsparse!`): the lines of the
direction's nodes in order, each once, df = b (d_a − d_b) where |df| > 1e-14; returns the node
slots (the direction's nodes, then the other ends in first-touch order) and g per slot (kept where
|g| > 1e-12).
"""
function sparse_g(sys, LM, nodes::AbstractVector{Int}, vals::AbstractVector{Float64})
    d = Dict{Int,Float64}()
    for (n, a) in zip(nodes, vals); d[n] = a; end
    order = Int[]; gd = Dict{Int,Float64}()
    seen = Set{Int}()
    for n in nodes, e in LM.nodelines[n]
        e in seen && continue
        push!(seen, e)
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        df = sys.line_b[e] * (get(d, a, 0.0) - get(d, b, 0.0))
        abs(df) <= 1e-14 && continue
        haskey(gd, a) || (push!(order, a); gd[a] = 0.0)
        gd[a] += df
        haskey(gd, b) || (push!(order, b); gd[b] = 0.0)
        gd[b] -= df
    end
    slots = copy(collect(nodes))
    for m in order; m in slots || push!(slots, m); end
    return slots, [get(d, n, 0.0) for n in slots], [abs(get(gd, n, 0.0)) > 1e-12 ? gd[n] : 0.0 for n in slots]
end

"""
    sparse_source(P, LM, ix; atoms, rates) -> (fam, τ, rng) → site spec or nothing

Host source of the sparse families: atoms for `:tcut…` from `atoms()` (a vector of (H, pieces) pairs:
the frames' proxy pieces and the learned pieces, `localize`; a mode uniformly, its atom ∝ c²), with
the sites sorted into the small and wide buckets by their node slots; for `:xloc`, `:xlocc` a node n
uniformly, a partner of `LM.xpart[n]` uniformly, a generator unit at each (the CPU's `prep_xloc!`);
for `:xpair` two nodes of one component and a unit at each, with the global re-routing.
"""
function sparse_source(P::ML.Problem, LM, ix::DeviceIndex; atoms=() -> Any[])
    E = P.E
    sys = E.sys
    qof = Dict(ix.gidx[q] => q for q in eachindex(ix.gidx))
    unitq((f, i)) = get(qof, (findfirst(g -> g.vf == f, E.gens), i), 0)
    wide(n) = length(n) > SPARSE_NS
    function atom(rng)
        srcs = [a for a in atoms() if !isempty(a[2])]
        isempty(srcs) && return nothing
        H, pieces = srcs[rand(rng, eachindex(srcs))]
        pk = pieces[rand(rng, eachindex(pieces))]
        isempty(pk) && return nothing
        tot = sum(last, pk); u = rand(rng) * tot; acc = 0.0; j = pk[end][1]
        for (jj, w) in pk
            acc += w
            acc >= u && (j = jj; break)
        end
        rg = ML.nzrange(H.U, j)
        return sparse_g(sys, LM, ML.rowvals(H.U)[rg], ML.nonzeros(H.U)[rg])
    end
    return function (fam, τ, rng)
        if fam in (:tcut, :tcutw, :tcutL, :tcutLw)
            sp = atom(rng)
            (sp === nothing || wide(sp[1]) != (fam in (:tcutw, :tcutLw))) && return nothing
            return (sp..., nothing)
        elseif fam in (:xloc, :xlocc)
            n = rand(rng, 1:sys.N)
            pn = LM.xpart[n]
            isempty(pn) && return nothing
            j = rand(rng, eachindex(pn)); m = pn[j]
            (isempty(LM.units[n]) || isempty(LM.units[m])) && return nothing
            q1, q2 = unitq(rand(rng, LM.units[n])), unitq(rand(rng, LM.units[m]))
            (q1 == 0 || q2 == 0) && return nothing
            xn, xv, lk, lg = LM.xnod[n][j], LM.xval[n][j], LM.xlkn[n][j], LM.xlkg[n][j]
            slots = vcat(n, m, [x for x in xn if x != n && x != m])
            for x in lk; x in slots || push!(slots, x); end
            for e in 1:length(sys.line_s)                         # far ends of the moving lines
                a, b = sys.line_bus0[e], sys.line_bus1[e]
                (a in xn || b in xn) && (a in slots || push!(slots, a); b in slots || push!(slots, b))
            end
            dmap = Dict(zip(xn, xv)); gmap = fam == :xlocc ? Dict(zip(lk, lg)) : Dict{Int,Float64}()
            return (slots, [get(dmap, x, 0.0) for x in slots], [get(gmap, x, 0.0) for x in slots], (q1, q2))
        elseif fam == :xpair
            n, m = rand(rng, 1:sys.N), rand(rng, 1:sys.N)
            (m == n || LM.comp[m] != LM.comp[n]) && return nothing
            (isempty(LM.units[n]) || isempty(LM.units[m])) && return nothing
            u1, u2 = rand(rng, LM.units[n]), rand(rng, LM.units[m])
            u1 == u2 && return nothing
            q1, q2 = unitq(u1), unitq(u2)
            (q1 == 0 || q2 == 0) && return nothing
            x = zeros(sys.N)
            ML.reroute!(x, LM.GS, n, m)
            nodes = LM.GS.nodes[LM.comp[n]]
            slots = vcat(n, m, [i for i in nodes if i != n && i != m])
            return (slots, [abs(x[i]) < 1e-15 ? 0.0 : x[i] for i in slots], zeros(length(slots)), (q1, q2))
        end
        return nothing
    end
end
