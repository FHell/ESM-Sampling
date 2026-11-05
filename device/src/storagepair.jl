# Storage pairs (CPU `prep_sxpair!`, `prep_sxlong!`, src/storage_cross.jl, docs/system/moves/storage.md):
# unit A's state of charge rises by t and unit B's (at A's node or a partner node of the re-routing
# templates) falls by ρ t, ρ = E_A/E_B; at the two end steps the nominal injection changes by ∓κ_A t at
# A and ±κ_A t at B, the flow between them is re-routed on the pair's ball template (angles c·Δθ),
# the leak at the ball's edge and the expected losses of the two units (`sx_qloss`: the loss share λ
# of a unit that only discharges) are compensated share-preservingly.
#
#   :sxpair, :sxpairn   single instant τ (steps τ−1 and τ), cross-node / same-node pair; the
#                       commitment follows at τ only (the CPU's `wa.follow = false`)
#   :sxlong, :sxlongn   cyclic interval a+1 … a+L (2 ≤ L ≤ T−1; steps a and b = a+L), as `:slong` for
#                       two units; the commitment follows at both end steps
#
# Template of a site: two end steps × node slots (A's node, B's node, the ball, the far ends of the
# moving lines, the leak nodes; the same-node families have the one node), line ends per slot (as
# `sparse.jl`), G compensation columns per slot and step, the two units' SoC terms (the end steps,
# and for `:sxlong` every other step of the cycle with its own balance slot per unit node). The
# sites come from a host source (the plan's `tp.ctx.sx`, `sx_source`): A uniformly among the units with at least
# the pair hours, B uniformly among the eligible units of A's partner slots (the CPU draws B
# ∝ min(k_A, k_B), a function of k; the device's choice is state-independent too), packed into rounds
# of disjoint node sets (per step window τ−2 … τ+1 for `:sxpair`, over the whole cycle for `:sxlong`). T2 only (models
# without requirement tracking).

"family → (interval, same node)"
const SX_FAMILIES = Dict(:sxpair => (false, false), :sxpairn => (false, true), :sxlong => (true, false), :sxlongn => (true, true))
const SX_NS = 16
"Rounds of `:sxlong`, `:sxlongn` per sweep (the default of a plan's `tp.ctx.sx_moves`)."
const SX_MOVES = Ref(8)
"Host source: (fam, τ or 0, rng) → (sA, sB, nodes, d, gleak) or nothing (`sx_source`); the default a plan made without `sources` copies."
const SX_SOURCE = Ref{Any}(nothing)

sx_ns(fam, tp) = SX_FAMILIES[fam][2] ? 1 : min(SX_NS, tp.N)

function sx_rounds(fam::Symbol, tp::Topo, M::DeviceModel, cls, rng)
    src = tp.ctx.sx
    src === nothing && return Vector{Vector{Tuple}}()
    tracking(M) && error(":$fam needs a model without line losses and lossy links")
    long, _ = SX_FAMILIES[fam]
    ns = sx_ns(fam, tp)
    T = tp.T
    rounds = Vector{Vector{Tuple}}()
    if !long
        for c in cls
            sites = Tuple[]
            used = Set{Tuple{Int,Int}}()       # (node, step): a site at τ writes τ−1, τ and reads τ−2 … τ+1
            for τ in c
                for _ in 1:tp.ctx.sparse_pack
                    sp = src(fam, τ, rng)
                    sp === nothing && continue
                    keys = [(n, mod1(τ + o, T)) for n in sp[3] for o in -2:1]
                    (length(sp[3]) > ns || any(in(used), keys)) && continue
                    union!(used, keys); push!(sites, (mod1(τ - 1, T), 1, sp))      # (a, L, spec): instant τ = a+1
                end
            end
            push!(rounds, sites)
        end
    else
        for _ in 1:tp.ctx.sx_moves
            sites = Tuple[]; used = Set{Int}()
            for _ in 1:2tp.N
                sp = src(fam, 0, rng)
                (sp === nothing || length(sp[3]) > ns || any(in(used), sp[3])) && continue
                union!(used, sp[3]); push!(sites, (rand(rng, 1:T), rand(rng, 2:T-1), sp))
            end
            push!(rounds, sites)
        end
    end
    return rounds
end

function sx_cols(fam::Symbol, tp::Topo, site)
    T = tp.T
    G = gmax(tp)
    long, same = SX_FAMILIES[fam]
    Ns, Dm = sx_ns(fam, tp), sparse_dm(tp)
    a, L, (sA, sB, nodes, d, gl) = site
    steps = long ? slong_steps(T, a, L) : [mod1(a, T), mod1(a + 1, T)]
    Tn = length(steps)
    raised(τ) = mod(τ - a - 1, T) < L
    dof = Dict(n => d[j] for (j, n) in enumerate(nodes))
    lnv, cfv, ddv = zeros(Int, 2Ns * Dm), zeros(Int, 2Ns * Dm), zeros(Ns * Dm)
    for (s, τ) in enumerate(steps[1:2]), (j, n) in enumerate(nodes)
        q = 0
        for e in tp.nodelines[n]
            far = tp.ebus0[e] == n ? tp.ebus1[e] : tp.ebus0[e]
            Δ = get(dof, n, 0.0) - get(dof, far, 0.0)
            Δ == 0.0 && continue
            q += 1
            q <= Dm || error("storage pair: more moving lines at node $n than $Dm")
            o = ((s - 1) * Ns + j - 1) * Dm + q
            lnv[o], cfv[o] = e, colτ(T, τ, far)
            s == 1 && (ddv[(j-1)*Dm+q] = Δ)
        end
    end
    gcs = [gencols(tp, n, τ, G) for τ in steps[1:2] for n in vcat(nodes, fill(0, Ns - length(nodes)))]
    nA, nB = tp.snode[sA], tp.snode[sB]
    padf(v) = vcat(v, zeros(Ns - length(v)))
    return (sA=sA, sB=sB, cθ=vcat(padv([colτ(T, steps[1], n) for n in nodes], Ns), padv([colτ(T, steps[2], n) for n in nodes], Ns)),
            dv=padf(d), gv=padf(gl), ln=lnv, cf=cfv, dd=ddv,
            gq=reduce(vcat, [c[1] for c in gcs]), gcm=reduce(vcat, [c[2] for c in gcs]), gc=reduce(vcat, [c[3] for c in gcs]),
            gc1=reduce(vcat, [c[4] for c in gcs]),
            ceA0=[colτ(T, τ, sA) for τ in steps], ceA1=[colτ(T, τ + 1, sA) for τ in steps],
            ceB0=[colτ(T, τ, sB) for τ in steps], ceB1=[colτ(T, τ + 1, sB) for τ in steps],
            wa=[Float64(raised(τ)) for τ in steps], wb=[Float64(raised(mod1(τ + 1, T))) for τ in steps],
            crA=[colτ(T, τ, nA) for τ in steps], crB=[colτ(T, τ, nB) for τ in steps],
            criA=[colτ(T, τ, nA) for τ in steps[3:end]], criB=same ? Int[] : [colτ(T, τ, nB) for τ in steps[3:end]],
            cew=vcat([raised(τ) ? colτ(T, τ, sA) : 0 for τ in steps], [raised(τ) ? colτ(T, τ, sB) : 0 for τ in steps]))
end

for (f, (long, same)) in SX_FAMILIES
    @eval begin
        family_rounds(::Val{$(QuoteNode(f))}, tp::Topo, M::DeviceModel, cls, rng) = sx_rounds($(QuoteNode(f)), tp, M, cls, rng)
        site_cols(::Val{$(QuoteNode(f))}, tp::Topo, site, D::Int) = sx_cols($(QuoteNode(f)), tp, site)
        device_extra(::Val{$(QuoteNode(f))}, ht, tp) = (gon=Float64.(ht.gq .> 0),)
        build(::Val{$(QuoteNode(f))}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) =
            build_sx(st, cp, sp, tab, r, k, Val($long), Val($same), $(QuoteNode(f)))
        commit_tb(::Val{$(QuoteNode(f))}, st, tab, r, t, dr, aux, tb) = commit_sx(st, tab, r, t, dr, aux, tb)
        nroots(::Val{$(QuoteNode(f))}) = Val(6)
    end
    FOOT[f] = [(:cθ, :θ, true), (:cθ, :r, true), (:cf, :θ, false), (:gc, :ω, true), (:gc, :uc, true), (:gcm, :ω, false),
               (:gcm, :uc, false), (:gc1, :uc, false), (:cew, :e, true), (:ceA0, :e, false), (:ceB0, :e, false),
               (:ceA1, :e, false), (:ceB1, :e, false), (:criA, :r, true), (:criB, :r, true)]
end

"Loss share λ of a storage unit (1 − η_out for a unit that only discharges), (C|1, B, 1, 1) of a unit table."
sx_lambda(sp, s) = c4(ifelse.(s3(sp.sηin, s) .> 0.0, 0.0, 1.0 .- s3(sp.sηout, s)))

function build_sx(st, cp, sp, tab, r, k, ::Val{long}, ::Val{same}, fam::Symbol) where {long,same}
    on = m3(tab.on[:, r])
    sA, sB = tab.sA[:, r], tab.sB[:, r]
    EA = g3(cp.ks, sA) .* s3(sp.sh, sA)
    EB = g3(cp.ks, sB) .* s3(sp.sh, sB)
    κA = EA ./ k.Δt
    okm = (on .> 0.0) .& (κA .>= k.κmin) .& (EB ./ k.Δt .>= k.κmin)
    z = 0.0 .* κA
    z4 = c4(z)
    m4 = c4(ifelse.(okm, 1.0, 0.0))
    ρ = c4(ifelse.(okm, EA ./ ifelse.(okm, EB, 1.0), 0.0))
    Ns = size(tab.cθ, 2) ÷ 2
    Dm = size(tab.dd, 2) ÷ Ns
    Tn = size(tab.ceA0, 2)
    G = size(tab.gq, 2) ÷ (2Ns)
    c4A = c4(κA)
    cs = (-c4A .* m4, c4A .* m4)                              # nominal change per unit t at A: step a, step b
    # SoC terms of both units at every column step (end steps a, b first)
    wa, wb = M4(tab.wa[:, :, r]) .* m4, M4(tab.wb[:, :, r]) .* m4
    rep(x) = c4(x) .+ zeros(1, 1, 1, Tn)                         # a broadcast, not a splat of Tn arguments (trace time)
    par(f) = cat(rep(f(sA)), rep(f(sB)); dims=4)
    slA = vcat(1, Ns + 1, 2Ns .+ (1:Tn-2))
    slB = same ? slA : vcat(2, Ns + 2, 2Ns + Tn - 2 .+ (1:Tn-2))
    soc = ssoc(z4, 2Tn; e0=cat(G4(st.e, tab.ceA0[:, :, r]), G4(st.e, tab.ceB0[:, :, r]); dims=4), ae=cat(wa, .-ρ .* wa; dims=4),
               e10=cat(G4(st.e, tab.ceA1[:, :, r]), G4(st.e, tab.ceB1[:, :, r]); dims=4), ae1=cat(wb, .-ρ .* wb; dims=4),
               k=par(s -> g3(cp.ks, s)), w=par(s -> g3(cp.ws, s)), h=par(s -> s3(sp.sh, s)), ηin=par(s -> s3(sp.sηin, s)),
               ηout=par(s -> s3(sp.sηout, s)), qt=par(s -> s3(sp.sqt, s)), mc=par(s -> s3(sp.smc, s)), slot=vcat(slA, slB))
    cθ = tab.cθ[:, :, r]
    rint = Tn > 2 ? (same ? G4(st.r, tab.crA[:, 3:Tn, r]) : cat(G4(st.r, tab.crA[:, 3:Tn, r]), G4(st.r, tab.crB[:, 3:Tn, r]); dims=4)) : nothing
    rend = G4(st.r, cθ)
    r0 = rint === nothing ? rend : cat(rend, rint; dims=4)
    # flows: the ball template times the nominal change, at both end steps
    θ = G4(st.θ, cθ)
    dv = M4(tab.dv[:, :, r])
    d = cat(dv .* cs[1], dv .* cs[2]; dims=4)                  # (C, B, 1, 2Ns)
    ln = tab.ln[:, :, r]
    near = slotmap(Dm, 2Ns)
    bl = S4(sp.lb, ln)
    dd = M4(tab.dd[:, :, r])
    line = sline(z4; f0=bl .* (θ[:, :, :, near] .- G4(st.θ, tab.cf[:, :, r])), df=bl .* cat(dd .* cs[1], dd .* cs[2]; dims=4),
                 s=S4(sp.ls, ln), α=S4(sp.lα, ln), near=near, far=zeros(Int, 2Ns * Dm), wc=fill(0.5, 2Ns * Dm))
    # requirement changes per (step, slot): leak, and the expected losses at A's and B's nodes
    λA, λB = sx_lambda(sp, sA), sx_lambda(sp, sB)
    e1(j) = reshape(Float64.((1:Ns) .== j), 1, 1, 1, Ns)
    gl = M4(tab.gv[:, :, r]) .+ (same ? (λA .- λB) .* e1(1) : λA .* e1(1) .- λB .* e1(2))
    gm = cat(gl .* cs[1], gl .* cs[2]; dims=4)
    # their requirement shifts (`sx_qloss`) from the units' SoC terms at the end steps
    function qloss(u, sidx, lam)                               # unit u's column of end step sidx
        c = (u - 1) * Tn + sidx
        kk_, h_, ηi, ηo, qt, e0, e1_ = (soc.k[:, :, :, c:c], soc.h[:, :, :, c:c], soc.ηin[:, :, :, c:c], soc.ηout[:, :, :, c:c],
                                         soc.qt[:, :, :, c:c], soc.e0[:, :, :, c:c], soc.e10[:, :, :, c:c])
        return .-soc_j.(soc_d.(k.Δt, e0, e1_, h_, qt), kk_, ηi, ηo) .+ (1.0 .- lam) .* kk_ .* h_ .* (e1_ .- e0) ./ k.Δt
    end
    zq = 0.0 .* z4 .* zeros(1, 1, 1, Ns)
    qa = [same ? (qloss(1, s, λA) .+ qloss(2, s, λB)) .* e1(1) .+ zq : qloss(1, s, λA) .* e1(1) .+ qloss(2, s, λB) .* e1(2) .+ zq for s in 1:2]
    qadj = cat(qa...; dims=4)
    q = tab.gq[:, :, r]
    gc, gcm, gc1 = tab.gc[:, :, r], tab.gcm[:, :, r], tab.gc1[:, :, r]
    gon = M4(tab.gon[:, :, r]) .> 0.0
    κ = G4(cp.κg, q)
    sl = slotmap(G, 2Ns)
    ω = G4(st.ω, gc)
    v = gon .& (absd.(κ) .>= k.κmin)
    dq = prorata4(κ, ω, v, rend, gm, 2Ns; qadj=qadj)
    kk = skinks(z4; Q=dq.Q, g=ifelse.(dq.act, gm, 0.0), P=dq.P, e=dq.e, nm=dq.nm, on=ifelse.(dq.act, 1.0, 0.0))
    x = ifelse.(dq.act[:, :, :, sl] .& v, ω ./ ifelse.(dq.act[:, :, :, sl], dq.P[:, :, :, sl], 1.0), 0.0)
    gen = gen4(st, cp, sp, z4, q, gc, sl; ak=x, kid=sl)
    nG = Ns * G
    uj = [j for j in 1:2nG if ucon(tab, j)]
    uc, ucq = nothing, nothing
    if !isempty(uj)
        ucq = ucflags(sp, q, uj, gon)
        xu, ku = x[:, :, :, uj], sl[uj]
        if long                                                # the commitment follows at both end steps
            sets = [(cy=gc[:, uj], y=xu, ky=ku, cu=gc[:, uj], u=xu .* ucq, ku=ku, cu1=gc1[:, uj], u1=nothing, ku1=nothing),
                    (cy=gcm[:, uj], y=nothing, ky=nothing, cu=gcm[:, uj], u=nothing, ku=nothing, cu1=gc[:, uj], u1=xu .* ucq, ku1=ku)]
            uc = uc4s(st, cp, sp, z4, q[:, uj], ucq, sets, true)
        else                                                   # at τ only: the step τ−1 columns move y, the step τ ones u
            j1 = [j for j in uj if j <= nG]; j2 = [j for j in uj if j > nG]
            parts = Any[]
            if !isempty(j1)                                    # term of step τ−1: y moves (kink at τ−1), u_τ follows step τ's columns
                q1 = ucflags(sp, q, j1, gon)
                push!(parts, uc4s(st, cp, sp, z4, q[:, j1], q1,
                                  [(cy=gcm[:, j1 .+ nG], y=x[:, :, :, j1], ky=sl[j1], cu=gcm[:, j1 .+ nG], u=nothing, ku=nothing,
                                    cu1=gc[:, j1 .+ nG], u1=x[:, :, :, j1 .+ nG] .* q1, ku1=sl[j1 .+ nG])], true))
            end
            if !isempty(j2)                                    # term of step τ: y and u move (kink at τ)
                q2 = ucflags(sp, q, j2, gon)
                push!(parts, uc4s(st, cp, sp, z4, q[:, j2], q2,
                                  [(cy=gc[:, j2], y=x[:, :, :, j2], ky=sl[j2], cu=gc[:, j2], u=x[:, :, :, j2] .* q2, ku=sl[j2],
                                    cu1=gc1[:, j2], u1=nothing, ku1=nothing)], true))
            end
            uc = uc_concat(parts)
            uj = j2; ucq = isempty(j2) ? nothing : ucflags(sp, q, j2, gon)
            isempty(j2) && (uj = Int[])
        end
    end
    tb = TermBatchS(z; gen=gen, uc=uc, line=line, soc=soc, kk=kk, r0=r0)
    return tb, (wa, ρ, d, x, ucq, uj, G, Ns, Tn, same, m4)
end

function commit_sx(st, tab, r, t, dr, aux, tb)
    wa, ρ, d, x, ucq, uj, G, Ns, Tn, same, m4 = aux
    t4 = c4(t)
    e = scadd4(scadd4(st.e, tab.ceA0[:, :, r], wa .* t4), tab.ceB0[:, :, r], .-ρ .* wa .* t4)
    cθ = tab.cθ[:, :, r]
    θ = scadd4(st.θ, cθ, d .* t4)
    inj = dr4(dr)
    rr = scadd4(st.r, cθ, msk(inj[:, :, :, 1:2Ns], m4 .+ 0.0 .* t4))
    if Tn > 2
        ni = Tn - 2
        rr = scadd4(rr, tab.crA[:, 3:Tn, r], msk(inj[:, :, :, 2Ns+1:2Ns+ni], m4 .+ 0.0 .* t4))
        same || (rr = scadd4(rr, tab.crB[:, 3:Tn, r], msk(inj[:, :, :, 2Ns+ni+1:2Ns+2ni], m4 .+ 0.0 .* t4)))
    end
    gc = tab.gc[:, :, r]
    Δ = x .* kinkφ4(tb, t)[:, :, :, slotmap(G, 2Ns)]
    ω = scadd4(st.ω, gc, Δ)
    uc = ucq === nothing ? st.uc : scadd4(st.uc, gc[:, uj], Δ[:, :, :, uj] .* ucq)
    return FlatState(θ, ω, uc, e, st.u, rr)
end

"""
    sx_source(P, LM, ix) -> (fam, τ, rng) → (sA, sB, nodes, d, gleak) or nothing

Host source of the storage pairs: A uniformly among the units with at least the pair hours
(`LM.sxunits`), a partner slot of A's node and B uniformly among its eligible units (same node for
`:sxpairn`/`:sxlongn`, a partner node otherwise); the node slots, the ball angles and the leak's
requirement changes of the template (`LM.sxnod`, `sxval`, `sxlkn`, `sxlkg`).
"""
function sx_source(P::ML.Problem, LM, ix::DeviceIndex)
    E = P.E
    sys = E.sys
    sof = Dict(ix.sidx[q] => q for q in eachindex(ix.sidx))
    return function (fam, τ, rng)
        isempty(LM.sxunits) && return nothing
        _, same = SX_FAMILIES[fam]
        si, i = rand(rng, LM.sxunits)
        n = E.stors[si].bus[i]
        js = same ? [1] : collect(2:length(LM.sxpart[n]))
        isempty(js) && return nothing
        j = rand(rng, js)
        m = LM.sxpart[n][j]
        cand = [u for u in LM.sxnodeu[m] if u != (si, i)]
        isempty(cand) && return nothing
        sj, jj = rand(rng, cand)
        sA, sB = sof[(si, i)], sof[(sj, jj)]
        same && return (sA, sB, [n], [0.0], [0.0])
        xn, xv, lk, lg = LM.sxnod[n][j], LM.sxval[n][j], LM.sxlkn[n][j], LM.sxlkg[n][j]
        slots = vcat(n, m, [x for x in xn if x != n && x != m])
        for e in 1:length(sys.line_s)
            a, b = sys.line_bus0[e], sys.line_bus1[e]
            (a in xn || b in xn) && (a in slots || push!(slots, a); b in slots || push!(slots, b))
        end
        for x in lk; x in slots || push!(slots, x); end
        dmap = Dict(zip(xn, xv)); gmap = Dict(zip(lk, lg))
        return (sA, sB, slots, [get(dmap, x, 0.0) for x in slots], [get(gmap, x, 0.0) for x in slots])
    end
end
