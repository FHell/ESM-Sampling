# Phase 3, part 1: compensated transport-in-time and angle moves (docs/reactant_design.md §3.3, §3.5).
#
#   :spair  UNIT rule: e of one storage unit at instant τ rises by t; unit q1 at τ−1 takes κ_s/κ₁,
#           unit q2 at τ takes −κ_s/κ₂ (κ_s = k h/Δt), as `prep_spair!`. The units are chosen per
#           site on the host (state-independent).
#   :tsocL  LINEAR rule: as `prep_tsoc!` with learned coefficients (`compensate_learned!`): every
#           unit q at the node takes β_{q,cls}·(±κ_s)/κ_q at τ−1 (+) and τ (−).
#   :tsoc   T2 rule: as `prep_tsoc!`/`prorata!`: the units follow pro rata with one kink per step
#           (requirement Q, change ±κ_s), plus the Jacobian independence step.
#   :tpotL  LINEAR θ: as `prep_tpot!(…; learn)`: θ of node n, and at n and its neighbours m the
#           units take β·g_m/κ (g_m: lossless requirement change of the flows).
#   :tpot   T2 θ: as `prep_tpot!`: one kink per node of N[n], Jacobian step.
#
# Requirement-tracking kinks (line losses or lossy links: the CPU evaluates Q(t) = Q + Δr(t) and
# walks events) are not ported: with those terms the device uses the LINEAR rule (direction
# constant, no kink, no Jacobian); `:tsoc`/`:tpot` refuse models with `tracking` = true.
#
# The commitment of committable units follows at the main step τ (CPU `ws.follow`), not in the
# linked previous step (`wa.follow = false`).

const PHASE3 = (:spair, :tsocL, :tsoc, :tpotL, :tpot)

"Whether T2 kinks would have to track the requirement (CPU `tracking_kinks`)."
tracking(M::DeviceModel) = any(>(0.0), M.lα) || any(<(1.0), M.η)

# ---------------------------------------------------------------------------
# schedule and index tables
# ---------------------------------------------------------------------------

function family_rounds(::Val{fam}, tp::Topo, M::DeviceModel, cls, rng) where {fam}
    T = tp.T
    rounds = Vector{Vector{Tuple}}()
    Sm = maximum(length.(tp.nodestor); init=0)
    if fam == :spair || fam == :spair0
        # with :spair0 in the list: :spair0 takes the pairs of units without fleet commitment (no
        # commitment terms), :spair the pairs with at least one committable unit
        split = fam == :spair0 || :spair0 in SCHED_FAMILIES[]
        ucq(q) = M.guc[q] > 0
        for j in 1:Sm, c in cls
            sites = Tuple[]
            for τ in c, n in 1:tp.N
                (length(tp.nodestor[n]) >= j && !isempty(tp.nodegens[n])) || continue
                gs = tp.nodegens[n]
                prs = [(a, b) for a in gs for b in gs]
                split && (prs = filter(p -> (ucq(p[1]) | ucq(p[2])) == (fam == :spair), prs))
                isempty(prs) && continue
                push!(sites, (tp.nodestor[n][j], τ, rand(rng, prs)...))
            end
            push!(rounds, sites)
        end
    elseif fam == :tsoc || fam == :tsocL
        (fam == :tsoc && tracking(M)) && error(":tsoc needs a model without line losses and lossy links (use :tsocL)")
        for j in 1:Sm, c in cls
            push!(rounds, Tuple[(tp.nodestor[n][j], τ) for τ in c for n in 1:tp.N if length(tp.nodestor[n]) >= j])
        end
    elseif fam == :tpot || fam == :tpotL
        (fam == :tpot && tracking(M)) && error(":tpot needs a model without line losses and lossy links (use :tpotL)")
        nc = colour_d2(tp)
        for col in 1:maximum(nc; init=0), c in cls
            push!(rounds, Tuple[(n, τ) for τ in c for n in findall(==(col), nc) if !isempty(tp.nodelines[n])])
        end
    else
        error("unknown family $fam")
    end
    return rounds
end

gmax(tp::Topo) = maximum(length.(tp.nodegens); init=1)
padv(v, n) = vcat(v, zeros(Int, n - length(v)))

"Unit columns of the generators of node n (padded to G): units, τ−1, τ, τ+1 columns."
function gencols(tp::Topo, n, τ, G)
    T = tp.T
    qs = n == 0 ? Int[] : tp.nodegens[n]
    return padv(qs, G), padv([colτ(T, τ - 1, q) for q in qs], G), padv([colτ(T, τ, q) for q in qs], G),
           padv([colτ(T, τ + 1, q) for q in qs], G)
end

function site_cols(::Val{fam}, tp::Topo, site, D::Int) where {fam}
    T = tp.T
    G = gmax(tp)
    if fam == :spair || fam == :spair0
        s, τ, q1, q2 = site
        n = tp.snode[s]
        return (s=s, ce=colτ(T, τ, s), cem=colτ(T, τ - 1, s), ce1=colτ(T, τ + 1, s), crm=colτ(T, τ - 1, n), cr=colτ(T, τ, n),
                q1=q1, cwa=colτ(T, τ - 1, q1), cwa1=colτ(T, τ, q1),
                q2=q2, cwb=colτ(T, τ, q2), cwbm=colτ(T, τ - 1, q2), cwb1=colτ(T, τ + 1, q2), same=Int(q1 == q2))
    elseif fam == :tsoc || fam == :tsocL
        s, τ = site
        n = tp.snode[s]
        gq, gcm, gc, gc1 = gencols(tp, n, τ, G)
        return (s=s, ce=colτ(T, τ, s), cem=colτ(T, τ - 1, s), ce1=colτ(T, τ + 1, s), crm=colτ(T, τ - 1, n), cr=colτ(T, τ, n),
                gq=gq, gcm=gcm, gc=gc, gc1=gc1, ccm=colτ(T, τ - 1, n), cc=colτ(T, τ, n))
    elseif fam == :tpot || fam == :tpotL
        n, τ = site
        ls = tp.nodelines[n]
        far = [tp.ebus0[e] == n ? tp.ebus1[e] : tp.ebus0[e] for e in ls]
        nodes = padv(vcat(n, far), D + 1)
        cols = [gencols(tp, m, τ, G) for m in nodes]
        return (cθ=colτ(T, τ, n), cr=colτ(T, τ, n), ln=padv(ls, D), cθf=padv([colτ(T, τ, m) for m in far], D),
                crf=padv([colτ(T, τ, m) for m in far], D),
                gq=reduce(vcat, [c[1] for c in cols]), gcm=reduce(vcat, [c[2] for c in cols]), gc=reduce(vcat, [c[3] for c in cols]),
                gc1=reduce(vcat, [c[4] for c in cols]), cc=[m == 0 ? 0 : colτ(T, τ, m) for m in nodes])
    end
    error("unknown family $fam")
end

device_extra(::Val{:tsoc}, ht) = (gon=Float64.(ht.gq .> 0),)
device_extra(::Val{:tsocL}, ht) = (gon=Float64.(ht.gq .> 0),)
device_extra(::Val{:tpot}, ht) = (gon=Float64.(ht.gq .> 0), lon=Float64.(ht.ln .> 0))
device_extra(::Val{:tpotL}, ht) = (gon=Float64.(ht.gq .> 0), lon=Float64.(ht.ln .> 0))
device_extra(::Val{:spair}, ht) = (samef=Float64.(ht.same),)
device_extra(::Val{:spair0}, ht) = (samef=Float64.(ht.same),)

let socf = [(:ce, :e, true), (:cem, :e, false), (:ce1, :e, false), (:crm, :r, true), (:cr, :r, true)],
    genf = [(:gcm, :ω, true), (:gcm, :uc, false), (:gc, :ω, true), (:gc, :uc, true), (:gc1, :uc, false)],
    thf = [(:cθ, :θ, true), (:cθf, :θ, false), (:cr, :r, true), (:crf, :r, true)],
    potg = [(:gc, :ω, true), (:gc, :uc, true), (:gcm, :ω, false), (:gcm, :uc, false), (:gc1, :uc, false)]
    FOOT[:spair] = vcat(socf, [(:cwa, :ω, true), (:cwa, :uc, false), (:cwa1, :uc, false), (:cwb, :ω, true), (:cwb, :uc, true),
                               (:cwbm, :ω, false), (:cwbm, :uc, false), (:cwb1, :uc, false)])
    FOOT[:spair0] = vcat(socf, [(:cwa, :ω, true), (:cwb, :ω, true)])
    FOOT[:tsoc] = vcat(socf, genf)
    FOOT[:tsocL] = vcat(socf, genf)
    FOOT[:tpot] = vcat(thf, potg)
    FOOT[:tpotL] = vcat(thf, potg)
end

# ---------------------------------------------------------------------------
# term-batch assembly helpers
# ---------------------------------------------------------------------------

"A generator box term of unit q at column cw (balance slot `slot`), linear slope a and/or kink (ak, kid)."
gent(st, cp, sp, q, cw, slot; a=0.0, ak=0.0, kid=0) =
    (y0=g3(st.ω, cw), a=a, κ=g3(cp.κg, q), w=g3(cp.wg, q), mc=s3(sp.gmc, q), clo=s3(sp.clo, q), chi=s3(sp.chi, q),
     plo=g3(cp.plo, q), hv=s3(sp.hv, q), yk=g3(cp.yk, q), slot=slot, ak=ak, kid=kid)

"A commitment term of unit q: arguments (output y, commitment u, next commitment u1), each x0 + a t + ak φ_kid."
uct(cp, sp, q; y0, u0, u10, on, ay=0.0, aky=0.0, kidy=0, au=0.0, aku=0.0, kidu=0, au1=0.0, aku1=0.0, kidu1=0) =
    (y0=y0, ay=ay, u0=u0, au=au, u10=u10, au1=au1, κ=g3(cp.κg, q), w=g3(cp.wg, q), gmin=s3(sp.gmin, q), su=s3(sp.su, q),
     nl=s3(sp.nl, q), on=on, aky=aky, kidy=kidy, aku=aku, kidu=kidu, aku1=aku1, kidu1=kidu1)

col(L, f) = [getfield(x, f) for x in L]
mkgen(L) = isempty(L) ? nothing :
    GenTerms(Any[col(L, :y0)...], Any[col(L, :a)...], Any[col(L, :κ)...], Any[col(L, :w)...], Any[col(L, :mc)...], Any[col(L, :clo)...],
             Any[col(L, :chi)...], Any[col(L, :plo)...], Any[col(L, :hv)...], Any[col(L, :yk)...], Int[col(L, :slot)...],
             Any[col(L, :ak)...], Int[col(L, :kid)...])
mkuc(L) = isempty(L) ? nothing :
    UCTerms(Any[col(L, :y0)...], Any[col(L, :ay)...], Any[col(L, :u0)...], Any[col(L, :au)...], Any[col(L, :u10)...], Any[col(L, :au1)...],
            Any[col(L, :κ)...], Any[col(L, :w)...], Any[col(L, :gmin)...], Any[col(L, :su)...], Any[col(L, :nl)...], Any[col(L, :on)...],
            Any[col(L, :aky)...], Int[col(L, :kidy)...], Any[col(L, :aku)...], Int[col(L, :kidu)...], Any[col(L, :aku1)...], Int[col(L, :kidu1)...])

"The two SoC terms of a storage entry e_τ (slope ae), balance slots 1 (τ−1) and 2 (τ)."
function socterms(st, cp, sp, tab, r, ae)
    s = tab.s[:, r]
    e0, em, e1 = g3(st.e, tab.ce[:, r]), g3(st.e, tab.cem[:, r]), g3(st.e, tab.ce1[:, r])
    k, w = g3(cp.ks, s), g3(cp.ws, s)
    h, ηi, ηo, qt, mc = s3(sp.sh, s), s3(sp.sηin, s), s3(sp.sηout, s), s3(sp.sqt, s), s3(sp.smc, s)
    z = 0.0                     # static zero slope
    return SocTerms([em, e0], [z, ae], [e0, e1], [ae, z], [k, k], [w, w], [h, h], [ηi, ηi], [ηo, ηo], [qt, qt], [mc, mc], [1, 2])
end

"Pro-rata data of node slot j (units gq[:, idx]) at columns gcol for a requirement change g: (P, Q, e, nm, active, ω's)."
function prorata_data(st, cp, k, tab, r, idx, gcol, rcol, gsign_ok)
    P = 0.0; Qs = 0.0; nu = 0.0
    ωs = Any[]; valid = Any[]
    for j in idx
        q = tab.gq[:, j, r]
        κ = g3(cp.κg, q)
        v = (m3(tab.gon[:, j, r]) .> 0.0) .& (absd.(κ) .>= k.κmin)
        ω = g3(st.ω, tab[gcol][:, j, r])
        P = P .+ ifelse.(v, κ .* ω, 0.0)
        Qs = Qs .+ ifelse.(v, κ .* clamp01.(ω), 0.0)
        nu = nu .+ ifelse.(v, 1.0, 0.0)
        push!(ωs, ω); push!(valid, v)
    end
    Q = g3(st.r, rcol) .+ Qs
    act = gsign_ok .& (nu .> 0.0) .& (absd.(P) .>= 1e-6)
    return (P=P, Q=Q, e=P .- max.(Q, 0.0), nm=nu .- 1.0, act=act, ω=ωs, valid=valid)
end

# ---------------------------------------------------------------------------
# :spair
# ---------------------------------------------------------------------------

build(::Val{:spair}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_spair(st, cp, sp, tab, r, k, Val(true))
build(::Val{:spair0}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_spair(st, cp, sp, tab, r, k, Val(false))

function build_spair(st, cp, sp, tab, r, k, ::Val{withuc}) where {withuc}
    on = m3(tab.on[:, r])
    s, q1, q2 = tab.s[:, r], tab.q1[:, r], tab.q2[:, r]
    κs = g3(cp.ks, s) .* s3(sp.sh, s) ./ k.Δt
    κ1, κ2 = g3(cp.κg, q1), g3(cp.κg, q2)
    ok = (on .> 0.0) .& (κs .>= k.κmin) .& (absd.(κ1) .>= k.κmin) .& (absd.(κ2) .>= k.κmin)
    ae = ifelse.(ok, 1.0, 0.0)
    a1 = ifelse.(ok, κs ./ ifelse.(ok, κ1, 1.0), 0.0)
    a2 = ifelse.(ok, .-κs ./ ifelse.(ok, κ2, 1.0), 0.0)
    same = m3(tab.samef[:, r])
    ucq1, ucq2 = s3(sp.ucq, q1), s3(sp.ucq, q2)
    au2 = a2 .* ucq2                                         # the commitment of q2 follows at τ
    cwa, cwa1, cwb, cwbm, cwb1 = tab.cwa[:, r], tab.cwa1[:, r], tab.cwb[:, r], tab.cwbm[:, r], tab.cwb1[:, r]
    gens = [gent(st, cp, sp, q1, cwa, 1; a=a1), gent(st, cp, sp, q2, cwb, 2; a=a2)]
    ucs = !withuc ? Any[] :
          [uct(cp, sp, q1; y0=g3(st.ω, cwa), ay=a1, u0=g3(st.uc, cwa), u10=g3(st.uc, cwa1), au1=same .* au2, on=ucq1),
           uct(cp, sp, q2; y0=g3(st.ω, cwb), ay=a2, u0=g3(st.uc, cwb), au=au2, u10=g3(st.uc, cwb1), on=ucq2),
           uct(cp, sp, q2; y0=g3(st.ω, cwbm), u0=g3(st.uc, cwbm), u10=g3(st.uc, cwb), au1=au2, on=ucq2 .* (1.0 .- same))]
    withuc || (au2 = 0.0)
    tb = TermBatch(mkgen(gens), mkuc(ucs), nothing, nothing, socterms(st, cp, sp, tab, r, ae),
                   Any[g3(st.r, tab.crm[:, r]), g3(st.r, tab.cr[:, r])])
    return tb, (ae, a1, a2, au2)
end

commit(::Val{:spair0}, st::FlatState, tab, r, t, dr, aux) = commit(Val(:spair), st, tab, r, t, dr, aux)
function commit(::Val{:spair}, st::FlatState, tab, r, t, dr, aux)
    ae, a1, a2, au2 = aux
    ω = scadd(scadd(st.ω, tab.cwa[:, r], flat(t .* a1)), tab.cwb[:, r], flat(t .* a2))
    uc = au2 isa Number ? st.uc : scadd(st.uc, tab.cwb[:, r], flat(t .* au2))
    e = scadd(st.e, tab.ce[:, r], flat(t .* ae))
    rr = scadd(scadd(st.r, tab.crm[:, r], flat(msk(dr[1], ae))), tab.cr[:, r], flat(msk(dr[2], ae)))
    return FlatState(st.θ, ω, uc, e, st.u, rr)
end

# ---------------------------------------------------------------------------
# stacked unit terms (shared by the compensated, block and cell families)
# ---------------------------------------------------------------------------

"Static column → node-slot map of a site with `nslot` slots of G unit columns each."
slotmap(G, nslot) = [m for m in 1:nslot for _ in 1:G]

"""
Generator columns of the units q (B, n) at the state columns cw (B, n) (one gather per field), kinked
(T2: `ak`, `kid`) or linear (`a`), in balance slots `slot`.
"""
gen4(st, cp, sp, z4, q, cw, slot; a=nothing, ak=nothing, kid=zeros(Int, length(slot))) =
    sgen(z4, length(slot); y0=G4(st.ω, cw), a=a, κ=G4(cp.κg, q), w=G4(cp.wg, q), mc=S4(sp.gmc, q), clo=S4(sp.clo, q),
         chi=S4(sp.chi, q), plo=G4(cp.plo, q), hv=S4(sp.hv, q), yk=G4(cp.yk, q), ak=ak, kid=kid, slot=slot)

"""
Commitment columns of the units q (B, nU) (commitment flags ucq (1, B, 1, nU)), one set of nU terms per
entry of `sets`: each a NamedTuple of the state columns (B, nU) of the arguments y, u, u1 (`cy`, `cu`,
`cu1`), their slopes (`y`, `u`, `u1`: (C, B, 1, nU) or `nothing` for a static argument) and kink ids
(`ky`, `ku`, `ku1`: Int vectors; T2 only). T2: the slopes are kinked coefficients.
"""
function uc4s(st, cp, sp, z4, q, ucq, sets, T2::Bool)
    nU = size(q, 2)
    ns = length(sets)
    cols(f) = reduce(hcat2, [s[f] for s in sets])
    qq = reduce(hcat2, fill(q, ns))
    zx = 0.0 .* ucq .+ z4
    slope(f) = all(s[f] === nothing for s in sets) ? nothing : cat((s[f] === nothing ? zx : s[f] for s in sets)...; dims=4)
    cnt(f) = vcat((fill(s[f] === nothing ? 0 : 1, nU) for s in sets)...)
    kid(f, kf) = vcat((s[f] === nothing ? zeros(Int, nU) : s[kf] for s in sets)...)
    common = (y0=G4(st.ω, cols(:cy)), u0=G4(st.uc, cols(:cu)), u10=G4(st.uc, cols(:cu1)), κ=G4(cp.κg, qq), w=G4(cp.wg, qq),
              gmin=S4(sp.gmin, qq), su=S4(sp.su, qq), nl=S4(sp.nl, qq), on=cat(fill(ucq, ns)...; dims=4),
              cy=cnt(:y), cu=cnt(:u), cu1=cnt(:u1))
    T2 && return suc(z4, ns * nU; common..., aky=slope(:y), kidy=kid(:y, :ky), aku=slope(:u), kidu=kid(:u, :ku),
                     aku1=slope(:u1), kidu1=kid(:u1, :ku1))
    return suc(z4, ns * nU; common..., ay=slope(:y), au=slope(:u), au1=slope(:u1))
end

"Commitment flags (1, B, 1, nU) of the unit columns `uj` of q (B, n) with the column mask gon (1, B, 1, n)."
ucflags(sp, q, uj, gon) = S4(sp.ucq, q[:, uj]) .* ifelse.(gon[:, :, :, uj], 1.0, 0.0)

# ---------------------------------------------------------------------------
# :tsocL, :tsoc
# ---------------------------------------------------------------------------

"β of unit q in the proxy class of (node, step) column cc."
βof(cp, sp, q, cc) = ifelse.(g3(cp.cls, cc) .> 0.5, s3(sp.β2, q), s3(sp.β1, q))
"β of the units q (B, n) in the proxy classes cl (C, B, 1, n|1) of their (node, step): (C, B, 1, n)."
βof4(cp, sp, q, cl) = ifelse.(cl .> 0.5, S4(sp.β2, q), S4(sp.β1, q))

"The two SoC terms of a storage entry e_τ (slope ae), balance slots 1 (τ−1) and 2 (τ), stacked."
function socterms4(st, cp, sp, tab, r, ae, z4)
    s = tab.s[:, r]
    e0, em, e1 = c4(g3(st.e, tab.ce[:, r])), c4(g3(st.e, tab.cem[:, r])), c4(g3(st.e, tab.ce1[:, r]))
    two(x) = (y = c4(x); cat(y, y; dims=4))
    a4 = c4(ae)
    za = 0.0 .* a4
    return ssoc(z4, 2; e0=cat(em, e0; dims=4), ae=cat(za, a4; dims=4), e10=cat(e0, e1; dims=4), ae1=cat(a4, za; dims=4),
                k=two(g3(cp.ks, s)), w=two(g3(cp.ws, s)), h=two(s3(sp.sh, s)), ηin=two(s3(sp.sηin, s)), ηout=two(s3(sp.sηout, s)),
                qt=two(s3(sp.sqt, s)), mc=two(s3(sp.smc, s)), slot=[1, 2], mve=[false, true], mve1=[true, false])
end

function build_tsoc(st, cp, sp, tab, r, k, ::Val{T2}, fam::Symbol=T2 ? :tsoc : :tsocL) where {T2}
    on = m3(tab.on[:, r])
    s = tab.s[:, r]
    κs = g3(cp.ks, s) .* s3(sp.sh, s) ./ k.Δt
    okm = (on .> 0.0) .& (κs .>= k.κmin)
    ae = ifelse.(okm, 1.0, 0.0)
    z = 0.0 .* κs
    z4 = c4(z)
    q = tab.gq[:, :, r]
    G = size(q, 2)
    gon = M4(tab.gon[:, :, r]) .> 0.0
    qq = hcat2(q, q)
    gcm, gc, gc1 = tab.gcm[:, :, r], tab.gc[:, :, r], tab.gc1[:, :, r]
    cg = hcat2(gcm, gc)                                       # columns: the units at τ−1, then at τ
    κ = G4(cp.κg, qq)
    sl = slotmap(G, 2)
    r0 = cat(c4(g3(st.r, tab.crm[:, r])), c4(g3(st.r, tab.cr[:, r])); dims=4)
    soc = socterms4(st, cp, sp, tab, r, ae, z4)
    gon2 = cat(gon, gon; dims=4)
    if T2
        ω = G4(st.ω, cg)
        v = gon2 .& (absd.(κ) .>= k.κmin)
        κs4, ok4 = c4(κs), c4(okm)
        gs = ifelse.(ok4, cat(κs4, .-κs4; dims=4), 0.0)       # requirement change per unit t: +κs at τ−1, −κs at τ
        so = soc
        d = prorata4(κ, ω, v, r0, gs, 2; qadj=soc_qadj4(k.Δt, so.k, so.h, so.ηin, so.ηout, so.qt, so.e0, so.e10))
        kk = skinks(z4; Q=d.Q, g=ifelse.(d.act, gs, 0.0), P=d.P, e=d.e, nm=d.nm, on=ifelse.(d.act, 1.0, 0.0))
        x = ifelse.(d.act[:, :, :, sl] .& v, ω ./ ifelse.(d.act[:, :, :, sl], d.P[:, :, :, sl], 1.0), 0.0)   # σ/κ = ω/P
        gen = gen4(st, cp, sp, z4, qq, cg, sl; ak=x, kid=sl)
    else
        κG = κ[:, :, :, 1:G]
        v = c4(okm) .& gon .& (absd.(κG) .>= k.κmin)
        βm = βof4(cp, sp, q, c4(g3(cp.cls, tab.ccm[:, r])))
        βt = βof4(cp, sp, q, c4(g3(cp.cls, tab.cc[:, r])))
        κs4 = c4(κs)
        xm = ifelse.(v .& (βm .!= 0.0), βm .* κs4 ./ ifelse.(v, κG, 1.0), 0.0)
        xt = ifelse.(v .& (βt .!= 0.0), .-βt .* κs4 ./ ifelse.(v, κG, 1.0), 0.0)
        x = cat(xm, xt; dims=4)
        kk = nothing
        gen = gen4(st, cp, sp, z4, qq, cg, sl; a=x)
    end
    uj = [j for j in 1:G if ucon(fam, j)]                    # committable unit columns (static)
    uc, ucq = nothing, nothing
    if !isempty(uj)
        ucq = ucflags(sp, q, uj, gon)
        nU = length(uj)
        xm, xt = x[:, :, :, uj], x[:, :, :, G .+ uj]
        # the commitment of a unit follows at τ: term τ−1 (y_{τ−1}, u_{τ−1}, u_τ), term τ (y_τ, u_τ, u_{τ+1})
        sets = [(cy=gcm[:, uj], y=xm, ky=fill(1, nU), cu=gcm[:, uj], u=nothing, ku=nothing, cu1=gc[:, uj], u1=xt .* ucq, ku1=fill(2, nU)),
                (cy=gc[:, uj], y=xt, ky=fill(2, nU), cu=gc[:, uj], u=xt .* ucq, ku=fill(2, nU), cu1=gc1[:, uj], u1=nothing, ku1=nothing)]
        uc = uc4s(st, cp, sp, z4, q[:, uj], ucq, sets, T2)
    end
    tb = TermBatchS(z; gen=gen, uc=uc, soc=soc, kk=kk, r0=r0)
    return tb, (ae, x, cg, ucq, uj, G, T2)
end
build(::Val{:tsocL}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_tsoc(st, cp, sp, tab, r, k, Val(false))
build(::Val{:tsoc}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_tsoc(st, cp, sp, tab, r, k, Val(true))

"The balance-slot changes of a commit (a vector of (C, B, 1)) as (C, B, 1, R)."
dr4(dr) = (x = cat(dr...; dims=3); rs(x, size(x, 1), size(x, 2), 1, size(x, 3)))

"Change of the unit columns at the drawn t: x t (LINEAR) or x φ_kid(t) (T2), (C, B, 1, n)."
unitΔ(x, tb, t, kid, T2) = T2 ? x .* kinkφ4(tb, t)[:, :, :, kid] : x .* c4(t)

function commit_tsoc(st, tab, r, t, dr, aux, tb)
    ae, x, cg, ucq, uj, G, T2 = aux
    Δ = unitΔ(x, tb, t, slotmap(G, 2), T2)
    ω = scadd4(st.ω, cg, Δ)
    uc = ucq === nothing ? st.uc : scadd4(st.uc, tab.gc[:, uj, r], Δ[:, :, :, G .+ uj] .* ucq)
    e = scadd(st.e, tab.ce[:, r], flat(t .* ae))
    rr = scadd(scadd(st.r, tab.crm[:, r], flat(msk(dr[1], ae))), tab.cr[:, r], flat(msk(dr[2], ae)))
    return FlatState(st.θ, ω, uc, e, st.u, rr)
end

# ---------------------------------------------------------------------------
# :tpotL, :tpot
# ---------------------------------------------------------------------------

function build_tpot(st, cp, sp, tab, r, k, ::Val{T2}, fam::Symbol=T2 ? :tpot : :tpotL) where {T2}
    on = m3(tab.on[:, r])
    z = 0.0 .* g3(st.θ, tab.cθ[:, r])
    z4 = c4(z)
    a4 = c4(on .+ z)
    θn = c4(g3(st.θ, tab.cθ[:, r]))
    ln = tab.ln[:, :, r]
    D = size(ln, 2)
    G = size(tab.gq, 2) ÷ (D + 1)
    bl = S4(sp.lb, ln)
    b = bl .* M4(tab.lon[:, :, r])
    line = sline(z4; f0=bl .* (θn .- G4(st.θ, tab.cθf[:, :, r])), df=b .* a4, s=S4(sp.ls, ln), α=S4(sp.lα, ln),
                 near=fill(1, D), far=collect(2:D+1))
    r0 = cat(c4(g3(st.r, tab.cr[:, r])), G4(st.r, tab.crf[:, :, r]); dims=4)
    gm = cat(sum(b; dims=4) .* a4, .-b .* a4; dims=4)        # requirement change per node slot (lossless), (C, B, 1, D+1)
    q = tab.gq[:, :, r]
    gc, gcm, gc1 = tab.gc[:, :, r], tab.gcm[:, :, r], tab.gc1[:, :, r]
    gon = M4(tab.gon[:, :, r]) .> 0.0
    κ = G4(cp.κg, q)
    sl = slotmap(G, D + 1)
    if T2
        ω = G4(st.ω, gc)
        v = gon .& (absd.(κ) .>= k.κmin)
        d = prorata4(κ, ω, v, r0, gm, D + 1)
        kk = skinks(z4; Q=d.Q, g=ifelse.(d.act, gm, 0.0), P=d.P, e=d.e, nm=d.nm, on=ifelse.(d.act, 1.0, 0.0))
        x = ifelse.(d.act[:, :, :, sl] .& v, ω ./ ifelse.(d.act[:, :, :, sl], d.P[:, :, :, sl], 1.0), 0.0)
        gen = gen4(st, cp, sp, z4, q, gc, sl; ak=x, kid=sl)
    else
        β = βof4(cp, sp, q, G4(cp.cls, tab.cc[:, :, r])[:, :, :, sl])
        gmc = gm[:, :, :, sl]
        v = (a4 .> 0.0) .& gon .& (absd.(κ) .>= k.κmin) .& (β .!= 0.0) .& (gmc .!= 0.0)
        x = ifelse.(v, β .* gmc ./ ifelse.(v, κ, 1.0), 0.0)
        kk = nothing
        gen = gen4(st, cp, sp, z4, q, gc, sl; a=x)
    end
    uj = [j for j in 1:(D+1)*G if ucon(fam, j)]
    uc, ucq = nothing, nothing
    if !isempty(uj)
        ucq = ucflags(sp, q, uj, gon)
        nU = length(uj)
        xu = x[:, :, :, uj]
        ku = sl[uj]
        sets = [(cy=gc[:, uj], y=xu, ky=ku, cu=gc[:, uj], u=xu .* ucq, ku=ku, cu1=gc1[:, uj], u1=nothing, ku1=nothing),
                (cy=gcm[:, uj], y=nothing, ky=nothing, cu=gcm[:, uj], u=nothing, ku=nothing, cu1=gc[:, uj], u1=xu .* ucq, ku1=ku)]
        uc = uc4s(st, cp, sp, z4, q[:, uj], ucq, sets, T2)
    end
    tb = TermBatchS(z; gen=gen, uc=uc, line=line, kk=kk, r0=r0)
    return tb, (a4, x, ucq, uj, G, D, T2)
end
build(::Val{:tpotL}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_tpot(st, cp, sp, tab, r, k, Val(false))
build(::Val{:tpot}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_tpot(st, cp, sp, tab, r, k, Val(true))

function commit_tpot(st, tab, r, t, dr, aux, tb)
    a4, x, ucq, uj, G, D, T2 = aux
    inj = dr4(dr)
    mr = cat(a4, a4 .* M4(tab.lon[:, :, r]); dims=4)          # padded sites and lines change nothing
    cr = tab.cr[:, r]
    rr = scadd4(st.r, hcat2(rs(cr, length(cr), 1), tab.crf[:, :, r]), msk(inj, mr))
    gc = tab.gc[:, :, r]
    Δ = unitΔ(x, tb, t, slotmap(G, D + 1), T2)
    ω = scadd4(st.ω, gc, Δ)
    uc = ucq === nothing ? st.uc : scadd4(st.uc, gc[:, uj], Δ[:, :, :, uj] .* ucq)
    return FlatState(scadd(st.θ, tab.cθ[:, r], flat(t .* rs(a4, size(a4, 1), size(a4, 2), 1))), ω, uc, st.e, st.u, rr)
end

# commits that need the term batch (kinks at t): do_round passes it through `commit_tb`
commit_tb(f::Val, st, tab, r, t, dr, aux, tb) = commit(f, st, tab, r, t, dr, aux)
commit_tb(::Val{:tsoc}, st, tab, r, t, dr, aux, tb) = commit_tsoc(st, tab, r, t, dr, aux, tb)
commit_tb(::Val{:tsocL}, st, tab, r, t, dr, aux, tb) = commit_tsoc(st, tab, r, t, dr, aux, tb)
commit_tb(::Val{:tpot}, st, tab, r, t, dr, aux, tb) = commit_tpot(st, tab, r, t, dr, aux, tb)
commit_tb(::Val{:tpotL}, st, tab, r, t, dr, aux, tb) = commit_tpot(st, tab, r, t, dr, aux, tb)
