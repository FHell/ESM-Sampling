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
# :tsocL, :tsoc
# ---------------------------------------------------------------------------

"β of unit q in the proxy class of (node, step) column cc."
βof(cp, sp, q, cc) = ifelse.(g3(cp.cls, cc) .> 0.5, s3(sp.β2, q), s3(sp.β1, q))

function build_tsoc(st, cp, sp, tab, r, k, ::Val{T2}) where {T2}
    on = m3(tab.on[:, r])
    s = tab.s[:, r]
    κs = g3(cp.ks, s) .* s3(sp.sh, s) ./ k.Δt
    okm = (on .> 0.0) .& (κs .>= k.κmin)
    ae = ifelse.(okm, 1.0, 0.0)
    G = size(tab.gq, 2)
    gens, ucs = Any[], Any[]
    am, at, aum, aut = Any[], Any[], Any[], Any[]                   # per unit: slopes (LINEAR) or kink coefficients (T2)
    kinks = nothing
    if T2
        dm = prorata_data(st, cp, k, tab, r, 1:G, :gcm, tab.crm[:, r], okm)
        dt = prorata_data(st, cp, k, tab, r, 1:G, :gc, tab.cr[:, r], okm)
        kinks = Kinks(Any[dm.Q, dt.Q], Any[ifelse.(dm.act, κs, 0.0), ifelse.(dt.act, .-κs, 0.0)], Any[dm.P, dt.P],
                      Any[dm.e, dt.e], Any[dm.nm, dt.nm], Any[ifelse.(dm.act, 1.0, 0.0), ifelse.(dt.act, 1.0, 0.0)])
    end
    for j in 1:G
        q = tab.gq[:, j, r]
        gon = m3(tab.gon[:, j, r])
        cgm, cg, cg1 = tab.gcm[:, j, r], tab.gc[:, j, r], tab.gc1[:, j, r]
        ucq = s3(sp.ucq, q) .* gon
        if T2
            xm = ifelse.(dm.act .& dm.valid[j], dm.ω[j] ./ ifelse.(dm.act, dm.P, 1.0), 0.0)   # σ/κ = ω/P
            xt = ifelse.(dt.act .& dt.valid[j], dt.ω[j] ./ ifelse.(dt.act, dt.P, 1.0), 0.0)
            push!(gens, gent(st, cp, sp, q, cgm, 1; ak=xm, kid=1), gent(st, cp, sp, q, cg, 2; ak=xt, kid=2))
            push!(ucs, uct(cp, sp, q; y0=g3(st.ω, cgm), aky=xm, kidy=1, u0=g3(st.uc, cgm), u10=g3(st.uc, cg), aku1=xt .* ucq, kidu1=2, on=ucq),
                       uct(cp, sp, q; y0=g3(st.ω, cg), aky=xt, kidy=2, u0=g3(st.uc, cg), aku=xt .* ucq, kidu=2, u10=g3(st.uc, cg1), on=ucq))
            push!(am, xm); push!(at, xt)
        else
            κ = g3(cp.κg, q)
            v = okm .& (gon .> 0.0) .& (absd.(κ) .>= k.κmin)
            βm, βt = βof(cp, sp, q, tab.ccm[:, r]), βof(cp, sp, q, tab.cc[:, r])
            xm = ifelse.(v .& (βm .!= 0.0), βm .* κs ./ ifelse.(v, κ, 1.0), 0.0)
            xt = ifelse.(v .& (βt .!= 0.0), .-βt .* κs ./ ifelse.(v, κ, 1.0), 0.0)
            push!(gens, gent(st, cp, sp, q, cgm, 1; a=xm), gent(st, cp, sp, q, cg, 2; a=xt))
            push!(ucs, uct(cp, sp, q; y0=g3(st.ω, cgm), ay=xm, u0=g3(st.uc, cgm), u10=g3(st.uc, cg), au1=xt .* ucq, on=ucq),
                       uct(cp, sp, q; y0=g3(st.ω, cg), ay=xt, u0=g3(st.uc, cg), au=xt .* ucq, u10=g3(st.uc, cg1), on=ucq))
            push!(am, xm); push!(at, xt)
        end
        push!(aut, at[end] .* ucq)
    end
    tb = TermBatch(mkgen(gens), mkuc(ucs), nothing, nothing, socterms(st, cp, sp, tab, r, ae),
                   Any[g3(st.r, tab.crm[:, r]), g3(st.r, tab.cr[:, r])], kinks)
    return tb, (ae, am, at, aut, T2)
end
build(::Val{:tsocL}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_tsoc(st, cp, sp, tab, r, k, Val(false))
build(::Val{:tsoc}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_tsoc(st, cp, sp, tab, r, k, Val(true))

"Change of a unit entry at the drawn t: a t (LINEAR) or ak φ_k(t) (T2)."
devΔ(x, t, φ, T2) = T2 ? x .* φ : x .* t

function commit_tsoc(st, tab, r, t, dr, aux, tb)
    ae, am, at, aut, T2 = aux
    φ = kinkφ(tb, t)
    φm, φt = T2 ? (φ[1], φ[2]) : (nothing, nothing)
    ω, uc = st.ω, st.uc
    for j in eachindex(am)
        ω = scadd(ω, tab.gcm[:, j, r], flat(devΔ(am[j], t, φm, T2)))
        ω = scadd(ω, tab.gc[:, j, r], flat(devΔ(at[j], t, φt, T2)))
        uc = scadd(uc, tab.gc[:, j, r], flat(devΔ(aut[j], t, φt, T2)))
    end
    e = scadd(st.e, tab.ce[:, r], flat(t .* ae))
    rr = scadd(scadd(st.r, tab.crm[:, r], flat(msk(dr[1], ae))), tab.cr[:, r], flat(msk(dr[2], ae)))
    return FlatState(st.θ, ω, uc, e, st.u, rr)
end

# ---------------------------------------------------------------------------
# :tpotL, :tpot
# ---------------------------------------------------------------------------

function build_tpot(st, cp, sp, tab, r, k, ::Val{T2}) where {T2}
    on = m3(tab.on[:, r])
    a = on
    cθ, cr = tab.cθ[:, r], tab.cr[:, r]
    θn = g3(st.θ, cθ)
    D = size(tab.ln, 2)
    G = size(tab.gq, 2) ÷ (D + 1)
    f0s, dfs, ss, αs, r0 = Any[], Any[], Any[], Any[], Any[g3(st.r, cr)]
    gm = Any[]                                                   # requirement change per node slot (lossless)
    gsum = a .* 0.0
    for i in 1:D
        ln = tab.ln[:, i, r]
        b = s3(sp.lb, ln) .* m3(tab.lon[:, i, r])
        push!(f0s, s3(sp.lb, ln) .* (θn .- g3(st.θ, tab.cθf[:, i, r])))
        push!(dfs, b .* a)
        push!(ss, s3(sp.ls, ln)); push!(αs, s3(sp.lα, ln))
        push!(r0, g3(st.r, tab.crf[:, i, r]))
        gsum = gsum .+ b
        push!(gm, .-b .* a)
    end
    pushfirst!(gm, gsum .* a)
    lt = LineTermsD(f0s, dfs, ss, αs, fill(1, D), collect(2:D+1))
    gens, ucs, xs, xus = Any[], Any[], Any[], Any[]
    kinks = nothing
    if T2
        ds = [prorata_data(st, cp, k, tab, r, (m-1)*G+1:m*G, :gc, m == 1 ? cr : tab.crf[:, m-1, r], (on .> 0.0) .& (gm[m] .!= 0.0))
              for m in 1:D+1]
        kinks = Kinks(Any[d.Q for d in ds], Any[ifelse.(ds[m].act, gm[m], 0.0) for m in 1:D+1], Any[d.P for d in ds],
                      Any[d.e for d in ds], Any[d.nm for d in ds], Any[ifelse.(d.act, 1.0, 0.0) for d in ds])
    end
    for m in 1:D+1, jj in 1:G
        j = (m - 1) * G + jj
        q = tab.gq[:, j, r]
        gon = m3(tab.gon[:, j, r])
        cg, cgm, cg1 = tab.gc[:, j, r], tab.gcm[:, j, r], tab.gc1[:, j, r]
        ucq = s3(sp.ucq, q) .* gon
        if T2
            d = ds[m]
            x = ifelse.(d.act .& d.valid[jj], d.ω[jj] ./ ifelse.(d.act, d.P, 1.0), 0.0)
            push!(gens, gent(st, cp, sp, q, cg, m; ak=x, kid=m))
            push!(ucs, uct(cp, sp, q; y0=g3(st.ω, cg), aky=x, kidy=m, u0=g3(st.uc, cg), aku=x .* ucq, kidu=m, u10=g3(st.uc, cg1), on=ucq),
                       uct(cp, sp, q; y0=g3(st.ω, cgm), u0=g3(st.uc, cgm), u10=g3(st.uc, cg), aku1=x .* ucq, kidu1=m, on=ucq))
        else
            κ = g3(cp.κg, q)
            β = βof(cp, sp, q, tab.cc[:, m, r])
            v = (on .> 0.0) .& (gon .> 0.0) .& (absd.(κ) .>= k.κmin) .& (β .!= 0.0) .& (gm[m] .!= 0.0)
            x = ifelse.(v, β .* gm[m] ./ ifelse.(v, κ, 1.0), 0.0)
            push!(gens, gent(st, cp, sp, q, cg, m; a=x))
            push!(ucs, uct(cp, sp, q; y0=g3(st.ω, cg), ay=x, u0=g3(st.uc, cg), au=x .* ucq, u10=g3(st.uc, cg1), on=ucq),
                       uct(cp, sp, q; y0=g3(st.ω, cgm), u0=g3(st.uc, cgm), u10=g3(st.uc, cg), au1=x .* ucq, on=ucq))
        end
        push!(xs, x); push!(xus, x .* ucq)
    end
    tb = TermBatch(mkgen(gens), mkuc(ucs), nothing, lt, nothing, r0, kinks)
    return tb, (a, xs, xus, G, T2)
end
build(::Val{:tpotL}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_tpot(st, cp, sp, tab, r, k, Val(false))
build(::Val{:tpot}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) = build_tpot(st, cp, sp, tab, r, k, Val(true))

function commit_tpot(st, tab, r, t, dr, aux, tb)
    a, xs, xus, G, T2 = aux
    φ = kinkφ(tb, t)
    rr = scadd(st.r, tab.cr[:, r], flat(msk(dr[1], a)))
    D = size(tab.ln, 2)
    for i in 1:D
        rr = scadd(rr, tab.crf[:, i, r], flat(msk(dr[1+i], a .* m3(tab.lon[:, i, r]))))
    end
    ω, uc = st.ω, st.uc
    for j in eachindex(xs)
        m = (j - 1) ÷ G + 1
        ω = scadd(ω, tab.gc[:, j, r], flat(devΔ(xs[j], t, T2 ? φ[m] : nothing, T2)))
        uc = scadd(uc, tab.gc[:, j, r], flat(devΔ(xus[j], t, T2 ? φ[m] : nothing, T2)))
    end
    return FlatState(scadd(st.θ, tab.cθ[:, r], flat(t .* a)), ω, uc, st.e, st.u, rr)
end

# commits that need the term batch (kinks at t): do_round passes it through `commit_tb`
commit_tb(f::Val, st, tab, r, t, dr, aux, tb) = commit(f, st, tab, r, t, dr, aux)
commit_tb(::Val{:tsoc}, st, tab, r, t, dr, aux, tb) = commit_tsoc(st, tab, r, t, dr, aux, tb)
commit_tb(::Val{:tsocL}, st, tab, r, t, dr, aux, tb) = commit_tsoc(st, tab, r, t, dr, aux, tb)
commit_tb(::Val{:tpot}, st, tab, r, t, dr, aux, tb) = commit_tpot(st, tab, r, t, dr, aux, tb)
commit_tb(::Val{:tpotL}, st, tab, r, t, dr, aux, tb) = commit_tpot(st, tab, r, t, dr, aux, tb)
