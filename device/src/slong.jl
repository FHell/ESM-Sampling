# Long-range storage moves (CPU `prep_slong!`, `storage_long_sweep!`, docs/system/moves/storage.md):
# `slong` raises the state of charge of one unit at the cyclic instants a+1 … a+L (energy from step a
# to step b = a+L), compensated at the two end steps at the unit's node (share-preserving with the
# nominal requirement `soc_qadj`, or the learned rule); `slevel` raises every instant (L = T, no
# compensation).
#
#   :slong   share-preserving (models without requirement tracking)
#   :slongL  linear
#
# A site holds every step of the cycle, in the order a, b, then the other steps from a+1 on: the SoC
# term of each step (its instants raised by the 0/1 profile of the site, host data `wa`, `wb`), the
# node's balance slot at each step (static slot k for the k-th column), and at the two end steps the
# node's G compensation columns (slots 1, 2) with their commitment terms. The level move uses the
# same template (slots 1, 2 at a and a+1, no compensation). The device draws L uniformly on 2 … T−1
# (the CPU on 1 … T−1: L = 1 is the `tsoc` move), a uniformly on 1 … T, one level move per unit and
# `SLONG_MOVES` plateaus, one storage slot per node and round, for the units with at least
# `SLONG_HOURS` hours of storage (κ_s ≥ κ_min per chain). Interior steps cost nothing while their
# instants stay in the box: their columns are exact zeros there, as the CPU's compact terms.

const SLONG_HOURS = Ref(24.0)
const SLONG_MOVES = Ref(4)

function slong_rounds(fam::Symbol, tp::Topo, M::DeviceModel, rng)
    (fam == :slong && tracking(M)) && error(":slong needs a model without line losses and lossy links (use :slongL)")
    T = tp.T
    T >= 4 || return Vector{Vector{Tuple}}()
    units = [[s for s in tp.nodestor[n] if M.sh[s] >= SLONG_HOURS[]] for n in 1:tp.N]
    Sm = maximum(length.(units); init=0)
    rounds = Vector{Vector{Tuple}}()
    for j in 1:Sm, mv in 0:SLONG_MOVES[]
        sites = Tuple[]
        for n in 1:tp.N
            length(units[n]) >= j || continue
            push!(sites, (units[n][j], rand(rng, 1:T), mv == 0 ? T : rand(rng, 2:T-1)))
        end
        push!(rounds, sites)
    end
    return rounds
end

"Relative step order of a site: a, b, then the other steps from a+1 on (cyclic); b = a+1 for the level move."
function slong_steps(T, a, L)
    b = L == T ? mod1(a + 1, T) : mod1(a + L, T)
    return vcat(a, b, [mod1(a + j, T) for j in 1:T-1 if mod1(a + j, T) != b])
end

function slong_cols(tp::Topo, site)
    T = tp.T
    s, a, L = site
    n = tp.snode[s]
    G = gmax(tp)
    steps = slong_steps(T, a, L)
    raised(τ) = L == T || mod(τ - a - 1, T) < L                # instant τ ∈ {a+1, …, a+L}
    gq, gcm, gc, gc1 = gencols(tp, n, steps[1], G)
    _, gcmb, gcb, gc1b = gencols(tp, n, steps[2], G)
    return (s=s, lev=Float64(L == T), ce0=[colτ(T, τ, s) for τ in steps], ce1=[colτ(T, τ + 1, s) for τ in steps],
            wa=[Float64(raised(τ)) for τ in steps], wb=[Float64(raised(mod1(τ + 1, T))) for τ in steps],
            cr=[colτ(T, τ, n) for τ in steps], gq=vcat(gq, gq), gcm=vcat(gcm, gcmb), gc=vcat(gc, gcb), gc1=vcat(gc1, gc1b),
            cc=[colτ(T, steps[1], n), colτ(T, steps[2], n)])
end

for (f, lin) in ((:slong, false), (:slongL, true))
    @eval begin
        family_rounds(::Val{$(QuoteNode(f))}, tp::Topo, M::DeviceModel, cls, rng) = slong_rounds($(QuoteNode(f)), tp, M, rng)
        site_cols(::Val{$(QuoteNode(f))}, tp::Topo, site, D::Int) = slong_cols(tp, site)
        device_extra(::Val{$(QuoteNode(f))}, ht) = (gon=Float64.(ht.gq .> 0),)
        build(::Val{$(QuoteNode(f))}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) =
            build_slong(st, cp, sp, tab, r, k, Val($lin), $(QuoteNode(f)))
        commit_tb(::Val{$(QuoteNode(f))}, st, tab, r, t, dr, aux, tb) = commit_slong(st, tab, r, t, dr, aux, tb)
    end
    FOOT[f] = [(:ce0, :e, true), (:cr, :r, true), (:gc, :ω, true), (:gc, :uc, true), (:gcm, :ω, false), (:gcm, :uc, false),
               (:gc1, :uc, false)]
end

function build_slong(st, cp, sp, tab, r, k, ::Val{lin}, fam::Symbol) where {lin}
    on = m3(tab.on[:, r])
    s = tab.s[:, r]
    κs = g3(cp.ks, s) .* s3(sp.sh, s) ./ k.Δt
    okm = (on .> 0.0) .& (κs .>= k.κmin)
    z = 0.0 .* κs
    z4 = c4(z)
    m4 = c4(ifelse.(okm, 1.0, 0.0))
    T = size(tab.ce0, 2)
    wa, wb = M4(tab.wa[:, :, r]) .* m4, M4(tab.wb[:, :, r]) .* m4
    rep(x) = c4(x) .+ zeros(1, 1, 1, T)                          # a broadcast: a splat of T arguments takes minutes to trace
    soc = ssoc(z4, T; e0=G4(st.e, tab.ce0[:, :, r]), ae=wa, e10=G4(st.e, tab.ce1[:, :, r]), ae1=wb,
               k=rep(g3(cp.ks, s)), w=rep(g3(cp.ws, s)), h=rep(s3(sp.sh, s)), ηin=rep(s3(sp.sηin, s)), ηout=rep(s3(sp.sηout, s)),
               qt=rep(s3(sp.sqt, s)), mc=rep(s3(sp.smc, s)), slot=collect(1:T))
    r0 = G4(st.r, tab.cr[:, :, r])
    lev = m3(tab.lev[:, r]) .> 0.0
    comp = c4(okm .& .!lev)
    κs4 = c4(κs)
    dg = ifelse.(comp, cat(κs4, .-κs4; dims=4), 0.0)          # +κs at step a, −κs at step b
    q = tab.gq[:, :, r]
    G = size(q, 2) ÷ 2
    gc, gcm, gc1 = tab.gc[:, :, r], tab.gcm[:, :, r], tab.gc1[:, :, r]
    gon = M4(tab.gon[:, :, r]) .> 0.0
    κ = G4(cp.κg, q)
    sl = slotmap(G, 2)
    if !lin
        ω = G4(st.ω, gc)
        v = gon .& (absd.(κ) .>= k.κmin)
        sq = (k=soc.k[:, :, :, 1:2], h=soc.h[:, :, :, 1:2], ηin=soc.ηin[:, :, :, 1:2], ηout=soc.ηout[:, :, :, 1:2], qt=soc.qt[:, :, :, 1:2],
              e0=soc.e0[:, :, :, 1:2], e10=soc.e10[:, :, :, 1:2])
        d = prorata4(κ, ω, v, r0[:, :, :, 1:2], dg, 2; qadj=soc_qadj4(k.Δt, sq.k, sq.h, sq.ηin, sq.ηout, sq.qt, sq.e0, sq.e10))
        kk = skinks(z4; Q=d.Q, g=ifelse.(d.act, dg, 0.0), P=d.P, e=d.e, nm=d.nm, on=ifelse.(d.act, 1.0, 0.0))
        x = ifelse.(d.act[:, :, :, sl] .& v, ω ./ ifelse.(d.act[:, :, :, sl], d.P[:, :, :, sl], 1.0), 0.0)
        gen = gen4(st, cp, sp, z4, q, gc, sl; ak=x, kid=sl)
    else
        β = βof4(cp, sp, q, G4(cp.cls, tab.cc[:, :, r])[:, :, :, sl])
        dgc = dg[:, :, :, sl]
        v = gon .& (absd.(κ) .>= k.κmin) .& (β .!= 0.0) .& (dgc .!= 0.0)
        x = ifelse.(v, β .* dgc ./ ifelse.(v, κ, 1.0), 0.0)
        kk = nothing
        gen = gen4(st, cp, sp, z4, q, gc, sl; a=x)
    end
    uj = [j for j in 1:2G if ucon(fam, j)]
    uc, ucq = nothing, nothing
    if !isempty(uj)
        ucq = ucflags(sp, q, uj, gon)
        xu = x[:, :, :, uj]; ku = sl[uj]
        # the commitment follows at the end steps: terms of the end step (y, u moving) and of the step before (u1 moving)
        sets = [(cy=gc[:, uj], y=xu, ky=ku, cu=gc[:, uj], u=xu .* ucq, ku=ku, cu1=gc1[:, uj], u1=nothing, ku1=nothing),
                (cy=gcm[:, uj], y=nothing, ky=nothing, cu=gcm[:, uj], u=nothing, ku=nothing, cu1=gc[:, uj], u1=xu .* ucq, ku1=ku)]
        uc = uc4s(st, cp, sp, z4, q[:, uj], ucq, sets, !lin)
    end
    tb = TermBatchS(z; gen=gen, uc=uc, soc=soc, kk=kk, r0=r0)
    return tb, (wa, x, ucq, uj, G, lin, m4)
end

function commit_slong(st, tab, r, t, dr, aux, tb)
    wa, x, ucq, uj, G, lin, m4 = aux
    e = scadd4(st.e, tab.ce0[:, :, r], wa .* c4(t))             # every raised instant once (the start instant of its step)
    gc = tab.gc[:, :, r]
    Δ = unitΔ(x, tb, t, slotmap(G, 2), !lin)
    ω = scadd4(st.ω, gc, Δ)
    uc = ucq === nothing ? st.uc : scadd4(st.uc, gc[:, uj], Δ[:, :, :, uj] .* ucq)
    rr = scadd4(st.r, tab.cr[:, :, r], msk(dr4(dr), m4 .+ 0.0 .* c4(t)))
    return FlatState(st.θ, ω, uc, e, st.u, rr)
end
