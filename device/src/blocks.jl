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

"Profiles of length L: the plateau, then the shape columns `shapes` ((L+1) × S, zero ends)."
function set_profiles!(L::Int, shapes=zeros(L + 1, 0))
    plat = vcat(0.0, ones(L))
    PROFILES[L] = hcat(plat, shapes)
    return PROFILES[L]
end
profiles(L) = get(PROFILES, L) do
    set_profiles!(L)
end

blksym(L, lin) = Symbol(lin ? :sblkL : :sblk, L)

"Rounds of block length L: offset o, stride L, blocks of one parity × storage slot j; a profile per site."
function block_rounds(L::Int, tp::Topo, rng; offset=nothing)
    T = tp.T
    o = offset === nothing ? rand(rng, 0:L-1) : offset
    starts = collect(1+o:L:T-L-1)                            # steps b0 … b0+L and the commitment term at b0−1
    starts = filter(b -> b >= 2, starts)
    Sm = maximum(length.(tp.nodestor); init=0)
    np = size(profiles(L), 2)
    rounds = Vector{Vector{Tuple}}()
    for j in 1:Sm, par in (1, 2)
        sites = Tuple[]
        for (bi, b0) in enumerate(starts)
            isodd(bi) == (par == 1) || continue
            for n in 1:tp.N
                length(tp.nodestor[n]) >= j || continue
                push!(sites, (tp.nodestor[n][j], b0, rand(rng, 1:np)))
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
             block_rounds($L, tp, rng))
        site_cols(::Val{$(QuoteNode(nm))}, tp::Topo, site, D::Int) = block_cols(tp, site, $L)
        device_extra(::Val{$(QuoteNode(nm))}, ht) = (gon=Float64.(ht.gq .> 0), W=copy(profiles($L)))
        build(::Val{$(QuoteNode(nm))}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) =
            build_block(st, cp, sp, tab, r, k, Val($L), Val($lin))
        commit(::Val{$(QuoteNode(nm))}, st::FlatState, tab, r, t, dr, aux) = error("block commits use commit_tb")
        commit_tb(::Val{$(QuoteNode(nm))}, st, tab, r, t, dr, aux, tb) = commit_block(st, tab, r, t, dr, aux, tb)
    end
    FOOT[nm] = [(:cew, :e, true), (:cer, :e, false), (:cr, :r, true), (:gw, :ω, true), (:gw, :uc, true),
                (:grd, :ω, false), (:grd, :uc, false)]
end

"Profile weights w_j (C|1, B, 1), j = 0 … L+1 (w_{L+1} = 0), of the sites' profile ids."
function block_weights(tab, r, L)
    pid = tab.p[:, r]
    W = tab.W                                                 # (L+1, P) static table (a device array)
    w = Any[]
    for j in 0:L
        push!(w, s3(W[j+1, :], pid))                          # gather the row's profile values
    end
    push!(w, w[1] .* 0.0)
    return w
end

function build_block(st, cp, sp, tab, r, k, ::Val{L}, ::Val{lin}) where {L,lin}
    on = m3(tab.on[:, r])
    s = tab.s[:, r]
    κs = g3(cp.ks, s) .* s3(sp.sh, s) ./ k.Δt
    okm = (on .> 0.0) .& (κs .>= k.κmin)
    w = [x .* ifelse.(on .> 0.0, 1.0, 0.0) for x in block_weights(tab, r, L)]
    G = size(tab.gq, 2)
    # SoC terms of steps b0+j (args e(b0+j), e(b0+j+1)) and balance slots
    kk, ww, h, ηi, ηo, qt, mc = g3(cp.ks, s), g3(cp.ws, s), s3(sp.sh, s), s3(sp.sηin, s), s3(sp.sηout, s), s3(sp.sqt, s), s3(sp.smc, s)
    e = [g3(st.e, tab.ce[:, j, r]) for j in 1:L+2]
    soc = SocTerms(Any[e[j+1] for j in 0:L], Any[w[j+1] for j in 0:L], Any[e[j+2] for j in 0:L], Any[w[j+2] for j in 0:L],
                   Any[kk for _ in 0:L], Any[ww for _ in 0:L], Any[h for _ in 0:L], Any[ηi for _ in 0:L], Any[ηo for _ in 0:L],
                   Any[qt for _ in 0:L], Any[mc for _ in 0:L], collect(1:L+1))
    r0 = Any[g3(st.r, tab.cr[:, j, r]) for j in 1:L+1]
    # requirement change of each step, compensation per step
    dg = [ifelse.(okm, κs .* (w[j+2] .- w[j+1]), 0.0) for j in 0:L]
    gc(j, jj) = tab.gcol[:, (j + 1) * G + jj, r]             # generator column of unit jj at step b0+j (j = −1 … L+1)
    x = [Any[] for _ in 0:L]                                  # per step, per unit: slope (LINEAR) or kink coefficient (T2)
    kinks = nothing
    if !lin
        ds = Any[]
        for j in 0:L
            P_, Qs, nu = on .* 0.0, on .* 0.0, on .* 0.0
            ωs, vs = Any[], Any[]
            for jj in 1:G
                q = tab.gq[:, jj, r]
                κ = g3(cp.κg, q)
                v = (m3(tab.gon[:, jj, r]) .> 0.0) .& (absd.(κ) .>= k.κmin)
                ω = g3(st.ω, gc(j, jj))
                P_ = P_ .+ ifelse.(v, κ .* ω, 0.0); Qs = Qs .+ ifelse.(v, κ .* clamp01.(ω), 0.0); nu = nu .+ ifelse.(v, 1.0, 0.0)
                push!(ωs, ω); push!(vs, v)
            end
            Q = r0[j+1] .+ Qs
            act = (dg[j+1] .!= 0.0) .& (nu .> 0.0) .& (absd.(P_) .>= 1e-6)
            push!(ds, (P=P_, Q=Q, e=P_ .- max.(Q, 0.0), nm=nu .- 1.0, act=act))
            for jj in 1:G
                push!(x[j+1], ifelse.(act .& vs[jj], ωs[jj] ./ ifelse.(act, P_, 1.0), 0.0))
            end
        end
        kinks = Kinks(Any[d.Q for d in ds], Any[ifelse.(d.act, dg[i], 0.0) for (i, d) in enumerate(ds)], Any[d.P for d in ds],
                      Any[d.e for d in ds], Any[d.nm for d in ds], Any[ifelse.(d.act, 1.0, 0.0) for d in ds])
    else
        for j in 0:L, jj in 1:G
            q = tab.gq[:, jj, r]
            κ = g3(cp.κg, q)
            β = βof(cp, sp, q, tab.cc[:, j+1, r])
            v = (m3(tab.gon[:, jj, r]) .> 0.0) .& (absd.(κ) .>= k.κmin) .& (β .!= 0.0) .& (dg[j+1] .!= 0.0)
            push!(x[j+1], ifelse.(v, β .* dg[j+1] ./ ifelse.(v, κ, 1.0), 0.0))
        end
    end
    gens, ucs, xs, xus = Any[], Any[], Any[], Any[]
    for jj in 1:G
        q = tab.gq[:, jj, r]
        ucq = s3(sp.ucq, q) .* m3(tab.gon[:, jj, r])
        for j in 0:L
            xj = x[j+1][jj]
            push!(gens, lin ? gent(st, cp, sp, q, gc(j, jj), j + 1; a=xj) : gent(st, cp, sp, q, gc(j, jj), j + 1; ak=xj, kid=j + 1))
        end
        for j in -1:L                                         # commitment term of step b0+j: (y_j, u_j, u_{j+1})
            ymv = 0 <= j <= L
            nxt = j + 1 <= L
            y0, u0, u10 = g3(st.ω, gc(j, jj)), g3(st.uc, gc(j, jj)), g3(st.uc, gc(j + 1, jj))
            xj = ymv ? x[j+1][jj] : 0.0
            xn = nxt ? x[j+2][jj] : 0.0
            if lin
                push!(ucs, uct(cp, sp, q; y0=y0, ay=xj, u0=u0, au=xj .* ucq, u10=u10, au1=xn .* ucq, on=ucq))
            else
                push!(ucs, uct(cp, sp, q; y0=y0, aky=xj, kidy=ymv ? j + 1 : 0, u0=u0, aku=xj .* ucq, kidu=ymv ? j + 1 : 0,
                               u10=u10, aku1=xn .* ucq, kidu1=nxt ? j + 2 : 0, on=ucq))
            end
        end
    end
    tb = TermBatch(mkgen(gens), mkuc(ucs), nothing, nothing, soc, r0, kinks)
    return tb, (on, w, x, G, lin, [s3(sp.ucq, tab.gq[:, jj, r]) .* m3(tab.gon[:, jj, r]) for jj in 1:G])
end

function commit_block(st, tab, r, t, dr, aux, tb)
    on, w, x, G, lin, ucq = aux
    L = length(x) - 1
    φ = kinkφ(tb, t)
    e = st.e
    for j in 1:L                                              # instants b0+1 … b0+L (w_0 = 0)
        e = scadd(e, tab.ce[:, j+1, r], flat(t .* w[j+1]))
    end
    ω, uc = st.ω, st.uc
    for j in 0:L, jj in 1:G
        col = tab.gcol[:, (j + 1) * G + jj, r]
        Δ = lin ? x[j+1][jj] .* t : x[j+1][jj] .* φ[j+1]
        ω = scadd(ω, col, flat(Δ))
        uc = scadd(uc, col, flat(Δ .* ucq[jj]))
    end
    rr = st.r
    for j in 1:L+1
        rr = scadd(rr, tab.cr[:, j, r], flat(msk(dr[j], on)))
    end
    return FlatState(st.θ, ω, uc, e, st.u, rr)
end
