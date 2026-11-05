# Learned dispatch shapes (CPU `prep_adshape!`, `adaptive_block_sweep!`, docs/system/moves/learned.md):
# two generator units at one node change by +φ_j t/κ₁ and −φ_j t/κ₂ at the steps b0+j−1, j = 1 … L,
# φ a learned dispatch shape of the block's season class: a balance-neutral re-dispatch with an
# intra-day or multi-day profile. No compensation; the commitment of committable units follows.
#
# Families `:dshp{L}` (L = 2, 4, 8, 16, 32): rounds of blocks of one parity (start offset, stride L)
# × one unit pair per node with at least two complete generator units; the pair and the profile per
# site drawn on the host. Terms of a site: 2L generator columns (L balance slots, the node at each
# step) and the commitment terms of the committable units at the L + 1 steps b0−1 … b0+L−1.

const DSHAPE_LENGTHS = (2, 4, 8, 16, 32)
"Module-level default tables of dispatch shapes, L → (L, P) (columns), and their season classes (0: any); a plan copies them (`sched_ctx`)."
const DSHAPES = Dict{Int,Matrix{Float64}}()
const DSHAPE_SEASON = Dict{Int,Vector{Int}}()
dshsym(L) = Symbol(:dshp, L)

"The default dispatch shapes of length L: the columns of `Φ` (L × P), with their season classes."
function set_dshapes!(L::Int, Φ::AbstractMatrix; seasons=zeros(Int, size(Φ, 2)))
    DSHAPES[L] = Matrix{Float64}(Φ)
    DSHAPE_SEASON[L] = collect(seasons)
    return DSHAPES[L]
end
"""
Learned dispatch shapes of AM (`AdaptiveModes.shapes[(:gen, L, season)]`), `nper` per season (zeros
until learned): the table (Φ, seasons) of length L, of a fixed shape (an upload when relearned).
"""
function learned_dshape_table(L::Int, AM=nothing; nper::Int=2)
    cols, seas = Any[], Int[]
    for cl in 1:4
        Φ = AM === nothing ? nothing : get(AM.shapes, (:gen, L, cl), nothing)
        for j in 1:nper
            push!(cols, Φ !== nothing && j <= size(Φ, 2) ? Φ[:, j] : zeros(L)); push!(seas, cl)
        end
    end
    return (Matrix{Float64}(hcat(cols...)), seas)
end
"`learned_dshape_table` as the default dispatch shapes of length L."
function set_learned_dshapes!(L::Int, AM=nothing; nper::Int=2)
    Φ, seas = learned_dshape_table(L, AM; nper=nper)
    return set_dshapes!(L, Φ; seasons=seas)
end
"The default dispatch shapes of length L (zeros until set)."
dshapes(L) = get(DSHAPES, L) do
    set_learned_dshapes!(L)
end
"The default entry (Φ, seasons) of length L, a copy (`sched_ctx`)."
dshape_entry(L) = (copy(dshapes(L)), copy(get(DSHAPE_SEASON, L, zeros(Int, size(dshapes(L), 2)))))
"The dispatch shapes Φ of length L of a plan (`tp.ctx.dshapes`)."
dshapes(tp::Topo, L) = tp.ctx.dshapes[L][1]

"The default generator-unit pairs of every node (device units, distinct; `nothing`: every pair of a node's units)."
const DSHAPE_PAIRS = Ref{Any}(nothing)

function dshape_rounds(L::Int, tp::Topo, M::DeviceModel, rng)
    T = tp.T
    o = rand(rng, 0:L-1)
    starts = filter(b -> b >= 2 && b + L <= T, collect(1+o:L:T-L))
    np = size(dshapes(tp, L), 2)
    seas = tp.ctx.dshapes[L][2]
    h = round(Int, M.Δt)
    compat(b0) = findall(s -> s == 0 || s == ML.season(b0, h), seas)
    prs = tp.ctx.dpairs === nothing ? [[(a, b) for a in g for b in g if a != b] for g in tp.nodegens] : tp.ctx.dpairs
    rounds = Vector{Vector{Tuple}}()
    for par in (1, 2)
        sites = Tuple[]
        for (bi, b0) in enumerate(starts)
            isodd(bi) == (par == 1) || continue
            cp_ = compat(b0)
            isempty(cp_) && continue
            for n in 1:tp.N
                isempty(prs[n]) && continue
                push!(sites, (rand(rng, prs[n])..., b0, rand(rng, cp_)))
            end
        end
        push!(rounds, sites)
    end
    return rounds
end

function dshape_cols(tp::Topo, site, L::Int)
    T = tp.T
    q1, q2, b0, p = site
    n = tp.gnode[q1]
    cw(q) = [colτ(T, b0 + j, q) for j in -1:L]                   # steps b0−1 … b0+L
    return (q1=q1, q2=q2, p=p, cw1=cw(q1), cw2=cw(q2), cr=[colτ(T, b0 + j, n) for j in 0:L-1],
            gq=[q1, q2], cww=vcat(cw(q1)[2:L+1], cw(q2)[2:L+1]), cwr=vcat(cw(q1)[[1, L+2]], cw(q2)[[1, L+2]]))
end

for L in DSHAPE_LENGTHS
    nm = dshsym(L)
    @eval begin
        family_rounds(::Val{$(QuoteNode(nm))}, tp::Topo, M::DeviceModel, cls, rng) = dshape_rounds($L, tp, M, rng)
        site_cols(::Val{$(QuoteNode(nm))}, tp::Topo, site, D::Int) = dshape_cols(tp, site, $L)
        device_extra(::Val{$(QuoteNode(nm))}, ht, tp) = (W=copy(tp === nothing ? dshapes($L) : dshapes(tp, $L)),)
        build(::Val{$(QuoteNode(nm))}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) =
            build_dshape(st, cp, sp, tab, r, k, Val($L), $(QuoteNode(nm)))
        commit_tb(::Val{$(QuoteNode(nm))}, st, tab, r, t, dr, aux, tb) = commit_dshape(st, tab, r, t, dr, aux, tb)
    end
    FOOT[nm] = [(:cww, :ω, true), (:cww, :uc, true), (:cwr, :ω, false), (:cwr, :uc, false), (:cr, :r, true)]
end

function build_dshape(st, cp, sp, tab, r, k, ::Val{L}, fam::Symbol) where {L}
    on = m3(tab.on[:, r])
    q1, q2 = tab.q1[:, r], tab.q2[:, r]
    κ1, κ2 = g3(cp.κg, q1), g3(cp.κg, q2)
    z = 0.0 .* κ1
    z4 = c4(z)
    ok = (on .> 0.0) .& (absd.(κ1) .>= k.κmin) .& (absd.(κ2) .>= k.κmin)
    φ = M4(permutedims(tab.W[:, tab.p[:, r]]))                   # (1, B, 1, L)
    o4 = c4(ok)
    x1 = ifelse.(o4, φ ./ ifelse.(o4, c4(κ1), 1.0), 0.0)
    x2 = ifelse.(o4, .-φ ./ ifelse.(o4, c4(κ2), 1.0), 0.0)
    cw1, cw2 = tab.cw1[:, :, r], tab.cw2[:, :, r]                # (B, L+2): steps b0−1 … b0+L
    rep(q) = reduce(hcat2, fill(rs(q, length(q), 1), L))
    qq = hcat2(rep(q1), rep(q2))
    cws = hcat2(cw1[:, 2:L+1], cw2[:, 2:L+1])
    x = cat(x1, x2; dims=4)
    gen = gen4(st, cp, sp, z4, qq, cws, vcat(1:L, 1:L); a=x)
    gq = tab.gq[:, :, r]                                         # (B, 2) for the committable flags
    uj = [j for j in 1:2 if ucon(tab, j)]
    uc, ucq = nothing, nothing
    if !isempty(uj)
        ucqs = [ucflags(sp, gq, [j], M4(ones(size(gq, 1), 2)) .> 0.0) for j in 1:2]
        xs = (x1, x2); cws_ = (cw1, cw2)
        parts = Any[]
        for j in uj
            xj, cj, qj = xs[j], cws_[j], gq[:, j:j]
            sets = [(cy=cj[:, s+2:s+2], y=0 <= s ? xj[:, :, :, s+1:s+1] : nothing, ky=nothing,
                     cu=cj[:, s+2:s+2], u=0 <= s ? xj[:, :, :, s+1:s+1] .* ucqs[j] : nothing, ku=nothing,
                     cu1=cj[:, s+3:s+3], u1=s + 1 <= L - 1 ? xj[:, :, :, s+2:s+2] .* ucqs[j] : nothing, ku1=nothing)
                    for s in -1:L-1]
            push!(parts, uc4s(st, cp, sp, z4, qj, ucqs[j], sets, false))
        end
        uc = uc_concat(parts)
        ucq = ucqs
    end
    tb = TermBatchS(z; gen=gen, uc=uc, r0=G4(st.r, tab.cr[:, :, r]))
    return tb, (x, cws, ucq, uj, L, ifelse.(ok, 1.0, 0.0))
end

function commit_dshape(st, tab, r, t, dr, aux, tb)
    x, cws, ucq, uj, L, m = aux
    Δ = x .* c4(t)
    ω = scadd4(st.ω, cws, Δ)
    uc = st.uc
    for j in uj
        uc = scadd4(uc, cws[:, (j-1)*L+1:j*L], Δ[:, :, :, (j-1)*L+1:j*L] .* ucq[j])
    end
    rr = scadd4(st.r, tab.cr[:, :, r], msk(dr4(dr), c4(m)))
    return FlatState(st.θ, ω, uc, st.e, st.u, rr)
end
