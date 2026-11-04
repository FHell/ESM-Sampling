# Bias change of a capacity move on the device (docs/system/device.md §5).
#
# b̂(x, k) = ĝ(x, k) − r Ē ĝ(·, k) + (1 − r) ĥ(k) at the chains' level (the bias is lifted
# there, src/bias.jl), with ĝ = Δt Σ_t F₁(u₁_t) + F₂(u₂_t), u₁ = (D − R(k))/D̄,
# u₂ = u₁ − firm(k)/D̄, F₁, F₂ from the CPU tables (piecewise linear: a gather of two table
# entries per point). The level is spatially complete, so a move of one unit changes the node
# capacity of one item: Δk_c = clamp(k + δ) − clamp(k). Items are the investable units whose
# capacity enters b̂, stacked: generators (non-gas: R_t changes by A_t Δk_c, with the chain's
# availability for ĝ and the sample steps' for Ē ĝ; gas: firm) and storage (firm: u₂ shifts by
# a constant). ĥ (separable, quadratic in the clamped coefficients, flat beyond its vertex when
# concave) changes by Δzc = Δk_c U[i, :], and the exact measure term (1 − r) log J by the item's
# n_u (log(k_c' + κ0) − log(k_c + κ0)).
# The per-chain aggregates (features per step of the chain's weather and per sample step, and
# the sums of F₁ + F₂ over them) come from the CPU bias cache and are updated on the device.

"Static part of the device bias."
struct BiasDev{F,V,Mx,A3}
    Δt::F; Dbar::F; ratio::F; wstep::F
    u01::F; h1::F; F1::V          # table of F₁ (grid u01 + (j − 1) h1)
    u02::F; h2::F; F2::V          # table of F₂
    Id::Mx                        # (Ni, Ni) identity: one-hot rows of the items
    Uh::Mx                        # (Ni, Mh) basis row of every item in the coefficients of ĥ
    lo::V; hi::V                  # boxes per item
    isfirm::V                     # 1 for firm items (gas, storage), 0 otherwise
    A0s::Mx                       # (Ni, n) availability of the non-firm items at the sample steps
    a::V; c::V; kscale::V         # ĥ coefficients of the fields of the items
    hlo::V; hhi::V                # ĥ: range of κ in the regression data per coefficient
    A0x::A3                       # (C, T, Ni) availability of the non-firm items in the chain's weather
    nJ::V; κ0::F                  # n_u of the measure term log J per item (0: not in log J), capacity floor
end

"Per-chain aggregates of the device bias."
mutable struct BiasState{A2}
    k::A2             # (C, Ni) raw node capacities of the items
    zc::A2            # (C, Mh) clamped coefficients
    u1x::A2; u2x::A2  # (C, T) features of the chain's weather
    u1s::A2; u2s::A2  # (C, n) features of the sample steps
    gx::A2; gs::A2    # (C, 1) Σ F₁ + F₂ over the steps and over the sample steps
end
bias_fields(BS::BiasState) = (BS.k, BS.zc, BS.u1x, BS.u2x, BS.u1s, BS.u2s, BS.gx, BS.gs)
function set_bias_fields!(BS::BiasState, t)
    BS.k, BS.zc, BS.u1x, BS.u2x, BS.u1s, BS.u2s, BS.gx, BS.gs = t
    return BS
end

"""
    pack_bias(P, ix, ss) -> (BiasDev, BiasState, gitem, sitem, ritem)

Device bias of the chains `ss` (finest-level States, at the level of the bias). `gitem[q]`,
`sitem[q]`: item of dispatchable unit q / storage unit q (0: not investable); `ritem[q]`: item of
renewable unit q in the order of `ren_units`.
"""
function pack_bias(P::ML.Problem, ix::DeviceIndex, ss::Vector{ML.State})
    E, b = P.E, P.bias
    L = ss[1].ℓ
    b.ℓb == L || error("pack_bias: the bias is at level $(b.ℓb), the chains at level $L (lift the bias first)")
    lk = P.lk[L+1]
    flds = Tuple{Int,Bool,Int,Int}[]             # (k field, firm?, generator group or 0, storage group or 0)
    for (gi, g) in enumerate(E.gens)
        g.kf > 0 && push!(flds, (g.kf, g.carrier == "gas", gi, 0))
    end
    for (si, st) in enumerate(E.stors)
        st.kf > 0 && push!(flds, (st.kf, true, 0, si))
    end
    sizes = [ML.nsupport(E.kspecs[f].B) for (f, _, _, _) in flds]
    msz = [lk.m[f] for (f, _, _, _) in flds]
    Ni, Mh = sum(sizes), sum(msz)
    ioff, moff = cumsum([0; sizes[1:end-1]]), cumsum([0; msz[1:end-1]])
    BCs = [ML.BiasCache(P, s) for s in ss]
    T, ns = BCs[1].T, length(b.sD)
    Uh = zeros(Ni, Mh)
    lo, hi, isfirm, nJ = zeros(Ni), zeros(Ni), zeros(Ni), zeros(Ni)
    A0s = zeros(Ni, ns)
    a, c, ks, hlo, hhi = zeros(Mh), zeros(Mh), zeros(Mh), zeros(Mh), zeros(Mh)
    C = length(ss)
    k, zc = zeros(C, Ni), zeros(C, Mh)
    A0x = zeros(C, T, Ni)
    for (j, (f, firm, gi, si)) in enumerate(flds)
        U = Matrix(ML.cols(E.kspecs[f].B, lk.m[f]))
        ri, rm = ioff[j]+1:ioff[j]+sizes[j], moff[j]+1:moff[j]+msz[j]
        Uh[ri, rm] .= U
        lo[ri] .= E.kspecs[f].lo; hi[ri] .= E.kspecs[f].hi
        isfirm[ri] .= firm
        nJ[ri] .= BCs[1].nJ[f]
        co = lk.off[f]+1:lk.off[f]+lk.m[f]
        a[rm] .= b.a[co]; c[rm] .= b.c[co]; ks[rm] .= b.kscale[co]; hlo[rm] .= b.κlo[co]; hhi[rm] .= b.κhi[co]
        if !firm
            A0s[ri, :] .= permutedims(b.sA[gi])
        end
        for (cc, BC) in enumerate(BCs)
            k[cc, ri] .= BC.kraw[f]
            zc[cc, rm] .= BC.zc[co]
            if !firm
                g = E.gens[gi]
                A0x[cc, :, ri] .= g.wf > 0 ? BC.At[gi] : repeat(permutedims(g.avail), T, 1)
            end
        end
    end
    item(gi, si, i) = (j = findfirst(x -> x[3] == gi && x[4] == si, flds); j === nothing ? 0 : ioff[j] + i)
    gitem = [item(gi, 0, i) for (gi, i) in ix.gidx]
    sitem = [item(0, si, i) for (si, i) in ix.sidx]
    ritem = [item(gi, 0, i) for (gi, i) in ren_units(P)]
    u1x, u2x, u1s, u2s = zeros(C, T), zeros(C, T), zeros(C, ns), zeros(C, ns)
    gx, gs = zeros(C, 1), zeros(C, 1)
    for (cc, BC) in enumerate(BCs)
        fd = BC.firm / b.Dbar
        u1x[cc, :] .= BC.U1x; u2x[cc, :] .= BC.U1x .- fd
        u1s[cc, :] .= BC.U1s; u2s[cc, :] .= BC.U1s .- fd
        gx[cc, 1] = sum(BC.F1x .+ BC.F2x); gs[cc, 1] = sum(BC.F1s .+ BC.F2s)
    end
    t1, t2 = ML.tables(b)
    BD = BiasDev(b.Δt, b.Dbar, b.ratio, b.wstep, t1.u0, t1.h, copy(t1.F), t2.u0, t2.h, copy(t2.F),
                 Matrix{Float64}(I, Ni, Ni), Uh, lo, hi, isfirm, A0s, a, c, ks, hlo, hhi, A0x, nJ, E.κ0)
    return BD, BiasState(k, zc, u1x, u2x, u1s, u2s, gx, gs), gitem, sitem, ritem
end


"Table value at u (any shape): F[j] + w (F[j+1] − F[j]) on the clamped segment j (the CPU's `tabval`)."
function tab_dev(F, u0, h, u)
    n = length(F)
    x = (u .- u0) ./ h
    j = clamp.(floor.(x), 0.0, n - 2.0)
    w = x .- j
    i = trunc.(Int64, j) .+ 1
    f0 = F[i]
    return f0 .+ w .* (F[i.+1] .- f0)
end

"""
ĥ per coordinate (the CPU's `hcoord`): inside the data range [lo, hi] a κ + ½ c κ², flat beyond
the vertex −a/c when c < 0; outside it the edge value continued with the edge slope clipped to
not push away from the data (≥ 0 above hi, ≤ 0 below lo).
"""
function hcoord_dev(a, c, lo, hi, κ)
    neg = c .< 0
    vtx = -a ./ ifelse.(neg, c, -1.0)
    above, below = κ .> hi, κ .< lo
    κc = ifelse.(above, hi, ifelse.(below, lo, κ))
    sat = neg .& (κc .> vtx)
    κs = ifelse.(sat, vtx, κc)
    d = ifelse.(sat, 0.0, a .+ c .* κc)
    s = ifelse.(above, max.(d, 0.0), ifelse.(below, min.(d, 0.0), 0.0))
    return a .* κs .+ 0.5 .* c .* κs .^ 2 .+ s .* (κ .- κc)
end

"Σ over the columns of F₁(u₁) + F₂(u₂): (C, n) features → (C, 1)."
gsum(BD::BiasDev, u1, u2) = sum(tab_dev(BD.F1, BD.u01, BD.h1, u1) .+ tab_dev(BD.F2, BD.u02, BD.h2, u2); dims=2)

"""
Bias change Δb̂ (C,) of a node-capacity change δ (C,) at item ι, and the aggregates after
the change (the tuple of `bias_fields`, for the commit).
"""
function delta_bias(BD::BiasDev, BS::BiasState, ι, δ)
    e = reshape(BD.Id[ι, :], 1, :)                                           # one-hot row of the item
    kι = vec(sum(BS.k .* e; dims=2))
    lo, hi = Reactant.@allowscalar(BD.lo[ι]), Reactant.@allowscalar(BD.hi[ι])
    kc0, kc1 = clamp.(kι, lo, hi), clamp.(kι .+ δ, lo, hi)
    Δkc = kc1 .- kc0                                                          # (C,)
    knew = BS.k .+ reshape(δ, :, 1) .* e
    fm = Reactant.@allowscalar BD.isfirm[ι]
    Δren = reshape(Δkc .* (1 .- fm), :, 1)
    sf = reshape(Δkc .* fm ./ BD.Dbar, :, 1)
    ΔRx = BD.A0x[:, :, ι] .* Δren                                             # (C, T)
    ΔRs = Δren .* reshape(BD.A0s[ι, :], 1, :)                                 # (C, n)
    u1xn, u1sn = BS.u1x .- ΔRx ./ BD.Dbar, BS.u1s .- ΔRs ./ BD.Dbar
    u2xn, u2sn = BS.u2x .- ΔRx ./ BD.Dbar .- sf, BS.u2s .- ΔRs ./ BD.Dbar .- sf
    gxn, gsn = gsum(BD, u1xn, u2xn), gsum(BD, u1sn, u2sn)
    zcn = BS.zc .+ reshape(Δkc, :, 1) .* reshape(BD.Uh[ι, :], 1, :)
    ksr = reshape(BD.kscale, 1, :)
    κ0, κ1 = BS.zc ./ ksr, zcn ./ ksr
    a, c, hlo, hhi = reshape(BD.a, 1, :), reshape(BD.c, 1, :), reshape(BD.hlo, 1, :), reshape(BD.hhi, 1, :)
    Δh = vec(sum(hcoord_dev(a, c, hlo, hhi, κ1) .- hcoord_dev(a, c, hlo, hhi, κ0); dims=2))
    nJι = Reactant.@allowscalar BD.nJ[ι]
    Δlj = nJι .* (log.(kc1 .+ BD.κ0) .- log.(kc0 .+ BD.κ0))
    Δb = BD.Δt .* vec(gxn .- BS.gx) .- BD.ratio .* BD.wstep .* vec(gsn .- BS.gs) .+ (1 - BD.ratio) .* (Δh .+ Δlj)
    return Δb, (knew, zcn, u1xn, u2xn, u1sn, u2sn, gxn, gsn)
end

"Commit the accepted changes (per-chain mask `ok`)."
commit_bias!(BS::BiasState, new, ok) = set_bias_fields!(BS, map((x, y) -> ifelse.(reshape(ok, :, 1), x, y), new, bias_fields(BS)))
