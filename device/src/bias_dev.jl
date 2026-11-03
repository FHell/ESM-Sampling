# Bias change of a capacity move on the device (docs/reactant_design.md §3.6, phase 4).
#
# b̂(x, k) = ĝ(x, k) − r Ē ĝ(·, k) + (1 − r) ĥ(k) at level 0 (src/bias.jl), with
# ĝ = Δt₀ Σ_t θ·φ(u₁_t, u₂_t), u₁ = (D − R(k))/D̄, u₂ = u₁ − firm(k)/D̄. A move of one unit
# changes the clamped level-0 node capacities of its field, Δk0 = clamp(k0 + P0[:, ι] δ) −
# clamp(k0), with P0 = U0 U0ᵀ the projection onto the level-0 span (k_at0 is the coefficient
# prefix). Items are the level-0 support positions of the fields involved, stacked:
#   * firm items (gas, storage): firm changes by Σ Δk0, so u₂ shifts by a constant;
#   * renewable items: R_t changes by Σ_items A0[t, item] Δk0[item] (A0 the level-0
#     availability: the chain's weather for ĝ, the exact samples for Ē ĝ), so u₁ and u₂ shift
#     step by step.
# ĥ (separable, quadratic in the clamped level-0 coefficients) changes by Δzc = U0ᵀ Δk0.
# The per-chain aggregates are computed on the host and updated incrementally on the device.

"Static part of the device bias."
struct BiasDev{F,V,Mx,A3}
    Δt0::F; Dbar::F; ratio::F; s::F
    θ1::F; θr1::V; τ1::V          # linear u₁ term and u₁ ramps
    θ2::F; θr2::V; τ2::V          # linear u₂ term and u₂ ramps
    P0::Mx                        # (Ni, Ni) block-diagonal projection
    U0::Mx                        # (Ni, M0) block-diagonal level-0 basis
    lo::V; hi::V                  # level-0 boxes per item
    isfirm::V                     # 1 for firm items, 0 for renewable items
    A0s::Mx                       # (Ni, S·T0) availability of the renewable items in the exact samples
    a::V; c::V; kscale::V         # ĥ coefficients of the level-0 coordinates of these fields
    A0x::A3                       # (C, T0, Ni) availability of the renewable items in the chain's weather
end

"Per-chain aggregates of the device bias."
mutable struct BiasState{A2}
    k0::A2            # (C, Ni) raw level-0 node capacities
    zc::A2            # (C, M0) clamped level-0 coefficients
    u1x::A2; u2x::A2  # (C, T0) features of the chain's level-0 weather
    u1s::A2; u2s::A2  # (C, S·T0) features of the exact weather samples
end
bias_fields(BS::BiasState) = (BS.k0, BS.zc, BS.u1x, BS.u2x, BS.u1s, BS.u2s)
function set_bias_fields!(BS::BiasState, t)
    BS.k0, BS.zc, BS.u1x, BS.u2x, BS.u1s, BS.u2s = t
    return BS
end

@inline softplus_d(x) = ifelse.(x .> 30, x, log1p.(exp.(min.(x, 30.0))))

"""
    pack_bias(P, ix, ss) -> (BiasDev, BiasState, gitem, sitem, ritem)

Device bias of the chains `ss` (finest-level States). `gitem[q]`, `sitem[q]`: item of
dispatchable unit q / storage unit q (0: not an investable firm unit); `ritem[q]`: item of
renewable unit q in the order of `ren_units`.
"""
function pack_bias(P::ML.Problem, ix::DeviceIndex, ss::Vector{ML.State})
    E, b = P.E, P.bias
    lk0 = P.lk[1]
    flds = Tuple{Int,Bool,Int}[]                 # (k field, firm?, generator group or 0)
    for (gi, g) in enumerate(E.gens)
        g.kf > 0 || continue
        if g.carrier == "gas"
            push!(flds, (g.kf, true, gi))
        elseif g.vf == 0 && g.wf > 0
            push!(flds, (g.kf, false, gi))
        end
    end
    for st in E.stors
        st.kf > 0 && push!(flds, (st.kf, true, 0))
    end
    sizes = [ML.nsupport(E.kspecs[f].B) for (f, _, _) in flds]
    msz = [lk0.m[f] for (f, _, _) in flds]
    Ni, M0 = sum(sizes), sum(msz)
    ioff, moff = cumsum([0; sizes[1:end-1]]), cumsum([0; msz[1:end-1]])
    T0 = P.lv[1].T
    S = length(b.sampD)
    P0, U0 = zeros(Ni, Ni), zeros(Ni, M0)
    lo, hi, isfirm = zeros(Ni), zeros(Ni), zeros(Ni)
    A0s = zeros(Ni, S * T0)
    a, c, ks = zeros(M0), zeros(M0), zeros(M0)
    for (j, (f, firm, gi)) in enumerate(flds)
        U = Matrix(ML.cols(E.kspecs[f].B, lk0.m[f]))
        ri, rm = ioff[j]+1:ioff[j]+sizes[j], moff[j]+1:moff[j]+msz[j]
        P0[ri, ri] .= U * U'
        U0[ri, rm] .= U
        lo[ri] .= E.kspecs[f].lo; hi[ri] .= E.kspecs[f].hi
        isfirm[ri] .= firm
        co = lk0.off[f]+1:lk0.off[f]+lk0.m[f]
        a[rm] .= b.a[co]; c[rm] .= b.c[co]; ks[rm] .= b.kscale[co]
        if !firm
            for si in 1:S
                A0s[ri, (si-1)*T0+1:si*T0] .= b.sampA[si][gi]
            end
        end
    end
    n1 = length(b.τ1)
    θ1, θr1 = b.θ[1], b.θ[2:1+n1]
    θ2, θr2 = b.θ[2+n1], b.θ[3+n1:end]
    item(kf, i) = (j = findfirst(x -> x[1] == kf, flds); j === nothing ? 0 : ioff[j] + i)
    gitem = [(g = E.gens[gi]; (g.carrier == "gas" && g.kf > 0) ? item(g.kf, i) : 0) for (gi, i) in ix.gidx]
    sitem = [(st = E.stors[si]; st.kf > 0 ? item(st.kf, i) : 0) for (si, i) in ix.sidx]
    ritem = [item(E.gens[gi].kf, i) for (gi, i) in ren_units(P)]
    # per-chain aggregates
    C = length(ss)
    k0m, zcm = zeros(C, Ni), zeros(C, M0)
    u1x, u2x, u1s, u2s = zeros(C, T0), zeros(C, T0), zeros(C, S * T0), zeros(C, S * T0)
    A0x = zeros(C, T0, Ni)
    for (cc, s) in enumerate(ss)
        zk0 = ML.k_at0(P, s.ℓ, s.k)
        X0 = ML.weather_nodes(P.W, 0, s.z0)
        W = [map(x -> ML.datamap(E.wspecs[f], x), X0[f]) for f in eachindex(X0)]
        for (j, (f, firm, gi)) in enumerate(flds)
            U = Matrix(ML.cols(E.kspecs[f].B, lk0.m[f]))
            kn = U * vec(ML.fieldview(zk0, lk0, E.kspecs, f))
            ri, rm = ioff[j]+1:ioff[j]+sizes[j], moff[j]+1:moff[j]+msz[j]
            k0m[cc, ri] .= kn
            zcm[cc, rm] .= U' * clamp.(kn, E.kspecs[f].lo, E.kspecs[f].hi)
            firm || (A0x[cc, :, ri] .= permutedims(W[E.gens[gi].wf]))
        end
        kc, _ = ML.node_capacities(E, lk0, zk0)
        R, firm = ML.residual_parts(E, kc, W, T0)
        D = vec(sum(W[1]; dims=1))
        u1x[cc, :] .= (D .- R) ./ b.Dbar
        u2x[cc, :] .= u1x[cc, :] .- firm / b.Dbar
        for si in 1:S
            Ds, A = b.sampD[si], b.sampA[si]
            Rs = zeros(T0)
            for (gj, g) in enumerate(E.gens)
                kv = g.kf > 0 ? kc[g.kf] : g.kfix
                g.carrier == "gas" && continue
                for t in 1:T0, i in eachindex(g.bus)
                    Rs[t] += A[gj][i, t] * kv[i]
                end
            end
            rr = (si-1)*T0+1:si*T0
            u1s[cc, rr] .= (Ds .- Rs) ./ b.Dbar
            u2s[cc, rr] .= u1s[cc, rr] .- firm / b.Dbar
        end
    end
    BD = BiasDev(Float64(P.levels[1][2]), b.Dbar, b.ratio, b.s, θ1, collect(θr1), copy(b.τ1), θ2, collect(θr2), copy(b.τ2),
                 P0, U0, lo, hi, isfirm, A0s, a, c, ks, A0x)
    return BD, BiasState(k0m, zcm, u1x, u2x, u1s, u2s), gitem, sitem, ritem
end

"Investable renewable units (renewables as load) in device order: (group, index)."
ren_units(P::ML.Problem) = [(gi, i) for (gi, g) in enumerate(P.E.gens) if g.vf == 0 && g.kf > 0 for i in eachindex(g.bus)]

"Sum over steps of θ·φ(u₁, u₂): (C, n) features → (C,)."
function gfull(BD::BiasDev, u1, u2)
    v = BD.θ1 .* u1 .+ BD.θ2 .* u2
    for j in 1:length(BD.τ1)
        θj = Reactant.@allowscalar BD.θr1[j]; τj = Reactant.@allowscalar BD.τ1[j]
        v = v .+ θj .* BD.s .* softplus_d((u1 .- τj) ./ BD.s)
    end
    for j in 1:length(BD.τ2)
        θj = Reactant.@allowscalar BD.θr2[j]; τj = Reactant.@allowscalar BD.τ2[j]
        v = v .+ θj .* BD.s .* softplus_d((u2 .- τj) ./ BD.s)
    end
    return vec(sum(v; dims=2))
end

"""
Bias change Δb̂ (C,) of a node-capacity change δ (C,) at item ι, and the aggregates after
the change (the tuple of `bias_fields`, for the commit).
"""
function delta_bias(BD::BiasDev, BS::BiasState, ι, δ)
    k0new = BS.k0 .+ reshape(δ, :, 1) .* reshape(BD.P0[:, ι], 1, :)
    lo, hi = reshape(BD.lo, 1, :), reshape(BD.hi, 1, :)
    Δkc = clamp.(k0new, lo, hi) .- clamp.(BS.k0, lo, hi)
    fm = reshape(BD.isfirm, 1, :)
    Δfirm = vec(sum(Δkc .* fm; dims=2))
    Δren = Δkc .* (1 .- fm)
    C, T0 = size(BS.u1x)
    ΔRx = dropdims(sum(BD.A0x .* reshape(Δren, C, 1, :); dims=3); dims=3)    # (C, T0)
    ΔRs = Δren * BD.A0s                                                       # (C, S·T0)
    sf = reshape(Δfirm ./ BD.Dbar, :, 1)
    u1xn, u1sn = BS.u1x .- ΔRx ./ BD.Dbar, BS.u1s .- ΔRs ./ BD.Dbar
    u2xn, u2sn = BS.u2x .- ΔRx ./ BD.Dbar .- sf, BS.u2s .- ΔRs ./ BD.Dbar .- sf
    S = size(BS.u1s, 2) ÷ max(T0, 1)
    Δg = BD.Δt0 .* (gfull(BD, u1xn, u2xn) .- gfull(BD, BS.u1x, BS.u2x))
    Δgm = BD.Δt0 .* (gfull(BD, u1sn, u2sn) .- gfull(BD, BS.u1s, BS.u2s)) ./ S
    zcn = BS.zc .+ Δkc * BD.U0
    ksr = reshape(BD.kscale, 1, :)
    κ0, κ1 = BS.zc ./ ksr, zcn ./ ksr
    a, c = reshape(BD.a, 1, :), reshape(BD.c, 1, :)
    Δh = vec(sum(a .* (κ1 .- κ0) .+ 0.5 .* c .* (κ1 .^ 2 .- κ0 .^ 2); dims=2))
    Δb = Δg .- BD.ratio .* Δgm .+ (1 - BD.ratio) .* Δh
    return Δb, (k0new, zcn, u1xn, u2xn, u1sn, u2sn)
end

"Commit the accepted changes (per-chain mask `ok`)."
commit_bias!(BS::BiasState, new, ok) = set_bias_fields!(BS, map((x, y) -> ifelse.(reshape(ok, :, 1), x, y), new, bias_fields(BS)))
