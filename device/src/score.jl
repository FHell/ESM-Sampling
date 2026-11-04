# Capacity score on the device (docs/system/device.md §5): per chain (replica) and investable
# unit, the sampled capacity score at fixed utilizations
#
#     s_k = −β_v ∂_k C(k, v) + ∂_k log J(k),
#
# in node coordinates (one value per unit), and the capex force β_k ∂_k K. The CPU reference is
# the capacity gradient of `energy_cost(…; grad=true)` (`gk_op`, chained through the clamp to the
# box) plus `log_measure_grad`, term by term: `gen_grad`/`gen_vint`/`uc_grad` for the dispatchable
# units, `soc_grad` for storage, and the generator loop with y = 1 for renewables as load. The
# balance enters through its subgradient Δt (c_shed [r > 0] − c_dump [r < 0]) at the unit's node.
# Elementwise maps, sums over the steps and one-hot contractions only: it runs inside the device
# programs, on Julia Arrays for the tests.

"Balance subgradient Δt (c_shed [r > 0] − c_dump [r < 0]) of every (chain, step, node), (C, T, N)."
bal_grad(KP, r) = KP.Δt .* ifelse.(r .> 0, KP.c_shed, ifelse.(r .< 0, -KP.c_dump, 0.0))

"Clamped capacities (C, U) of one unit class: the box for investable units, the fixed capacity otherwise."
clamp_cap(k, inv, lo, hi, kfix) = ifelse.(reshape(inv, 1, :), clamp.(k, reshape(lo, 1, :), reshape(hi, 1, :)), reshape(kfix, 1, :))

"Units strictly inside their capacity box (the CPU's mask of the chained capacity gradient)."
inside_box(k, inv, lo, hi) = reshape(inv, 1, :) .& (reshape(lo, 1, :) .< k) .& (k .< reshape(hi, 1, :))

"∂C/∂k_c per (chain, step) of every dispatchable unit (C, T, U_g): `gen_grad` (+ `uc_grad`) at fixed y."
function gen_dk(KP, kc, ω, uc, Gn)
    Δt, cv = KP.Δt, KP.c_viol
    kcr = rowk(kc)
    A = rowv(KP.gav)
    mc, mc0, k0 = rowv(KP.gmc), rowv(KP.gmc0), rowv(KP.gk0)
    yc = clamp.(ω, 0.0, 1.0)
    p = A .* kcr .* yc
    hasv = (k0 .> 0) .& (mc0 .!= mc)
    q = min.(k0, kcr)
    oldfirst = mc0 .<= mc
    plo = A .* ifelse.(oldfirst, q, kcr .- q)
    clo, chi = ifelse.(oldfirst, mc0, mc), ifelse.(oldfirst, mc, mc0)
    mcp = ifelse.(p .<= plo, clo, chi)
    # gen_vint: dC/dk at fixed y (the kink moves with k when the new vintage is cheaper)
    dck = ifelse.(hasv, mcp .* A .* yc .+ ifelse.(.!oldfirst .& (k0 .< kcr) .& (p .> plo), A .* (clo .- chi), 0.0),
                  mc .* A .* yc)
    dk = Δt .* dck .- A .* yc .* Gn .+ cv .* Δt .* abs.(ω .- yc)           # running cost, balance, box penalty (w = k + κ0)
    # fleet unit commitment (uc_grad)
    ucm = rowv(KP.guc) .> 0
    gmin, su, nl = rowv(KP.ggmin), rowv(KP.gsu), rowv(KP.gnl)
    ucc, uc1 = clamp.(uc, 0.0, 1.0), clamp.(nextτ(uc), 0.0, 1.0)
    duc = A .* (Δt .* nl .* ucc .+ su .* max.(uc1 .- ucc, 0.0)) .+
          cv .* Δt .* (max.(yc .- ucc, 0.0) .+ max.(gmin .* ucc .- yc, 0.0) .+ abs.(uc .- ucc))
    return dk .+ ifelse.(ucm, duc, 0.0)
end

"∂C/∂k_c per (chain, step) of every storage unit (C, T, U_s): `soc_grad` at fixed states of charge."
function stor_dk(KP, ks, e, Gn)
    Δt, cv = KP.Δt, KP.c_viol
    h, ηin, ηout, qt, smc = rowv(KP.sh), rowv(KP.sηin), rowv(KP.sηout), rowv(KP.sqt), rowv(KP.smc)
    ec, ec1 = clamp.(e, 0.0, 1.0), clamp.(nextτ(e), 0.0, 1.0)
    d = h .* (ec1 .- ec) ./ Δt .- qt
    pos = d .>= 0
    pc = d ./ ifelse.(ηin .> 0, ηin, 1.0)
    mc = -d .* ηout
    mcc = min.(mc, 1.0)
    dch = ifelse.(ηin .<= 0, cv .* Δt .* d, Δt .* smc .* pc .+ Gn .* pc .+ ifelse.(pc .> 1, cv .* Δt .* (pc .- 1), 0.0))
    ddis = Δt .* smc .* mcc .- Gn .* mcc .+ ifelse.((mc .> 1) .& (qt .<= 0), cv .* Δt .* (mc .- 1), 0.0)
    return ifelse.(pos, dch, ddis) .+ cv .* h .* abs.(e .- ec)
end

"Capex force β_k ∂K/∂k (C, U) of one unit class (the box penalty γ_k outside the box; 0 for fixed units)."
capex_force(k, kc, inv, cap, γk, βk) = ifelse.(reshape(inv, 1, :), βk .* reshape(cap, 1, :) .* (1 .+ γk .* sign.(k .- kc)), 0.0)

"""
    cap_score(KP, RP, ω, uc, e, kg, ks, kr, Ar, r, Val(physical); βv, βk) -> NamedTuple

Capacity scores s_k = −β_v ∂_k C + ∂_k log J per chain and unit, at fixed utilizations, for the
dispatchable units (`sg`, (C, U_g)), the storage units (`ss`, (C, U_s)) and the renewable units
(`sr`, (C, U_r), `ren_units` order), and the capex forces β_k ∂K (`fg`, `fs`, `fr`). Inputs: node
values ω, uc (C, T, U_g), e (C, T, U_s), raw capacities kg, ks, kr, the clamped renewable
availability Ar (C, T, U_r) and the residual r (C, T, N). Units outside their box get a zero
operation score (the operation sees the clamped capacity); fixed units get zeros.
"""
function cap_score(KP, RP, ω, uc, e, kg, ks, kr, Ar, r, ::Val{physical}; βv, βk) where {physical}
    T = size(ω, 2)
    Gr = bal_grad(KP, r)
    # dispatchable units
    kgc = clamp_cap(kg, KP.ginv, KP.glo, KP.ghi, KP.gkfix)
    dg = dropdims(sum(gen_dk(KP, kgc, ω, uc, tmulT(Gr, KP.G)); dims=2); dims=2)
    ing = inside_box(kg, KP.ginv, KP.glo, KP.ghi)
    sg = ifelse.(ing, -βv .* dg, 0.0)
    if physical                                   # log J = Σ n_u log(k_u + κ0), n_u = T (2T with commitment)
        ng = T .* (1 .+ reshape(KP.guc, 1, :))
        sg = sg .+ ifelse.(ing, ng ./ (kg .+ KP.κ0), 0.0)
    end
    # storage
    ksc = clamp_cap(ks, KP.sinv, KP.slo, KP.shi, KP.skfix)
    ds = dropdims(sum(stor_dk(KP, ksc, e, tmulT(Gr, KP.S)); dims=2); dims=2)
    ins = inside_box(ks, KP.sinv, KP.slo, KP.shi)
    ss = ifelse.(ins, -βv .* ds, 0.0)
    physical && (ss = ss .+ ifelse.(ins, T ./ (ks .+ KP.κ0), 0.0))
    # renewables as load (y = 1; utilization measure)
    krc = clamp.(kr, reshape(RP.lo, 1, :), reshape(RP.hi, 1, :))
    dr = dropdims(sum(KP.Δt .* rowv(RP.mc) .* Ar .- Ar .* tmulT(Gr, RP.H); dims=2); dims=2)
    inr = (reshape(RP.lo, 1, :) .< kr) .& (kr .< reshape(RP.hi, 1, :))
    sr = ifelse.(inr, -βv .* dr, 0.0)
    fg = capex_force(kg, kgc, KP.ginv, KP.gcap, KP.γk, βk)
    fs = capex_force(ks, ksc, KP.sinv, KP.scap, KP.γk, βk)
    fr = βk .* reshape(RP.cap, 1, :) .* (1 .+ RP.γk .* sign.(kr .- krc))
    return (; sg, ss, sr, fg, fs, fr)
end

"Score of one unit class in the Langevin coordinates w = log(k + κ_s): ∂_w = (k + κ_s) ∂_k."
wscore(k, s, κs) = (k .+ reshape(κs, 1, :)) .* s
