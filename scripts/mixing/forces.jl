# Forces on the log capacities (docs/mixing_diagnosis.md).
#
# For a unit with capacity k and u = log k, two parametrizations of a capacity change give
# the derivative of the log marginal density of u, d/du log m_u = E_{v|k}[lf'(0)]:
#   kscale (fixed physical dispatch):  lf_ks(u) = -Δ(β_k K + β_v C) + (1-n) u + n log((k e^u+κ0)/(k+κ0))
#   kop    (fixed utilization, pro-rata compensation at the node):
#          lf_kop(u) = -Δ(β_k K + β_v C) + n log((k e^u+κ0)/(k+κ0)) + log J_prorata(u) + u
# Both expectations agree when v is in equilibrium given (k, x). The parts: capex
# -β_k c k, measure n k/(k+κ0), group Jacobian, and the cost (+ pro-rata Jacobian) part.

include(joinpath(@__DIR__, "kop.jl"))

"Per-unit force decomposition at state s; returns a vector of named tuples."
function unit_forces(P, LM, s; ε::Float64=1e-4, kinds=(:renewable, :storage, :dispatch))
    E = P.E
    LC = ML.LocalCost(P, LM, s)
    Kraw = ML.knodes(P, LM.ℓ, s.k)
    dp = zeros(LM.T)
    out = NamedTuple[]
    for grp in kop_units(P, LM; kinds=kinds), (kind, gi, i) in grp
        kf = kind == 1 ? E.gens[gi].kf : E.stors[gi].kf
        k = Kraw[kf][i]
        k >= KOP_KMIN || continue
        kt = kop_term(P, LM, LC, Kraw, kind, gi, i)
        fp, fm = kop_eval(P, LM, LC, kt, k * exp(ε)), kop_eval(P, LM, LC, kt, k * exp(-ε))
        dlf = (fp[1] - fm[1]) / (2ε)
        dlj = (fp[2] - fm[2]) / (2ε)
        sp = E.kspecs[kf]
        inbox = sp.lo[i] < k < sp.hi[i]
        capex = -P.βk * kt.cap * k * (inbox ? 1.0 : 1.0 + E.γk)
        meas = inbox ? kt.nmeas * k / (k + E.κ0) : 0.0
        cost_kop = dlf - dlj - capex - meas
        n = kind == 1 ? (E.gens[gi].vf > 0 ? LM.T : 0) : LM.T
        dks = -(ML.kscale_delta!(dp, P, LM, LC, Kraw, kind, gi, i, ε) - ML.kscale_delta!(dp, P, LM, LC, Kraw, kind, gi, i, -ε)) / (2ε)
        cost_ks = dks - capex
        total_ks = dks + (1 - n) + meas
        total_kop = dlf + 1
        # α = 1: derivative of −b̂ along u (the same for both parametrizations: b̂ depends on k only)
        dbias = 0.0
        if P.use_bias
            lk = P.lk[LM.ℓ+1]; Ut = E.kspecs[kf].B.Ut
            bat(uu) = (kk = copy(s.k); for q in nzrange(Ut, i); kk[lk.off[kf]+rowvals(Ut)[q]] += k * (exp(uu) - 1) * nonzeros(Ut)[q]; end;
                       ML.bias_term(P, ML.k_at0(P, LM.ℓ, kk), s.z0))
            dbias = -(bat(ε) - bat(-ε)) / (2ε)
            total_ks += dbias; total_kop += dbias
        end
        carrier = kind == 1 ? E.gens[gi].carrier : E.stors[gi].carrier
        push!(out, (; carrier, i, node=kt.n, k, capex, meas, cost_kop, jac_kop=dlj, cost_ks, total_ks, total_kop, dbias))
    end
    return out
end

"Average the forces over a list of states (same capacities, different operation)."
function mean_forces(P, LM, states; kw...)
    F = [unit_forces(P, LM, s; kw...) for s in states]
    keys_ = (:capex, :meas, :cost_kop, :jac_kop, :cost_ks, :total_ks, :total_kop, :dbias)
    return [merge(F[1][j], NamedTuple{keys_}(Tuple(mean(getfield(F[q][j], kk) for q in eachindex(F)) for kk in keys_)))
            for j in eachindex(F[1])]
end

"Print per-carrier medians of the force parts."
function print_forces(F)
    for c in unique(f.carrier for f in F)
        G = [f for f in F if f.carrier == c]
        @printf("%-8s n=%2d  k med %8.0f  capex %9.1f  meas %7.1f  cost_kop %9.1f  jac_kop %7.1f  cost_ks %9.1f | total_ks %9.1f  total_kop %9.1f\n",
                c, length(G), median(f.k for f in G), median(f.capex for f in G), median(f.meas for f in G),
                median(f.cost_kop for f in G), median(f.jac_kop for f in G), median(f.cost_ks for f in G),
                median(f.total_ks for f in G), median(f.total_kop for f in G))
    end
end
