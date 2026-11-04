# Prototype of a capacity dynamics without a learned bias (review proposal P1): the gas
# capacities follow a Langevin step at the investment temperature β_k whose drift is the
# capex gradient plus the sampled capacity score (score identity: ∂_k log Z = E[−β_v ∂_k C|_ω] +
# ∂_k log J), i.e. the gradient of −β_k [K + F_x(k)] estimated from the current dispatch, and the
# dispatch follows at fixed utilization and is re-equilibrated by operation sweeps. Weather fixed
# (one year), so the target of the capacities is exp(−β_k [K(k) + F_x(k)]) for that year: two
# chains started apart must meet. Stochastic-gradient Langevin: exact as the step → 0.
import ESMSampling as ML
using Random, LinearAlgebra, Statistics, Printf

"Set node capacities of k field `f` to `knew` (raw) and scale the unit's utilizations so that its output is unchanged."
function set_capacities_fixed_output!(P, s, f, knew)
    E = P.E
    lk, lv = P.lk[s.ℓ+1], P.lv[s.ℓ+1]
    U = ML.cols(E.kspecs[f].B, lk.m[f])
    kold = U * vec(ML.fieldview(s.k, lk, E.kspecs, f))
    ML.fieldview(s.k, lk, E.kspecs, f) .= reshape(U' * knew, :, 1)
    sp = E.kspecs[f]
    for g in E.gens
        (g.kf == f && g.vf > 0) || continue
        Uv = ML.cols(E.vspecs[g.vf].B, lv.m[g.vf])
        Z = ML.fieldview(s.v, lv, E.vspecs, g.vf)
        Y = Uv * Z
        for i in axes(Y, 1)
            Y[i, :] .*= clamp(kold[i], sp.lo[i], sp.hi[i]) / clamp(knew[i], sp.lo[i], sp.hi[i])
        end
        Z .= Uv' * Y
    end
    return ML.evaluate!(P, s)
end

"""
    sgld_run!(rng, P, S, s, f, nsweeps; η, kmin, every) -> trace

Langevin steps on the node capacities of k field `f` interleaved with operation sweeps.
Per sweep: `every` operation sweeps, then the score of the state (node coordinates),
then k ← k + η (−β_k ∂K + r s_k) + √(2η) ξ, reflected at `kmin`. Returns per sweep the
total capacity, the mean drift and the shed and the cost parts.
"""
function sgld_run!(rng, P, S, s, f, nsweeps; η=1e4, kmin=1.0, every=1, log_every=10)
    Kk = S.loc
    LMk = ML.local_model(Kk, P, s.ℓ)
    s = copy(s)
    r = P.βk / P.βv
    trace = NamedTuple[]
    U = ML.cols(P.E.kspecs[f].B, P.lk[s.ℓ+1].m[f])
    for it in 1:nsweeps
        for _ in 1:every
            s, _ = ML.local_sweep!(rng, P, LMk, s; wsets=Kk.wsets, extra=Kk.extra)
            if rand(rng) < 0.5 && LMk.complete[P.E.fθ]
                isempty(Kk.pools) && append!(Kk.pools, [ML.BlockPool(P, 140) for _ in 1:Threads.maxthreadid()])
                s, _ = ML.storage_ms_sweep!(rng, P, LMk, s; pools=Kk.pools)
            end
        end
        ns = node_scores(P, s)
        k = U * vec(ML.fieldview(s.k, P.lk[s.ℓ+1], P.E.kspecs, f))
        drift = -ns.fk[f] .+ r .* ns.sk[f]
        knew = k .+ η .* drift .+ sqrt(2η) .* randn(rng, length(k))
        knew = [x < kmin ? 2kmin - x : x for x in knew]
        knew = max.(knew, kmin)
        s = set_capacities_fixed_output!(P, s, f, knew)
        push!(trace, (; it, total=sum(knew), drift=mean(drift), score=mean(ns.sk[f]), capex=s.parts.capex, op=s.parts.op, shed=s.parts.shed))
        if it % log_every == 0
            @printf("  sgld %4d: total %.1f GW, mean drift %.3g, mean score %.3g nats/MW, capex %.3e op %.3e shed %.3g\n",
                    it, sum(knew) / 1e3, mean(drift), mean(ns.sk[f]), s.parts.capex, s.parts.op, s.parts.shed)
            flush(stdout)
        end
    end
    return trace, s
end
