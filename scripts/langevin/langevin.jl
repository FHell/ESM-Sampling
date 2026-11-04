# Langevin capacity step without a learned bias (the follow-up of the α = 1 review,
# scripts/review/sgld_gas.jl): every investable node capacity follows
#   k ← k + η (−β_k ∂K + r ŝ_k) + √(2η) ξ,   ŝ_k = −β_v ∂_k C + ∂_k log J,
# the sampled capacity score at the current dispatch (by the score identity an unbiased
# estimate of ∂_k log Z_x(k) when the dispatch is at equilibrium), so that the capacities
# target exp(−β_k K(k) + r log Z_x(k)) for the chain's weather (one year fixed here),
# and the operation is re-equilibrated by operation sweeps between the steps.
#
# The step is taken in w = log(k + κ_s) per unit (κ_s a per-carrier offset), i.e. with the
# position-dependent step η(k) = η_w (k + κ_s)², which is an exact change of variables (drift
# (k + κ_s) g + 1 in w), reflected at k = κ_min. The operation follows the step:
#   dispatchable generators: at fixed output (utilizations scaled by c(k0)/c(k1), the kscale map);
#   renewables (as load) and storage: at fixed utilization with the dispatchable generators of
#   the node following share-preservingly (the kop curve, src/kop.jl).
import ESMSampling as ML
using Random, LinearAlgebra, Statistics, Printf

"Capacity scores in node coordinates: s_k = −β_v ∂_k C + ∂_k log J and the capex force β_k ∂_k K (with the box penalty)."
function lgv_scores(P, s)
    E = P.E
    lk, lv = P.lk[s.ℓ+1], P.lv[s.ℓ+1]
    Xn = ML.weather_nodes(P.W, s.ℓ, s.xs[end])
    parts, (gkc, gko, gv, gX) = ML.energy_cost(E, lk, lv, s.k, s.v, Xn; grad=true)
    sk = -P.βv .* gko .+ ML.log_measure_grad(E, lk, lv, s.k)
    fk = P.βk .* gkc
    ng(g) = [ML.cols(E.kspecs[f].B, lk.m[f]) * vec(ML.fieldview(g, lk, E.kspecs, f)) for f in eachindex(E.kspecs)]
    return ng(sk), ng(fk)
end

"Investable units at the complete level: (:disp | :ren | :stor, group, index, k field)."
function lgv_units(P, LM)
    E = P.E
    lk = P.lk[LM.ℓ+1]
    units = Tuple{Symbol,Int,Int,Int}[]
    for (gi, g) in enumerate(E.gens)
        (g.kf > 0 && ML.iscomplete(lk, E.kspecs, g.kf)) || continue
        append!(units, [(g.vf > 0 ? :disp : :ren, gi, i, g.kf) for i in eachindex(g.bus)])
    end
    for (si, st) in enumerate(E.stors)
        (st.kf > 0 && ML.iscomplete(lk, E.kspecs, st.kf)) || continue
        append!(units, [(:stor, si, i, st.kf) for i in eachindex(st.bus)])
    end
    return units
end

"Move the raw capacity of a dispatchable unit to k1 at fixed output (utilizations and commitment scaled by c(k0)/c(k1))."
function lgv_set_disp!(P, LM, LC, s, gi, i, k0, k1)
    E = P.E
    lk, lv = P.lk[s.ℓ+1], P.lv[s.ℓ+1]
    g = E.gens[gi]
    sp = E.kspecs[g.kf]
    kc0, kc1 = clamp(k0, sp.lo[i], sp.hi[i]), clamp(k1, sp.lo[i], sp.hi[i])
    Ut = sp.B.Ut
    for q in ML.nzrange(Ut, i)
        s.k[lk.off[g.kf]+ML.rowvals(Ut)[q]] += (k1 - k0) * ML.nonzeros(Ut)[q]
    end
    sc = ML.kc_ratio(kc0, kc1)
    sc == 1.0 && return
    for f in (g.ucf > 0 ? (g.vf, g.ucf) : (g.vf,))
        m = lv.m[f]
        Utv = E.vspecs[f].B.Ut
        for τ in 1:LM.T
            Δy = LC.Y[f][i, τ] * (sc - 1)
            o = lv.off[f] + (τ - 1) * m
            for q in ML.nzrange(Utv, i)
                s.v[o+ML.rowvals(Utv)[q]] += Δy * ML.nonzeros(Utv)[q]
            end
        end
    end
end

"""
    langevin_step!(rng, P, LM, s, units; ηw, κs, κmin, clip, box, scores) -> (s, info)

One Langevin step of all investable capacities (ηw, κs: per k field). `clip` bounds the
drift per step in w; `box = true` reflects at the upper capacity bound (the box penalty
γ_k = 10 × capex makes the box nearly a wall; reflecting avoids overshooting steps there);
`scores = (sk, fk)` replaces the scores of the current state (e.g. averages over the
preceding operation sweeps). Returns the state and the per-field node vectors of the score r ŝ_k and
the capex force β_k ∂K used in the step, and the number of clipped drifts.
"""
function langevin_step!(rng, P, LM, s, units; ηw, κs, κmin=1.0, clip=Inf, box::Bool=false, scores=nothing)
    E = P.E
    r = P.βk / P.βv
    sk, fk = scores === nothing ? lgv_scores(P, s) : scores
    Kraw = ML.knodes(P, s.ℓ, s.k)
    knew = deepcopy(Kraw)
    nclip = 0
    for kf in unique(u[4] for u in units)
        k = Kraw[kf]
        g = -fk[kf] .+ r .* sk[kf]
        w = log.(k .+ κs[kf])
        # drift in w, clipped to ±clip per step (a trust region far from equilibrium; inactive near it)
        dw0 = ηw[kf] .* ((k .+ κs[kf]) .* g .+ 1)
        nclip += count(abs.(dw0) .> clip)
        dw = clamp.(dw0, -clip, clip)
        w1 = w .+ dw .+ sqrt(2ηw[kf]) .* randn(rng, length(k))
        wmin = log(κmin + κs[kf])
        w1 = [x < wmin ? 2wmin - x : x for x in w1]                 # reflection at k = κmin
        if box
            wmax = log.(max.(E.kspecs[kf].hi, κmin + 1.0) .+ κs[kf])
            w1 = [x > wm && w0 <= wm ? max(2wm - x, wmin) : x for (x, wm, w0) in zip(w1, wmax, w)]
        end
        knew[kf] = exp.(w1) .- κs[kf]
    end
    # dispatchable generators at fixed output
    LC = ML.LocalCost(P, LM, s)
    for (kind, gi, i, kf) in units
        kind == :disp || continue
        lgv_set_disp!(P, LM, LC, s, gi, i, Kraw[kf][i], knew[kf][i])
    end
    # renewables and storage at fixed utilization, the node's dispatchable generators following
    LC = ML.LocalCost(P, LM, s)
    Kr = ML.knodes(P, s.ℓ, s.k)
    for (kind, gi, i, kf) in units
        kind == :disp && continue
        k0 = Kr[kf][i]
        mv = ML.kop_prepare(P, LM, LC, s, [(kind == :ren ? 1 : 2, gi, i)]; κmin=κmin, Kraw=Kr)
        if mv === nothing                                            # k0 < κmin: set the capacity alone
            Ut = E.kspecs[kf].B.Ut
            for q in ML.nzrange(Ut, i)
                s.k[P.lk[s.ℓ+1].off[kf]+ML.rowvals(Ut)[q]] += (knew[kf][i] - k0) * ML.nonzeros(Ut)[q]
            end
            LC = ML.LocalCost(P, LM, s); Kr = ML.knodes(P, s.ℓ, s.k)
            continue
        end
        ML.kop_commit!(P, LM, LC, s, mv, log(knew[kf][i] / k0); Kraw=Kr)
    end
    return ML.evaluate!(P, s), (; rs=[r .* x for x in sk], fk, nclip)
end

"Operation sweep as in the fine step of the local kernel (local sweep, storage multiscale with probability 1/2)."
function lgv_op!(rng, P, K, LM, s)
    s, _ = ML.local_sweep!(rng, P, LM, s; wsets=K.wsets, extra=K.extra)
    if rand(rng) < 0.5 && LM.complete[P.E.fθ]
        isempty(K.pools) && append!(K.pools, [ML.BlockPool(P, 140) for _ in 1:Threads.maxthreadid()])
        s, _ = ML.storage_ms_sweep!(rng, P, LM, s; pools=K.pools)
    end
    return s
end

"""
    langevin_run!(rng, P, S, s, nsweeps; ηw, κs, inner, log_every) -> (trace, s)

`nsweeps` Langevin steps, each followed by `inner` operation sweeps. With `avg = true` the
step uses the scores averaged over the `inner` sweeps since the previous step (a time average
over the operation chain at fixed k), else the scores of the current state. Trace per step:
raw totals per k field, cost parts, total cost, per field the node means of r ŝ_k and β_k ∂K
and (`nodes = true`) the node vectors of k, r ŝ_k and β_k ∂K.
"""
function langevin_run!(rng, P, S, s, nsweeps; ηw, κs, inner=1, log_every=10, label="", clip=Inf, trace=NamedTuple[], it0=0,
                       box::Bool=false, avg::Bool=false, nodes::Bool=false)
    K = S.loc
    LM = ML.local_model(K, P, s.ℓ)
    units = lgv_units(P, LM)
    s = copy(s)
    t0 = time()
    scores = nothing
    for it in it0+1:it0+nsweeps
        kn = ML.knodes(P, s.ℓ, s.k)
        s, inf = langevin_step!(rng, P, LM, s, units; ηw, κs, clip, box, scores)
        acc = nothing
        for _ in 1:inner
            s = lgv_op!(rng, P, K, LM, s)
            if avg
                sk, fk = lgv_scores(P, s)
                acc = acc === nothing ? (sk ./ inner, fk ./ inner) : (acc[1] .+ sk ./ inner, acc[2] .+ fk ./ inner)
            end
        end
        scores = acc
        tot = [sum(k) for k in ML.knodes(P, s.ℓ, s.k)]
        push!(trace, (; it, tot, capex=s.parts.capex, op=s.parts.op, shed=s.parts.shed, cost=s.parts.capex + s.parts.op,
                      rs=mean.(inf.rs), fk=mean.(inf.fk), nclip=inf.nclip, t=time() - t0,
                      kn=nodes ? kn : nothing, rsn=nodes ? inf.rs : nothing, fkn=nodes ? inf.fk : nothing))
        if it % log_every == 0
            @printf("%s %4d (%5.0f s): k [%s] GW, capex %.4e op %.4e total %.4e shed %.3g MWh, clipped %d\n", label, it, time() - t0,
                    join([@sprintf("%.1f", x / 1e3) for x in tot], ", "), s.parts.capex, s.parts.op,
                    s.parts.capex + s.parts.op, s.parts.shed, inf.nclip)
            flush(stdout)
        end
    end
    return trace, s
end
