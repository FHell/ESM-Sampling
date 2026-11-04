# Weather-refresh relaxation: from an equilibrated state, replace the weather by a fresh exact
# draw from its prior (capacities and the dispatch coordinates kept) and record the cost and
# the capacity scores along the operation sweeps that follow, against a control that keeps the
# old weather. This prices a "weather from its prior by refresh" step of the sampler.
# Needs the session of setup40.jl (P, S, LM, lgv_op!, lgv_scores).

"Node-mean capacity scores r ŝ_k per k field (interior nodes of `mask`)."
function mean_scores(P, s, mask)
    r = P.βk / P.βv
    sk, _ = lgv_scores(P, s)
    return [any(mask[f]) ? mean(r .* sk[f][mask[f]]) : NaN for f in eachindex(sk)]
end

"""
    refresh_trace(rng, P, S, LM, s, n; newweather) -> trace

`n` operation sweeps from `s` after (newweather = true) replacing its weather by a draw from
the prior; per sweep the cost parts and the node-mean capacity scores.
"""
function refresh_trace(rng, P, S, LM, s, n; newweather::Bool, wrng=rng)
    if newweather
        m, z0, ws = ML.sample_weather(wrng, P.W, s.ℓ)
        s = ML.make_state(P, s.ℓ, m, z0, ws, s.k, s.v)
    else
        s = copy(s)
    end
    Kr = ML.knodes(P, s.ℓ, s.k)
    mask = [(Kr[f] .> 50) .& (Kr[f] .< P.E.kspecs[f].hi .- 1) for f in eachindex(Kr)]
    tr = NamedTuple[(; it=0, op=s.parts.op, shed=s.parts.shed, pen=s.parts.penalty, sc=mean_scores(P, s, mask))]
    for it in 1:n
        s = lgv_op!(rng, P, S.loc, LM, s)
        push!(tr, (; it, op=s.parts.op, shed=s.parts.shed, pen=s.parts.penalty, sc=mean_scores(P, s, mask)))
    end
    return tr, s
end

"""
    refresh_weather(P, LM, s, wrng; comp) -> s′

The state with its weather replaced by a draw from the prior. With `comp = true` the
dispatchable generators at every node and step take up the change of the node residual
(demand minus supply) caused by the new weather: an increase in proportion to their headroom
(up to it), a decrease in proportion to their output; the rest stays on the residual.
"""
function refresh_weather(P, LM, s, wrng; comp::Bool=true)
    E = P.E
    lv = P.lv[s.ℓ+1]
    r0 = copy(ML.LocalCost(P, LM, s).r)
    m, z0, ws = ML.sample_weather(wrng, P.W, s.ℓ)
    s1 = ML.make_state(P, s.ℓ, m, z0, ws, s.k, s.v)
    comp || return s1
    LC = ML.LocalCost(P, LM, s1)
    Δ = LC.r .- r0
    for n in axes(Δ, 1), τ in axes(Δ, 2)
        need = Δ[n, τ]
        abs(need) < 1e-6 && continue
        us = [(f, i) for (f, i) in LM.genunits[n] if E.gens[LM.role[f][2]].wf == 0]
        isempty(us) && continue
        κs_ = [ML.κunit(P, LM, LC, f, i, τ) for (f, i) in us]
        ps = [κ * clamp(LC.Y[f][i, τ], 0.0, 1.0) for ((f, i), κ) in zip(us, κs_)]
        ws_ = need > 0 ? max.(κs_ .- ps, 0.0) : ps
        tot = sum(ws_)
        tot > 1e-9 || continue
        a = need > 0 ? min(need, tot) / tot : max(need, -tot) / tot
        for ((f, i), κ, wq) in zip(us, κs_, ws_)
            abs(κ) < 1e-9 && continue
            Δy = a * wq / κ
            Δy == 0.0 && continue
            m_ = lv.m[f]; o = lv.off[f] + (τ - 1) * m_
            Ut = E.vspecs[f].B.Ut
            for q in ML.nzrange(Ut, i)
                j = ML.rowvals(Ut)[q]
                j <= m_ && (s1.v[o+j] += Δy * ML.nonzeros(Ut)[q])
            end
        end
    end
    return ML.evaluate!(P, s1)
end
