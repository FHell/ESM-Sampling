# Sampler state of a chain and the level-0 base kernel.
#
# A `ChainSampler` holds the kernels of one chain: the fine-level local kernel
# (`LocalKernel`: local models, workspaces, flow frames, adaptive modes) and the
# level-0 kernel of the bias warmup (`GibbsState`: slice widths, plus the step
# sizes of the carrier scaling move).
#
# Level 0 (`base_step` = `local_base_step`, local.jl): a random scan over
# π₀-invariant moves (local operation sweep, weather slice updates, weather with
# pro-rata shear, capacity slice updates, FFBS, carrier scaling). Carrier scaling:
# for one investable carrier multiply its capacity coefficients by c and all its
# utilization coefficients by 1/c. Nodal dispatch is unchanged, so this follows
# the ridge k·ω = const along which investments move; MH with Jacobian
# c^(m_k − m_ω T).

mutable struct ChainSampler
    σscale::Vector{Float64}          # carrier scaling: log-scale step per k field
    nscale::Vector{Int}
    accscale::Vector{Int}
    adapt::Bool                      # adapt the level-0 step sizes (warmup)
    gibbs::GibbsState                # level-0 slice widths and workspaces (level0.jl)
    loc::LocalKernel                 # fine-level (and level-0 local) kernel (local.jl)
end

"Quantities of interest Q_ℓ(s): capacity totals, costs, shed, regime frequencies, weather means."
function quantities(P::Problem, s::State)
    E = P.E
    lk = P.lk[s.ℓ+1]
    caps = capacity_totals(E, lk, s.k)
    p = s.parts
    K = P.W.K
    rf = [count(==(k), s.m) / length(s.m) for k in 1:K]
    lay = P.W.lays[1]
    wm = [mean(fieldview(s.z0, lay, P.W.specs, f)[1, :]) for f in eachindex(P.W.specs)]
    return vcat(caps, [p.capex, p.op, p.capex + p.op, p.shed, p.penalty], rf, wm, [p.co2])
end

function quantity_names(P::Problem)
    vcat([String(s.name) for s in P.E.kspecs], ["capex", "opex", "total", "shed_MWh", "penalty"],
         ["regime$k" for k in 1:P.W.K], ["z0mean_" * String(s.name) for s in P.W.specs], ["co2_t"])
end

"v fields scaled together with k field f (utilizations of that carrier)."
function scaled_vfields(E::EnergyModel, f::Int)
    for g in E.gens
        g.kf == f && return g.vf > 0 ? (g.ucf > 0 ? [g.vf, g.ucf] : [g.vf]) : Int[]
    end
    for st in E.stors
        st.kf == f && return [st.fe]
    end
    return Int[]
end

function scale_step(rng, P::Problem, s::State, S::ChainSampler)
    E = P.E
    lk, lv = P.lk[s.ℓ+1], P.lv[s.ℓ+1]
    f = rand(rng, 1:length(E.kspecs))
    logc = S.σscale[f] * randn(rng)
    c = exp(logc)
    sn = copy(s)
    fieldview(sn.k, lk, E.kspecs, f) .*= c
    nlogJ = lk.m[f]
    for vf in scaled_vfields(E, f)
        fieldview(sn.v, lv, E.vspecs, vf) ./= c
        nlogJ -= lv.m[vf] * ntime(lv, E.vspecs[vf])
    end
    evaluate!(P, sn)
    logα = sn.L - s.L + nlogJ * logc
    acc = isfinite(logα) ? min(1.0, exp(logα)) : 0.0
    S.nscale[f] += 1
    if S.adapt
        S.σscale[f] *= exp((acc - 0.3) / sqrt(S.nscale[f]))
    end
    if rand(rng) < acc
        S.accscale[f] += 1
        return sn
    end
    return s
end

"One π₀-invariant level-0 transition (the level-0 kernel of the bias warmup)."
base_step(rng, P::Problem, s::State, S::ChainSampler) = local_base_step(rng, P, s, S)
