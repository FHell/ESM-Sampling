# Level models π_ℓ (docs/system/ensemble.md) and states.
#
#   π_ℓ(s_ℓ) ∝ e^{−β_k capex_ℓ} p^w_ℓ(x_ℓ) e^{−β_v C^{(ℓ)}} e^{−b̂(x,k)}
#
# with x_ℓ = (m, z0, w₁…w_ℓ). All constraints are priced in C. States cache
# L_ℓ = log π_ℓ − log p^w_ℓ (the weather prior enters the moves that change the
# weather). The bias is evaluated at its level ℓ_b (level 0 during the level-0
# warmup, the finest level after the lift) on the restriction of (x, k) to that
# level, so states at levels ℓ ≥ ℓ_b can be evaluated.

struct Problem
    E::EnergyModel
    W::WeatherModel
    levels::Vector{Tuple{Int,Int}}
    lk::Vector{Layout}
    lv::Vector{Layout}
    trk::Vector{Transfer}
    trv::Vector{Transfer}
    βk::Float64
    βv::Float64
    bias::BiasModel
    use_bias::Bool
end

nlevels(P::Problem) = length(P.levels)

mutable struct State
    ℓ::Int
    m::Vector{Int}
    z0::Vector{Float64}
    ws::Vector{Vector{Float64}}
    xs::Vector{Vector{Float64}}      # standardized anomalies at levels 0..ℓ
    k::Vector{Float64}
    v::Vector{Float64}
    L::Float64
    parts::CostParts
end

Base.copy(s::State) = State(s.ℓ, copy(s.m), copy(s.z0), copy.(s.ws), copy.(s.xs), copy(s.k), copy(s.v), s.L, s.parts)

function build_problem(E::EnergyModel, W::WeatherModel, levels, βk, βv, bias; use_bias=true)
    lk = [Layout(E.kspecs, M, h) for (M, h) in levels]
    lv = [Layout(E.vspecs, M, h) for (M, h) in levels]
    nl = length(levels)
    trk = [Transfer(E.kspecs, lk[j], lk[j+1]) for j in 1:nl-1]
    trv = [Transfer(E.vspecs, lv[j], lv[j+1]) for j in 1:nl-1]
    return Problem(E, W, collect(levels), lk, lv, trk, trv, βk, βv, bias, use_bias)
end

with_bias(P::Problem, use_bias::Bool) =
    Problem(P.E, P.W, P.levels, P.lk, P.lv, P.trk, P.trv, P.βk, P.βv, P.bias, use_bias)

k_at0(P::Problem, ℓ, k) = restrict_to(k, P.lk, P.trk, P.E.kspecs, ℓ, 0)

"Capacity coefficients and weather anomalies of a level-ℓ state restricted to the level ℓ_b of the bias."
function bias_restrict(P::Problem, ℓ::Int, k, x, z0)
    ℓb = P.bias.ℓb
    ℓ >= ℓb || error("the bias is evaluated at level $ℓb, which a level-$ℓ state does not resolve")
    kb = ℓ == ℓb ? k : restrict_to(k, P.lk, P.trk, P.E.kspecs, ℓ, ℓb)
    xb = ℓ == ℓb ? x : (ℓb == 0 ? z0 : restrict_to(x, P.W.lays, P.W.trs, P.W.specs, ℓ, ℓb))
    return kb, xb
end

"b̂ of a level-ℓ state with capacity coefficients k, anomalies x of level ℓ and level-0 anomalies z0 (0 at α = 0)."
function bias_term(P::Problem, ℓ::Int, k, x, z0)
    P.use_bias || return 0.0
    kb, xb = bias_restrict(P, ℓ, k, x, z0)
    ℓb = P.bias.ℓb
    return bias_value(P.bias, P.E, P.lk[ℓb+1], kb, weather_nodes(P.W, ℓb, xb))
end
bias_term(P::Problem, s::State) = bias_term(P, s.ℓ, s.k, s.xs[end], s.z0)

"Incremental bias cache of a state at the level ℓ_b of the bias (bias.jl; nothing at α = 0)."
function BiasCache(P::Problem, s::State)
    P.use_bias || return nothing
    s.ℓ == P.bias.ℓb || error("the bias is evaluated at level $(P.bias.ℓb), the state is at level $(s.ℓ): " *
                             "lift the bias to the chains' level first (lift_bias!)")
    return BiasCache(P.bias, P.E, P.lk[s.ℓ+1], s.k, weather_nodes(P.W, s.ℓ, s.xs[end]))
end

"Lift the bias to level ℓ (bias.jl `lift_bias!`), with sample steps drawn from `rng`."
lift_bias!(P::Problem, ℓ::Int; rng, nsamples::Int=P.bias.nsamp, nsteps::Int=4096) =
    lift_bias!(P.bias, P.E, P.W, P.lk, P.trk, ℓ; rng=rng, nsamples=nsamples, nsteps=nsteps)

"L_ℓ = log π_ℓ − log p^w_ℓ, and the cost parts."
function logtarget(P::Problem, ℓ::Int, k, v, x, z0)
    Xn = weather_nodes(P.W, ℓ, x)
    parts = energy_cost(P.E, P.lk[ℓ+1], P.lv[ℓ+1], k, v, Xn)
    b = bias_term(P, ℓ, k, x, z0)
    return -P.βk * parts.capex - P.βv * parts.op - b + log_measure(P.E, P.lk[ℓ+1], P.lv[ℓ+1], k), parts
end

function evaluate!(P::Problem, s::State)
    s.L, s.parts = logtarget(P, s.ℓ, s.k, s.v, s.xs[end], s.z0)
    return s
end

"State at level ℓ from weather draw (m, z0, ws) and level-ℓ coordinates k, v."
function make_state(P::Problem, ℓ, m, z0, ws, k, v)
    xs = [z0]
    for j in 1:ℓ
        h0 = P.W.lays[1].h
        ξ = details_from_w(P.W.details[j], P.W.specs, P.W.lays[j], P.W.lays[j+1], P.W.trs[j], xs[end], m, ws[j], h0)
        push!(xs, merge_state(P.W.trs[j], P.W.lays[j], P.W.lays[j+1], P.W.specs, xs[end], ξ))
    end
    s = State(ℓ, copy(m), copy(z0), copy.(ws[1:ℓ]), xs, copy(k), copy(v), -Inf,
              CostParts(NaN, NaN, NaN, NaN, NaN, NaN, NaN))
    return evaluate!(P, s)
end

# --------------------------------------------------------------------------
# level-0 potential on q = [k; v; z0] (for the optional pre-descent)
# --------------------------------------------------------------------------

function level0_potential(P::Problem, m)
    lk, lv, lw = P.lk[1], P.lv[1], P.W.lays[1]
    nk, nv = lk.n, lv.n
    return function (q)
        k = view(q, 1:nk)
        v = view(q, nk+1:nk+nv)
        z0 = q[nk+nv+1:end]
        Xn = weather_nodes(P.W, 0, z0)
        parts, (gkc, gko, gv, gX) = energy_cost(P.E, lk, lv, k, v, Xn; grad=true)
        lp, glp = logp_z0(P.W, m, z0)
        U = P.βk * parts.capex + P.βv * parts.op - lp
        gk = P.βk .* gkc .+ P.βv .* gko
        gXs = [P.βv .* g for g in gX]
        if P.use_bias
            P.bias.ℓb == 0 || error("level0_potential needs the bias at level 0")
            b, gbX, gbk = bias_value(P.bias, P.E, lk, collect(k), Xn; grad=true)
            U += b
            gk .+= gbk
            for f in eachindex(gXs)
                gXs[f] .+= gbX[f]
            end
        end
        gz = weather_grad_to_x(P.W, 0, gXs) .- glp
        return U, vcat(gk, P.βv .* gv, gz)
    end
end
