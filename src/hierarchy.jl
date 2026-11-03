# Level models π_ℓ (§4) and states.
#
#   π_ℓ(s_ℓ) ∝ e^{−β_k capex_ℓ} p^w_ℓ(x_ℓ) e^{−β_v C^{(ℓ)}} 1_{Ω_ℓ} e^{−b̂(x,k)}
#
# with x_ℓ = (m, z0, w₁…w_ℓ). The weather prior p^w_ℓ cancels exactly in the
# level increments, so states cache L_ℓ = log π_ℓ − log p^w_ℓ. The bias is
# learned at level 0 and evaluated on the level-0 restriction of (x, k) at all
# levels, so b̂_{ℓ+1} − b̂_ℓ = 0. F̂_ℓ (adaptive error model) is zero.

struct Problem
    E::EnergyModel
    W::WeatherModel
    levels::Vector{Tuple{Int,Int}}
    lk::Vector{Layout}
    lv::Vector{Layout}
    trk::Vector{Transfer}
    trv::Vector{Transfer}
    cons::Vector{Constraints}        # on [k; v]
    cons0::Constraints               # level 0 on [k; v; z0] (for HMC)
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
    cons = [build_constraints(E, lk[j], lv[j]) for j in 1:nl]
    cons0 = build_constraints(E, lk[1], lv[1]; nextra=W.lays[1].n)
    return Problem(E, W, collect(levels), lk, lv, trk, trv, cons, cons0, βk, βv, bias, use_bias)
end

with_temperatures(P::Problem, βk, βv) =
    Problem(P.E, P.W, P.levels, P.lk, P.lv, P.trk, P.trv, P.cons, P.cons0, βk, βv, P.bias, P.use_bias)

with_bias(P::Problem, use_bias::Bool) =
    Problem(P.E, P.W, P.levels, P.lk, P.lv, P.trk, P.trv, P.cons, P.cons0, P.βk, P.βv, P.bias, use_bias)

k_at0(P::Problem, ℓ, k) = restrict_to(k, P.lk, P.trk, P.E.kspecs, ℓ, 0)

function bias_term(P::Problem, k0, z0)
    P.use_bias || return 0.0
    X0 = weather_nodes(P.W, 0, z0)
    return bias_value(P.bias, P.E, P.lk[1], k0, X0, Float64(P.levels[1][2]))
end

"L_ℓ = log π_ℓ − log p^w_ℓ, and the cost parts."
function logtarget(P::Problem, ℓ::Int, k, v, x, z0)
    isfeasible(P.cons[ℓ+1], vcat(k, v)) || return -Inf, CostParts(NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    Xn = weather_nodes(P.W, ℓ, x)
    parts = energy_cost(P.E, P.lk[ℓ+1], P.lv[ℓ+1], k, v, Xn)
    b = bias_term(P, k_at0(P, ℓ, k), z0)
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

"Coarse part of a level-ℓ state (ℓ ≥ 1) and its fine modes (w_ℓ, ξ_k, ξ_v)."
function split_level(P::Problem, s::State)
    ℓ = s.ℓ
    kc, ξk = split_state(P.trk[ℓ], P.lk[ℓ], P.lk[ℓ+1], P.E.kspecs, s.k)
    vc, ξv = split_state(P.trv[ℓ], P.lv[ℓ], P.lv[ℓ+1], P.E.vspecs, s.v)
    sc = State(ℓ - 1, s.m, s.z0, s.ws[1:ℓ-1], s.xs[1:ℓ], kc, vc, -Inf, s.parts)
    return evaluate!(P, sc), (s.ws[ℓ], ξk, ξv)
end

"Fine state from a coarse state and fine modes."
function assemble(P::Problem, sc::State, w, ξk, ξv)
    ℓ = sc.ℓ + 1
    W = P.W
    ξ = details_from_w(W.details[ℓ], W.specs, W.lays[ℓ], W.lays[ℓ+1], W.trs[ℓ], sc.xs[end], sc.m, w, W.lays[1].h)
    x = merge_state(W.trs[ℓ], W.lays[ℓ], W.lays[ℓ+1], W.specs, sc.xs[end], ξ)
    k = merge_state(P.trk[ℓ], P.lk[ℓ], P.lk[ℓ+1], P.E.kspecs, sc.k, ξk)
    v = merge_state(P.trv[ℓ], P.lv[ℓ], P.lv[ℓ+1], P.E.vspecs, sc.v, ξv)
    s = State(ℓ, sc.m, sc.z0, vcat(sc.ws, [w]), vcat(sc.xs, [x]), k, v, -Inf, sc.parts)
    return evaluate!(P, s)
end

# --------------------------------------------------------------------------
# level-0 potential for HMC on q = [k; v; z0]
# --------------------------------------------------------------------------

function level0_potential(P::Problem, m)
    lk, lv, lw = P.lk[1], P.lv[1], P.W.lays[1]
    nk, nv = lk.n, lv.n
    Δt = Float64(P.levels[1][2])
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
            b, gbX, gbk = bias_value(P.bias, P.E, lk, collect(k), Xn, Δt; grad=true)
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
