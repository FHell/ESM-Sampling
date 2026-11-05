# The replica–Langevin sampler (docs/system/sampler.md §11, docs/system/ensemble.md §4):
#
#   * M weather replicas (m_j, z0_j, w_j, v_j) share the capacities k;
#   * the weather of every replica follows a prior-reversible process of its own (pCN in the
#     non-centred innovations, regime window draws, optionally the centred FFBS), driven by a
#     random stream of its own and never reading k or v: every replica's weather is an
#     autonomous Markov chain whose stationary law is the prior ρ^x;
#   * the operation of every replica follows its weather and k by operation sweeps
#     (`operation_sweep!`, the fine kernel's), after a compensated initialisation when its
#     weather moved;
#   * the capacities take one Langevin step in w = log(k + κ_s) per step, driven by the
#     replica-averaged sampled capacity score −β_k∂K + r s̄_k (weights over the relaxed replicas,
#     times the optional stress tilt exp(λΦ)), applied to every replica with its dispatch
#     following (fixed output for dispatchable generators, the kop curve for renewables and
#     storage);
#   * a lag probe compares the in-flight score of a relaxed replica with its equilibrated score
#     at frozen (k, x) and adapts the Langevin step per carrier during the warmup.
#
# Building blocks with explicit random numbers (the device port's reference): `pcn_weather`
# (normals given), `regime_window` (uniforms given), `langevin_update` (normals given); the rng
# forms draw them with scalar `randn(rng)` / `rand(rng)` calls in a documented order, so that a
# `MockRNG` feeds the same numbers.

# ======================================================================================
# 1. the weather process (reads neither k nor v)
# ======================================================================================

"Square roots of the VAR covariances (the factors of `sample_weather`): `Lk[k] Lk[k]' = Σ_k`, `L0 L0' = Σ_0`."
struct VarFactors
    Lk::Vector{LowerTriangular{Float64,Matrix{Float64}}}
    L0::LowerTriangular{Float64,Matrix{Float64}}
end
VarFactors(W::WeatherModel) =
    VarFactors([cholesky(Symmetric(inv(W.Σinv[k]))).L for k in 1:W.K], cholesky(Symmetric(inv(W.Σ0inv))).L)

"VAR mean of level-0 step τ ≥ 2 given z0(τ−1) in regime with coefficients B (the expression of `sample_weather`)."
@inline function var_mean(B, zprev, τ, T0, n)
    c, cs, sn = harmonics(τ, T0)
    return B[:, 1:n] * zprev .+ c .* B[:, n+1] .+ cs .* B[:, n+2] .+ sn .* B[:, n+3]
end

"""
    z0_innovations(W, F, m, z0) -> E (n0 × T0)

Innovations of the level-0 VAR at the regime path m: z0(1) = L₀e₁,
z0(τ) = μ_m(τ) + L_{m_τ} e_τ. For fixed m the map e ↦ z0 is affine and invertible with a
constant Jacobian, so the prior p(z0 | m) is N(0, I) in e.
"""
function z0_innovations(W::WeatherModel, F::VarFactors, m, z0)
    Z = z0matrix(z0, W.lays[1], W.specs)
    n, T0 = size(Z)
    E = zeros(n, T0)
    E[:, 1] .= F.L0 \ Z[:, 1]
    for τ in 2:T0
        E[:, τ] .= F.Lk[m[τ]] \ (Z[:, τ] .- var_mean(W.Bm[m[τ]], Z[:, τ-1], τ, T0, n))
    end
    return E
end

"z0 from its innovations at the regime path m (the recursion of `sample_weather`, bitwise)."
function z0_from_innovations(W::WeatherModel, F::VarFactors, m, E::AbstractMatrix)
    n, T0 = size(E)
    Z = zeros(n, T0)
    Z[:, 1] .= F.L0 * E[:, 1]
    for τ in 2:T0
        Z[:, τ] .= var_mean(W.Bm[m[τ]], Z[:, τ-1], τ, T0, n) .+ F.Lk[m[τ]] * E[:, τ]
    end
    return z0vector(Z, W.lays[1], W.specs)
end

"""
    pcn_rotate(u, ξ, θ) -> cos θ u + sin θ ξ

The pCN step of angle θ for N(0, I): the transition density N(u′; cos θ u, sin²θ I) satisfies
N(u) q(u → u′) = N(u′) q(u′ → u) (reversible). Its reverse noise is `pcn_reverse_noise`.
"""
pcn_rotate(u, ξ, θ::Float64) = cos(θ) .* u .+ sin(θ) .* ξ

"Noise ξ′ that takes u′ back to u in one pCN step: ξ′ = (u − cos θ u′)/sin θ (then |u|² + |ξ|² = |u′|² + |ξ′|²)."
pcn_reverse_noise(u, u1, θ::Float64) = (u .- cos(θ) .* u1) ./ sin(θ)

"""
    draw_pcn_noise(rng, n0, T0, ws) -> (ξe, ξw)

Standard normals of one pCN step: ξe (n0 × T0, column by column, i.e. step by step) then ξw
per level 1…L (length of that level's innovations), each from scalar `randn(rng)` calls.
"""
function draw_pcn_noise(rng, n::Int, T0::Int, ws)
    ξe = zeros(n, T0)
    for τ in 1:T0, i in 1:n
        ξe[i, τ] = randn(rng)
    end
    ξw = [[randn(rng) for _ in eachindex(w)] for w in ws]
    return ξe, ξw
end

"""
    pcn_weather(W, F, m, z0, ws, θ, ξe, ξw) -> (z0′, ws′)
    pcn_weather(rng, W, F, m, z0, ws, θ)

pCN step of angle θ in the non-centred coordinates (e, w₁…w_L) at fixed regimes: the level-0
innovations e of z0 (`z0_innovations`) and the detail innovations w move by `pcn_rotate`, z0
follows from e. Reversible for the prior p(z0 | m) ∏_ℓ N(w_ℓ; 0, I).
"""
function pcn_weather(W::WeatherModel, F::VarFactors, m, z0, ws, θ::Float64, ξe::AbstractMatrix, ξw)
    E = pcn_rotate(z0_innovations(W, F, m, z0), ξe, θ)
    return z0_from_innovations(W, F, m, E), [pcn_rotate(w, x, θ) for (w, x) in zip(ws, ξw)]
end
function pcn_weather(rng, W::WeatherModel, F::VarFactors, m, z0, ws, θ::Float64)
    ξe, ξw = draw_pcn_noise(rng, n0(W), W.lays[1].T, ws)
    return pcn_weather(W, F, m, z0, ws, θ, ξe, ξw)
end

"Index k of the categorical distribution ∝ p (unnormalised) at the uniform u: the first k with u Σp ≤ Σ_{i≤k} p_i."
function categorical(p::AbstractVector, u::Float64)
    t = u * sum(p)
    c = 0.0
    for k in eachindex(p)
        c += p[k]
        t <= c && return k
    end
    return length(p)
end

"""
    regime_window_messages(W, m, a, b) -> β (K × (b − a + 1))

Backward messages of the regime window [a, b] given the regime m[b+1] after it:
β_τ(i) = (P^{b+1−τ})[i, m_{b+1}] (all ones at b = T0, no constraint).
"""
function regime_window_messages(W::WeatherModel, m, a::Int, b::Int)
    T0, K = length(m), W.K
    β = ones(K, b - a + 1)
    if b < T0
        β[:, end] .= W.Ptr[:, m[b+1]]
        for τ in b-1:-1:a
            β[:, τ-a+1] .= W.Ptr * β[:, τ-a+2]
        end
    end
    return β
end

"Forward probabilities (unnormalised) of window step τ given the regime before it: (π₀ at τ = 1, else P[m_{τ−1}, :]) ⊙ β_τ."
regime_window_probs(W::WeatherModel, β, a::Int, τ::Int, mprev::Int) =
    (τ == 1 ? W.π0 : W.Ptr[mprev, :]) .* β[:, τ-a+1]

"""
    regime_window(W, m, a, b, us) -> m′
    regime_window(rng, W, m, a, b)

Exact draw of the regimes m[a..b] from the Markov-chain prior given the regimes outside the
window: backward messages (`regime_window_messages`), then forward sampling with the uniforms
`us[1..b−a+1]` (`categorical`). In the non-centred coordinates (m, e, w) the regimes are a
priori independent of (e, w), so this is the exact Gibbs conditional of the prior.
"""
function regime_window(W::WeatherModel, m, a::Int, b::Int, us::AbstractVector)
    β = regime_window_messages(W, m, a, b)
    m1 = copy(m)
    for τ in a:b
        m1[τ] = categorical(regime_window_probs(W, β, a, τ, τ == 1 ? 0 : m1[τ-1]), us[τ-a+1])
    end
    return m1
end
regime_window(rng, W::WeatherModel, m, a::Int, b::Int) = regime_window(W, m, a, b, [rand(rng) for _ in a:b])

"Window [a, b] of length B (cut at T0) with a uniform start from the uniform u (state-independent)."
window_bounds(T0::Int, B::Int, u::Float64) = (a = min(T0, 1 + floor(Int, u * T0)); (a, min(T0, a + B - 1)))

"""
    weather_process_step(rng, P, F, s, θ; window, ffbs) -> s′

One step of a replica's weather process, from its own stream `rng`: the pCN step of angle θ
(θ > 0; `draw_pcn_noise` order), then with `window = B > 0` a regime window move (one uniform for
the start, then B uniforms; z0 re-built from its innovations at the new regimes, the detail
innovations kept), then with `ffbs = true` the centred FFBS of the regimes given the anomalies
(`fine_regime_step!`; x unchanged). Each block is reversible for the prior; none reads k or v.
The capacities and the operation coefficients of `s` are kept (the state is re-evaluated).
"""
function weather_process_step(rng, P::Problem, F::VarFactors, s::State, θ::Float64; window::Int=0, ffbs::Bool=false)
    W = P.W
    m, z0, ws = s.m, s.z0, s.ws
    changed = false
    if θ > 0
        z0, ws = pcn_weather(rng, W, F, m, z0, ws, θ)
        changed = true
    end
    if window > 0
        a, b = window_bounds(length(m), window, rand(rng))
        E = z0_innovations(W, F, m, z0)
        m = regime_window(rng, W, m, a, b)
        z0 = z0_from_innovations(W, F, m, E)
        changed = true
    end
    s = changed ? make_state(P, s.ℓ, m, z0, ws, s.k, s.v) : copy(s)
    ffbs && fine_regime_step!(rng, P, s)
    return s
end

"""
    compensate_weather!(P, LM, r0, s) -> s

Compensated initialisation of the operation after a weather change: at every node and step the
weather-independent dispatchable generators take up the change of the node residual from `r0`
(before the change) to that of `s`, an increase in proportion to their headroom (up to it), a
decrease in proportion to their output; the rest stays on the residual. Changes v only.
"""
function compensate_weather!(P::Problem, LM::LocalModel, r0::AbstractMatrix, s::State)
    E = P.E
    lv = P.lv[s.ℓ+1]
    LC = LocalCost(P, LM, s)
    Δ = LC.r .- r0
    for n in axes(Δ, 1), τ in axes(Δ, 2)
        need = Δ[n, τ]
        abs(need) < 1e-6 && continue
        us = [(f, i) for (f, i) in LM.genunits[n] if E.gens[LM.role[f][2]].wf == 0]
        isempty(us) && continue
        κs = [κunit(P, LM, LC, f, i, τ) for (f, i) in us]
        ps = [κ * clamp(LC.Y[f][i, τ], 0.0, 1.0) for ((f, i), κ) in zip(us, κs)]
        wq = need > 0 ? max.(κs .- ps, 0.0) : ps
        tot = sum(wq)
        tot > 1e-9 || continue
        a = need > 0 ? min(need, tot) / tot : max(need, -tot) / tot
        for ((f, i), κ, w) in zip(us, κs, wq)
            abs(κ) < 1e-9 && continue
            Δy = a * w / κ
            Δy == 0.0 && continue
            m_ = lv.m[f]; o = lv.off[f] + (τ - 1) * m_
            Ut = E.vspecs[f].B.Ut
            for q in nzrange(Ut, i)
                j = rowvals(Ut)[q]
                j <= m_ && (s.v[o+j] += Δy * nonzeros(Ut)[q])
            end
        end
    end
    return evaluate!(P, s)
end

# ======================================================================================
# 2. weather features and the stress tilt
# ======================================================================================

"""
    weather_features(P, x, kref) -> NamedTuple

Features of level-L anomalies x after the data map: annual mean system load (GW), annual mean
capacity factors of the weather-dependent carriers (node means), and the residual load
D_t − Σ k_ref a_t at the fixed reference capacities `kref` (node vectors per k field): annual
mean, peak step and worst 7-step mean (GW).
"""
function weather_features(P::Problem, x::AbstractVector, kref)
    E = P.E
    L = length(P.W.lays) - 1
    Xn = weather_nodes(P.W, L, x)
    D = vec(sum(max.(Xn[1], 0.0), dims=1))
    R = zeros(length(D))
    cf = Float64[]
    for g in E.gens
        g.wf > 0 || continue
        A = clamp.(Xn[g.wf], 0.0, 1.0)
        push!(cf, mean(A))
        g.kf > 0 && (R .+= vec(kref[g.kf]' * A))
    end
    res = (D .- R) ./ 1e3
    T = length(res)
    w7 = maximum(mean(res[mod1.(t:t+6, T)]) for t in 1:T)
    return (; load=mean(D) / 1e3, cf, resmean=mean(res), respeak=maximum(res), res7=w7)
end

"Feature vector of a replica: `weather_features` plus the regime frequencies and the number of regime switches."
replica_features(P::Problem, s::State, kref) =
    (f = weather_features(P, s.xs[end], kref);
     vcat(f.load, f.cf, f.resmean, f.respeak, f.res7, [count(==(k), s.m) / length(s.m) for k in 1:P.W.K], count(diff(s.m) .!= 0)))
replica_feature_names(P::Problem) =
    vcat("load_GW", ["cf_" * String(P.E.wspecs[g.wf].name) for g in P.E.gens if g.wf > 0], "resload_mean_GW", "resload_peak_GW",
         "resload_7step_GW", ["regime_$k" for k in 1:P.W.K], "regime_switches")

"""
The declared tilt functional Φ(x): the annual mean residual load at the fixed reference
capacities `kref`, standardised by its prior mean μ and sd σ (estimated from exact draws).
"""
struct TiltFunctional
    kref::Vector{Vector{Float64}}
    μ::Float64
    σ::Float64
end
(T::TiltFunctional)(P::Problem, x) = (weather_features(P, x, T.kref).resmean - T.μ) / T.σ

"`TiltFunctional` with μ and σ from `n` exact weather draws at the finest level (stream `rng`)."
function TiltFunctional(P::Problem, kref; rng, n::Int=200)
    L = nlevels(P) - 1
    v = [begin m, z0, ws = sample_weather(rng, P.W, L); weather_features(P, weather_at_level(P.W, L, m, z0, ws), kref).resmean end
         for _ in 1:n]
    return TiltFunctional(kref, mean(v), std(v))
end

"""
    tilt_weights(λ, Φs, active) -> ω

Self-normalised weights exp(λΦ_j)/Σ_active exp(λΦ) over the active replicas (0 for the others;
all replicas if none is active). At λ = 0 they are exactly 1/M_active, bitwise those of
`uniform_weights`.
"""
function tilt_weights(λ::Float64, Φs::AbstractVector, active::AbstractVector{Bool})
    any(active) || (active = trues(length(Φs)))
    a = [act ? λ * Φ : -Inf for (act, Φ) in zip(active, Φs)]
    w = exp.(a .- maximum(a))
    return w ./ sum(w)
end
"Weights 1/M_active over the active replicas (all if none is active): the untilted drift."
function uniform_weights(active::AbstractVector{Bool})
    any(active) || (active = trues(length(active)))
    w = [a ? 1.0 : 0.0 for a in active]
    return w ./ sum(w)
end

# ======================================================================================
# 3. the capacity score, the Langevin update and the dispatch-following maps
# ======================================================================================

"""
    capacity_scores(P, s) -> (sk, fk)

Per k field, node vectors of the sampled capacity score ŝ_k = −β_v ∂_k C + ∂_k log J and of
the capex force β_k ∂_k K (with the box penalty), in node capacity coordinates (MW). The
Langevin drift in k is −fk + r ŝ_k (r = β_k/β_v); in w = log(k + κ_s) it is (k + κ_s)(−fk + r ŝ_k) + 1
(`langevin_drift_w`). Without the bias, −fk + ŝ_k is the gradient of the log target at fixed
(v, x) (checked against finite differences in test/test_replicas.jl).
"""
function capacity_scores(P::Problem, s::State)
    E = P.E
    lk, lv = P.lk[s.ℓ+1], P.lv[s.ℓ+1]
    Xn = weather_nodes(P.W, s.ℓ, s.xs[end])
    _, (gkc, gko, _, _) = energy_cost(E, lk, lv, s.k, s.v, Xn; grad=true)
    sk = -P.βv .* gko .+ log_measure_grad(E, lk, lv, s.k)
    fk = P.βk .* gkc
    ng(g) = [cols(E.kspecs[f].B, lk.m[f]) * vec(fieldview(g, lk, E.kspecs, f)) for f in eachindex(E.kspecs)]
    return ng(sk), ng(fk)
end

"Investable units at a complete level: (:disp | :ren | :stor, group, index, k field)."
function capacity_units(P::Problem, LM::LocalModel)
    E = P.E
    lk = P.lk[LM.ℓ+1]
    units = Tuple{Symbol,Int,Int,Int}[]
    for (gi, g) in enumerate(E.gens)
        (g.kf > 0 && iscomplete(lk, E.kspecs, g.kf)) || continue
        append!(units, [(g.vf > 0 ? :disp : :ren, gi, i, g.kf) for i in eachindex(g.bus)])
    end
    for (si, st) in enumerate(E.stors)
        (st.kf > 0 && iscomplete(lk, E.kspecs, st.kf) && LM.complete[st.fe]) || continue
        append!(units, [(:stor, si, i, st.kf) for i in eachindex(st.bus)])
    end
    return units
end

"Drift of the Langevin step in w = log(k + κ_s): η ((k + κ_s) g + 1), g the drift in k (the +1 is the Jacobian of the change of variables)."
langevin_drift_w(k, g, η::Float64, κs::Float64) = η .* ((k .+ κs) .* g .+ 1)

"""
    langevin_update(k, g, ξ; η, κs, κmin, clip, hi) -> (k′, nclip)

The Langevin step of one k field in w = log(k + κ_s) with given standard normals ξ (one per
node): w′ = w + clamp(η((k + κ_s)g + 1), ±clip) + √(2η) ξ; reflected at w_min = log(κmin + κ_s)
(w′ < w_min → 2w_min − w′) and, with `hi` (the box, node vector), at w_max = log(max(hi, κmin + 1) + κ_s)
when the step crosses it from below (then max(2w_max − w′, w_min)). Returns k′ = e^{w′} − κ_s and
the number of clipped drifts.
"""
function langevin_update(k::AbstractVector, g::AbstractVector, ξ::AbstractVector; η::Float64, κs::Float64, κmin::Float64=1.0,
                         clip::Float64=Inf, hi=nothing)
    w = log.(k .+ κs)
    dw0 = langevin_drift_w(k, g, η, κs)
    nclip = count(abs.(dw0) .> clip)
    w1 = w .+ clamp.(dw0, -clip, clip) .+ sqrt(2η) .* ξ
    wmin = log(κmin + κs)
    w1 = [x < wmin ? 2wmin - x : x for x in w1]
    if hi !== nothing
        wmax = log.(max.(hi, κmin + 1.0) .+ κs)
        w1 = [x > wm && w0 <= wm ? max(2wm - x, wmin) : x for (x, wm, w0) in zip(w1, wmax, w)]
    end
    return exp.(w1) .- κs, nclip
end

"""
    langevin_capacities(rng, P, units, Kraw, sk̄, fk; ηw, κs, κmin, clip, box) -> (knew, nclip)

The Langevin step of every investable k field (in field order; fields with ηw = 0 are frozen
and draw no noise): drift −fk + r s̄_k from the (replica-averaged) score, one standard normal per
node from scalar `randn(rng)` calls, `langevin_update` per field (`box`: reflection at the
upper bound).
"""
function langevin_capacities(rng, P::Problem, units, Kraw, sk̄, fk; ηw, κs, κmin::Float64=1.0, clip::Float64=Inf, box::Bool=true)
    r = P.βk / P.βv
    knew = deepcopy(Kraw)
    nclip = 0
    for kf in unique(u[4] for u in units)
        ηw[kf] == 0 && continue
        ξ = [randn(rng) for _ in eachindex(Kraw[kf])]
        knew[kf], nc = langevin_update(Kraw[kf], -fk[kf] .+ r .* sk̄[kf], ξ; η=ηw[kf], κs=κs[kf], κmin=κmin, clip=clip,
                                       hi=box ? P.E.kspecs[kf].hi : nothing)
        nclip += nc
    end
    return knew, nclip
end

"""
    set_dispatchable_capacity!(P, LM, LC, s, gi, i, k0, k1)

Move the raw capacity of dispatchable generator i of group gi from k0 to k1 at fixed output:
the utilizations (and the commitment) at every step scaled by c(k0)/c(k1), c the clamp to the
box (the map of `kscale`). Reads the utilizations from `LC` (not updated).
"""
function set_dispatchable_capacity!(P::Problem, LM::LocalModel, LC::LocalCost, s::State, gi::Int, i::Int, k0::Float64, k1::Float64)
    E = P.E
    lk, lv = P.lk[s.ℓ+1], P.lv[s.ℓ+1]
    g = E.gens[gi]
    sp = E.kspecs[g.kf]
    Ut = sp.B.Ut
    for q in nzrange(Ut, i)
        s.k[lk.off[g.kf]+rowvals(Ut)[q]] += (k1 - k0) * nonzeros(Ut)[q]
    end
    sc = kc_ratio(clamp(k0, sp.lo[i], sp.hi[i]), clamp(k1, sp.lo[i], sp.hi[i]))
    sc == 1.0 && return s
    for f in (g.ucf > 0 ? (g.vf, g.ucf) : (g.vf,))
        m = lv.m[f]
        Utv = E.vspecs[f].B.Ut
        for τ in 1:LM.T
            Δy = LC.Y[f][i, τ] * (sc - 1)
            o = lv.off[f] + (τ - 1) * m
            for q in nzrange(Utv, i)
                s.v[o+rowvals(Utv)[q]] += Δy * nonzeros(Utv)[q]
            end
        end
    end
    return s
end

"""
    follow_capacities!(P, LM, s, units, Kraw, knew; κmin) -> (s, LC)

Move the capacities of `s` from the raw node capacities `Kraw` to `knew` with the dispatch
following: dispatchable generators at fixed output (`set_dispatchable_capacity!`), then every
renewable and storage unit in `units` order along its kop curve (`kop_prepare`, `kop_commit!` at
u = log(k′/k): fixed utilizations, the node's dispatchable generators following
share-preservingly); a unit below κmin moves its capacity alone. Returns the evaluated state and
its local caches (equal to a rebuild). The same `Kraw`, `knew` give bitwise the same k in every
replica.
"""
function follow_capacities!(P::Problem, LM::LocalModel, s::State, units, Kraw, knew; κmin::Float64=1.0)
    E = P.E
    LC = LocalCost(P, LM, s)
    for (kind, gi, i, kf) in units
        kind == :disp || continue
        set_dispatchable_capacity!(P, LM, LC, s, gi, i, Kraw[kf][i], knew[kf][i])
    end
    LC = LocalCost(P, LM, s)
    Kr = knodes(P, s.ℓ, s.k)
    for (kind, gi, i, kf) in units
        kind == :disp && continue
        k0 = Kr[kf][i]
        mv = kop_prepare(P, LM, LC, s, [(kind == :ren ? 1 : 2, gi, i)]; κmin=κmin, Kraw=Kr)
        if mv === nothing                                     # k0 < κmin: the capacity alone
            Ut = E.kspecs[kf].B.Ut
            for q in nzrange(Ut, i)
                s.k[P.lk[s.ℓ+1].off[kf]+rowvals(Ut)[q]] += (knew[kf][i] - k0) * nonzeros(Ut)[q]
            end
            LC = LocalCost(P, LM, s); Kr = knodes(P, s.ℓ, s.k)
            continue
        end
        kop_commit!(P, LM, LC, s, mv, log(knew[kf][i] / k0); Kraw=Kr)
    end
    return evaluate!(P, s), LC
end

"Interior nodes per k field: capacity above `lo` MW and below the box (the score diagnostics average over them)."
interior_nodes(P::Problem, Kr; lo::Float64=50.0) = [(Kr[f] .> lo) .& (Kr[f] .< P.E.kspecs[f].hi .- 1) for f in eachindex(Kr)]
"Mean of a node vector per field over the masked nodes (NaN where none)."
masked_means(v, mask) = [any(mask[f]) ? mean(v[f][mask[f]]) : NaN for f in eachindex(v)]

# ======================================================================================
# 4. replica sets, controls, one step
# ======================================================================================

"Settings of the replica sampler (from `Config`, `ReplicaCtl(cfg, P)`)."
Base.@kwdef mutable struct ReplicaCtl
    M::Int = 8                            # replicas per chain
    inner::Int = 1                        # operation sweeps per replica and step
    θ::Float64 = 0.0                      # continuous pCN angle per step (0: off)
    burst_every::Int = 40                 # a pCN burst of angle φ per replica every that many steps, staggered (0: off)
    φ::Float64 = 0.3                      # burst angle
    window::Int = 60                      # regime window length B (level-0 steps; 0: no window moves)
    window_prob::Float64 = 0.5            # probability of a window move with a burst (continuous mode: per step)
    ffbs::Bool = false                    # centred FFBS of the regimes with every burst (or every step, continuous mode)
    comp::Bool = true                     # compensated initialisation after a weather step
    gate_min::Int = 10                    # steps after a pCN burst before a replica can be active again
    gate_max::Int = 30                    # ... after which it is active whatever its indicator
    gate_min_window::Int = 20             # the same after a burst with a regime window
    gate_max_window::Int = 60
    gate_win::Int = 8                     # relaxation indicator: halves of the last gate_win steps
    gate_tol::Float64 = 0.3               # ... active when the halves' carrier scores (units of β_k∂K) differ by less
    ηw::Vector{Float64} = Float64[]       # initial Langevin step per k field
    κs::Vector{Float64} = Float64[]       # offsets of w = log(k + κ_s) per k field (MW)
    κmin::Float64 = 1.0
    clip::Float64 = 0.02                  # drift clip per step in w
    box::Bool = true                      # reflection at the upper capacity bound
    probe_every::Int = 20                 # lag probe every that many steps (0: off)
    probe_len::Int = 30                   # operation sweeps of a probe at frozen (k, x)
    probe_tail::Int = 15                  # equilibrated score: mean over the last sweeps of the probe
    probe_win::Int = 10                   # in-flight score: mean over the replica's last active steps (a replica needs that many)
    probe_avg::Int = 3                    # probes averaged per adaptation
    adapt::Bool = true                    # adapt ηw per carrier from the probes (warmup only)
    gap_hi::Float64 = 0.10                # |mean gap| (units of β_k∂K) above which the step shrinks
    gap_lo::Float64 = 0.05                # below which it grows
    ηmin::Float64 = 2e-5
    ηmax::Float64 = 2e-3
    shrink::Float64 = 0.7
    grow::Float64 = 1.2
    λ::Float64 = 0.0                      # stress tilt
    Φ::Any = nothing                      # declared tilt functional (TiltFunctional; nothing: none, untilted)
end

"Default offsets κ_s of w = log(k + κ_s) per carrier (MW); other carriers `langevin_kappa_default`."
const LANGEVIN_KAPPA = Dict("onwind" => 500.0, "solar" => 1000.0, "offwind" => 200.0, "gas" => 300.0, "battery" => 100.0, "H2" => 100.0)

"Carrier name of a k field (the field name without the `k_` prefix)."
kcarrier(sp::FieldSpec) = (n = String(sp.name); startswith(n, "k_") ? n[3:end] : n)

"The controls of the replica sampler from the configuration (per k field: κ_s by carrier name, the initial step)."
function ReplicaCtl(cfg::Config, P::Problem)
    nk = length(P.E.kspecs)
    κs = [get(cfg.langevin_kappa, kcarrier(sp), get(LANGEVIN_KAPPA, kcarrier(sp), cfg.langevin_kappa_default)) for sp in P.E.kspecs]
    return ReplicaCtl(; M=cfg.replicas, inner=cfg.replica_inner, θ=cfg.replica_theta, burst_every=cfg.replica_burst_every,
                      φ=cfg.replica_burst_angle, window=cfg.replica_window, window_prob=cfg.replica_window_prob,
                      ffbs=cfg.replica_ffbs, comp=cfg.replica_compensate, gate_min=cfg.replica_gate_min,
                      gate_max=cfg.replica_gate_max, gate_min_window=cfg.replica_gate_min_window,
                      gate_max_window=cfg.replica_gate_max_window, gate_win=cfg.replica_gate_win, gate_tol=cfg.replica_gate_tol,
                      ηw=fill(cfg.langevin_eta, nk), κs, κmin=cfg.langevin_kmin, clip=cfg.langevin_clip, box=cfg.langevin_box,
                      probe_every=cfg.replica_probe_every, probe_len=cfg.replica_probe_len, probe_tail=cfg.replica_probe_tail,
                      probe_win=cfg.replica_probe_win, probe_avg=cfg.replica_probe_avg, adapt=cfg.langevin_adapt,
                      gap_hi=cfg.langevin_gap_hi, gap_lo=cfg.langevin_gap_lo, ηmin=cfg.langevin_eta_min, ηmax=cfg.langevin_eta_max,
                      λ=cfg.tilt_lambda)
end

"""
The replicas of one chain: states (sharing k), their weather streams, the gate (steps since
the last weather burst, whether it had a regime window, active flags, the normalised carrier
scores since the burst), the in-flight score windows of the probe, the Langevin step per k
field and the buffer of probe gaps.
"""
mutable struct ReplicaSet
    reps::Vector{State}
    wrngs::Vector{Any}
    since::Vector{Int}
    window::Vector{Bool}
    active::Vector{Bool}
    gbuf::Vector{Vector{Vector{Float64}}}
    hist::Vector{Vector{Vector{Vector{Float64}}}}
    ηw::Vector{Float64}
    gaps::Vector{Vector{Float64}}
end
ReplicaSet(reps, wrngs, ηw) = (M = length(reps);
    ReplicaSet(reps, wrngs, fill(10^9, M), falses(M), trues(M), [Vector{Float64}[] for _ in 1:M],
               [Vector{Vector{Float64}}[] for _ in 1:M], copy(ηw), Vector{Float64}[]))

"""
    weather_schedule(ctl, it, j) -> (θ, burst)

The weather angle of replica j at step `it` and whether it is a burst: a fixed, state-independent
schedule (continuous angle θ every step; a burst of angle φ when (it + (j−1)·⌊burst_every/M⌋)
mod burst_every = 0).
"""
function weather_schedule(ctl::ReplicaCtl, it::Int, j::Int)
    nb = ctl.burst_every
    burst = nb > 0 && mod(it + (j - 1) * (nb ÷ ctl.M), nb) == 0
    θ = burst ? (ctl.θ > 0 ? acos(clamp(cos(ctl.θ) * cos(ctl.φ), -1.0, 1.0)) : ctl.φ) : ctl.θ
    return θ, burst
end

"""
    replica_weather!(R, j, P, LM, F, ctl, it) -> moved

The weather step of replica j at step `it` from its own stream: the scheduled angle, a regime
window move with probability `window_prob` (one uniform of the stream; with a burst, or every
step in continuous mode), FFBS with `ffbs`, then the compensated initialisation of its
operation. Resets the gate at a burst.
"""
function replica_weather!(R::ReplicaSet, j::Int, P::Problem, LM::LocalModel, F::VarFactors, ctl::ReplicaCtl, it::Int)
    θ, burst = weather_schedule(ctl, it, j)
    wr = R.wrngs[j]
    regime = (burst || ctl.burst_every == 0) && (ctl.window > 0 || ctl.ffbs)
    win = regime && ctl.window > 0 && rand(wr) < ctl.window_prob
    ffbs = regime && ctl.ffbs
    if burst
        R.since[j] = 0
        R.window[j] = win
        R.active[j] = false
        empty!(R.gbuf[j]); empty!(R.hist[j])
    else
        R.since[j] += 1
    end
    (θ > 0 || win || ffbs) || return false
    s = R.reps[j]
    r0 = ctl.comp ? copy(LocalCost(P, LM, s).r) : nothing
    s = weather_process_step(wr, P, F, s, θ; window=win ? ctl.window : 0, ffbs=ffbs)
    r0 === nothing || compensate_weather!(P, LM, r0, s)
    R.reps[j] = s
    return true
end

"""
    replica_operation(rng, P, K, LM, s, inner) -> (s, sk, fk)

`inner` operation sweeps of a replica (`operation_sweep!`) with the mean of its capacity scores
after each (`capacity_scores`).
"""
function replica_operation(rng, P::Problem, K::LocalKernel, LM::LocalModel, s::State, inner::Int)
    acc, fk = nothing, nothing
    for _ in 1:inner
        s = operation_sweep!(rng, P, K, LM, s)
        sk, fk = capacity_scores(P, s)
        acc = acc === nothing ? sk ./ inner : acc .+ sk ./ inner
    end
    return s, acc, fk
end

"""
    relaxation_indicator(gbuf, win) -> D

Relaxation of a replica after its weather burst: the largest difference, over the k fields,
between the mean normalised carrier score (r ŝ_k/β_k∂K on interior nodes) of the last win/2
steps and of the win/2 steps before (Inf until there are `win` steps). It depends on the
replica's state and so on its weather: the gate built on it selects replicas for the drift by a
weather-dependent rule, bounded by `gate_max` (docs/system/sampler.md §11.3).
"""
function relaxation_indicator(gbuf, win::Int)
    h = win ÷ 2
    length(gbuf) >= 2h && h >= 1 || return Inf
    n = length(gbuf)
    a = mean(gbuf[n-2h+1:n-h]); b = mean(gbuf[n-h+1:n])
    d = [abs(x - y) for (x, y) in zip(a, b) if isfinite(x) && isfinite(y)]
    return isempty(d) ? 0.0 : maximum(d)
end

"""
    update_gate!(R, j, g, ctl) -> active

Record the normalised carrier scores `g` of replica j and update its active flag: continuous
mode (no bursts) always active; else active from `gate_min` steps after its burst on (the window
values after a regime window) once `relaxation_indicator` is below `gate_tol`, or from `gate_max`
steps on; active stays active until the next burst.
"""
function update_gate!(R::ReplicaSet, j::Int, g::Vector{Float64}, ctl::ReplicaCtl)
    ctl.burst_every == 0 && return (R.active[j] = true)
    R.active[j] && return true
    push!(R.gbuf[j], g)
    length(R.gbuf[j]) > ctl.gate_win && popfirst!(R.gbuf[j])
    gmin, gmax = R.window[j] ? (ctl.gate_min_window, ctl.gate_max_window) : (ctl.gate_min, ctl.gate_max)
    R.since[j] >= gmin || return false
    R.active[j] = R.since[j] >= gmax || relaxation_indicator(R.gbuf[j], ctl.gate_win) < ctl.gate_tol
    R.active[j] && empty!(R.gbuf[j])
    return R.active[j]
end

"Replica weights of the drift: `tilt_weights` with the declared Φ (any λ, also 0), `uniform_weights` without one."
drift_weights(ctl::ReplicaCtl, Φs, active) = ctl.Φ === nothing ? uniform_weights(active) : tilt_weights(ctl.λ, Φs, active)

"""
    replica_step!(rng, P, Ks, LM, units, F, R, ctl, it) -> NamedTuple

One step of the replica sampler for one chain: per replica (in order) its weather step
(`replica_weather!`, own stream), its operation sweeps and scores (`replica_operation`, chain
stream `rng`), its gate; then the drift weights, the replica-weighted score
s̄_k = Σ_j ω_j ŝ_k^{(j)}, one Langevin step of the shared capacities (`langevin_capacities`, chain
stream) and the dispatch-following map on every replica (`follow_capacities!`). Returns the
step's record (scores per replica for the probe windows, weights, Φ, diagnostics).
"""
function replica_step!(rng, P::Problem, Ks, LM::LocalModel, units, F::VarFactors, R::ReplicaSet, ctl::ReplicaCtl, it::Int)
    M = length(R.reps)
    r = P.βk / P.βv
    nk = length(P.E.kspecs)
    sks = Vector{Vector{Vector{Float64}}}(undef, M)
    fk = nothing
    Kraw = knodes(P, R.reps[1].ℓ, R.reps[1].k)
    mask = interior_nodes(P, Kraw)
    moved = falses(M)
    tw = @elapsed for j in 1:M
        moved[j] = replica_weather!(R, j, P, LM, F, ctl, it)
        R.reps[j], sks[j], fk = replica_operation(rng, P, Ks[j], LM, R.reps[j], ctl.inner)
        fkm = masked_means(fk, mask)
        update_gate!(R, j, masked_means([r .* x for x in sks[j]], mask) ./ fkm, ctl)
        if R.active[j]
            push!(R.hist[j], [r .* x for x in sks[j]])
            length(R.hist[j]) > ctl.probe_win && popfirst!(R.hist[j])
        end
    end
    Φs = ctl.Φ === nothing ? zeros(M) : [ctl.Φ(P, s.xs[end]) for s in R.reps]
    wts = drift_weights(ctl, Φs, R.active)
    sk̄ = [sum(wts[j] .* sks[j][f] for j in 1:M) for f in 1:nk]
    knew, nclip = langevin_capacities(rng, P, units, Kraw, sk̄, fk; ηw=R.ηw, κs=ctl.κs, κmin=ctl.κmin, clip=ctl.clip, box=ctl.box)
    for j in 1:M
        R.reps[j], _ = follow_capacities!(P, LM, R.reps[j], units, Kraw, knew; κmin=ctl.κmin)
    end
    all(s -> s.k == R.reps[1].k, R.reps) || error("replica capacities diverged")
    return (; it, sks, wts, Φs, moved, nclip, Kraw, knew, rs=masked_means([r .* x for x in sk̄], mask), fk=masked_means(fk, mask),
            tw)
end

# ======================================================================================
# 5. the lag probe and the step adaptation
# ======================================================================================

"""
    probe_candidate(R, ctl) -> j or nothing

The replica to probe: among those with a full in-flight window (`probe_win` steps active since
their last burst, so their weather has relaxed), the one longest after its burst.
"""
function probe_candidate(R::ReplicaSet, ctl::ReplicaCtl)
    cand = [j for j in eachindex(R.reps) if R.active[j] && length(R.hist[j]) >= ctl.probe_win]
    isempty(cand) && return nothing
    return cand[argmax([R.since[j] for j in cand])]
end

"""
    lag_probe(prng, P, K, LM, s, hist, ctl) -> NamedTuple

Freeze (k, x) on a copy of replica `s`, run `probe_len` operation sweeps (stream `prng`, the
replica is not changed), and compare the mean score of the last `probe_tail` sweeps
(equilibrated) with the replica's in-flight score (mean of its window `hist`), per k field on
the interior nodes: gap = ⟨r ŝ_in − r ŝ_eq⟩/⟨β_k∂K⟩. Since the probed replica has been active for
the whole window, the gap measures the lag of the dispatch behind the moving capacities, not the
relaxation after its weather burst.
"""
function lag_probe(prng, P::Problem, K::LocalKernel, LM::LocalModel, s::State, hist, ctl::ReplicaCtl)
    r = P.βk / P.βv
    c = copy(s)
    acc, fk = nothing, nothing
    for q in 1:ctl.probe_len
        c = operation_sweep!(prng, P, K, LM, c)
        if q > ctl.probe_len - ctl.probe_tail
            sk, fk = capacity_scores(P, c)
            acc = acc === nothing ? [r .* x ./ ctl.probe_tail for x in sk] : acc .+ [r .* x ./ ctl.probe_tail for x in sk]
        end
    end
    inflight = sum(hist) ./ length(hist)
    mask = interior_nodes(P, knodes(P, s.ℓ, s.k))
    rs_eq, rs_in, fkm = masked_means(acc, mask), masked_means(inflight, mask), masked_means(fk, mask)
    return (; gap=(rs_in .- rs_eq) ./ fkm, rs_eq, rs_in, fk=fkm, op_in=s.parts.op, op_eq=c.parts.op, shed_in=s.parts.shed,
            shed_eq=c.parts.shed)
end

"""
    adapt_step!(R, gap, ctl) -> adapted

Add a probe's gaps to the buffer; once it holds `probe_avg` probes, per k field with a finite
mean gap ḡ: ηw ← max(ηmin, shrink·ηw) if |ḡ| > gap_hi, min(ηmax, grow·ηw) if |ḡ| < gap_lo; the
buffer is then emptied (every adaptation uses fresh probes).
"""
function adapt_step!(R::ReplicaSet, gap::Vector{Float64}, ctl::ReplicaCtl)
    push!(R.gaps, copy(gap))
    length(R.gaps) >= ctl.probe_avg || return false
    for f in eachindex(R.ηw)
        v = [g[f] for g in R.gaps if isfinite(g[f])]
        isempty(v) && continue
        ḡ = mean(v)
        if abs(ḡ) > ctl.gap_hi
            R.ηw[f] = max(ctl.ηmin, R.ηw[f] * ctl.shrink)
        elseif abs(ḡ) < ctl.gap_lo
            R.ηw[f] = min(ctl.ηmax, R.ηw[f] * ctl.grow)
        end
    end
    empty!(R.gaps)
    return true
end

# ======================================================================================
# 6. the driver
# ======================================================================================

"""
    init_replicas(rng, P, K, LM, s0, wrngs, ctl, nsweeps) -> ReplicaSet

M replicas at the capacities and operation of `s0`, each with a fresh exact weather draw from
its own stream, the compensated initialisation and `nsweeps` operation sweeps at fixed k.
"""
function init_replicas(rng, P::Problem, Ks, LM::LocalModel, s0::State, wrngs, ctl::ReplicaCtl, nsweeps::Int)
    r0 = copy(LocalCost(P, LM, s0).r)
    reps = State[]
    for j in eachindex(wrngs)
        m, z0, ws = sample_weather(wrngs[j], P.W, s0.ℓ)
        s = compensate_weather!(P, LM, r0, make_state(P, s0.ℓ, m, z0, ws, s0.k, s0.v))
        for _ in 1:nsweeps
            s = operation_sweep!(rng, P, Ks[j], LM, s)
        end
        push!(reps, s)
    end
    return ReplicaSet(reps, wrngs, ctl.ηw)
end

"Operation kernels of a chain's replicas: one shared kernel (local), one per replica (flow, adaptive: their frames follow (k, x)); all share `LM`."
function replica_kernels(P::Problem, s::State, cfg::Config, M::Int, LM::LocalModel)
    mk() = (K = new_sampler(P, s, cfg).loc; K.models[LM.ℓ] = LM; K)
    K1 = mk()
    return K1.mode == :local ? fill(K1, M) : [j == 1 ? K1 : mk() for j in 1:M]
end

"Names of the per-step replica records (`replica_record`)."
const REPLICA_TRACE_KEYS = ("ktot", "capex", "op", "shed", "penalty", "pfcost", "weights", "active", "since", "phi", "rs", "fk",
                            "eta_w", "nclip", "moved", "time", "features")

"Per-step record of a chain's replicas (columns of `replicas/<c>/<key>`; features stacked replica by replica)."
function replica_record(P::Problem, R::ReplicaSet, st, kref, t)
    reps = R.reps
    return Dict("ktot" => [sum(k) for k in knodes(P, reps[1].ℓ, reps[1].k)], "capex" => [reps[1].parts.capex],
                "op" => [s.parts.op for s in reps], "shed" => [s.parts.shed for s in reps], "penalty" => [s.parts.penalty for s in reps],
                "pfcost" => [s.parts.capex + s.parts.gen + s.parts.storage for s in reps], "weights" => copy(st.wts),
                "active" => Float64.(R.active), "since" => Float64.(min.(R.since, 10^6)), "phi" => copy(st.Φs), "rs" => st.rs,
                "fk" => st.fk, "eta_w" => copy(R.ηw), "nclip" => [Float64(st.nclip)], "moved" => Float64.(st.moved), "time" => [t, st.tw],
                "features" => reduce(vcat, [replica_features(P, s, kref) for s in reps]))
end

"Probe record (columns of `replicas/<c>/probe/<key>`)."
probe_record(it, j, pr, η0, η1, adapted) =
    Dict("it" => [Float64(it), Float64(j), Float64(adapted)], "gap" => pr.gap, "rs_eq" => pr.rs_eq, "rs_in" => pr.rs_in, "fk" => pr.fk,
         "cost" => [pr.op_in, pr.op_eq, pr.shed_in, pr.shed_eq], "eta_before" => η0, "eta_after" => η1)

"Monitored quantities of a chain: the mean of `monitor_quantities` over its replicas (capacities are shared; costs and weather averaged)."
replica_monitor(P::Problem, LM::LocalModel, R::ReplicaSet) = mean([monitor_quantities(P, s; LM=LM) for s in R.reps])

"""
    run_replicas(cfg; hook=nothing) -> results

The replica–Langevin driver (`sampler = "replicas"`, docs/system/sampler.md §11): `setup`
(the bias is not used), then `nchains` chains (starts cycled from `chain_starts`), each with
`replicas` weather replicas initialised by `init_replicas` (`replica_init_sweeps` sweeps), then
`mlda_warmup` + `nsamples` steps of `replica_step!`; probes every `replica_probe_every` steps, the
Langevin step adapted from them during the warmup and frozen in production. Outputs:
`chains_report` on the replica-mean monitored quantities and the replica records
(docs/system/data.md §3). Snapshots and resume as `run_chains` (stop file, exit code 75; fresh
streams after a resume; the adapted steps and the gates are restored).
"""
function run_replicas(cfg::Config; hook=nothing)
    C = Checkpointer(cfg; hook=hook)
    P, sL, info = setup(cfg; rng=MersenneTwister(cfg.seed))
    cfg.use_bias && logmsg(cfg, "sampler = replicas: the learned bias is not used (use_bias ignored)")
    P = with_bias(P, false)
    sL = evaluate!(P, sL)
    L = nlevels(P) - 1
    t0 = time()
    snap = read_replica_snapshot(cfg.checkpoint, cfg, P)
    r = snap === nothing ? 0 : snap.restarts + 1
    C.restarts = r
    snap === nothing || logmsg(cfg, "resuming from snapshot $(cfg.checkpoint) (restart $r): $(snap.where_)")
    snap === nothing || snap.co2 === P.E.co2[] || set_co2_price!(P.E, snap.co2)
    C.last = time()
    nc, M = cfg.nchains, cfg.replicas
    nwarm, nprod = cfg.mlda_warmup, cfg.nsamples
    LM = LocalModel(P, L; cell_power=cfg.cell_level_power, xr_partner=cfg.xloc_partner_hops, xr_ball=cfg.xloc_ball_hops)
    units = capacity_units(P, LM)
    F = VarFactors(P.W)
    kref = knodes(P, sL.ℓ, sL.k)                               # reference capacities of the features and of Φ (set-up start)
    ctl = ReplicaCtl(cfg, P)
    if cfg.tilt_functional == "resload_mean"
        ctl.Φ = TiltFunctional(P, kref; rng=MersenneTwister(cfg.seed + 5151), n=cfg.tilt_samples)
        logmsg(cfg, @sprintf("tilt: λ = %g, Φ = standardised annual mean residual load (μ %.2f GW, σ %.2f GW, %d draws)", ctl.λ,
                             ctl.Φ.μ, ctl.Φ.σ, cfg.tilt_samples))
    end
    kinds = [cfg.chain_starts[mod1(c, length(cfg.chain_starts))] for c in 1:nc]
    names = monitor_names(P)
    rngs = [MersenneTwister(restart_seed(cfg, r, 1000c)) for c in 1:nc]
    prngs = [MersenneTwister(restart_seed(cfg, r, 1000c + 500)) for c in 1:nc]
    wrngs = [[MersenneTwister(restart_seed(cfg, r, 1000c + j)) for j in 1:M] for c in 1:nc]
    sets = ReplicaSet[]
    Kss = Vector{Vector{LocalKernel}}()
    if snap === nothing
        it0 = 0
        traces = [Vector{Vector{Float64}}() for _ in 1:nc]
        rtr = [Dict{String,Vector{Vector{Float64}}}() for _ in 1:nc]
        ptr = [Dict{String,Vector{Vector{Float64}}}() for _ in 1:nc]
        for c in 1:nc
            s0 = chain_start(MersenneTwister(restart_seed(cfg, r, 7c)), P, sL, kinds[c], false)
            Ks = replica_kernels(P, s0, cfg, M, LM)
            push!(Kss, Ks)
            push!(sets, init_replicas(rngs[c], P, Ks, LM, s0, wrngs[c], ctl, cfg.replica_init_sweeps))
            logmsg(cfg, @sprintf("chain %d (%s): %d replicas initialised (%d sweeps each), op %s  [%.0fs]", c, kinds[c], M,
                                 cfg.replica_init_sweeps, join((@sprintf("%.3e", s.parts.op) for s in sets[c].reps), " "), time() - t0))
        end
    else
        length(snap.sets) == nc || error("snapshot $(cfg.checkpoint) has $(length(snap.sets)) chains, nchains = $nc")
        it0 = snap.it
        kinds = snap.kinds
        traces, rtr, ptr = snap.traces, snap.rtraces, snap.ptraces
        for c in 1:nc
            sv = snap.sets[c]
            length(sv.reps) == M || error("snapshot $(cfg.checkpoint): $(length(sv.reps)) replicas, replicas = $M")
            R = ReplicaSet(sv.reps, wrngs[c], sv.ηw)
            R.since .= sv.since; R.window .= sv.window; R.active .= sv.active
            push!(sets, R)
            push!(Kss, replica_kernels(P, R.reps[1], cfg, M, LM))
        end
    end
    resume_states = snap === nothing ? nothing : [copy(s) for R in sets for s in R.reps]
    snapdata = () -> (; states=[s for R in sets for s in R.reps], sets, kinds, traces, rtraces=rtr, ptraces=ptr)
    try
        logmsg(cfg, "$nc chains × $M replicas (starts: $(join(kinds, ", "))), $nwarm warmup + $nprod steps",
               it0 > 0 ? ", from step $it0" : "")
        it0 == 0 && boundary!(C, cfg, P, :replicas, 0, snapdata; force=true, writer=write_replica_snapshot)
        for it in it0+1:nwarm+nprod
            for c in 1:nc
                R = sets[c]
                st = replica_step!(rngs[c], P, Kss[c], LM, units, F, R, ctl, it)
                push!(traces[c], replica_monitor(P, LM, R))
                for (k, v) in replica_record(P, R, st, kref, time() - t0)
                    push!(get!(rtr[c], k, Vector{Vector{Float64}}()), v)
                end
                if ctl.probe_every > 0 && it % ctl.probe_every == 0
                    j = probe_candidate(R, ctl)
                    if j !== nothing
                        pr = lag_probe(prngs[c], P, Kss[c][j], LM, R.reps[j], R.hist[j], ctl)
                        η0 = copy(R.ηw)
                        ad = ctl.adapt && it <= nwarm && adapt_step!(R, pr.gap, ctl)
                        for (k, v) in probe_record(it, j, pr, η0, copy(R.ηw), ad)
                            push!(get!(ptr[c], k, Vector{Vector{Float64}}()), v)
                        end
                    end
                end
            end
            if it % max(1, (nwarm + nprod) ÷ 20) == 0
                for c in 1:nc
                    R = sets[c]
                    logmsg(cfg, @sprintf("  step %4d/%d chain %d: k [%s] GW, active %d/%d, op %.4e, shed %.3g MWh, ηw [%s]  [%.0fs]",
                                         it, nwarm + nprod, c, join((@sprintf("%.1f", x / 1e3) for x in rtr[c]["ktot"][end]), ", "),
                                         count(R.active), M, mean(s.parts.op for s in R.reps), mean(s.parts.shed for s in R.reps),
                                         join((@sprintf("%.1e", x) for x in R.ηw), ", "), time() - t0))
                end
            end
            it < nwarm + nprod && boundary!(C, cfg, P, :replicas, it, snapdata; writer=write_replica_snapshot)
        end
        res = chains_report(cfg, P, names, traces, kinds, nothing, nwarm, nprod, t0)
        write_replica_outputs(cfg, P, info, ctl, rtr, ptr, r, snap)
        write_replica_summary(cfg, P, ctl, rtr, ptr, kinds, nwarm, res.passed, time() - t0)
        return (; res..., P, sets, info, status=:finished, exitcode=0, resume_states, restarts=r)
    catch e
        e isa CheckpointStop || rethrow()
        logmsg(cfg, "stop file $(cfg.stop_file): stopped after the snapshot $(e.path) ($(e.where_)); exit code $EXIT_CHECKPOINTED")
        return (; status=:stopped, exitcode=EXIT_CHECKPOINTED, states=e.states, sets, resume_states, restarts=r, P, info)
    end
end

"Matrix of a record list (one column per step)."
record_matrix(v) = isempty(v) ? zeros(0, 0) : reduce(hcat, v)

"Outputs of the replica driver appended to `cfg.out` (docs/system/data.md §3)."
function write_replica_outputs(cfg::Config, P::Problem, info, ctl::ReplicaCtl, rtr, ptr, r, snap)
    h5open(cfg.out, "r+") do f
        f["replicas/carriers"] = [kcarrier(sp) for sp in P.E.kspecs]
        f["replicas/feature_names"] = replica_feature_names(P)
        f["replicas/M"] = ctl.M
        f["replicas/kappa_s"] = ctl.κs
        f["replicas/tilt"] = ctl.Φ === nothing ? [ctl.λ, NaN, NaN] : [ctl.λ, ctl.Φ.μ, ctl.Φ.σ]
        for c in eachindex(rtr)
            for (k, v) in rtr[c]
                f["replicas/$c/$k"] = record_matrix(v)
            end
            for (k, v) in ptr[c]
                f["replicas/$c/probe/$k"] = record_matrix(v)
            end
        end
        f["weather/transition"] = P.W.Ptr
        f["calibration/beta"] = [P.βk, P.βv]
        f["config/levels"] = reduce(hcat, [[a, b] for (a, b) in cfg.levels])
        f["run/restarts"] = r
        f["run/resumed_at_sweep"] = snap === nothing ? -1 : snap.it
    end
end

"Path of the text summary of a replica run: `summary_replicas.txt` next to `out` (collected by hpc/collect.sh)."
replica_summary_path(cfg::Config) = joinpath(dirname(abspath(cfg.out)), "summary_replicas.txt")

"""
    write_replica_summary(cfg, P, ctl, rtr, ptr, kinds, nwarm, passed, t)

Small text summary of a replica run (`replica_summary_path`), over the production steps (all
steps if there are none) per chain: carrier totals (mean, sd, last), capex and the replicas'
penalty-free cost, opex and shed, active replicas, the equilibrated scores of the probes
⟨r ŝ_eq⟩/⟨β_k∂K⟩ and their gaps (mean, number of probes), the final Langevin steps, and the
replica weather features (mean, sd over replicas and steps; prior mean and sd of Φ's residual
load).
"""
function write_replica_summary(cfg::Config, P::Problem, ctl::ReplicaCtl, rtr, ptr, kinds, nwarm::Int, passed::Bool, t::Float64;
                               probe_warm::Int=nwarm)   # warmup length in steps for the probe selection (the device driver records once per checkpoint, nwarm counts records)
    cars = [kcarrier(sp) for sp in P.E.kspecs]
    fn = replica_feature_names(P)
    fmt(x) = @sprintf("%.6g", x)
    fmean(v) = (u = filter(isfinite, v); isempty(u) ? NaN : mean(u))
    row(io, name, v) = println(io, rpad(name, 26), join((lpad(fmt(x), 13) for x in v), " "))
    atomic_open(replica_summary_path(cfg)) do io
        println(io, "# replica–Langevin run (src/replicas.jl): $(cfg.out)")
        println(io, "# chains $(length(rtr)), replicas $(ctl.M), steps $(length(rtr[1]["ktot"])) (warmup $nwarm), ",
                "λ $(ctl.λ), wall $(round(t; digits=0)) s, convergence $(passed ? "passed" : "not passed")")
        ctl.Φ === nothing || println(io, @sprintf("# Φ: prior residual load mean %.4f GW, sd %.4f GW", ctl.Φ.μ, ctl.Φ.σ))
        for c in eachindex(rtr)
            d = rtr[c]
            n = length(d["ktot"])
            w = nwarm < n ? (nwarm+1:n) : (1:n)
            M(k) = record_matrix(d[k])[:, w]
            println(io, "\n## chain $c ($(kinds[c])), steps $(first(w))–$(last(w))")
            println(io, rpad("", 26), join((lpad(x, 13) for x in cars), " "))
            K = M("ktot") ./ 1e3
            row(io, "k total GW, mean", vec(mean(K; dims=2))); row(io, "k total GW, sd", vec(std(K; dims=2)))
            row(io, "k total GW, last", K[:, end])
            row(io, "eta_w, last", d["eta_w"][end])
            rs, fk = M("rs"), M("fk")
            row(io, "drift r s̄/β_k∂K, mean", [fmean(rs[f, :] ./ fk[f, :]) for f in eachindex(cars)])
            pr = ptr[c]
            if haskey(pr, "gap")
                it = record_matrix(pr["it"])[1, :]
                sel = findall(>(nwarm < length(rtr[c]["ktot"]) ? probe_warm : -1), it)   # production probes (steps)
                isempty(sel) && (sel = eachindex(it))
                eq, fkp, gp = record_matrix(pr["rs_eq"])[:, sel], record_matrix(pr["fk"])[:, sel], record_matrix(pr["gap"])[:, sel]
                fm(A) = [fmean(A[f, :]) for f in axes(A, 1)]
                row(io, "probe r ŝ_eq/β_k∂K", fm(eq ./ fkp)); row(io, "probe gap/β_k∂K", fm(gp))
                println(io, rpad("probes", 26), length(sel))
            end
            for (k, lab) in (("capex", "capex €/a"), ("pfcost", "pf cost €/a (replicas)"), ("op", "opex €/a (replicas)"),
                             ("shed", "shed MWh (replicas)"))
                A = M(k)
                println(io, rpad(lab, 26), "mean ", fmt(mean(A)), "  sd ", fmt(std(vec(A))))
            end
            A = M("active")
            println(io, rpad("active replicas", 26), "mean ", fmt(mean(sum(A; dims=1))), " of $(ctl.M)")
            Fm = M("features")
            nf = length(fn)
            X = reshape(Fm, nf, :)
            println(io, "replica weather features (all replicas and steps): mean, sd")
            for i in 1:nf
                println(io, "  ", rpad(fn[i], 24), lpad(fmt(mean(X[i, :])), 13), lpad(fmt(std(X[i, :])), 13))
            end
        end
    end
end

# ======================================================================================
# 7. snapshots# ======================================================================================
# 7. snapshots of the replica driver (checkpoint.jl; layout in docs/system/data.md §5)
# ======================================================================================

"Write a record dictionary under group `g` (one matrix per key)."
function write_records!(f, g::String, d)
    for (k, v) in d
        isempty(v) || (f["$g/$k"] = record_matrix(v))
    end
end
"Read the record dictionary of group `g` (empty if absent)."
function read_records(f, g::String)
    d = Dict{String,Vector{Vector{Float64}}}()
    haskey(f, g) || return d
    for k in keys(f[g])
        f[g][k] isa HDF5.Dataset || continue
        A = read(f[g][k])
        d[k] = [A[:, j] for j in 1:size(A, 2)]
    end
    return d
end

"""
    write_replica_snapshot(C, cfg, P, data) -> path

Snapshot of the replica driver (`boundary!` writer): `driver = "replicas"`, the step counter,
per chain its start kind, the monitored trace, the replica records and probe records so far,
the Langevin steps, the gates and every replica's state.
"""
function write_replica_snapshot(C::Checkpointer, cfg::Config, P::Problem, data::NamedTuple)
    t = time()
    where_ = ckpt_where(data.phase, data.it, cfg)
    atomic_write(C.path; sync=true) do tmp
        h5open(tmp, "w") do f
            f["version"] = SNAPSHOT_VERSION
            f["driver"] = "replicas"
            f["phase"] = String(data.phase)
            f["it"] = data.it
            f["where"] = where_
            f["restarts"] = C.restarts
            f["written"] = t
            f["co2_price"] = P.E.co2[]
            f["config/seed"] = cfg.seed
            f["config/levels"] = reduce(hcat, [[a, b] for (a, b) in cfg.levels])
            f["nchains"] = length(data.sets)
            for (c, R) in enumerate(data.sets)
                g = "chains/$c"
                f["$g/start"] = data.kinds[c]
                f["$g/replicas"] = length(R.reps)
                f["$g/eta_w"] = R.ηw
                f["$g/since"] = R.since
                f["$g/window"] = Int.(R.window)
                f["$g/active"] = Int.(R.active)
                isempty(data.traces[c]) || (f["$g/Q"] = reduce(hcat, data.traces[c]))
                write_records!(f, "$g/trace", data.rtraces[c])
                write_records!(f, "$g/probe", data.ptraces[c])
                for (j, s) in enumerate(R.reps)
                    h = "$g/replica/$j"
                    f["$h/level"] = s.ℓ
                    f["$h/m"] = s.m
                    f["$h/z0"] = s.z0
                    for i in 1:s.ℓ
                        f["$h/w/$i"] = s.ws[i]
                    end
                    f["$h/k"] = s.k
                    f["$h/v"] = s.v
                end
            end
        end
    end
    C.last = time()
    C.nwritten += 1
    logmsg(cfg, @sprintf("snapshot %s: %s  [%.1f MB, %.1fs]", C.path, where_, filesize(C.path) / 1e6, C.last - t))
    return C.path
end

"""
    read_replica_snapshot(path, cfg, P) -> NamedTuple or nothing

Read a snapshot of the replica driver (nothing if there is none): `it`, `restarts`, `co2`,
`kinds`, `traces`, `rtraces`, `ptraces` and per chain (`sets`) the replica states (rebuilt with
`make_state`), `ηw`, `since`, `window`, `active`. Checks the sampler, the version, the levels,
the seed and the sizes.
"""
function read_replica_snapshot(path::String, cfg::Config, P::Problem)
    (isempty(path) || !isfile(path)) && return nothing
    h5open(path, "r") do f
        (haskey(f, "driver") && read(f["driver"]) == "replicas") || error("snapshot $path is not a snapshot of the replica driver")
        v = read(f["version"])
        v == SNAPSHOT_VERSION || error("snapshot $path: format version $v, the code reads $SNAPSHOT_VERSION")
        read(f["config/levels"]) == reduce(hcat, [[a, b] for (a, b) in cfg.levels]) || error("snapshot $path: levels differ from the configuration")
        read(f["config/seed"]) == cfg.seed || error("snapshot $path: seed $(read(f["config/seed"])), the configuration has $(cfg.seed)")
        nc = read(f["nchains"])
        sets, kinds, traces, rtraces, ptraces = [], String[], Vector{Vector{Vector{Float64}}}(), [], []
        for c in 1:nc
            g = "chains/$c"
            push!(kinds, read(f["$g/start"]))
            Q = haskey(f, "$g/Q") ? read(f["$g/Q"]) : zeros(0, 0)
            push!(traces, [Q[:, j] for j in 1:size(Q, 2)])
            push!(rtraces, read_records(f, "$g/trace"))
            push!(ptraces, read_records(f, "$g/probe"))
            reps = State[]
            for j in 1:read(f["$g/replicas"])
                h = "$g/replica/$j"
                ℓ = read(f["$h/level"])
                m, z0, k, vv = read(f["$h/m"]), read(f["$h/z0"]), read(f["$h/k"]), read(f["$h/v"])
                (length(k) == P.lk[ℓ+1].n && length(vv) == P.lv[ℓ+1].n && length(z0) == P.W.lays[1].n) ||
                    error("snapshot $path: replica $j of chain $c does not fit the problem at level $ℓ")
                push!(reps, make_state(P, ℓ, m, z0, [read(f["$h/w/$i"]) for i in 1:ℓ], k, vv))
            end
            push!(sets, (; reps, ηw=read(f["$g/eta_w"]), since=read(f["$g/since"]), window=Bool.(read(f["$g/window"])),
                         active=Bool.(read(f["$g/active"]))))
        end
        return (; it=read(f["it"]), restarts=read(f["restarts"]), where_=read(f["where"]), written=read(f["written"]),
                co2=read(f["co2_price"]), sets, kinds, traces, rtraces, ptraces)
    end
end
