# Configuration, set-up and the pieces of a run shared by the drivers:
# data → energy and weather models → start (reference LP or heuristic) and
# temperatures → bias → finest-level start state; the level-0 warmup chain that
# learns the bias. The CPU driver itself (`run` → `run_chains`) is in chains.jl,
# the device driver in device/.

Base.@kwdef mutable struct Config
    data::String = "data/processed/model.h5"
    out::String = "results/run.h5"
    levels::Vector{Tuple{Int,Int}} = [(10, 24), (40, 24)]
    seed::Int = 1
    K::Int = 4                      # weather regimes
    c_shed::Float64 = 3000.0
    c_dump::Float64 = 5.0
    c_viol::Float64 = 3000.0        # price of constraint violations
    local_extra::Float64 = 4.0      # random non-atom moves per node and step and sweep
    fine_kernel::String = "auto"    # fine level: "local", "flow", "adaptive" or "auto" (flow if basis = "flow", else local)
    cell_level_power::Float64 = 1.0 # partition level of cell moves drawn ∝ (number of cells)^power
    xloc_partner_hops::Int = 1      # partners of the localized pair moves (xloc, xlocc) within this many hops
    xloc_ball_hops::Int = 0         # ... flow re-routed on the nodes within this many hops of the pair (0: the pair only)
    kop::Bool = false               # capacity moves with their operation (kop.jl) in the fine step; off until α = 1 is stable with them
    measure::String = "utilization" # reference measure of dispatchable units: "utilization" or "physical" (MW); open choice
    storage::String = "soc"         # storage form: "soc" (state-of-charge trajectory; the only form)
    renewables::String = "load"     # renewables: "load" (output A·k; curtailment = nodal dump; the only form)
    flow_theta::String = "nested"   # basis = "flow": "nested" (modes are the coarse angle coordinates) or "haar" (sparse angles, modes as moves)
    # model completion (docs/system/model.md)
    co2_price::Float64 = NaN        # canonical CO₂ price μ (€/t); NaN: the price contained in the data file
    co2_cap::Float64 = 0.0          # > 0: tune μ during the finest-level warmup so that the mean annual emissions equal this cap (t/yr); frozen after
    brownfield::Bool = false        # existing capacities as floors (capex on k − k0), merged vintages, fixed thermal fleets
    losses::Bool = false            # quadratic line losses (piecewise-linear interpolant on a fixed grid)
    loss_segments::Int = 4          # grid cells per thermal rating for the loss interpolant
    thermal::String = "hard"        # line limits: "hard" (wall priced at c_viol) or "soft" (convex overload band)
    thermal_band::Float64 = 0.1     # soft limits: overload band (fraction of the rating) over which the price rises to c_viol
    link_losses::Bool = false       # signed lossy DC links (efficiency from the data)
    uc::Vector{String} = String[]   # carriers with fleet unit commitment (e.g. ["gas", "coal", "lignite"])
    fine_weather::Float64 = 0.0     # fraction of the weather coordinates moved per fine-level step (0: weather fixed at the fine level)
    init::String = "lp"             # initial state: "lp" (reference LP) or "heuristic" (no LP)
    basis::String = "haar"          # spatial basis: "haar" (graph Haar, countries → nodes) or "flow" (Haar plus transport modes)
    use_bias::Bool = true           # α = 1 (learned bias) or α = 0
    beta_v::Float64 = 4e-5          # operational temperature [1/€], fixed by experience
    beta_k::Float64 = 4e-8          # investment temperature [1/€]
    calibrate_beta::Bool = false    # instead derive β from the LP by equipartition
    eps_v::Float64 = 0.01           # ... with these target slacks (fractions of the optimal cost)
    eps_k::Float64 = 0.05
    warmup0::Int = 1500             # level-0 warmup iterations (bias learning)
    bias_start::Int = 150
    bias_every::Int = 25
    bias_samples::Int = 16          # exact weather draws for Ē ĝ (redrawn at the chains' level when the bias is lifted)
    bias_sample_steps::Int = 4096   # ... of whose steps this many are kept (uniformly, weighted; 0: all)
    bias_table_res::Int = 100       # grid points per smoothing width s of the F₁, F₂ tables
    bias_warmstart::Float64 = 20.0  # weight of the LP-score warm start (in samples), 0 = off
    descent_iters::Int = 4000       # descent of the level-0 operation before the level-0 warmup (0: off)
    mlda_warmup::Int = 100          # finest-level warmup sweeps (device: checkpoints)
    nsamples::Int = 300             # finest-level production sweeps (device: checkpoints)
    bias_monotone::Bool = true      # fit ĝ with a monotone, bounded implied price (no tail leak)
    bias_online::Bool = true        # refit b̂ from the finest chains' scores during the finest-level warmup, frozen after
    bias_online_every::Int = 1      # ... observing every n-th finest-level step
    bias_post::Bool = true          # α = 1: refit b̂ after the run and reweight the finest-level samples (weight ESS reported)
    # chains and convergence diagnostics
    nchains::Int = 4                # finest-level chains (mlda_warmup + nsamples sweeps each)
    chain_starts::Vector{String} = ["low", "high", "heuristic", "heuristic"]   # capacities ×0.5, ×2, as set up (cycled over the chains)
    rhat_max::Float64 = 1.01        # convergence: rank-normalized split-R̂ below this
    ess_min::Float64 = 400.0        # ... bulk and tail ESS (all chains) at least this
    mcse_max::Float64 = 0.05        # ... MCSE of the mean / posterior sd at most this
    rhat_screen::Float64 = 1.05     # threshold of the sliding-window warmup detection
    # device runs (device/, ESMDevice.run_device): nchains chains, mlda_warmup + nsamples checkpoints
    device_backend::String = "gpu"  # Reactant backend: "gpu" or "cpu"
    device_sweeps::Int = 10         # operation sweeps per device call (one capacity pass and one checkpoint after each)
    kmove_nprop::Int = 2            # Metropolis proposals per unit and capacity pass
    kmove_sigma::Float64 = 0.3      # kscale proposal scale (log capacity)
    kpeak_sigma::Float64 = 0.05     # kpeak proposal scale, fraction of the unit's initial capacity (≥ 100 MW)
    verbose::Bool = true
end

"""
Removed keys that old configuration files may still contain, with the only value
that matches the code (any other value is an error).
"""
const REMOVED_KEYS = Dict{String,Any}("kernel" => "local", "p_fine" => 1.0, "alpha0" => false, "regularized" => true)

function load_config(path::String)
    cfg = Config()
    d = TOML.parsefile(path)
    for (k, v) in d
        if haskey(REMOVED_KEYS, k)
            v == REMOVED_KEYS[k] || error("config key $k = $v is no longer supported (the code always uses $(REMOVED_KEYS[k]))")
            @warn "$path: config key $k is obsolete (always $(REMOVED_KEYS[k])); remove it"
            continue
        end
        s = Symbol(k)
        hasfield(Config, s) || error("unknown config key $k")
        if s == :levels
            v = [(Int(a[1]), Int(a[2])) for a in v]
        elseif fieldtype(Config, s) <: Vector
            v = convert(fieldtype(Config, s), v)
        end
        setfield!(cfg, s, convert(fieldtype(Config, s), v))
    end
    check_config(cfg)
    return cfg
end

"Values of the choice keys that the code supports."
function check_config(cfg::Config)
    cfg.storage == "soc" || error("storage = \"$(cfg.storage)\": only the state-of-charge form \"soc\" exists")
    cfg.renewables == "load" || error("renewables = \"$(cfg.renewables)\": only renewables as load \"load\" exist")
    cfg.basis in ("haar", "flow") || error("basis = \"$(cfg.basis)\": \"haar\" or \"flow\"")
    cfg.init in ("lp", "heuristic") || error("init = \"$(cfg.init)\": \"lp\" or \"heuristic\"")
    cfg.measure in ("utilization", "physical") || error("measure = \"$(cfg.measure)\": \"utilization\" or \"physical\"")
    cfg.fine_kernel in ("auto", "local", "flow", "adaptive") || error("fine_kernel = \"$(cfg.fine_kernel)\"")
    return cfg
end

logmsg(cfg, args...) = cfg.verbose && (println(args...); flush(stdout))

function check_levels(levels, N)
    for j in 2:length(levels)
        (M0, h0), (M1, h1) = levels[j-1], levels[j]
        @assert M1 >= M0 && h0 % h1 == 0 "levels must refine: $(levels[j-1]) → $(levels[j])"
        @assert (M1, h1) != (M0, h0) "repeated level $(levels[j])"
    end
    ref = findfirst(l -> l[1] >= N, levels)
    @assert ref !== nothing "one level must resolve all $N nodes (spatially complete)"
    @assert all(l -> l[1] >= N, levels[ref:end])
    return ref - 1   # 0-based level index
end

"""
    setup(cfg) -> (P, s_init, info)

Builds the energy and weather models, the start (reference LP or heuristic) and
the temperatures, and returns the problem with an initial finest-level state.
"""
function setup(cfg::Config; rng=MersenneTwister(cfg.seed))
    check_config(cfg)
    t0 = time()
    sys = load_system(cfg.data)
    E = build_energy_model(sys; c_shed=cfg.c_shed, c_dump=cfg.c_dump, c_viol=cfg.c_viol,
                           basis=Symbol(cfg.basis), measure=Symbol(cfg.measure),
                           flow_theta=Symbol(cfg.flow_theta), co2_price=isnan(cfg.co2_price) ? nothing : cfg.co2_price,
                           brownfield=cfg.brownfield, losses=cfg.losses, loss_segments=cfg.loss_segments,
                           thermal=Symbol(cfg.thermal), thermal_band=cfg.thermal_band, link_losses=cfg.link_losses, uc=cfg.uc)
    ref = check_levels(cfg.levels, sys.N)
    L = length(cfg.levels) - 1
    logmsg(cfg, "system: $(sys.N) nodes, $(length(sys.line_s)) lines, $(length(sys.link_cap)) DC links, ",
           "$(length(sys.years)) weather years; levels $(cfg.levels), LP/reference level $ref")
    W = weather_model_cached(E, cfg.levels; K=cfg.K, seed=cfg.seed, cachedir=joinpath(dirname(cfg.out), "cache"), datafile=cfg.data)
    logmsg(cfg, @sprintf("weather model: %d regimes, coarse dim %d per step, EM loglik %.1f (%d its)  [%.0fs]",
                         W.K, n0(W), W.loglik[end], length(W.loglik), time() - t0))

    # initial weather draw and calibration LP at the reference level
    m, z0, ws = sample_weather(rng, W, L)
    xref = weather_at_level(W, ref, m, z0, ws)
    Xn = weather_nodes(W, ref, xref)
    Wd = [map(x -> datamap(E.wspecs[f], x), Xn[f]) for f in eachindex(Xn)]
    lkr, lvr = Layout(E.kspecs, cfg.levels[ref+1]...), Layout(E.vspecs, cfg.levels[ref+1]...)
    # (the storage and renewable entries keep the cache keys of earlier runs valid)
    keyt = (data_fingerprint(sys, cfg.data), cfg.levels[ref+1], cfg.seed, cfg.K, cfg.c_shed, cfg.c_dump, cfg.levels,
            cfg.storage, cfg.renewables, cfg.basis, :wcache)
    key = model_key(cfg) === nothing ? hash(keyt) : hash((keyt..., model_key(cfg)))
    if cfg.init == "heuristic"
        @assert !cfg.calibrate_beta "init = heuristic needs fixed β"
        lp = heuristic_start(E, lkr, lvr, Wd)
        logmsg(cfg, "no reference LP: heuristic capacities (proxy renewables, gas at nodal peak load), interior dispatch")
    else
        lp = solve_lp_cached(E, lkr, lvr, Wd, joinpath(dirname(cfg.out), "cache", "lp_$(string(key; base=16)).h5"))
    end
    T_L, T_ref = 8760 ÷ cfg.levels[end][2], lvr.T
    # equipartition: one unit of β×cost per cost-confined direction; interior
    # investment coordinates are confined by curvature and carry one half
    n_v = (lp.n_v_total - lp.n_v_free) * T_L / T_ref
    n_k = (lp.n_k_total - lp.n_k_free) + 0.5 * lp.n_k_free
    βv = cfg.calibrate_beta ? n_v / (cfg.eps_v * lp.objective) : cfg.beta_v
    βk = cfg.calibrate_beta ? n_k / (cfg.eps_k * lp.objective) : cfg.beta_k
    logmsg(cfg, @sprintf("LP: total %.3e €/a (capex %.3e), confined n_v*≈%.0f (finest), n_k=%.1f → β_v=%.3e, β_k=%.3e  [%.0fs]",
                         lp.objective, lp.capex, n_v, n_k, βv, βk, time() - t0))

    bias = init_bias(E, W, Layout(E.kspecs, cfg.levels[1]...), βk / βv; rng=rng, nsamples=cfg.bias_samples,
                     nsteps=cfg.bias_sample_steps, res=cfg.bias_table_res)
    P = build_problem(E, W, cfg.levels, βk, βv, bias; use_bias=cfg.use_bias)
    # LP-score warm start only with a reference LP; LP-free starts learn the bias from the chain alone
    cfg.use_bias && cfg.bias_warmstart > 0 && cfg.init == "lp" && warmstart_bias!(P, lp, xref, ref; weight=cfg.bias_warmstart,
                                                        monotone=bias_monotone_args(P, cfg))

    # interior initial state at the reference level, then up the hierarchy
    δ = 0.02
    center = interior_center(E, lvr)
    vvals = [(1 - δ) .* lp.v[f] .+ δ .* center[f] for f in eachindex(center)]
    kvals = [clamp.(lp.k[f], 1e-3 .* E.kspecs[f].hi, (1 - 1e-3) .* E.kspecs[f].hi) for f in eachindex(lp.k)]
    kref = to_modes(lkr, E.kspecs, [reshape(k, :, 1) for k in kvals])
    vref = to_modes(lvr, E.vspecs, vvals)
    kL = prolong_to(kref, P.lk, P.trk, E.kspecs, ref, L)
    vL = prolong_to(vref, P.lv, P.trv, E.vspecs, ref, L)
    s = make_state(P, L, m, z0, ws, kL, vL)
    info = (; lp, n_v, n_k, ref, setup_time=time() - t0)
    return P, s, info
end

"Restrict a finest-level state to level ℓ (weather prefix and k, v restrictions)."
function restrict_state(P::Problem, s::State, ℓ::Int)
    k = restrict_to(s.k, P.lk, P.trk, P.E.kspecs, s.ℓ, ℓ)
    v = restrict_to(s.v, P.lv, P.trv, P.E.vspecs, s.ℓ, ℓ)
    sc = State(ℓ, copy(s.m), copy(s.z0), s.ws[1:ℓ], s.xs[1:ℓ+1], k, v, -Inf, s.parts)
    return evaluate!(P, sc)
end

"""
Warm start of the bias from the zero-temperature scores of the LP (nodal and
availability shadow prices), restricted to level-0 coordinates. The gradient of
a block-averaged coordinate is the block sum of the fine gradient (adjoint of
the prolongation); spatially it is the mode prefix.
"""
function warmstart_bias!(P::Problem, lp::LPSolution, xref, ref::Int; weight=20.0, monotone=nothing)
    W, E = P.W, P.E
    gz = weather_grad_to_x(W, ref, [-P.βv .* g for g in lp.dF_dX])
    for ℓ in ref:-1:1
        gz, _ = split_state(W.trs[ℓ], W.lays[ℓ], W.lays[ℓ+1], W.specs, gz)
        gz .*= W.trs[ℓ].r
    end
    z0 = restrict_to(xref, W.lays, W.trs, W.specs, ref, 0)
    lkr = P.lk[ref+1]
    gk = zeros(lkr.n)
    kref = zeros(lkr.n)
    for f in eachindex(E.kspecs)
        add_field_grad!(gk, lkr, E.kspecs, f, reshape(-P.βv .* lp.dF_dk[f], :, 1))
        add_field_grad!(kref, lkr, E.kspecs, f, reshape(lp.k[f], :, 1))
    end
    gk0 = restrict_to(gk, P.lk, P.trk, E.kspecs, ref, 0)   # k fields: restriction = mode prefix
    k0 = restrict_to(kref, P.lk, P.trk, E.kspecs, ref, 0)
    X0 = weather_nodes(W, 0, z0)
    P.bias.ℓb == 0 || error("the LP warm start needs the bias at level 0")
    bias_observe!(P.bias, E, W, P.lk[1], k0, X0, gz, gk0; weight=weight)
    bias_update!(P.bias; monotone=monotone)
end

"Arguments of the monotone-price bias fit (nothing: unconstrained)."
bias_monotone_args(P::Problem, cfg::Config) = cfg.bias_monotone ? (P.βv, P.E.c_shed, P.E.c_dump) : nothing

"""
    lift_bias!(P, cfg) -> P

After the level-0 warmup: lift the bias to the finest level (θ kept, a and c prolongated,
regression sums reset, `bias_samples` exact draws redrawn there with `bias_sample_steps`
kept), from a stream of its own so that every driver gets the same sample steps.
"""
function lift_bias!(P::Problem, cfg::Config)
    P.use_bias || return P
    L = nlevels(P) - 1
    lift_bias!(P, L; rng=MersenneTwister(cfg.seed + 4243), nsamples=cfg.bias_samples, nsteps=cfg.bias_sample_steps)
    logmsg(cfg, @sprintf("bias lifted to level %d (%d sample steps of %d draws)", L, length(P.bias.sD), P.bias.nsamp))
    return P
end

"""
    run_level0!(rng, P, s, S, n, cfg; learn, label) -> (state, Q trace, weather diagnostics trace, mean k)

Level-0 chain with the level-0 kernel (`base_step`). With `learn = true` (the bias
warmup) the slice widths and step sizes adapt and the bias is learned from the
chain's scores (from iteration `bias_start`, refit every `bias_every`).
"""
function run_level0!(rng, P::Problem, s::State, S::ChainSampler, n::Int, cfg::Config; learn=false, label="level 0")
    Q = Vector{Vector{Float64}}()
    D = Vector{Vector{Float64}}()
    ksum = zeros(length(s.k))
    t0 = time()
    S.adapt = learn
    S.gibbs.adapt = learn
    P.use_bias && P.bias.ℓb != 0 && error("the level-0 chain needs the bias at level 0 (it has been lifted to level $(P.bias.ℓb))")
    for it in 1:n
        s = base_step(rng, P, s, S)
        if learn
            if it >= cfg.bias_start && P.use_bias && P.bias.active
                Xn, sz, sk, kb = bias_scores(P, s)
                bias_observe!(P.bias, P.E, P.W, P.lk[1], kb, Xn, sz, sk)
                it % cfg.bias_every == 0 && (bias_update!(P.bias; monotone=bias_monotone_args(P, cfg)); evaluate!(P, s))
            end
        end
        push!(Q, quantities(P, s))
        push!(D, weather_diagnostics(P, s.m, s.z0))
        ksum .+= s.k
        if it % max(1, n ÷ 10) == 0
            ktxt = @sprintf("scale-move acc %.2f", sum(S.accscale) / max(sum(S.nscale), 1))
            logmsg(cfg, @sprintf("  %s %5d/%d  %s  total %.4e  [%.0fs]", label, it, n, ktxt,
                                 Q[end][length(P.E.kspecs)+3], time() - t0))
        end
    end
    S.adapt = false
    S.gibbs.adapt = false
    return s, Q, D, ksum ./ n
end

"Seasonal means of the first (system-mean) mode per weather field, and regime frequencies."
function weather_diagnostics(P::Problem, m, z0)
    lay = P.W.lays[1]
    h0 = lay.h
    out = Float64[]
    for f in eachindex(P.W.specs)
        Z = fieldview(z0, lay, P.W.specs, f)
        for sea in 1:4
            idx = [τ for τ in 1:lay.T if season(τ, h0) == sea]
            push!(out, mean(Z[1, idx]))
        end
    end
    append!(out, [count(==(k), m) / length(m) for k in 1:P.W.K])
    return out
end

"""
    descend!(q, potential; iters, lr, mask) -> U

Descent towards the mode, used to pre-equilibrate a chain: the position moves
along a normalized (Adam-like) negative gradient, a step is kept only if the
potential does not increase (the step size grows by 5 % after an accepted step
and halves after a rejected one). Coordinates with `mask = 0` are held fixed.
"""
function descend!(q, potential; iters=200, lr=1e-2, β1=0.9, β2=0.999, mask=ones(length(q)))
    U, g = potential(q)
    m1, m2 = zeros(length(q)), zeros(length(q))
    scale = max.(abs.(q), 1e-3)
    for it in 1:iters
        @. m1 = β1 * m1 + (1 - β1) * g
        @. m2 = β2 * m2 + (1 - β2) * g^2
        p = @. -lr * mask * scale * (m1 / (1 - β1^it)) / (sqrt(m2 / (1 - β2^it)) + 1e-12)
        qn = q .+ p
        Un, gn = potential(qn)
        if isfinite(Un) && Un <= U
            q .= qn
            U, g = Un, gn
            lr *= 1.05
        else
            lr /= 2
        end
    end
    return U
end

"""
Descent of the operation v of a level-0 state at fixed (k, x) (burn-in aid).
Joint minimization over k or x would ignore the operational entropy that the
bias accounts for, so those stay at their initial values.
"""
function predescend(P::Problem, s0::State, cfg::Config)
    cfg.descent_iters > 0 || return s0
    q = vcat(s0.k, s0.v, s0.z0)
    pot = level0_potential(P, s0.m)
    U0 = pot(q)[1]
    mask = vcat(zeros(length(s0.k)), ones(length(s0.v)), zeros(length(s0.z0)))
    t = @elapsed U = descend!(q, pot; iters=cfg.descent_iters, mask=mask)
    logmsg(cfg, @sprintf("descent at level 0: U %.4e → %.4e  [%.0fs]", U0, U, t))
    s = copy(s0)
    nk, nv = length(s.k), length(s.v)
    s.k, s.v, s.z0 = q[1:nk], q[nk+1:nk+nv], q[nk+nv+1:end]
    s.xs = [s.z0]
    return evaluate!(P, s)
end

"The kernels of one chain: the fine-level kernel and the level-0 kernel of the bias warmup."
function new_sampler(P::Problem, s::State, cfg::Config)
    nkf = length(P.E.kspecs)
    return ChainSampler(fill(1e-3, nkf), zeros(Int, nkf), zeros(Int, nkf), false, GibbsState(P),
                        LocalKernel(P; extra=cfg.local_extra, mode=Symbol(cfg.fine_kernel), weather=cfg.fine_weather,
                                    lmkw=(cell_power=cfg.cell_level_power, xr_partner=cfg.xloc_partner_hops,
                                          xr_ball=cfg.xloc_ball_hops), kop=cfg.kop))
end

"""
    run(cfg) -> results

The CPU driver: the multi-chain run of `run_chains` (chains.jl).
"""
run(cfg::Config) = run_chains(cfg)

"Content fingerprint of the model data used by the LP (network, units, weather)."
data_fingerprint(sys::System, path::String) =
    hash((sys.line_bus0, sys.line_bus1, sys.line_b, sys.line_s, sys.link_cap,
          [sys.gen[k] for k in (:carrier, :bus, :invest, :kbar, :k, :capex, :mc, :avail)],
          [sys.stor[k] for k in sort(collect(keys(sys.stor)))], weather_fingerprint(path)))

"Model options that change the LP (nothing for the defaults, so old cache keys stay valid)."
function model_key(cfg::Config)
    k = (isnan(cfg.co2_price) ? nothing : cfg.co2_price, cfg.brownfield, cfg.link_losses, cfg.uc)
    return k == (nothing, false, false, String[]) ? nothing : k
end

"""
LP-free start (large models): proxy renewable capacities (potentials scaled to
70 % of the load), gas at the nodal peak load, batteries at 5 % of it, no H₂; the
interior centre for the operation. Returned in the shape of an `LPSolution`
(objective NaN, no scores) so that the rest of `setup` is unchanged.
"""
function heuristic_start(E::EnergyModel, lk::Layout, lv::Layout, W)
    sys = E.sys
    khat = proxy_capacities(sys)
    peak = vec(maximum(W[1]; dims=2))                       # nodal peak load at the level
    kvals = Vector{Vector{Float64}}()
    for (f, sp) in enumerate(E.kspecs)
        k = zeros(nsupport(sp.B))
        for g in E.gens
            g.kf == f || continue
            u = findall((sys.gen[:carrier] .== g.carrier))
            for (i, n) in enumerate(g.bus)
                if g.carrier == "gas"
                    k[i] = peak[n]
                else
                    j = u[findfirst(==(n), sys.gen[:bus][u])]
                    k[i] = khat[j]
                end
            end
        end
        for st in E.stors
            st.kf == f || continue
            for (i, n) in enumerate(st.bus)
                k[i] = st.carrier == "battery" ? 0.05 * peak[n] : 0.0
            end
        end
        push!(kvals, clamp.(k, sp.lo, sp.hi))
    end
    vvals = interior_center(E, lv)
    nv = sum(length, vvals)
    return LPSolution(NaN, NaN, kvals, vvals, nv, nv, 0, sum(length, kvals),
                      [zeros(size(w)) for w in W], [zeros(length(k)) for k in kvals])
end
