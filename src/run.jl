# End-to-end driver: data → weather model → LP calibration → level-0 warmup
# with bias learning → level-0 production chain → MLDA on the finest level →
# estimator, diagnostics and outputs (§7).

Base.@kwdef mutable struct Config
    data::String = "data/processed/model.h5"
    out::String = "results/run.h5"
    levels::Vector{Tuple{Int,Int}} = [(4, 24), (12, 24), (40, 24), (40, 6)]
    seed::Int = 1
    K::Int = 4                      # weather regimes
    c_shed::Float64 = 3000.0
    c_dump::Float64 = 5.0
    c_viol::Float64 = 3000.0        # price of constraint violations (regularized Ω)
    regularized::Bool = true        # price all constraints instead of hard walls
    kernel::String = "gibbs"        # base kernel: "local" (sparse exact local moves, needs basis = "haar"), "gibbs" (coordinate slice Gibbs) or "hmc"
    local_extra::Float64 = 4.0      # kernel = "local": random non-atom moves per node and step and sweep
    fine_kernel::String = "auto"    # kernel = "local", fine levels: "local", "flow", "adaptive" or "auto" (flow if basis = "flow")
    cell_level_power::Float64 = 1.0 # kernel = "local": partition level of cell moves drawn ∝ (number of cells)^power
    xloc_partner_hops::Int = 1      # kernel = "local": partners of the localized pair moves (xloc, xlocc) within this many hops
    xloc_ball_hops::Int = 0         # ... flow re-routed on the nodes within this many hops of the pair (0: the pair only)
    measure::String = "utilization" # reference measure of dispatchable units: "utilization" or "physical" (MW)
    storage::String = "lp"          # storage form: "lp" (charge, discharge, SoC) or "soc" (SoC trajectory only)
    renewables::String = "dispatch" # "dispatch" (curtailment share per unit) or "load" (output A·k; curtailment = nodal dump)
    flow_theta::String = "nested"   # basis = "flow": "nested" (modes are the coarse angle coordinates) or "haar" (sparse angles, modes as moves)
    # model completion (docs/model_completion.md)
    co2_price::Float64 = NaN        # canonical CO₂ price μ (€/t); NaN: the price contained in the data file
    co2_cap::Float64 = 0.0          # > 0: tune μ during the finest-level warmup so that the mean annual emissions equal this cap (t/yr); frozen after
    brownfield::Bool = false        # existing capacities as floors (capex on k − k0), merged vintages, fixed thermal fleets
    losses::Bool = false            # quadratic line losses (piecewise-linear interpolant on a fixed grid)
    loss_segments::Int = 4          # grid cells per thermal rating for the loss interpolant
    thermal::String = "hard"        # line limits: "hard" (wall priced at c_viol) or "soft" (convex overload band)
    thermal_band::Float64 = 0.1     # soft limits: overload band (fraction of the rating) over which the price rises to c_viol
    link_losses::Bool = false       # signed lossy DC links (efficiency from the data)
    uc::Vector{String} = String[]   # carriers with fleet unit commitment (e.g. ["gas", "coal", "lignite"])
    fine_weather::Float64 = 0.0     # kernel = "local": fraction of the weather coordinates moved per fine-level step (0: weather fixed at fine levels)
    init::String = "lp"             # initial state: "lp" (reference LP), "heuristic" (no LP) or "decomposed" (recursive small LPs)
    decomp_cap::Int = 50            # init = "decomposed": maximal region size (nodes)
    decomp_block_days::Int = 7      # init = "decomposed": block length of the operation LPs
    basis::String = "laplace"       # spatial generating system: "laplace" (modes) or "haar" (countries → nodes)
    use_bias::Bool = true           # α = 1 (learned bias) or α = 0
    p_fine::Float64 = 0.5           # levels ≥ 1: probability of a Gibbs sweep at fixed level instead of an MLDA step (v2 §5.2)
    beta_v::Float64 = 4e-5          # operational temperature [1/€], fixed by experience
    beta_k::Float64 = 4e-8          # investment temperature [1/€]
    calibrate_beta::Bool = false    # instead derive β from the LP by equipartition (§1.5)
    eps_v::Float64 = 0.01           # ... with these target slacks (fractions of the optimal cost)
    eps_k::Float64 = 0.05
    nleap::Int = 20
    warmup0::Int = 1500
    mass_windows::Vector{Int} = [75, 150, 300, 600]
    bias_start::Int = 150
    bias_every::Int = 25
    bias_samples::Int = 16
    bias_warmstart::Float64 = 20.0  # weight of the LP-score warm start (in samples), 0 = off
    descent_iters::Int = 4000       # reflective descent before the level-0 warmup
    n0::Int = 1500                  # level-0 production samples (Q_0 term)
    J::Vector{Int} = [4, 3, 2]      # subchain lengths J_ℓ, ℓ = 1..L
    mlda_warmup::Int = 100
    nsamples::Int = 300
    frac::Float64 = 0.05
    ϱ::Float64 = 0.05
    p_ffbs::Float64 = 0.1
    p_scale::Float64 = 0.3
    alpha0::Bool = true             # also run the α = 0 comparison at level 0
    bias_monotone::Bool = true      # fit ĝ with a monotone, bounded implied price (no tail leak)
    bias_online::Bool = true        # keep learning b̂ from the finest chain's scores (diminishing adaptation)
    bias_online_every::Int = 1      # ... every n-th finest-level step
    bias_freeze::Int = typemax(Int) # ... and never after this many steps
    bias_post::Bool = true          # α = 1: refit b̂ after the run and reweight the finest-level samples (weight ESS reported)
    # several chains and convergence diagnostics (docs/validation_plan.md Part A)
    nchains::Int = 0                # > 0: run this many finest-level chains (mlda_warmup + nsamples sweeps each) instead of the MLDA driver
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

function load_config(path::String)
    cfg = Config()
    d = TOML.parsefile(path)
    for (k, v) in d
        s = Symbol(k)
        hasfield(Config, s) || error("unknown config key $k")
        if s == :levels
            v = [(Int(a[1]), Int(a[2])) for a in v]
        elseif fieldtype(Config, s) <: Vector
            v = convert(fieldtype(Config, s), v)
        end
        setfield!(cfg, s, convert(fieldtype(Config, s), v))
    end
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
    @assert ref !== nothing "one level must contain all $N Laplace modes (spatially complete)"
    @assert all(l -> l[1] >= N, levels[ref:end])
    return ref - 1   # 0-based level index
end

"""
    setup(cfg) -> (P, s_init, info)

Builds the energy and weather models, solves the calibration LP and returns the
problem at calibrated temperatures with an initial finest-level state.
"""
function setup(cfg::Config; rng=MersenneTwister(cfg.seed))
    t0 = time()
    sys = load_system(cfg.data)
    E = build_energy_model(sys; c_shed=cfg.c_shed, c_dump=cfg.c_dump, c_viol=cfg.c_viol, regularized=cfg.regularized,
                           basis=Symbol(cfg.basis), measure=Symbol(cfg.measure),
                           storage=Symbol(cfg.storage), renewables=Symbol(cfg.renewables),
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
    keyt = (data_fingerprint(sys, cfg.data), cfg.levels[ref+1], cfg.seed, cfg.K, cfg.c_shed, cfg.c_dump, cfg.levels,
            cfg.storage == "lp" ? nothing : cfg.storage, cfg.renewables == "dispatch" ? nothing : cfg.renewables,
            cfg.basis, :wcache)
    key = model_key(cfg) === nothing ? hash(keyt) : hash((keyt..., model_key(cfg)))
    decomp = nothing
    if cfg.init == "decomposed"
        @assert !cfg.calibrate_beta "init = decomposed needs fixed β"
        xL = weather_at_level(W, L, m, z0, ws)
        XL = weather_nodes(W, L, xL)
        td = @elapsed kd, vd, dinfo = decomposed_start(E, XL; h_fine=cfg.levels[end][2], cap=cfg.decomp_cap,
                                                     block_days=cfg.decomp_block_days, verbose=cfg.verbose)
        decomp = (; kd, vd, dinfo)
        logmsg(cfg, @sprintf("decomposed LP start: depths %d, LP time %.0f s (allocation %.0f s, operation %.0f s), total %.0f s",
                             length(dinfo.parts), dinfo.lp_time, dinfo.alloc_time, dinfo.op_time, td))
        lp = heuristic_start(E, lkr, lvr, Wd)
        lp = LPSolution(NaN, NaN, kd, lp.v, lp.n_v_free, lp.n_v_total, lp.n_k_free, lp.n_k_total, lp.dF_dX, lp.dF_dk)
    elseif cfg.init == "heuristic"
        @assert !cfg.calibrate_beta "init = heuristic needs fixed β"
        lp = heuristic_start(E, lkr, lvr, Wd)
        logmsg(cfg, "no reference LP: heuristic capacities (proxy renewables, gas at nodal peak load), interior dispatch")
    else
        lp = solve_lp_cached(E, lkr, lvr, Wd, joinpath(dirname(cfg.out), "cache", "lp_$(string(key; base=16)).h5"))
    end
    T_L, T_ref = 8760 ÷ cfg.levels[end][2], lvr.T
    # equipartition (§1.5): one unit of β×cost per cost-confined direction; interior
    # investment coordinates are confined by curvature and carry one half
    n_v = (lp.n_v_total - lp.n_v_free) * T_L / T_ref
    n_k = (lp.n_k_total - lp.n_k_free) + 0.5 * lp.n_k_free
    βv = cfg.calibrate_beta ? n_v / (cfg.eps_v * lp.objective) : cfg.beta_v
    βk = cfg.calibrate_beta ? n_k / (cfg.eps_k * lp.objective) : cfg.beta_k
    logmsg(cfg, @sprintf("LP: total %.3e €/a (capex %.3e), confined n_v*≈%.0f (finest), n_k=%.1f → β_v=%.3e, β_k=%.3e  [%.0fs]",
                         lp.objective, lp.capex, n_v, n_k, βv, βk, time() - t0))

    bias = init_bias(E, W, Layout(E.kspecs, cfg.levels[1]...), βk / βv; rng=rng, nsamples=cfg.bias_samples)
    P = build_problem(E, W, cfg.levels, βk, βv, bias; use_bias=cfg.use_bias)
    # LP-score warm start only with a reference LP; LP-free starts learn the bias from the chain alone
    cfg.use_bias && cfg.bias_warmstart > 0 && cfg.init == "lp" && warmstart_bias!(P, lp, xref, ref; weight=cfg.bias_warmstart,
                                                        monotone=bias_monotone_args(P, cfg))

    # strictly interior initial state at the reference level, then up/down the hierarchy
    δ = 0.02
    center = interior_center(E, lvr)
    vvals = [(1 - δ) .* lp.v[f] .+ δ .* center[f] for f in eachindex(center)]
    kvals = [clamp.(lp.k[f], 1e-3 .* E.kspecs[f].hi, (1 - 1e-3) .* E.kspecs[f].hi) for f in eachindex(lp.k)]
    kref = to_modes(lkr, E.kspecs, [reshape(k, :, 1) for k in kvals])
    vref = to_modes(lvr, E.vspecs, vvals)
    kL = prolong_to(kref, P.lk, P.trk, E.kspecs, ref, L)
    vL = prolong_to(vref, P.lv, P.trv, E.vspecs, ref, L)
    if decomp !== nothing          # node-level operation at the finest level from the decomposition
        lvL = P.lv[L+1]
        centerL = interior_center(E, lvL)
        vL = to_modes(lvL, E.vspecs, [(1 - δ) .* decomp.vd[f] .+ δ .* centerL[f] for f in eachindex(centerL)])
    end
    s = make_state(P, L, m, z0, ws, kL, vL)
    for ℓ in 0:L
        @assert isfeasible(P.cons[ℓ+1], vcat(restrict_to(kL, P.lk, P.trk, E.kspecs, L, ℓ),
                                            restrict_to(vL, P.lv, P.trv, E.vspecs, L, ℓ))) "initial state infeasible at level $ℓ"
    end
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

function initial_minv(P::Problem)
    lk, lv, lw = P.lk[1], P.lv[1], P.W.lays[1]
    mk = zeros(lk.n)
    for (f, s) in enumerate(P.E.kspecs)
        fieldview(mk, lk, P.E.kspecs, f) .= (1e-4 * maximum(s.hi))^2
    end
    mv = fill(1e-6, lv.n)
    fieldview(mv, lv, P.E.vspecs, P.E.fθ) .= 1.0
    return vcat(mk, mv, fill(1e-4, lw.n))
end

"Score samples at a level-0 state for the bias regression (state coordinates)."
function scores(P::Problem, s::State)
    lk, lv = P.lk[1], P.lv[1]
    Xn = weather_nodes(P.W, 0, s.z0)
    _, (_, gko, _, gX) = energy_cost(P.E, lk, lv, s.k, s.v, Xn; grad=true)
    return Xn, weather_grad_to_x(P.W, 0, [-P.βv .* g for g in gX]), -P.βv .* gko
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
    bias_observe!(P.bias, E, W, P.lk[1], k0, X0, Float64(P.levels[1][2]), gz, gk0; weight=weight)
    bias_update!(P.bias; monotone=monotone)
end

"Arguments of the monotone-price bias fit (nothing: unconstrained)."
bias_monotone_args(P::Problem, cfg::Config) = cfg.bias_monotone ? (P.βv, P.E.c_shed, P.E.c_dump) : nothing

"""
Level-0 chain. With `learn=true` adapts step size, mass matrix and the bias.
Returns (state, Q trace, weather diagnostics trace).
"""
function run_level0!(rng, P::Problem, s::State, S::MLDASampler, n::Int, cfg::Config; learn=false, label="level 0")
    Q = Vector{Vector{Float64}}()
    D = Vector{Vector{Float64}}()
    ksum = zeros(length(s.k))
    windows = cumsum(cfg.mass_windows)
    t0 = time()
    S.adapt = learn
    S.gibbs === nothing || (S.gibbs.adapt = learn)
    for it in 1:n
        s = base_step(rng, P, s, S)
        if learn && S.kernel == :hmc
            accumulate_mass!(S.hmc, vcat(s.k, s.v, s.z0))
            if it in windows
                update_mass!(S.hmc)
            end
        end
        if learn
            if it >= cfg.bias_start && P.use_bias && P.bias.active
                Xn, sz, sk = scores(P, s)
                bias_observe!(P.bias, P.E, P.W, P.lk[1], s.k, Xn, Float64(P.levels[1][2]), sz, sk)
                it % cfg.bias_every == 0 && (bias_update!(P.bias; monotone=bias_monotone_args(P, cfg)); evaluate!(P, s))
            end
        end
        push!(Q, quantities(P, s))
        push!(D, weather_diagnostics(P, s.m, s.z0))
        ksum .+= s.k
        if it % max(1, n ÷ 10) == 0
            ktxt = S.kernel == :hmc ?
                   @sprintf("hmc acc %.2f ε %.2e", mean(S.hmc_acc[max(1, end - 99):end]), S.hmc.ε) :
                   @sprintf("scale-move acc %.2f", sum(S.accscale) / max(sum(S.nscale), 1))
            logmsg(cfg, @sprintf("  %s %5d/%d  %s  total %.4e  [%.0fs]", label, it, n, ktxt,
                                 Q[end][length(P.E.kspecs)+3], time() - t0))
        end
    end
    learn && S.kernel == :hmc && finish_stepsize!(S.hmc)
    S.adapt = false
    S.gibbs === nothing || (S.gibbs.adapt = false)
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

"F̂(x,k) ≈ −(ĝ + ĥ)/β_v for weather years and per regime (§7 weather-cost map)."
function weather_cost_map(P::Problem, k0)
    W = P.W
    Δt = Float64(P.levels[1][2])
    h, _ = hhat(P.bias, clamped_kcoeffs(P.E, P.lk[1], k0)[1])
    years = Float64[]
    perregime = zeros(W.K)
    wsum = zeros(W.K)
    for y in eachindex(W.data[1])
        z0 = W.data[1][y]
        Xn = weather_nodes(W, 0, z0)
        push!(years, -(ghat(P.bias, P.E, P.lk[1], k0, Xn, Δt) + h) / P.βv)
        # per-step contributions, weighted by regime responsibilities
        T = size(Xn[1], 2)
        for t in 1:T
            Xt = [x[:, t:t] for x in Xn]
            gt = ghat(P.bias, P.E, P.lk[1], k0, Xt, Δt)
            for k in 1:W.K
                perregime[k] += W.γ[y][k, t] * (-gt / P.βv)
                wsum[k] += W.γ[y][k, t]
            end
        end
    end
    return years, perregime ./ wsum
end

"""
Reflective descent of the operation v of a level-0 state at fixed (k, x) (burn-in
aid). Joint minimization over k or x would ignore the operational entropy that
the bias accounts for, so those stay at their initial values.
"""
function predescend(P::Problem, s0::State, cfg::Config)
    cfg.descent_iters > 0 || return s0
    q = vcat(s0.k, s0.v, s0.z0)
    pot = level0_potential(P, s0.m)
    U0 = pot(q)[1]
    mask = vcat(zeros(length(s0.k)), ones(length(s0.v)), zeros(length(s0.z0)))
    t = @elapsed U = descend!(q, pot, P.cons0, BilliardWS(P.cons0); iters=cfg.descent_iters, mask=mask)
    logmsg(cfg, @sprintf("reflective descent at level 0: U %.4e → %.4e  [%.0fs]", U0, U, t))
    s = copy(s0)
    nk, nv = length(s.k), length(s.v)
    s.k, s.v, s.z0 = q[1:nk], q[nk+1:nk+nv], q[nk+nv+1:end]
    s.xs = [s.z0]
    return evaluate!(P, s)
end

function new_sampler(P::Problem, s::State, cfg::Config)
    L = nlevels(P) - 1
    σk, σv = Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
    cur = s
    chain = [s]
    for ℓ in L:-1:1
        sc, (w, ξk, ξv) = split_level(P, chain[1])
        pushfirst!(chain, sc)
        pushfirst!(σk, detail_scales(P.trk[ℓ], P.E.kspecs, ξk))
        pushfirst!(σv, detail_scales(P.trv[ℓ], P.E.vspecs, ξv))
    end
    J = length(cfg.J) >= L ? cfg.J[1:L] : vcat(cfg.J, fill(cfg.J[end], L - length(cfg.J)))
    hmc = HMCState(1e-2, cfg.nleap, initial_minv(P))
    nkf = length(P.E.kspecs)
    return MLDASampler(J, hmc, BilliardWS(P.cons0), cfg.p_ffbs, cfg.p_scale, fill(1e-3, nkf), zeros(Int, nkf), zeros(Int, nkf),
                       fill(cfg.ϱ, L), σk, σv, zeros(L), cfg.frac,
                       false, zeros(Int, L), [LevelStats() for _ in 1:L], false, [Tuple{Vector{Float64},Vector{Float64}}[] for _ in 1:L],
                       Float64[], Symbol(cfg.kernel), cfg.kernel in ("gibbs", "local") ? GibbsState(P) : nothing,
                       cfg.p_fine, zeros(Int, L), cfg.kernel == "local" ? LocalKernel(P; extra=cfg.local_extra, mode=Symbol(cfg.fine_kernel),
                                                                   weather=cfg.fine_weather,
                                                                   lmkw=(cell_power=cfg.cell_level_power, xr_partner=cfg.xloc_partner_hops, xr_ball=cfg.xloc_ball_hops)) : nothing)
end

function run(cfg::Config)
    cfg.nchains > 0 && return run_chains(cfg)
    rng = MersenneTwister(cfg.seed)
    P, sL, info = setup(cfg; rng=rng)
    L = nlevels(P) - 1
    S = new_sampler(P, sL, cfg)
    names = quantity_names(P)
    t0 = time()

    # ---- level 0: warmup with step size / mass adaptation and bias learning
    s0 = restrict_state(P, sL, 0)
    s0 = predescend(P, s0, cfg)
    logmsg(cfg, "level-0 warmup and bias learning ($(cfg.warmup0) its, dim $(P.cons0.n))")
    s0, Qw, Dw, _ = run_level0!(rng, P, s0, S, cfg.warmup0, cfg; learn=true, label="warmup")
    P.bias.active = false
    evaluate!(P, s0)

    # ---- level 0 production chain (Q_0 term of the estimator, diagnostics)
    logmsg(cfg, "level-0 production chain ($(cfg.n0) its)")
    s0, Q0, D0, k0mean = run_level0!(rng, P, s0, S, cfg.n0, cfg; label="level 0")

    # ---- exact weather draws for the diagnostic §6.4
    Dexact = [weather_diagnostics(P, sample_weather(rng, P.W, 0)[1:2]...) for _ in 1:max(cfg.n0, 500)]

    # ---- MLDA on the finest level, started from the (LP-based) initial finest state;
    #      lifting the level-0 state is not always possible since fine fibers can be empty
    QL = Vector{Vector{Float64}}()
    pacc = nothing
    PB = nothing
    OB = OnlineBias()
    online = false
    if L > 0
        evaluate!(P, sL)
        logmsg(cfg, "MLDA warmup ($(cfg.mlda_warmup) its) and production ($(cfg.nsamples) its) on level $L, dim $(length(sL.k) + length(sL.v))")
        OB = OnlineBias(; every=cfg.bias_online_every, freeze=cfg.bias_freeze)
        online = cfg.use_bias && cfg.bias_online
        S.adapt = true
        for it in 1:cfg.mlda_warmup
            sL = level_step(rng, P, sL, S, L)
            online && ((sL, _) = online_bias_step!(P, sL, OB; monotone=bias_monotone_args(P, cfg)))
            if cfg.co2_cap > 0                 # canonical price tuned to the mean cap (Robbins–Monro, frozen after warmup)
                μ = P.E.co2[] * exp(0.5 / it^0.6 * clamp(sL.parts.co2 / cfg.co2_cap - 1, -1.0, 1.0))
                set_co2_price!(P.E, μ)
                sL = evaluate!(P, sL)
                it == cfg.mlda_warmup && logmsg(cfg, @sprintf("CO₂ price tuned to the cap: μ = %.1f €/t (emissions %.3g t)", μ, sL.parts.co2))
            end
        end
        S.adapt = false
        foreach(st -> (st.nprop = 0; st.nacc = 0; st.ninfeasible = 0; empty!(st.violated)), S.stats)
        S.record = true
        pacc = nothing
        raccs = Dict{String,RegionPriceAcc}()
        LMf = nothing
        if !isempty(P.E.parts) && P.E.regularized
            LMf = LocalModel(P, L)
            pacc = PriceAcc(P.E.sys.N, P.lv[L+1].T)
            raccs = Dict("country" => RegionPriceAcc(P.E.sys.country), "system" => RegionPriceAcc(fill("all", P.E.sys.N)))
        end
        PB = (P.use_bias && cfg.bias_post) ? PostBias() : nothing
        for it in 1:cfg.nsamples
            sL = level_step(rng, P, sL, S, L)
            online && ((sL, _) = online_bias_step!(P, sL, OB; monotone=bias_monotone_args(P, cfg)))
            push!(QL, quantities(P, sL))
            PB === nothing || post_record!(PB, P, sL, QL[end])
            if pacc !== nothing
                LCf = LocalCost(P, LMf, sL)
                accumulate_prices!(pacc, P, LMf, LCf)
                foreach(a -> accumulate_region_prices!(a, P, LMf, LCf, sL), values(raccs))
            end
            if it % max(1, cfg.nsamples ÷ 10) == 0
                accs = join([@sprintf("ℓ%d %.2f", ℓ, st.nacc / max(st.nprop, 1)) for (ℓ, st) in enumerate(S.stats)], ", ")
                logmsg(cfg, @sprintf("  MLDA %4d/%d  acceptance %s  total %.4e  [%.0fs]", it, cfg.nsamples, accs,
                                     QL[end][length(P.E.kspecs)+3], time() - t0))
            end
        end
    end

    # ---- estimator
    est = mean(Q0)
    corr = Vector{Vector{Float64}}()
    for ℓ in 1:L
        pr = S.pairs[ℓ]
        c = isempty(pr) ? zeros(length(names)) : mean(first.(pr)) .- mean(last.(pr))
        push!(corr, c)
        est = est .+ c
    end

    # ---- α = 0 comparison at level 0
    Qa0 = Vector{Vector{Float64}}()
    if cfg.alpha0
        logmsg(cfg, "α = 0 comparison (no bias) at level 0 ($(cfg.n0) its)")
        Pa0 = with_bias(P, false)
        sa = evaluate!(Pa0, copy(s0))
        sa, Qa0, _, _ = run_level0!(rng, Pa0, sa, S, cfg.n0, cfg; label="α=0")
    end

    years_F, regime_F = weather_cost_map(P, k0mean)

    mkpath(dirname(cfg.out))
    h5open(cfg.out, "w") do f
        f["quantity_names"] = names
        f["level0/warmup_Q"] = reduce(hcat, Qw)
        f["level0/Q"] = reduce(hcat, Q0)
        f["level0/weather_diag"] = reduce(hcat, D0)
        f["level0/warmup_weather_diag"] = reduce(hcat, Dw)
        f["exact/weather_diag"] = reduce(hcat, Dexact)
        f["level0/hmc_acc"] = S.hmc_acc
        isempty(QL) || (f["finest/Q"] = reduce(hcat, QL))
        L > 0 && online && (f["bias/online_updates"] = OB.updates)
        # convergence diagnostics per quantity: ESS and split-R̂ of the single chains
        f["diagnostics/ess_level0"] = [ess([q[i] for q in Q0]) for i in eachindex(names)]
        f["diagnostics/rhat_level0"] = [split_rhat([[q[i] for q in Q0]]) for i in eachindex(names)]
        if !isempty(QL)
            f["diagnostics/ess_finest"] = [ess([q[i] for q in QL]) for i in eachindex(names)]
            f["diagnostics/rhat_finest"] = [split_rhat([[q[i] for q in QL]]) for i in eachindex(names)]
        end
        if L > 0 && PB !== nothing && !isempty(PB.bused)
            pc = post_correct(P, PB; monotone=bias_monotone_args(P, cfg))
            f["post/weights"] = pc.w
            f["post/logw"] = pc.logw
            f["post/ess"] = pc.ess
            f["post/Q_weighted"] = pc.Qw
            f["post/Q_unweighted"] = pc.Q
            f["post/weather_diag_weighted"] = pc.Dw
            f["post/weather_diag_unweighted"] = pc.D
            f["post/weather_diag_exact"] = mean(Dexact)
            logmsg(cfg, @sprintf("bias post-correction: weight ESS %.1f of %d samples (log-weight range %.2f)",
                                 pc.ess, length(pc.w), maximum(pc.logw) - minimum(pc.logw)))
        end
        if pacc !== nothing && pacc.n > 0
            f["prices/sampled"] = prices(pacc)            # €/MWh per node and finest step
            info.ref == L && (f["prices/lp"] = info.lp.dF_dX[1] ./ P.levels[end][2])   # LP duals at the same level
            for (rn, a) in raccs                           # per region and season (DJF, MAM, JJA, SON)
                for (key, val) in region_price_summary(a)
                    f["prices/regions/$rn/$key"] = val
                end
            end
        end
        for ℓ in 1:L
            pr = S.pairs[ℓ]
            isempty(pr) && continue
            f["pairs/level$ℓ/fine"] = reduce(hcat, first.(pr))
            f["pairs/level$ℓ/coarse"] = reduce(hcat, last.(pr))
        end
        if L > 0
            f["mlda/acceptance"] = [st.nacc / max(st.nprop, 1) for st in S.stats]
            f["mlda/infeasible"] = [st.ninfeasible / max(st.nprop, 1) for st in S.stats]
            for (j, st) in enumerate(S.stats), (fld, c) in st.violated
                f["mlda/violated/level$j/$fld"] = c / max(st.nprop, 1)
            end
        end
        f["estimate/multilevel"] = est
        isempty(QL) || (f["estimate/finest_chain"] = mean(QL))
        f["estimate/level0"] = mean(Q0)
        isempty(Qa0) || (f["alpha0/Q"] = reduce(hcat, Qa0))
        f["bias/theta"] = P.bias.θ
        f["bias/a"] = P.bias.a
        f["bias/c"] = P.bias.c
        f["costmap/years"] = years_F
        f["costmap/year_labels"] = P.E.sys.years
        f["costmap/regimes"] = regime_F
        f["weather/transition"] = P.W.Ptr
        f["weather/em_loglik"] = P.W.loglik
        f["calibration/beta"] = [P.βk, P.βv]
        f["calibration/lp_objective"] = info.lp.objective
        f["calibration/lp_capacity"] = [sum(k) for k in info.lp.k]
        f["config/levels"] = reduce(hcat, [[a, b] for (a, b) in cfg.levels])
        if S.loc !== nothing          # learned quantities of the adaptive fine kernel
            for (ℓ, AM) in S.loc.adapt
                g = "adaptive/level$ℓ"
                f["$g/updates"] = AM.updates
                f["$g/transport_modes"] = AM.Dθ
                for ((kind, L, cl), Φ) in AM.shapes
                    f["$g/shapes/$(kind)_L$(L)_season$cl"] = Φ
                end
                nz = [n for n in eachindex(AM.runits) if !isempty(AM.runits[n])]
                f["$g/response/nodes"] = nz
                f["$g/response/beta_deficit"] = reduce(vcat, [AM.β[n][:, 1] for n in nz]; init=Float64[])
                f["$g/response/beta_surplus"] = reduce(vcat, [AM.β[n][:, 2] for n in nz]; init=Float64[])
                f["$g/response/units_per_node"] = [length(AM.runits[n]) for n in nz]
            end
        end
    end
    for (ℓ, st) in enumerate(S.stats)
        logmsg(cfg, @sprintf("level %d: %d proposals, accepted %.3f, infeasible %.3f; violated fields: %s", ℓ, st.nprop,
                             st.nacc / max(st.nprop, 1), st.ninfeasible / max(st.nprop, 1),
                             join(["$k $(round(v / max(st.nprop, 1); digits=2))" for (k, v) in sort(collect(st.violated); by=x -> -x[2])], ", ")))
    end
    logmsg(cfg, "wrote $(cfg.out)  [total $(round(time() - t0 + info.setup_time; digits=0))s]")
    return (; P, S, est, Q0, QL, Qa0, names, info)
end

"""
    init_by_prolongation(rng, P, S, s0, sw; nsweeps) -> sL

LP-free initialization of the finest level (docs/flow_hierarchy.md §7): start
from an (equilibrated) level-0 state `s0`, prolong k and v level by level with
zero details, keep the weather of the reference state `sw` (finest level), and
relax each level with `nsweeps[ℓ]` fixed-level sweeps of the base kernel.
"""
function init_by_prolongation(rng, P::Problem, S, s0::State, sw::State; nsweeps=fill(20, nlevels(P) - 1))
    L = nlevels(P) - 1
    s = s0
    for ℓ in 1:L
        k = prolong_to(s.k, P.lk, P.trk, P.E.kspecs, ℓ - 1, ℓ)
        v = prolong_to(s.v, P.lv, P.trv, P.E.vspecs, ℓ - 1, ℓ)
        s = State(ℓ, copy(sw.m), copy(sw.z0), sw.ws[1:ℓ], sw.xs[1:ℓ+1], k, v, -Inf, s.parts)
        evaluate!(P, s)
        for _ in 1:nsweeps[ℓ]
            s = S.kernel == :local ? fine_local_step(rng, P, s, S) : fine_gibbs_step(rng, P, s, S.gibbs)
        end
    end
    return s
end


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
LP-free start (large models, docs/flow_hierarchy.md): proxy renewable capacities
(potentials scaled to 70 % of the load), gas at the nodal peak load, batteries at
5 % of it, no H₂; the interior centre for the operation. Returned in the shape of
an `LPSolution` (objective NaN, no scores) so that the rest of `setup` is unchanged.
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
