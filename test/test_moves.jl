# Deterministic exactness checks of every move family on single moves (docs/system/testing.md §5).
#
# For every family of the table `FAMILIES` (one entry per family), about 20 sites with a fixed
# seed; every site draws its discrete choices (step, unit, partner, interval, direction,
# template) from a generator of its own, so that the same site can be prepared again at
# another state. Per site:
#   1. draw consistency: the potential βΔC is linear between consecutive walk points (checked
#      at 1/4, 1/2, 3/4 of every segment); the recorded walk evaluation equals the plain one
#      (bitwise); the cost change the move reports for the drawn t equals the global change
#      after the commit;
#   2. reversibility: prepared again at the moved state with the same seed, the same discrete
#      choices come out (the fixed part of the direction, the shares of the compensation, the
#      draws consumed); the excess of every share-preserving compensation is kept and its
#      requirement shifted by g t; committing −t returns to the start; the reverse line
#      density is the forward one shifted, f_rev(u) − f_rev(0) = f(t + u) − f(t), with the
#      Jacobian, prior and bias terms; the Jacobians of the two directions are inverse; for
#      slice-sampled capacity moves the slice width is the same at both ends and the lower
#      bound shifts with the move; for Metropolis forms, negating the proposal returns exactly;
#   3. caches: after one threaded sweep of the family the cost caches (LocalCost, and the
#      BiasCache where the family changes k or x) equal a rebuild.
# Two setups: A, configs/haar_poc.toml at level 1 (40 nodes daily, no flow basis, no losses:
# the share-preserving compensation with a fixed requirement change and `soc_qadj`), with the
# bias at α = 1; B, a 5-node sub-network (DE/NL, two lossy DC links) daily with the flow basis,
# line losses, soft limits, lossy links, gas commitment, the physical measure, the bias and
# adapted modes (requirement-tracking compensation, the transport, learned and adaptive
# families). The older per-family tests (test_local.jl, test_flow.jl, test_kbox.jl,
# test_kop.jl, test_wblock.jl, test_model.jl) check parts of this; see the comments in the table.
using Test, Random, LinearAlgebra, Statistics
using ESMSampling
const MV = ESMSampling
@isdefined(MockRNG) || include(joinpath(@__DIR__, "..", "device", "test", "mockrng.jl"))

const NSITES = 20                    # sites per family and setup

# ---------------------------------------------------------------------------
# setups
# ---------------------------------------------------------------------------

"Mutable test context: problem, level model, state and its caches, frames and adaptive modes, workspaces."
mutable struct MCtx
    name::Symbol
    P::Any
    LM::Any
    s::Any
    LC::Any
    BC::Any
    FF::Any
    AM::Any
    ws::Any
    wa::Any
    pool::Any
    jac::Vector{NTuple{5,Float64}}
end
function MCtx(name, P, s; FF=nothing, AM=nothing)
    LM = MV.LocalModel(P, s.ℓ)
    MV.evaluate!(P, s)
    c = MCtx(name, P, LM, s, MV.LocalCost(P, LM, s), P.use_bias ? MV.BiasCache(P, s) : nothing, FF, AM,
             MV.MoveWS(P), MV.MoveWS(P), MV.BlockPool(P, 128), NTuple{5,Float64}[])
    return c
end
rebuild!(c::MCtx) = (c.LC = MV.LocalCost(c.P, c.LM, c.s); c.BC = c.P.use_bias ? MV.BiasCache(c.P, c.s) : nothing; c)

"Setup A: configs/haar_poc.toml at level 1 with the bias of test_kop.jl."
function setup_moves_A()
    cfg = MV.load_config(joinpath(@__DIR__, "..", "configs", "haar_poc.toml")); cfg.verbose = false
    cfg.use_bias = true; cfg.bias_warmstart = 20.0; cfg.bias_monotone = true; cfg.measure = "physical"
    P, sL, _ = MV.setup(cfg; rng=MersenneTwister(cfg.seed))
    MV.lift_bias!(P, 1; rng=MersenneTwister(5), nsteps=2048)
    P.bias.θ .*= 3.0                                  # a bias of realistic size (as in test_bias.jl)
    MV.rebuild_tables!(P.bias)
    return MCtx(:A, P, MV.restrict_state(P, sL, 1))
end

"Setup B: 5-node sub-network with the flow basis and all model terms, adapted modes (four adaptive sweeps)."
function setup_moves_B(θ)
    sys = MV.subsystem(MV.load_system(joinpath(@__DIR__, "..", "data", "processed", "model.h5")), [6, 10, 13, 17, 36])
    sys.line_s .*= 0.25                               # congested: predicted binding lines (tline) and line-limit kinks
    E = MV.build_energy_model(sys; basis=:flow, flow_theta=:haar, flow_kw=(K_max=2, nyears=1), measure=:physical,
                              losses=true, thermal=:soft, link_losses=true, uc=["gas"])
    levels = [(2, 120), (5, 24)]
    W = MV.fit_weather_model(MersenneTwister(1), E, levels; K=2)
    bias = MV.init_bias(E, W, MV.Layout(E.kspecs, levels[1]...), 1e-3; rng=MersenneTwister(2), nsamples=4)
    P = MV.build_problem(E, W, levels, 4e-8, 4e-5, bias; use_bias=true)
    MV.lift_bias!(P, 1; rng=MersenneTwister(5), nsteps=512)
    P.bias.θ .= θ                                     # the coefficients of setup A (no LP warm start here)
    MV.rebuild_tables!(P.bias)
    m, z0, ws = MV.sample_weather(MersenneTwister(3), W, 1)
    lk, lv = P.lk[2], P.lv[2]
    Xn = MV.weather_nodes(W, 1, MV.weather_at_level(W, 1, m, z0, ws))
    Wd = [map(x -> MV.datamap(E.wspecs[f], x), Xn[f]) for f in eachindex(Xn)]
    h = MV.heuristic_start(E, lk, lv, Wd)
    k = MV.to_modes(lk, E.kspecs, [reshape(clamp.(x, 0.0, sp.hi), :, 1) for (x, sp) in zip(h.k, E.kspecs)])
    s = MV.make_state(P, 1, m, z0, ws, k, MV.to_modes(lv, E.vspecs, h.v))
    LM = MV.LocalModel(P, 1)
    FF = MV.FlowFrames(P, 1, s)
    AM = MV.AdaptiveModes(P, 1; K=2, lengths=[2, 4, 8])
    rng = MersenneTwister(31)
    for _ in 1:4
        s, _, _ = MV.adaptive_sweep!(rng, P, LM, FF, AM, s; extra=1.0, slong=(p=1.0, nmoves=4, hmin=24.0))
    end
    @assert !isempty(AM.updates) && size(AM.Dθ, 2) > 0
    return MCtx(:B, P, s; FF=FF, AM=AM)
end

# ---------------------------------------------------------------------------
# line moves: a prepared site
# ---------------------------------------------------------------------------

"A prepared line move: mover, Jacobian terms, signature of its choices, prior quadratic, bias change, commit."
struct LSite
    mv::Any
    jac::Vector{NTuple{5,Float64}}
    sig::Any
    quad::NTuple{2,Float64}
    bias::Any                        # x -> Δb̂(x)
    commit!::Any                     # t -> commit into c.s, c.LC, c.BC
    state::Any                       # () -> the weather state to compare after the round trip (or nothing)
end

"""
Signature of the discrete choices of a prepared move: indices (steps, entries, coefficient
indices, kink ids, weather positions) and values (the fixed part of the direction, the shares
of the share-preserving compensation, its requirement changes and Jacobian exponents, the
weather changes), and the next draw of the site's generator (the draws consumed).
"""
function signature(mv, r; extra=())
    idx, val = Int[], Float64[]
    append!(idx, extra)
    mv isa MV.BlockMove && append!(idx, mv.τs[1:mv.n])
    for ws in MV.mv_wss(mv)
        push!(idx, -1)
        for (f, i) in sort(ws.ent); push!(idx, f, i); end
        append!(idx, ws.zidx, ws.zkidx, ws.zkid, ws.wf, ws.wpos, ws.knode)
        append!(val, ws.zval, ws.zkval, ws.wdX, ws.knm)
        append!(val, (isnan(g) ? 0.0 : g for g in ws.kg))
    end
    return (; idx, val, next=rand(r))
end
function same_sig(a, b)
    a.idx == b.idx && a.next == b.next || return false
    length(a.val) == length(b.val) || return false
    return all(abs(x - y) <= 1e-9 * (abs(x) + abs(y)) + 1e-12 for (x, y) in zip(a.val, b.val))
end

"Two-step site of the move dispatch of `local_sweep!`: f!(r, wa, ws, τm, τ, jac) prepares the family."
function stepsite(c::MCtx, r, f!)
    P, LM, LC = c.P, c.LM, c.LC
    T = LM.T
    τ = rand(r, 1:T); τm = mod(τ - 2, T) + 1
    ws, wa = c.ws, c.wa
    MV.clear!(ws); MV.clear!(wa); wa.follow = false
    empty!(c.jac)
    f!(r, wa, ws, τm, τ, c.jac) || return nothing
    MV.finalize!(ws, P, LM, LC, τ)
    isempty(ws.ent) && return nothing
    MV.finalize!(wa, P, LM, LC, τm)
    MV.link_soc!(wa, ws, LM)
    bm = MV.BlockMove([wa, ws], 2, [τm, τ], P, LM, LC, Float64[], Float64[])
    append!(bm.bps, wa.bps); append!(bm.bps, ws.bps)
    return LSite(bm, c.jac, signature(bm, r), (0.0, 0.0), x -> 0.0, t -> MV.commit_block!(bm, c.s.v, t), () -> nothing)
end

newbm(c::MCtx) = MV.BlockMove(c.pool.wss, 0, Int[], c.P, c.LM, c.LC, Float64[], Float64[])
blocksite(c::MCtx, r, bm; extra=()) =
    LSite(bm, c.jac, signature(bm, r; extra=extra), (0.0, 0.0), x -> 0.0, t -> MV.commit_block!(bm, c.s.v, t), () -> nothing)

"Storage units, single or as a cell–class aggregate (as `storage_ms_sweep!`)."
function stor_units(c::MCtx, r)
    E = c.P.E
    units = [(si, i) for (si, st) in enumerate(E.stors) for i in eachindex(st.bus) if c.LM.complete[st.fe]]
    cellof = E.flow === nothing ? E.parts[2] : E.flow.cell
    if rand(r) < 0.5
        groups = Dict{Tuple{Int,Int},Vector{NTuple{2,Int}}}()
        for (si, i) in units
            push!(get!(groups, (cellof[E.stors[si].bus[i]], si), NTuple{2,Int}[]), (si, i))
        end
        return groups[sort!(collect(keys(groups)))[rand(r, 1:length(groups))]]
    end
    return [rand(r, units)]
end

"Storage multiscale move (plateau, proxy shape, learned shape) on a dyadic block."
function msite(c::MCtx, r; shape::Symbol=:plateau, learn::Bool=false)
    E, LM = c.P.E, c.LM
    L = rand(r, shape == :learned ? c.AM.lengths : [2, 4, 8])
    b0 = rand(r, 1:LM.T-L)
    us = stor_units(c, r)
    shp = nothing
    if shape == :proxy
        Φ = MV.storage_shapes(E.flow, E.sys, c.P.lv[LM.ℓ+1].h, L, E.flow.cell[E.stors[us[1][1]].bus[us[1][2]]])
        size(Φ, 2) == 0 && return nothing
        shp = Φ[:, rand(r, 1:size(Φ, 2))]
    elseif shape == :learned
        Φ = get(c.AM.shapes, (:soc, L, MV.season(b0, c.P.lv[LM.ℓ+1].h)), nothing)
        Φ === nothing && return nothing
        shp = Φ[:, rand(r, 1:size(Φ, 2))]
    end
    bm = newbm(c)
    lrn = learn ? (AM=c.AM, FF=c.FF) : nothing
    MV.prep_storage_ms!(bm, c.P, LM, c.LC, us, b0, L, shp, c.jac; learn=lrn) || return nothing
    return blocksite(c, r, bm; extra=(b0, L, (x for u in us for x in u)...))
end

"Learned dispatch-pair shape (`adshape`) at one node."
function adsite(c::MCtx, r)
    LM = c.LM
    L = rand(r, c.AM.lengths)
    b0 = rand(r, 1:LM.T-L)
    Φ = get(c.AM.shapes, (:gen, L, MV.season(b0, c.P.lv[LM.ℓ+1].h)), nothing)
    Φ === nothing && return nothing
    nodes = [n for n in 1:c.P.E.sys.N if count(u -> LM.role[u[1]][1] == MV.ROLE_GEN && LM.complete[u[1]], LM.units[n]) >= 2]
    gu = LM.genunits[rand(r, nodes)]
    u1, u2 = rand(r, gu), rand(r, gu)
    u1 == u2 && return nothing
    bm = newbm(c)
    empty!(c.jac)
    MV.prep_adshape!(bm, c.P, LM, c.LC, u1, u2, view(Φ, :, rand(r, 1:size(Φ, 2))), b0, L) || return nothing
    return blocksite(c, r, bm)
end

"Long-range storage move (`slong` with L < T, `slevel` with L = T)."
function slsite(c::MCtx, r; level::Bool=false, learn::Bool=false)
    E, LM = c.P.E, c.LM
    T = LM.T
    units = [(si, i) for (si, st) in enumerate(E.stors) for i in eachindex(st.bus) if c.LM.complete[st.fe]]
    si, i = rand(r, units)
    a = rand(r, 1:T)
    L = level ? T : rand(r, 1:T-1)
    m = MV.SLong(c.pool.wss[1], c.pool.wss[2], c.pool.wss[3], c.P, LM, c.LC)
    MV.prep_slong!(m, si, i, a, L, c.jac; learn=learn ? (AM=c.AM, FF=c.FF) : nothing) || return nothing
    return LSite(m, c.jac, signature(m, r; extra=(si, i, a, L, m.τi...)), (0.0, 0.0), x -> 0.0,
                 t -> MV.commit_slong!(m, c.s.v, t), () -> nothing)
end

"Level model of the storage pairs with every storage unit pairing (`sx_hmin = 0`: setup B has no long-duration capacity)."
const SXLM = Dict{Symbol,Any}()
sxLM(c::MCtx) = get!(() -> MV.LocalModel(c.P, c.s.ℓ; sx_hmin=0.0), SXLM, c.name)

"Storage pair over a cyclic interval (`sxlong`): unit, partner, start and length from the site's generator."
function sxsite(c::MCtx, r; learn::Bool=false)
    m = MV.SXLong(c.pool.wss[1], c.pool.wss[2], c.pool.wss[3], c.P, sxLM(c), c.LC)
    MV.prep_sxlong_random!(r, m, c.jac; learn=learn ? (AM=c.AM, FF=c.FF) : nothing) || return nothing
    return LSite(m, c.jac, signature(m, r; extra=(m.us..., m.j, m.a, m.L, m.τi...)), (0.0, 0.0), x -> 0.0,
                 t -> MV.commit_sxlong!(m, c.s.v, t), () -> nothing)
end

"Weather state of a context (compared after the round trip of a weather move)."
wstate(c::MCtx) = (copy(c.s.z0), [copy(w) for w in c.s.ws], copy(c.s.xs[end]))

"Fine weather coordinate move (z0 for ℓ = 0, an innovation of level ℓ ≥ 1), compensated, with the bias change."
function wsite(c::MCtx, r, ℓ::Int)
    P, W = c.P, c.P.W
    ncoord = ℓ == 0 ? W.lays[1].n : W.trs[ℓ].n
    q = rand(r, 1:ncoord)
    bm = newbm(c)
    pr = MV.weather_coord_prepare!(P, c.LM, c.LC, c.s, bm, c.jac, ℓ, q)
    pr === nothing && return nothing
    ch, quad, iz = pr
    BC = c.BC
    bias = x -> BC === nothing ? 0.0 : MV.bias_delta_weather(BC, P.bias, P.E, bm, x)
    commit! = t -> (MV.weather_coord_commit!(P, c.s, bm, ch, iz, ℓ, q, t; BC=BC); BC === nothing || MV.bias_resum!(BC, P.bias))
    return LSite(bm, c.jac, signature(bm, r; extra=(ℓ, q)), quad, bias, commit!, () -> wstate(c))
end

"Joint z0 block move over B level-0 steps along one of the fixed directions."
function zsite(c::MCtx, r)
    P, W = c.P, c.P.W
    V = MV.z0_block_directions(W)
    B = rand(r, (1, 2, 4))                            # (the 16- and 64-step blocks: test_wblock.jl)
    a = rand(r, 1:W.lays[1].T-B+1)
    j = rand(r, 1:size(V, 2))
    bm = newbm(c)
    pr = MV.z0_block_prepare!(bm, P, c.LM, c.LC, c.s, c.jac, a, a + B - 1, V[:, j])
    pr === nothing && return nothing
    ch, dz, quad = pr
    BC = c.BC
    bias = x -> BC === nothing ? 0.0 : MV.bias_delta_weather(BC, P.bias, P.E, bm, x)
    commit! = t -> (MV.z0_block_commit!(P, c.s, bm, ch, dz, t; BC=BC); BC === nothing || MV.bias_resum!(BC, P.bias))
    return LSite(bm, c.jac, signature(bm, r; extra=(a, B, j)), quad, bias, commit!, () -> wstate(c))
end

# ---------------------------------------------------------------------------
# the checks of one line site
# ---------------------------------------------------------------------------

"Largest deviation of two vectors of log densities, relative to 1 + 1e-9 |a| (equal infinities agree)."
ddev(a, b) = maximum((x == y ? 0.0 : abs(x - y) / (1 + 1e-9 * abs(y)) for (x, y) in zip(a, b)); init=0.0)

"""
Log line density f(x) − (constant) of a prepared site, −βΔC − prior + log J − Δb̂, and the sum
of the magnitudes of its terms (the scale of its round-off).
"""
function linedens(c::MCtx, st::LSite, x)
    κ, b = st.quad
    lj = isempty(st.jac) ? 0.0 : MV.logjac(st.jac, x, st.mv)
    a1, a2, a4 = -c.P.βv * MV.mv_eval!(st.mv, x), -(0.5 * κ * x^2 + b * x), -st.bias(x)
    return a1 + a2 + lj + a4, abs(a1) + abs(a2) + abs(lj) + abs(a4)
end

"Accumulated worst deviations of one family."
mutable struct Worst
    n::Int; nt::Int; nk::Int
    lin::Float64; rec::Float64; glob::Float64; sig::Int; exc::Float64; ret::Float64; dens::Float64; jac::Float64; jacsweep::Int
end
Worst() = Worst(0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0)

"""
The checks of §1 and §2 at one site (`seed`) of a line family. The state returns to the start
(to round-off) and stays there for the next site.
"""
function check_line_site!(wst::Worst, c::MCtx, @nospecialize(fam), seed::Int)
    P = c.P
    β = P.βv
    st = fam.prep(c, MersenneTwister(seed))
    st === nothing && return
    jacf = copy(st.jac)
    (!isempty(jacf) && !fam.jac) && (wst.jacsweep += 1)           # Jacobian terms that the sweep would not apply
    t = MV.sample_line!(MersenneTwister(seed + 10^6), st.mv, β, Float64[]; quad=st.quad)
    t == 0.0 && return
    wst.n += 1
    # 1a. the potential is linear between consecutive walk points
    lb = MV.linebuf(st.mv)
    pts, cs = copy(lb.pts), copy(lb.cs)
    for j in 1:length(pts)-1
        pts[j+1] > pts[j] || continue
        wst.nt += 1
        for a in (0.25, 0.5, 0.75)
            cm = β * MV.mv_eval!(st.mv, pts[j] + a * (pts[j+1] - pts[j]))
            wst.lin = max(wst.lin, abs(cm - ((1 - a) * cs[j] + a * cs[j+1])))
        end
    end
    # 1b. recorded walk evaluation = plain evaluation (bitwise)
    ΔC = MV.mv_eval!(st.mv, t)
    Δc = (MV.mv_fill!(st.mv); MV.mv_evalc!(st.mv, t))
    wst.rec = max(wst.rec, abs(Δc - ΔC))
    # a parameter beyond the nearest kink of the compensation (drawn parameters rarely reach it),
    # 1.5 times its distance, if that lies within ten times the range of the walk (weather moves:
    # where the potential with the prior is within 80 nats of its minimum; the prior's round-off
    # grows as t²)
    xk = Inf
    for ws in MV.mv_wss(st.mv), k in eachindex(ws.kQ)
        x = -ws.kQ[k] / ws.kg[k]
        (isfinite(x) && x != 0.0 && abs(x) < abs(xk)) && (xk = x)
    end
    tk = 1.5 * xk
    g = cs .+ 0.5 .* st.quad[1] .* pts .^ 2 .+ st.quad[2] .* pts
    near = st.quad[1] > 0 ? pts[g .<= minimum(g) + 80.0] : [10 * pts[1], 10 * pts[end]]
    (isfinite(tk) && minimum(near) <= tk <= maximum(near)) || (tk = Inf)
    roundtrip!(wst, c, fam, seed, st, t)
    if isfinite(tk)
        wst.nk += 1
        roundtrip!(wst, c, fam, seed, fam.prep(c, MersenneTwister(seed)), tk)
    end
    return
end

"""
§1c and §2 for the prepared site `st` at parameter t: commit, global change, the site prepared
again at the moved state, the reverse density and Jacobian, −t back to the start.
"""
function roundtrip!(wst::Worst, c::MCtx, @nospecialize(fam), seed::Int, st::LSite, t::Float64)
    P = c.P
    β = P.βv
    jacf = copy(st.jac)
    ΔC = MV.mv_eval!(st.mv, t)
    # forward line density at t + u (before the commit) and the Jacobian at t
    us = (-t, -0.37t, 0.61t, 1.5t)
    f0, m0 = linedens(c, st, t)
    fm = [linedens(c, st, t + u) for u in us]
    ff = [x[1] - f0 for x in fm]
    mag = [x[2] + m0 for x in fm]
    ljf = isempty(jacf) ? 0.0 : MV.logjac(jacf, t, st.mv)
    Δb = st.bias(t)
    v0 = copy(c.s.v); w0 = st.state()
    L0, parts0 = c.s.L, c.s.parts
    # 1c. commit: global change = reported change
    MV.mv_eval!(st.mv, t)
    st.commit!(t)
    MV.evaluate!(P, c.s)
    wst.glob = max(wst.glob, abs(c.s.L - L0 - (-β * ΔC - Δb)) / (1e-12 * abs(L0) + 1e-9 * (1 + abs(β * ΔC))))
    # 2. the same site at the moved state
    sigf = st.sig
    rv = fam.prep(c, MersenneTwister(seed))
    if rv === nothing || !same_sig(sigf, rv.sig)
        wst.sig += 1
    else
        # share-preserving compensation: the excess is kept, the requirement moves by g t
        if length(rv.jac) == length(jacf)
            for (a, b) in zip(jacf, rv.jac)
                sc = max(1.0, abs(a[1]))
                wst.exc = max(wst.exc, abs(a[2] - b[2]) / sc)
                isnan(a[4]) || (wst.exc = max(wst.exc, abs(a[3] + a[4] * t - b[3]) / sc))
            end
        else
            wst.sig += 1
        end
        fr0 = linedens(c, rv, 0.0)[1]
        fr = [linedens(c, rv, u)[1] - fr0 for u in us]
        wst.dens = max(wst.dens, maximum(abs.(fr .- ff) ./ (1 .+ 1e-9 .* mag)))
        ljr = isempty(rv.jac) ? 0.0 : MV.logjac(rv.jac, -t, rv.mv)
        wst.jac = max(wst.jac, abs(ljf + ljr))
        MV.mv_eval!(rv.mv, -t)
        rv.commit!(-t)
    end
    w1 = st.state()
    err = maximum(abs, c.s.v .- v0) / (1 + maximum(abs, v0))
    if w0 !== nothing
        err = max(err, maximum(abs, w1[1] .- w0[1]), maximum(abs, w1[3] .- w0[3]),
                  maximum(maximum(abs, a .- b; init=0.0) for (a, b) in zip(w1[2], w0[2])))
    end
    wst.ret = max(wst.ret, err)
    if err > 1e-6                                     # not back at the start: rebuild from the state reached
        MV.evaluate!(P, c.s); rebuild!(c)
    else
        c.s.L, c.s.parts = L0, parts0
    end
    return
end

# ---------------------------------------------------------------------------
# capacity moves
# ---------------------------------------------------------------------------

mutable struct WorstK
    n::Int; nl::Int
    glob::Float64; sig::Int; ret::Float64; dens::Float64; jac::Float64; anti::Float64; width::Float64; lo::Float64
end
WorstK() = WorstK(0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

"Units of kscale (kind, group, index) and of kpeak (dispatchable generators)."
function kscale_units(c::MCtx)
    E, LM, lk = c.P.E, c.LM, c.P.lk[c.LM.ℓ+1]
    u = NTuple{3,Int}[]
    for (gi, g) in enumerate(E.gens)
        g.kf > 0 && MV.iscomplete(lk, E.kspecs, g.kf) && (g.vf == 0 || LM.complete[g.vf]) || continue
        append!(u, [(1, gi, i) for i in eachindex(g.bus)])
    end
    for (si, st) in enumerate(E.stors)
        st.kf > 0 && LM.complete[st.fe] && append!(u, [(2, si, i) for i in eachindex(st.bus)])
    end
    return u
end
kpeak_units(c::MCtx) = [x for x in kscale_units(c) if x[1] == 1 && c.P.E.gens[x[2]].vf > 0]

"""
One capacity move of unit `unit` through the sweep (`kscale_sweep!`/`kpeak_sweep!`, restricted
by `select`), with the test hooks `rec`, `probe`, `fix` and, for the Metropolis forms, a mock
generator with the proposal z and the uniform 0 (accepted).
"""
function kmove!(c::MCtx, move::Symbol, unit, rng; mh::Bool=false, z=0.0, probe=nothing, fix=nothing, learned::Bool=false)
    rec = []
    sel = (kd, g, j) -> (kd, g, j) == unit
    nu = 4000
    r = mh ? MockRNG(fill(z, nu), fill(0.0, nu)) : rng
    if move == :kscale
        c.s, _ = MV.kscale_sweep!(r, c.P, c.LM, c.s; LC=c.LC, BC=c.BC, mh=mh, σ=1.0, order=:fixed, select=sel, rec=rec,
                                  probe=probe, fix=fix)
    else
        c.s, _ = MV.kpeak_sweep!(r, c.P, c.LM, c.s; LC=c.LC, BC=c.BC, AM=learned ? c.AM : nothing, mh=mh, σk=ones(nu),
                                 order=:fixed, select=sel, rec=rec, probe=probe, fix=fix)
    end
    return isempty(rec) ? nothing : rec[1]                # nothing: k < κmin, not moved
end

"""
The checks at one site of a capacity family: forward move (slice draw, or the Metropolis
proposal z), reverse move from the moved state (slice form: `fix = −x`; Metropolis: −z).
"""
function check_kmove_site!(wk::WorstK, c::MCtx, @nospecialize(fam), seed::Int)
    r = MersenneTwister(seed)
    move, mh, learned = fam.move, fam.mh, fam.learned
    units = move == :kscale ? kscale_units(c) : kpeak_units(c)
    unit = rand(r, units)
    k0, v0 = copy(c.s.k), copy(c.s.v)
    L0 = c.s.L
    us = [-0.37, 0.61, 1.5]
    # forward: x the sampler coordinate (u, or δ for the Metropolis form of kpeak)
    z = mh ? (move == :kscale ? 0.3 : 0.2 * MV.knodes(c.P, c.LM.ℓ, c.s.k)[c.P.E.gens[unit[2]].kf][unit[3]]) * randn(r) : 0.0
    rf = kmove!(c, move, unit, MersenneTwister(seed); mh=mh, z=z, probe=x -> [x; x .+ us .* x], learned=learned)
    (rf === nothing || rf.x == 0.0) && return
    wk.n += 1
    move == :kpeak && (wk.nl += rf.learned)
    # incremental against global: the log target changes by the line density without its Jacobian
    wk.glob = max(wk.glob, abs(c.s.L - L0 - (rf.lf - rf.logjac)) / (1e-12 * abs(L0) + 1e-8 * (1 + abs(rf.lf))))
    x = rf.x
    ff = rf.lfp[2:end] .- rf.lfp[1]
    # reverse: the same unit and seed at the moved state, at −x
    rr = mh ? kmove!(c, move, unit, MersenneTwister(seed); mh=true, z=-z, probe=_ -> us .* x, learned=learned) :
              kmove!(c, move, unit, MersenneTwister(seed); fix=-x, probe=_ -> us .* x, learned=learned)
    rr === nothing && (wk.sig += 1; return)
    if move == :kpeak && (rr.S != rf.S || rr.learned != rf.learned ||
                          length(rr.comps) != length(rf.comps) ||
                          any(a[1:2] != b[1:2] || abs(a[3] - b[3]) > 1e-12 for (a, b) in zip(rr.comps, rf.comps)))
        wk.sig += 1
    end
    wk.dens = max(wk.dens, ddev(rr.lfp, ff))
    wk.anti = max(wk.anti, abs(rr.lf + rf.lf) / (1 + 1e-9 * abs(rf.lf)))
    wk.jac = max(wk.jac, abs(rr.logjac + rf.logjac))
    if !mh
        wk.width = max(wk.width, abs(rr.w - rf.w))
        wk.lo = max(wk.lo, abs(rr.lo - (rf.lo - x)) / (1 + abs(rf.lo)))
    end
    err = max(maximum(abs, c.s.k .- k0) / (1 + maximum(abs, k0)), maximum(abs, c.s.v .- v0) / (1 + maximum(abs, v0)))
    wk.ret = max(wk.ret, err)
    err > 1e-9 && (c.s.k .= k0; c.s.v .= v0; MV.evaluate!(c.P, c.s); rebuild!(c))
    return
end

"Pro-rata data of a kop move fixed along its curve: units, compensators with their shares, excess and exponents."
kopsig(mv) = [(kt.kind, kt.gi, kt.i, kt.e, kt.nm, [[(x[1], x[2], x[5]) for x in cp] for cp in kt.comp]) for kt in mv.terms]
function same_kopsig(a, b, sc)
    length(a) == length(b) || return false
    for (x, y) in zip(a, b)
        x[1:3] == y[1:3] && x[5] == y[5] || return false
        maximum(abs.(x[4] .- y[4]) ./ max.(1.0, abs.(x[4])); init=0.0) < 1e-6 || return false
        for (p, q) in zip(x[6], y[6])
            length(p) == length(q) || return false
            all(u[1:2] == w[1:2] && abs(u[3] - w[3]) < 1e-9 for (u, w) in zip(p, q)) || return false
        end
    end
    return true
end

"The checks at one site of `kop` (one unit or one carrier): the per-move functions of `kop_sweep!`."
function check_kop_site!(wk::WorstK, c::MCtx, @nospecialize(fam), seed::Int)
    P, LM = c.P, c.LM
    r = MersenneTwister(seed)
    groups = MV.kop_units(P, LM)
    g = rand(r, groups)
    grp = fam.carrier ? g : [rand(r, g)]
    mv = MV.kop_prepare(P, LM, c.LC, c.s, grp)
    mv === nothing && return
    lf = x -> MV.kop_logdensity(P, LM, c.LC, mv, x; BC=c.BC)[1]
    u = MV.slice_scalar(r, lf, 0.0, 0.3, mv.lo)                    # as kop_sweep!
    u == 0.0 && return
    wk.n += 1
    us = [-0.37, 0.61, 1.5] .* u
    f0, lj, _ = MV.kop_logdensity(P, LM, c.LC, mv, u; BC=c.BC)
    ff = [lf(u + y) - f0 for y in us]
    k0, v0, L0 = copy(c.s.k), copy(c.s.v), c.s.L
    sg = kopsig(mv)
    MV.kop_commit!(P, LM, c.LC, c.s, mv, u; BC=c.BC)
    MV.evaluate!(P, c.s)
    wk.glob = max(wk.glob, abs(c.s.L - L0 - (f0 - lj - length(mv.terms) * u)) / (1e-12 * abs(L0) + 1e-8 * (1 + abs(f0))))
    mv2 = MV.kop_prepare(P, LM, c.LC, c.s, grp)
    if mv2 === nothing || !same_kopsig(sg, kopsig(mv2), 1.0)
        wk.sig += 1
        c.s.k .= k0; c.s.v .= v0; MV.evaluate!(P, c.s); rebuild!(c)
        return
    end
    wk.lo = max(wk.lo, abs(mv2.lo - (mv.lo - u)) / (1 + abs(mv.lo)))
    lf2 = x -> MV.kop_logdensity(P, LM, c.LC, mv2, x; BC=c.BC)[1]
    wk.dens = max(wk.dens, ddev([lf2(y) for y in us], ff))
    f2, lj2, _ = MV.kop_logdensity(P, LM, c.LC, mv2, -u; BC=c.BC)
    wk.anti = max(wk.anti, abs(f2 + f0) / (1 + 1e-9 * abs(f0)))
    wk.jac = max(wk.jac, abs(lj + lj2))
    MV.kop_commit!(P, LM, c.LC, c.s, mv2, -u; BC=c.BC)
    MV.evaluate!(P, c.s)
    err = max(maximum(abs, c.s.k .- k0) / (1 + maximum(abs, k0)), maximum(abs, c.s.v .- v0) / (1 + maximum(abs, v0)))
    wk.ret = max(wk.ret, err)
    err > 1e-9 && (c.s.k .= k0; c.s.v .= v0; MV.evaluate!(P, c.s); rebuild!(c))
    return
end

# ---------------------------------------------------------------------------
# caches after a sweep
# ---------------------------------------------------------------------------

"Largest deviations of the caches from a rebuild: (LocalCost, cache cost, BiasCache)."
function cache_errors(c::MCtx)
    P, LC = c.P, c.LC
    LC2 = MV.LocalCost(P, c.LM, c.s)
    mx(a, b) = maximum(abs.(a .- b); init=0.0)
    eL = max(mx(LC.S, LC2.S), mx(LC.r, LC2.r), mx(LC.F, LC2.F),
             maximum(mx(LC.Y[f], LC2.Y[f]) for f in eachindex(LC.Y) if f != P.E.fθ),
             maximum(mx(LC.X[f], LC2.X[f]) for f in eachindex(LC.X)),
             maximum(mx(LC.kgen[g], LC2.kgen[g]) for g in eachindex(LC.kgen)),
             maximum(mx(LC.kst[g], LC2.kst[g]) for g in eachindex(LC.kst)))
    eC = abs(MV.cache_cost(P, c.LM, LC) - c.s.parts.op) / abs(c.s.parts.op)
    eB = 0.0
    if c.BC !== nothing
        BC, BC2 = c.BC, MV.BiasCache(P, c.s)
        b = MV.bias_term(P, c.s)
        eB = max(abs(MV.bias_value(BC, P.bias) - b) / (1 + abs(b)), mx(BC.U1x, BC2.U1x), mx(BC.U1s, BC2.U1s),
                 mx(BC.zc, BC2.zc) / 1e3, abs(BC.firm - BC2.firm) / 1e3, mx(BC.Dx, BC2.Dx) / 1e3,
                 maximum(mx(BC.At[g], BC2.At[g]) for g in eachindex(BC.At)),
                 maximum(mx(BC.Kw[f], BC2.Kw[f]) for f in eachindex(BC.Kw)) / 1e3)
    end
    return eL, eC, eB
end

"Sweep of the plain, flow and adaptive families through `local_sweep!` (atoms only for :atom)."
function opsweep!(c::MCtx, rng, fam::Symbol)
    N = c.P.E.sys.N
    c.s, st = MV.local_sweep!(rng, c.P, c.LM, c.s; families=[(fam == :atom ? :unit : fam, 1.0)], atoms=fam == :atom,
                              extra=fam == :atom ? 0.0 : 2.0 / N, LC=c.LC, frames=c.FF, adapt=c.AM)
    return st[1, 2] + st[end, 2]
end

# ---------------------------------------------------------------------------
# the family table
# ---------------------------------------------------------------------------
#
# One entry per family: `name`; `setups` it runs on; `kind` (:line, :kmove, :kop); for line
# families `prep(c, r)` (an `LSite` prepared with the site's generator r, or nothing) and `jac`
# (whether the sweep applies the Jacobian step to its terms); `sweep(c, rng)` (one threaded sweep
# of the family, returning the number of accepted moves; checked against the cache rebuild).
# The comment names the older test of the same condition, if any.

stepfam(name, f!; setups=(:A, :B), jac=false) =
    (; name, setups, kind=:line, jac, prep=(c, r) -> stepsite(c, r, (r_, wa, ws, τm, τ, jac_) -> f!(c, r_, wa, ws, τm, τ, jac_)),
     sweep=(c, rng) -> opsweep!(c, rng, name))
rnd(fam) = (c, r, wa, ws, τm, τ, jac) -> MV.prep_random!(r, ws, c.P, c.LM, c.LC, fam, τ)
lrn(c) = (AM=c.AM, FF=c.FF)
"Sweep through a function returning (state, stats with the accepted moves in column 2); returns the accepted count."
function swept!(c::MCtx, f)
    s, st = f()
    c.s = MV.evaluate!(c.P, s)
    return st isa Tuple ? st[2] : sum(st[:, 2])
end
msweep(; p_shape=0.0, learn=false) = (c, rng) -> swept!(c, () -> MV.storage_ms_sweep!(rng, c.P, c.LM, c.s; lengths=[2, 8], nmoves=1,
                                       p_shape=p_shape, LC=c.LC, learn=learn ? lrn(c) : nothing))
slsweep(; level=false, learn=false) = (c, rng) -> swept!(c, () -> MV.storage_long_sweep!(rng, c.P, c.LM, c.s; nmoves=level ? 0 : 2,
                                       p_level=level ? 1.0 : 0.0, hmin=0.0, LC=c.LC, learn=learn ? lrn(c) : nothing))
sxsweep(; learn=false) = (c, rng) -> swept!(c, () -> MV.storage_long_sweep!(rng, c.P, sxLM(c), c.s; nmoves=0, p_level=0.0,
                                       nxmoves=2, LC=c.LC, learn=learn ? lrn(c) : nothing))
function sxpsweep(c::MCtx, rng)
    c.s, st = MV.local_sweep!(rng, c.P, sxLM(c), c.s; families=[(:sxpair, 1.0)], atoms=false, extra=2.0 / c.P.E.sys.N, LC=c.LC)
    return st[1, 2]
end
ksweep(f) = (c, rng) -> ((s, n) = f(c, rng); c.s = s; n)

const FAMILIES = [
    # single coordinates and the plain local families (test_local.jl: incremental = global, recorded = plain, caches)
    stepfam(:atom, (c, r, wa, ws, τm, τ, jac) -> (f = rand(r, eachindex(c.P.E.vspecs));
                                                   MV.prep_atom!(ws, c.P, c.LM, f, rand(r, 1:c.P.lv[c.LM.ℓ+1].m[f]), τ); true);
            setups=(:B,)),                            # (the atom sweep at 40 nodes is ~10^5 moves)
    stepfam(:unit, rnd(:unit)),
    stepfam(:pair, rnd(:pair)),
    stepfam(:xpair, rnd(:xpair)),
    stepfam(:xloc, rnd(:xloc)),
    stepfam(:xlocc, (c, r, wa, ws, τm, τ, jac) -> MV.prep_xloc!(r, ws, c.P, c.LM, c.LC, τ, jac); jac=true),
    stepfam(:potential, rnd(:potential)),
    stepfam(:cell, rnd(:cell)),
    stepfam(:cellpair, rnd(:cellpair)),
    # storage (test_local.jl: walk linearity for spair and tsoc, the curve of tsoc and the plateaus)
    stepfam(:spair, (c, r, wa, ws, τm, τ, jac) -> MV.prep_spair!(r, wa, ws, c.P, c.LM, c.LC, τm, τ)),
    stepfam(:tsoc, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tsoc!(r, wa, ws, c.P, c.LM, c.LC, τm, τ, jac); jac=true),
    stepfam(:tsocL, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tsoc!(r, wa, ws, c.P, c.LM, c.LC, τm, τ, jac; learn=lrn(c));
            setups=(:B,)),
    (; name=:splateau, setups=(:A, :B), kind=:line, jac=true, prep=(c, r) -> msite(c, r), sweep=msweep()),
    (; name=:sshape, setups=(:B,), kind=:line, jac=true, prep=(c, r) -> msite(c, r; shape=:proxy), sweep=msweep(p_shape=1.0)),
    (; name=:splateauL, setups=(:B,), kind=:line, jac=true, prep=(c, r) -> msite(c, r; learn=true), sweep=msweep(learn=true)),
    # long-range storage (test_local.jl: incremental = global, caches, the curve, walk linearity)
    (; name=:slong, setups=(:A, :B), kind=:line, jac=true, prep=(c, r) -> slsite(c, r), sweep=slsweep()),
    (; name=:slongL, setups=(:B,), kind=:line, jac=true, prep=(c, r) -> slsite(c, r; learn=true), sweep=slsweep(learn=true)),
    (; name=:slevel, setups=(:A, :B), kind=:line, jac=true, prep=(c, r) -> slsite(c, r; level=true), sweep=slsweep(level=true)),
    # storage pairs across nodes (test_local.jl: footprints, lazy interior = full evaluation, disjoint rounds)
    (; name=:sxpair, setups=(:A, :B), kind=:line, jac=true, sweep=sxpsweep,
     prep=(c, r) -> stepsite(c, r, (r_, wa, ws, τm, τ, jac) -> MV.prep_sxpair!(r_, wa, ws, c.P, sxLM(c), c.LC, τm, τ, jac))),
    (; name=:sxlong, setups=(:A, :B), kind=:line, jac=true, prep=(c, r) -> sxsite(c, r), sweep=sxsweep()),
    (; name=:sxlongL, setups=(:B,), kind=:line, jac=true, prep=(c, r) -> sxsite(c, r; learn=true), sweep=sxsweep(learn=true)),
    # transport (test_flow.jl: incremental = global for tmode/tline/tmodeL, shares and excess across the kink of tline)
    stepfam(:tpot, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tpot!(r, ws, c.P, c.LM, c.LC, τ, jac); jac=true),
    stepfam(:tpotL, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tpot!(r, ws, c.P, c.LM, c.LC, τ, jac; learn=lrn(c)); setups=(:B,)),
    stepfam(:tmode, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tmode!(r, ws, c.P, c.LM, c.LC, c.FF, τ, jac); setups=(:B,), jac=true),
    stepfam(:tmode0, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tmode!(r, ws, c.P, c.LM, c.LC, c.FF, τ, jac; comp=false); setups=(:B,)),
    stepfam(:tmodeL, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tmode!(r, ws, c.P, c.LM, c.LC, c.FF, τ, jac; learn=lrn(c)); setups=(:B,)),
    stepfam(:tline, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tline!(r, ws, c.P, c.LM, c.LC, c.FF, τ, jac); setups=(:B,), jac=true),
    stepfam(:tline0, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tline!(r, ws, c.P, c.LM, c.LC, c.FF, τ, jac; comp=false); setups=(:B,)),
    stepfam(:tlineL, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tline!(r, ws, c.P, c.LM, c.LC, c.FF, τ, jac; learn=lrn(c)); setups=(:B,)),
    stepfam(:tcut, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tcut!(r, ws, c.P, c.LM, c.LC, c.FF.H, c.FF.pieces, τ, jac); setups=(:B,), jac=true),
    stepfam(:tcutL, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tcut!(r, ws, c.P, c.LM, c.LC, c.FF.H, c.FF.pieces, τ, jac; learn=lrn(c));
            setups=(:B,)),
    # adaptive and learned (test_flow.jl: caches after adaptive sweeps)
    stepfam(:amode, (c, r, wa, ws, τm, τ, jac) -> MV.prep_amode!(r, ws, c.P, c.LM, c.LC, c.FF, c.AM, τ, jac); setups=(:B,), jac=true),
    stepfam(:amode0, (c, r, wa, ws, τm, τ, jac) -> MV.prep_amode!(r, ws, c.P, c.LM, c.LC, c.FF, c.AM, τ, jac; comp=false); setups=(:B,)),
    stepfam(:amodeL, (c, r, wa, ws, τm, τ, jac) -> MV.prep_amode!(r, ws, c.P, c.LM, c.LC, c.FF, c.AM, τ, jac; learn=lrn(c));
            setups=(:B,)),
    stepfam(:acut, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tcut!(r, ws, c.P, c.LM, c.LC, c.AM.H, c.AM.pieces, τ, jac); setups=(:B,), jac=true),
    stepfam(:acutL, (c, r, wa, ws, τm, τ, jac) -> MV.prep_tcut!(r, ws, c.P, c.LM, c.LC, c.AM.H, c.AM.pieces, τ, jac; learn=lrn(c));
            setups=(:B,)),
    (; name=:ashape, setups=(:B,), kind=:line, jac=true, prep=(c, r) -> msite(c, r; shape=:learned),
     sweep=(c, rng) -> swept!(c, () -> MV.adaptive_block_sweep!(rng, c.P, c.LM, c.AM, c.s; nmoves=1, LC=c.LC))),
    (; name=:ashapeL, setups=(:B,), kind=:line, jac=true, prep=(c, r) -> msite(c, r; shape=:learned, learn=true),
     sweep=(c, rng) -> swept!(c, () -> MV.adaptive_block_sweep!(rng, c.P, c.LM, c.AM, c.s; nmoves=1, LC=c.LC, learn=lrn(c)))),
    (; name=:adshape, setups=(:B,), kind=:line, jac=true, prep=(c, r) -> adsite(c, r), sweep=nothing),   # (swept with ashape)
    # capacity (test_kbox.jl: density against the global target, reverse move, caches; test_kop.jl: the same for kop)
    (; name=:kscale, setups=(:A, :B), kind=:kmove, move=:kscale, mh=false, learned=false,
     sweep=ksweep((c, rng) -> MV.kscale_sweep!(rng, c.P, c.LM, c.s; LC=c.LC, BC=c.BC))),
    (; name=:kscaleMH, setups=(:A, :B), kind=:kmove, move=:kscale, mh=true, learned=false,
     sweep=ksweep((c, rng) -> MV.kscale_sweep!(rng, c.P, c.LM, c.s; LC=c.LC, BC=c.BC, mh=true))),
    (; name=:kpeak, setups=(:A, :B), kind=:kmove, move=:kpeak, mh=false, learned=false,
     sweep=ksweep((c, rng) -> MV.kpeak_sweep!(rng, c.P, c.LM, c.s; LC=c.LC, BC=c.BC))),
    (; name=:kpeakL, setups=(:B,), kind=:kmove, move=:kpeak, mh=false, learned=true,
     sweep=ksweep((c, rng) -> MV.kpeak_sweep!(rng, c.P, c.LM, c.s; LC=c.LC, BC=c.BC, AM=c.AM))),
    (; name=:kpeakMH, setups=(:A, :B), kind=:kmove, move=:kpeak, mh=true, learned=false,
     sweep=ksweep((c, rng) -> MV.kpeak_sweep!(rng, c.P, c.LM, c.s; LC=c.LC, BC=c.BC, mh=true,
                                              σk=fill(100.0, length(kpeak_units(c)))))),
    (; name=:kop, setups=(:A, :B), kind=:kop, carrier=false,
     sweep=ksweep((c, rng) -> MV.kop_sweep!(rng, c.P, c.LM, c.s; LC=c.LC, BC=c.BC, mode=:unit))),
    (; name=:kopC, setups=(:A, :B), kind=:kop, carrier=true, nsites=8,         # (a few carriers; 40 units per move)
     sweep=ksweep((c, rng) -> MV.kop_sweep!(rng, c.P, c.LM, c.s; LC=c.LC, BC=c.BC, mode=:carrier))),
    # fine weather (test_model.jl: incremental = global, caches; test_wblock.jl: the z0 blocks, reverse move)
    (; name=:wz0, setups=(:A, :B), kind=:line, jac=true, known_lin=true, prep=(c, r) -> wsite(c, r, 0),
     sweep=(c, rng) -> swept!(c, () -> MV.fine_weather_sweep!(rng, c.P, c.LM, c.s; frac=0.0, frac0=0.002, LC=c.LC, BC=c.BC))),
    (; name=:wfine, setups=(:A, :B), kind=:line, jac=true, known_lin=true, prep=(c, r) -> wsite(c, r, 1),
     sweep=(c, rng) -> swept!(c, () -> MV.fine_weather_sweep!(rng, c.P, c.LM, c.s; frac=0.002, frac0=0.0, LC=c.LC, BC=c.BC))),
    (; name=:wblock, setups=(:A, :B), kind=:line, jac=true, known_lin=true, prep=(c, r) -> zsite(c, r),
     sweep=(c, rng) -> swept!(c, () -> MV.z0_block_sweep!(rng, c.P, c.LM, c.s, 2; LC=c.LC, BC=c.BC, lengths=(4,)))),
]

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

"Run the table on a context; returns per family (name => (worst, sweep errors))."
function run_families(c::MCtx; families=FAMILIES, nsites=NSITES, verbose=false)
    out = Dict{Symbol,Any}()
    for fam in families
        c.name in fam.setups || continue
        t0 = time()
        w = fam.kind == :line ? Worst() : WorstK()
        for j in 1:get(fam, :nsites, nsites)
            seed = 1000 * j + 7
            fam.kind == :line ? check_line_site!(w, c, fam, seed) :
            fam.kind == :kmove ? check_kmove_site!(w, c, fam, seed) : check_kop_site!(w, c, fam, seed)
        end
        se = nothing
        if fam.sweep !== nothing
            nacc = fam.sweep(c, MersenneTwister(77))
            se = (nacc, cache_errors(c)...)
            rebuild!(c)
        end
        out[fam.name] = (w, se, time() - t0)
        verbose && println(rpad("$(c.name) $(fam.name)", 16), round(time() - t0; digits=2), " s  ", w, "  ", se)
    end
    return out
end

"The tests on the results of `run_families`."
function test_families(c::MCtx, res)
    for (name, (w, se, _)) in sort!(collect(res); by=first)
        fam = FAMILIES[findfirst(f -> f.name == name, FAMILIES)]
        @testset "$(c.name) $name" begin
            @test w.n >= 5                                     # enough sites gave a move
            if w isa Worst
                if get(fam, :known_lin, false) && w.lin >= 1e-6
                    # known: the event walk of requirement-tracking moves places some kinks ~1e-7
                    # (relative) off, up to ~1e-5 nats between points (docs/system/testing.md §5)
                    @test_broken w.lin < 1e-6
                else
                    @test w.lin < 1e-6                         # nats, between walk points
                end
                @test w.rec == 0.0                             # recorded = plain evaluation
                @test w.glob < 1.0                             # global change within 1e-12 |L| + 1e-9 (1 + |βΔC|)
                @test w.sig == 0                               # the same choices at the moved state
                @test w.exc < 1e-6                             # excess kept, requirement shifted by g t
                @test w.ret < 1e-12                            # back at the start after −t
                @test w.dens < 1e-6                            # reverse density = shifted forward density
                @test w.jac < 1e-9                             # inverse Jacobians
                @test w.jacsweep == 0                          # Jacobian terms only where the sweep applies them
            else
                @test w.glob < 1.0
                @test w.sig == 0
                @test w.ret < 1e-12
                @test w.dens < 1e-6
                @test w.anti < 1e-6
                @test w.jac < 1e-9
                @test w.width == 0.0                           # slice width identical at both ends
                @test w.lo < 1e-12                             # its lower bound shifted by the move
                get(fam, :learned, false) && @test w.nl > 0     # the learned variant came up
            end
            if se !== nothing
                nacc, eL, eC, eB = se
                @test nacc > 0
                @test eL < 1e-6 && eC < 1e-10 && eB < 1e-9
            end
        end
    end
end

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    @testset "move families: deterministic exactness checks" begin
        tA = @elapsed ctxA = setup_moves_A()
        tB = @elapsed ctxB = setup_moves_B(ctxA.P.bias.θ)
        trun = @elapsed begin
            resA = run_families(ctxA)
            resB = run_families(ctxB)
        end
        test_families(ctxA, resA)
        test_families(ctxB, resB)
        println("  test_moves: setup A $(round(tA; digits=1)) s, B $(round(tB; digits=1)) s; checks $(round(trun; digits=1)) s")
    end
end
