# End-to-end exactness test (docs/todo.md item 1; ≈ 2–3 min): on a 3-node sub-network at daily resolution
# with fixed weather, the full adaptive kernel (all local, flow, learned and capacity moves,
# with adaptation) must reproduce the equilibrium of a simple reference chain (exact coordinate
# draws on the basis atoms for the operation, slice sampling of each capacity coefficient on
# the full target), for a set of observables, within Monte Carlo error.
using Test, Random, Statistics, LinearAlgebra
using ESMSampling
const MX = ESMSampling

function exact_setup()
    sys0 = MX.load_system(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    # three connected nodes of one country with generators and storage
    c = "DE"
    cand = findall(==(c), sys0.country)
    adj = Dict(n => Int[] for n in cand)
    for (a, b) in zip(sys0.line_bus0, sys0.line_bus1)
        a in cand && b in cand && (push!(adj[a], b); push!(adj[b], a))
    end
    n1 = first(n for n in cand if length(adj[n]) >= 2)
    nodes = sort(unique(vcat(n1, adj[n1][1:2])))
    sys = MX.subsystem(sys0, nodes)
    E = MX.build_energy_model(sys; basis=:flow, flow_theta=:haar,
                              flow_kw=(K_max=2, nyears=1))
    levels = [(1, 120), (3, 120)]                  # 73 steps: quick
    W = MX.fit_weather_model(MersenneTwister(1), E, levels; K=2)
    bias = MX.init_bias(E, W, MX.Layout(E.kspecs, levels[1]...), 1e-3; rng=MersenneTwister(2), nsamples=4)
    P = MX.build_problem(E, W, levels, 4e-8, 4e-5, bias; use_bias=false)
    m, z0, ws = MX.sample_weather(MersenneTwister(3), W, 1)
    lk, lv = P.lk[2], P.lv[2]
    Xn = MX.weather_nodes(W, 1, MX.weather_at_level(W, 1, m, z0, ws))
    Wd = [map(x -> MX.datamap(E.wspecs[f], x), Xn[f]) for f in eachindex(Xn)]
    h = MX.heuristic_start(E, lk, lv, Wd)
    k = MX.to_modes(lk, E.kspecs, [reshape(clamp.(x, 0.0, sp.hi), :, 1) for (x, sp) in zip(h.k, E.kspecs)])
    v = MX.to_modes(lv, E.vspecs, h.v)
    s = MX.make_state(P, 1, m, z0, ws, k, v)
    return P, s
end

"Reference transition: exact atom draws for v (no other families), slice on each k coefficient."
function reference_step!(rng, P, LM, s, wk)
    s, _ = MX.local_sweep!(rng, P, LM, s; extra=0.0)
    for i in shuffle(rng, collect(eachindex(s.k)))
        lf = () -> MX.logtarget(P, 1, s.k, s.v, s.xs[end], s.z0)[1]
        MX.slice_coord!(rng, s.k, i, lf, wk[i])
    end
    return MX.evaluate!(P, s)
end

function observables(P, s)
    E = P.E
    lv = P.lv[2]
    gasf = findfirst(sp -> sp.name == :k_gas, E.kspecs)
    kn = MX.knodes(P, 1, s.k)
    θ = Matrix(MX.nodevals(s.v, lv, E.vspecs, E.fθ))
    socf = [st.fe for st in E.stors]
    soc = isempty(socf) ? 0.0 : mean(mean(Matrix(MX.nodevals(s.v, lv, E.vspecs, f))) for f in socf)
    flow = isempty(E.sys.line_s) ? 0.0 : E.sys.line_b[1] * mean(θ[E.sys.line_bus0[1], :] .- θ[E.sys.line_bus1[1], :])
    return [s.parts.op, s.parts.shed, sum(kn[gasf]), soc, flow]
end

function compare(names, R, Tm; zmax=4.0)
    for j in axes(R, 1)
        r, t = R[j, :], Tm[j, :]
        se = sqrt(var(r) / max(MX.ess(r), 2) + var(t) / max(MX.ess(t), 2))
        z = (mean(t) - mean(r)) / max(se, 1e-12 * abs(mean(r)) + 1e-12)
        println("  ", rpad(names[j], 14), " ref ", round(mean(r); sigdigits=5), " test ", round(mean(t); sigdigits=5), "  z = ", round(z; digits=2))
        @test abs(z) < zmax
    end
end

function run_chain(step!, s, burn, n)
    for _ in 1:burn; s = step!(s); end
    out = zeros(5, n)
    for i in 1:n
        s = step!(s)
        out[:, i] .= observables(P_, s)
    end
    return out
end

const names_ = ["op", "shed", "gas capacity", "mean SoC", "line flow"]

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5")) && get(ENV, "EXACT_SKIP", "") != "1"
    global P_, s0 = exact_setup()
    LM = MX.LocalModel(P_, 1)
    rng = MersenneTwister(11)
    @testset "exactness: operation at fixed (k, x) on a 3-node network" begin
        # reference: atoms plus the plain linear families (no Jacobians, no learning; exact by
        # construction). Atoms alone do not mix in a usable time (balance-coupled directions).
        plain = [(:unit, 0.3), (:pair, 0.25), (:xpair, 0.15), (:potential, 0.1), (:cellpair, 0.1), (:cell, 0.1), (:spair, 0.3)]
        ref(x) = MX.local_sweep!(rng, P_, LM, x; extra=4.0, families=plain)[1]
        FF = MX.FlowFrames(P_, 1, s0)
        AM = MX.AdaptiveModes(P_, 1; K=2, lengths=[2, 4, 8])
        full(x) = MX.adaptive_sweep!(rng, P_, LM, FF, AM, x)[1]          # all families, learned, adaptive
        R = run_chain(ref, copy(s0), 200, 2000)
        Tm = run_chain(full, copy(s0), 200, 1000)
        compare(names_[[1, 2, 4, 5]], R[[1, 2, 4, 5], :], Tm[[1, 2, 4, 5], :])
    end
    @testset "exactness: capacity moves on a 3-node network" begin
        op(x) = MX.local_sweep!(rng, P_, LM, x; extra=4.0)[1]
        refk(x) = MX.kscale_sweep!(rng, P_, LM, op(x))[1]
        fullk(x) = MX.kpeak_sweep!(rng, P_, LM, MX.kscale_sweep!(rng, P_, LM, op(x))[1])[1]
        R = run_chain(refk, copy(s0), 200, 2000)
        Tm = run_chain(fullk, copy(s0), 200, 2000)
        compare(names_, R, Tm)
    end
end
