# The building blocks of the replica–Langevin sampler (src/replicas.jl; docs/system/sampler.md
# §11, docs/system/testing.md §6). Deterministic checks only:
#   - the prior invariance of the weather process: the innovation-form generator against
#     `sample_weather` (bitwise), the round trip z0 → e → z0, the constant Jacobian, the pCN
#     balance identity of the implemented step, the regime window conditional against
#     enumeration, the weather step independent of k and v;
#   - the capacity score against finite differences of the global log target;
#   - the Langevin update and its reflections in w with given noise;
#   - the dispatch-following maps (fixed output, fixed utilizations, k′ exact, caches = rebuild);
#   - the gate, the schedule, the weights (λ = 0 bitwise untilted), the step adaptation;
#   - snapshots and resume of `run_replicas`.
using Test, Random, LinearAlgebra, Statistics, HDF5
using ESMSampling
const RP = ESMSampling
@isdefined(MockRNG) || include(joinpath(@__DIR__, "..", "device", "test", "mockrng.jl"))

@testset "replicas: schedule, gate, weights, adaptation" begin
    ctl = RP.ReplicaCtl(; M=4, burst_every=8, φ=0.3, gate_min=2, gate_max=5, gate_win=4, gate_tol=0.1, ηw=fill(1e-3, 2),
                        κs=fill(100.0, 2), probe_avg=2)
    # bursts staggered by burst_every ÷ M, every burst_every steps
    @test [findall(it -> RP.weather_schedule(ctl, it, j)[2], 1:16) for j in 1:4] == [[8, 16], [6, 14], [4, 12], [2, 10]]
    @test RP.weather_schedule(ctl, 8, 1) == (0.3, true) && RP.weather_schedule(ctl, 7, 1) == (0.0, false)
    ctl.θ = 0.1                                                 # continuous angle composed with the burst
    @test RP.weather_schedule(ctl, 8, 1)[1] ≈ acos(cos(0.1) * cos(0.3)) && RP.weather_schedule(ctl, 7, 1)[1] == 0.1
    ctl.θ = 0.0
    # categorical draw by inversion
    p = [0.2, 0.0, 0.5, 0.3]
    @test [RP.categorical(p, u) for u in (0.0, 0.1, 0.2, 0.2000001, 0.7, 0.70001, 1.0)] == [1, 1, 1, 3, 3, 4, 4]
    # relaxation indicator and the gate
    @test RP.relaxation_indicator([[1.0, 2.0]], 4) == Inf
    @test RP.relaxation_indicator([[1.0, 2.0], [1.0, 2.0], [1.5, 2.0], [1.5, 2.2]], 4) ≈ 0.5
    R = RP.ReplicaSet(RP.State[], Any[nothing], [0], [false], [false], [Vector{Float64}[]], [Vector{Vector{Float64}}[]],
                      copy(ctl.ηw), Vector{Float64}[])
    ctl.gate_max = 10
    acts = Bool[]
    for g in ([5.0, 1.0], [2.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
        push!(acts, RP.update_gate!(R, 1, g, ctl)); R.since[1] += 1
    end
    @test acts == [false, false, false, false, false, true, true]   # the indicator (halves of 4 steps) below 0.1
    ctl.gate_max = 5
    R.since[1] = 0; R.active[1] = false; empty!(R.gbuf[1])
    acts = [(a = RP.update_gate!(R, 1, [Float64(it), 0.0], ctl); R.since[1] += 1; a) for it in 1:7]
    @test acts == [false, false, false, false, false, true, true]   # never relaxed: active from gate_max = 5 on
    # weights: λ = 0 is the untilted drift, bitwise; inactive replicas get 0; normalised
    Φs = [0.7, -1.3, 0.2, 2.1]
    act = [true, false, true, true]
    @test RP.tilt_weights(0.0, Φs, act) == RP.uniform_weights(act) == [1 / 3, 0.0, 1 / 3, 1 / 3]
    w = RP.tilt_weights(2.0, Φs, act)
    @test sum(w) ≈ 1 && w[2] == 0 && w[4] / w[1] ≈ exp(2 * (2.1 - 0.7))
    @test RP.uniform_weights(falses(4)) == fill(0.25, 4)       # no active replica: all of them
    # step adaptation from the mean of probe_avg probes
    R.ηw .= [1e-3, 1e-3]
    @test !RP.adapt_step!(R, [0.5, 0.01], ctl)
    @test RP.adapt_step!(R, [-0.1, 0.03], ctl)               # means 0.2 (shrink), 0.02 (grow)
    @test R.ηw ≈ [0.7e-3, 1.2e-3] && isempty(R.gaps)
end

@testset "replicas: the Langevin update in w" begin
    k = [0.0, 50.0, 1000.0, 5000.0]
    g = [1e-3, -2e-3, 5e-4, 0.0]
    κs, η = 100.0, 1e-3
    ξ = [0.1, -0.2, 0.3, -0.4]
    k1, nc = RP.langevin_update(k, g, ξ; η=η, κs=κs, κmin=1.0)
    w = log.(k .+ κs)
    w1 = w .+ η .* ((k .+ κs) .* g .+ 1) .+ sqrt(2η) .* ξ
    wmin = log(1.0 + κs)
    w1 = [x < wmin ? 2wmin - x : x for x in w1]
    @test k1 ≈ exp.(w1) .- κs rtol = 1e-14
    @test nc == 0 && all(k1 .>= 1.0 - 1e-12)
    @test RP.langevin_drift_w(k, g, η, κs) ≈ η .* ((k .+ κs) .* g .+ 1)
    # reflection at κmin: a step below w_min lands as far above it
    kr, _ = RP.langevin_update([2.0], [0.0], [-50.0]; η=η, κs=κs, κmin=1.0)
    x = log(2.0 + κs) + η - sqrt(2η) * 50
    @test log(kr[1] + κs) ≈ 2wmin - x
    # clip: |drift| > clip counted and clamped
    kc, nc = RP.langevin_update([1000.0, 1000.0], [1.0, -1.0], [0.0, 0.0]; η=η, κs=κs, clip=0.02)
    @test nc == 2 && log.(kc .+ κs) ≈ log(1000.0 + κs) .+ [0.02, -0.02]
    # reflection at the box when crossing it from below, none from above
    hi = [1100.0, 1100.0]
    kb, _ = RP.langevin_update([1090.0, 1200.0], [0.0, 0.0], [5.0, 5.0]; η=η, κs=κs, hi=hi)
    wmax = log(1100.0 + κs)
    x1 = log(1090.0 + κs) + η + sqrt(2η) * 5
    @test log(kb[1] + κs) ≈ 2wmax - x1 && kb[1] < 1100
    @test log(kb[2] + κs) ≈ log(1200.0 + κs) + η + sqrt(2η) * 5
    # the rng form draws one normal per node, field by field (MockRNG = the explicit form)
end

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    cfgR = load_config(joinpath(@__DIR__, "..", "configs", "haar_poc.toml"))
    cfgR.data = joinpath(@__DIR__, "..", "data", "processed", "model.h5")
    cfgR.levels = [(10, 24), (40, 24)]
    cfgR.use_bias, cfgR.init, cfgR.measure, cfgR.verbose = false, "heuristic", "physical", false
    cfgR.bias_samples, cfgR.bias_sample_steps = 2, 64
    PR, sR, _ = RP.setup(cfgR; rng=MersenneTwister(cfgR.seed))
    sR = RP.evaluate!(PR, sR)
    WR = PR.W
    FR = RP.VarFactors(WR)
    LMR = RP.LocalModel(PR, 1)

    @testset "replicas: the weather process leaves the prior invariant" begin
        T0, n = WR.lays[1].T, RP.n0(WR)
        # the innovation-form generator is sample_weather (bitwise), from the same stream
        rA, rB = MersenneTwister(5), MersenneTwister(5)
        m, z0, ws = RP.sample_weather(rA, WR, 1)
        mB = zeros(Int, T0); mB[1] = RP.sample_log(rB, log.(WR.π0))
        for τ in 2:T0; mB[τ] = RP.sample_log(rB, log.(WR.Ptr[mB[τ-1], :])); end
        Eb = zeros(n, T0); for τ in 1:T0; Eb[:, τ] .= randn(rB, n); end
        @test mB == m
        @test RP.z0_from_innovations(WR, FR, mB, Eb) == z0
        # round trip z0 → e → z0
        e = RP.z0_innovations(WR, FR, m, z0)
        @test maximum(abs, RP.z0_from_innovations(WR, FR, m, e) .- z0) < 1e-12 * (1 + maximum(abs, z0))
        @test maximum(abs, e .- Eb) < 1e-10
        # constant Jacobian: log p(z0 | m) + |e|²/2 does not depend on e
        vals = [begin eq = randn(MersenneTwister(100 + q), n, T0) .* (1 + q)
                    RP.logp_z0(WR, m, RP.z0_from_innovations(WR, FR, m, eq))[1] + 0.5 * sum(abs2, eq) end for q in 1:5]
        @test maximum(vals) - minimum(vals) < 1e-9 * maximum(abs, vals) + 1e-6
        # pCN: the implemented step with given noise, its reverse noise, the balance identity
        θ = 0.3
        ξe, ξw = RP.draw_pcn_noise(MersenneTwister(9), n, T0, ws)
        z1, ws1 = RP.pcn_weather(WR, FR, m, z0, ws, θ, ξe, ξw)
        e1 = RP.z0_innovations(WR, FR, m, z1)
        @test maximum(abs, e1 .- (cos(θ) .* e .+ sin(θ) .* ξe)) < 1e-9
        ξe′ = RP.pcn_reverse_noise(e, e1, θ)
        ξw′ = [RP.pcn_reverse_noise(a, b, θ) for (a, b) in zip(ws, ws1)]
        lhs = sum(abs2, e) + sum(abs2, ξe) + sum(sum(abs2, w) for w in ws) + sum(sum(abs2, x) for x in ξw)
        rhs = sum(abs2, e1) + sum(abs2, ξe′) + sum(sum(abs2, w) for w in ws1) + sum(sum(abs2, x) for x in ξw′)
        @test abs(lhs - rhs) < 1e-10 * lhs                    # N(u) N(ξ) = N(u′) N(ξ′): q(u→u′)N(u) symmetric
        z2, ws2 = RP.pcn_weather(WR, FR, m, z1, ws1, θ, ξe′, ξw′)
        @test maximum(abs, z2 .- z0) < 1e-9 && maximum(abs, ws2[1] .- ws[1]) < 1e-12
        # the rng form consumes normals as draw_pcn_noise (MockRNG with the same numbers)
        zs = vcat(vec(ξe), ξw[1])
        zm, wm = RP.pcn_weather(MockRNG(zs, Float64[]), WR, FR, m, z0, ws, θ)
        @test zm == z1 && wm == ws1
        # regime window: the draw's conditionals multiply to the enumerated prior conditional
        K = WR.K
        for (a, b) in ((100, 102), (1, 3), (T0 - 2, T0))
            β = RP.regime_window_messages(WR, m, a, b)
            tot, dev = 0.0, 0.0
            ex = Dict{Vector{Int},Float64}()
            for p in Iterators.product(fill(1:K, b - a + 1)...)
                path = collect(p)
                w = a == 1 ? WR.π0[path[1]] : WR.Ptr[m[a-1], path[1]]
                for i in 2:length(path); w *= WR.Ptr[path[i-1], path[i]]; end
                b < T0 && (w *= WR.Ptr[path[end], m[b+1]])
                ex[path] = w
            end
            Z = sum(values(ex))
            for (path, w) in ex
                pa, prev = 1.0, a == 1 ? 0 : m[a-1]
                for (i, τ) in enumerate(a:b)
                    q = RP.regime_window_probs(WR, β, a, τ, prev)
                    pa *= q[path[i]] / sum(q); prev = path[i]
                end
                tot += pa; dev = max(dev, abs(pa - w / Z))
            end
            @test abs(tot - 1) < 1e-12 && dev < 1e-12
            # the draw with given uniforms is the categorical inversion of these conditionals
            us = [0.13, 0.77, 0.42]
            m1 = RP.regime_window(WR, m, a, b, us)
            prev = a == 1 ? 0 : m[a-1]
            for (i, τ) in enumerate(a:b)
                @test m1[τ] == RP.categorical(RP.regime_window_probs(WR, β, a, τ, prev), us[i]); prev = m1[τ]
            end
            @test m1[1:a-1] == m[1:a-1] && m1[b+1:end] == m[b+1:end]
        end
        # the weather step reads neither k nor v; FFBS keeps x
        s1 = RP.make_state(PR, 1, m, z0, ws, sR.k, sR.v)
        s2 = RP.make_state(PR, 1, m, z0, ws, 0.5 .* sR.k, 0.9 .* sR.v)
        a1 = RP.weather_process_step(MersenneTwister(4), PR, FR, s1, 0.3; window=60, ffbs=true)
        a2 = RP.weather_process_step(MersenneTwister(4), PR, FR, s2, 0.3; window=60, ffbs=true)
        @test a1.m == a2.m && a1.z0 == a2.z0 && a1.ws == a2.ws && a1.xs == a2.xs
        @test a1.k == s1.k && a1.v == s1.v
        b1 = RP.weather_process_step(MersenneTwister(4), PR, FR, s1, 0.0; ffbs=true)
        @test b1.xs[end] == s1.xs[end] && b1.z0 == s1.z0
        @test RP.make_state(PR, 1, b1.m, b1.z0, b1.ws, b1.k, b1.v).xs[end] ≈ s1.xs[end] atol = 1e-9   # ws re-derived: same x
    end

    @testset "replicas: capacity score against finite differences of the log target" begin
        s = copy(sR)
        sk, fk = RP.capacity_scores(PR, s)
        lk = PR.lk[2]
        E = PR.E
        Xn = RP.weather_nodes(WR, 1, s.xs[end])
        Lop(k) = (p = RP.energy_cost(E, lk, PR.lv[2], k, s.v, Xn); -PR.βv * p.op + RP.log_measure(E, lk, PR.lv[2], k))
        Lcap(k) = -PR.βk * RP.energy_cost(E, lk, PR.lv[2], k, s.v, Xn).capex
        Kr = RP.knodes(PR, 1, s.k)
        ntest = 0
        for f in eachindex(E.kspecs), i in shuffle(MersenneTwister(f), findall(x -> 10 < x < 0.9 * minimum(E.kspecs[f].hi), Kr[f]))[1:min(8, end)]
            Ut = E.kspecs[f].B.Ut
            dk(h) = (d = zeros(length(s.k)); for q in RP.nzrange(Ut, i); d[lk.off[f]+RP.rowvals(Ut)[q]] = h * RP.nonzeros(Ut)[q]; end; d)
            h = 1e-3
            L0 = Lop(s.k)
            fwd, bwd = (Lop(s.k .+ dk(h)) - L0) / h, (L0 - Lop(s.k .- dk(h))) / h
            abs(fwd - bwd) > 1e-6 * (1 + abs(fwd)) && continue        # a kink of the piecewise-linear cost within ±h
            @test sk[f][i] ≈ (fwd + bwd) / 2 rtol = 1e-5 atol = 1e-5
            @test fk[f][i] ≈ -(Lcap(s.k .+ dk(h)) - Lcap(s.k .- dk(h))) / 2h rtol = 1e-6
            ntest += 1
        end
        @test ntest >= 10
        # Lop + Lcap is the full log target at α = 0, so −fk + ŝ_k is its gradient at fixed (v, x)
        @test RP.logtarget(PR, 1, s.k, s.v, s.xs[end], s.z0)[1] ≈ Lop(s.k) + Lcap(s.k)
    end

    @testset "replicas: dispatch-following maps" begin
        s = copy(sR)
        units = RP.capacity_units(PR, LMR)
        E = PR.E
        @test Set(u[1] for u in units) == Set([:disp, :ren, :stor])
        Kraw = RP.knodes(PR, 1, s.k)
        r = MersenneTwister(3)
        knew = [k .* exp.(0.1 .* randn(r, length(k))) for k in Kraw]
        LC0 = RP.LocalCost(PR, LMR, s)
        # dispatchable generators alone: their output c(k)·ω is kept at every step
        disp = Set(kf for (kind, gi, i, kf) in units if kind == :disp)
        kd = [f in disp ? knew[f] : copy(Kraw[f]) for f in eachindex(Kraw)]
        sd, _ = RP.follow_capacities!(PR, LMR, copy(s), units, Kraw, kd)
        LCd = RP.LocalCost(PR, LMR, sd)
        for (kind, gi, i, kf) in units
            kind == :disp || continue
            g = E.gens[gi]
            @test LCd.kgen[gi][i] .* LCd.Y[g.vf][i, :] ≈ LC0.kgen[gi][i] .* LC0.Y[g.vf][i, :] rtol = 1e-9 atol = 1e-6
        end
        @test maximum(abs, LCd.r .- LC0.r) < 1e-6
        # all units: k′ exact, caches equal a rebuild, storage keeps its SoC fractions
        s1, LC1 = RP.follow_capacities!(PR, LMR, copy(s), units, Kraw, knew)
        K1 = RP.knodes(PR, 1, s1.k)
        for (kind, gi, i, kf) in units
            @test K1[kf][i] ≈ knew[kf][i] rtol = 1e-12
        end
        LC2 = RP.LocalCost(PR, LMR, s1)
        mx(a, b) = maximum(abs.(a .- b); init=0.0)
        @test mx(LC1.r, LC2.r) < 1e-6 && mx(LC1.S, LC2.S) < 1e-6 && mx(LC1.F, LC2.F) < 1e-6
        @test all(mx(LC1.Y[f], LC2.Y[f]) < 1e-9 for f in eachindex(LC1.Y) if f != E.fθ)
        @test all(mx(LC1.kgen[g], LC2.kgen[g]) < 1e-9 for g in eachindex(LC1.kgen))
        @test all(mx(LC1.kst[g], LC2.kst[g]) < 1e-9 for g in eachindex(LC1.kst))
        @test abs(RP.cache_cost(PR, LMR, LC1) - s1.parts.op) < 1e-10 * abs(s1.parts.op)
        for (kind, gi, i, kf) in units
            kind == :stor && @test LC2.Y[E.stors[gi].fe][i, :] ≈ LC0.Y[E.stors[gi].fe][i, :] atol = 1e-12
        end
        @test s1.L == RP.evaluate!(PR, copy(s1)).L
        # the same map on another replica (other weather) gives bitwise the same capacities
        mo, zo, wo = RP.sample_weather(MersenneTwister(8), WR, 1)
        so = RP.make_state(PR, 1, mo, zo, wo, s.k, s.v)
        so1, _ = RP.follow_capacities!(PR, LMR, so, units, Kraw, knew)
        @test so1.k == s1.k
    end

    @testset "replicas: one step; λ = 0 is the untilted step bitwise" begin
        cfg2 = deepcopy(cfgR); cfg2.replicas = 3; cfg2.sampler = "replicas"
        ctl = RP.ReplicaCtl(cfg2, PR)
        ctl.burst_every, ctl.φ = 3, 0.3                             # replica 3 bursts at step 1 (its gate closes)
        Ks = RP.replica_kernels(PR, sR, cfg2, 3, LMR)
        units = RP.capacity_units(PR, LMR)
        R0 = RP.init_replicas(MersenneTwister(1), PR, Ks, LMR, sR, [MersenneTwister(20 + j) for j in 1:3], ctl, 1)
        Φ = RP.TiltFunctional(PR, RP.knodes(PR, 1, sR.k); rng=MersenneTwister(2), n=20)
        @test isfinite(Φ.μ) && Φ.σ > 0
        cp(R) = RP.ReplicaSet([copy(s) for s in R.reps], [copy(w) for w in R.wrngs], copy(R.since), copy(R.window), copy(R.active),
                              deepcopy(R.gbuf), deepcopy(R.hist), copy(R.ηw), deepcopy(R.gaps))
        runs = Dict()
        for (lab, Φx, λ) in ((:none, nothing, 0.0), (:zero, Φ, 0.0), (:two, Φ, 2.0))
            R = cp(R0); c = deepcopy(ctl); c.Φ, c.λ = Φx, λ
            st = RP.replica_step!(MersenneTwister(7), PR, Ks, LMR, units, FR, R, c, 1)
            runs[lab] = (R, st)
        end
        (Rn, sn), (Rz, sz), (R2, s2) = runs[:none], runs[:zero], runs[:two]
        @test sz.wts == sn.wts && Rz.reps[1].k == Rn.reps[1].k && all(a.v == b.v for (a, b) in zip(Rz.reps, Rn.reps))
        @test sn.moved == [false, false, true] && Rn.active == [true, true, false] && sn.wts == [0.5, 0.5, 0.0]
        @test s2.wts != sz.wts && R2.reps[1].k != Rz.reps[1].k   # the tilt moves the drift ...
        @test all(a.z0 == b.z0 && a.m == b.m && a.ws == b.ws for (a, b) in zip(R2.reps, Rz.reps))   # ... never the weather
        @test all(s -> s.k == Rn.reps[1].k, Rn.reps)
        # the rng form of the Langevin step: one normal per node, field by field (MockRNG = langevin_update per field)
        Kraw = sn.Kraw
        sk̄ = [sum(sn.wts[j] .* sn.sks[j][f] for j in 1:3) for f in eachindex(Kraw)]
        _, fk0 = RP.capacity_scores(PR, R0.reps[1])
        kfs = unique(u[4] for u in units)
        zs = randn(MersenneTwister(99), sum(length(Kraw[f]) for f in kfs))
        ka, _ = RP.langevin_capacities(MockRNG(zs, Float64[]), PR, units, Kraw, sk̄, fk0; ηw=ctl.ηw, κs=ctl.κs, clip=ctl.clip)
        o = 0
        for f in kfs
            n = length(Kraw[f])
            kb, _ = RP.langevin_update(Kraw[f], -fk0[f] .+ (PR.βk / PR.βv) .* sk̄[f], zs[o+1:o+n]; η=ctl.ηw[f], κs=ctl.κs[f],
                                       clip=ctl.clip, hi=PR.E.kspecs[f].hi)
            @test ka[f] == kb
            o += n
        end
    end

    @testset "replicas: snapshots, stop and resume" begin
        d = mktempdir()
        function rcfg()
            c = deepcopy(cfgR)
            c.sampler, c.out = "replicas", joinpath(d, "rep.h5")
            c.nchains, c.chain_starts, c.replicas, c.replica_init_sweeps = 1, ["low"], 2, 1
            c.mlda_warmup, c.nsamples = 3, 3
            c.replica_burst_every, c.replica_gate_min, c.replica_gate_max = 2, 1, 1
            c.replica_probe_every, c.replica_probe_len, c.replica_probe_tail, c.replica_probe_win, c.replica_probe_avg = 1, 2, 1, 1, 1
            c.tilt_samples = 5
            c.checkpoint, c.stop_file, c.checkpoint_minutes = joinpath(d, "snapshot.h5"), joinpath(d, "STOP"), 1e6
            return c
        end
        same(a, b) = a.ℓ == b.ℓ && a.m == b.m && a.z0 == b.z0 && a.ws == b.ws && a.k == b.k && a.v == b.v
        stop_at(n) = c -> ((p, it) -> (p == :replicas && it == n && touch(c.stop_file); nothing))
        c = rcfg()
        A = RP.run(c; hook=stop_at(2)(c))
        @test A.exitcode == RP.EXIT_CHECKPOINTED && A.status == :stopped && !isfile(c.stop_file)
        @test length(A.states) == 2 && A.resume_states === nothing
        ηA = copy(A.sets[1].ηw)
        @test h5open(f -> read(f["chains/1/eta_w"]), c.checkpoint) == ηA
        @test h5open(f -> (read(f["driver"]), read(f["it"]), read(f["restarts"])), c.checkpoint) == ("replicas", 2, 0)
        @test_throws ErrorException RP.read_snapshot(c.checkpoint, c, PR)             # not a run_chains snapshot
        c = rcfg()
        B = RP.run(c; hook=stop_at(4)(c))
        @test all(same.(B.resume_states, A.states))
        @test B.exitcode == 75 && h5open(f -> (read(f["it"]), read(f["restarts"])), c.checkpoint) == (4, 1)
        trB = h5open(f -> read(f["chains/1/trace/ktot"]), c.checkpoint)
        @test size(trB, 2) == 4
        c = rcfg()
        D = RP.run(c)
        @test all(same.(D.resume_states, B.states)) && D.exitcode == 0 && D.restarts == 2
        h5open(c.out, "r") do f
            @test size(read(f["chains/1/Q"]), 2) == 6
            @test read(f["replicas/1/ktot"])[:, 1:4] == trB
            @test size(read(f["replicas/1/features"]), 1) == 2 * length(read(f["replicas/feature_names"]))
            @test size(read(f["replicas/1/eta_w"]), 2) == 6 && read(f["replicas/1/probe/it"])[2, :] ⊆ [1.0, 2.0]
            @test haskey(f, "replicas/1/probe/gap") && read(f["run/restarts"]) == 2 && read(f["run/resumed_at_sweep"]) == 4
        end
    end
end
