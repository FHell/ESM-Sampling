# The learned bias (docs/system/ensemble.md §4.4, sampler.md §6–§7): the monotone-price fit,
# the tables of F₁, F₂, the lift to the finest level, the incremental bias cache against the
# global evaluation (capacity moves of every unit kind, fine weather moves, commits, sweeps),
# the scores and design against finite differences, the post-correction records, and the
# invariance of the Metropolis bias correction of the fine weather moves (quadrature).
using Test, Random, LinearAlgebra, Statistics
using ESMSampling
const MB = ESMSampling

"Analytic ĝ (softplus basis, no tables) at the level of the bias."
function ghat_analytic(b, E, lk, zk, Xraw)
    W = [map(x -> MB.datamap(E.wspecs[f], x), Xraw[f]) for f in eachindex(Xraw)]
    T = size(W[1], 2)
    kc, _ = MB.node_capacities(E, lk, zk)
    R, firm = MB.residual_parts(E, kc, W, T)
    D = vec(sum(W[1]; dims=1))
    nb = MB.nθ(b)
    φ, d1, d2 = zeros(nb), zeros(nb), zeros(nb)
    val = 0.0
    for t in 1:T
        u1 = (D[t] - R[t]) / b.Dbar
        MB.bias_basis!(φ, d1, d2, b, u1, u1 - firm / b.Dbar)
        val += b.Δt * dot(b.θ, φ)
    end
    return val
end

"Set the raw node capacity of unit i of generator group (kind 1) or storage group (kind 2) gi to knew."
function set_node_capacity!(P, s, kind, gi, i, knew)
    E = P.E
    lk = P.lk[s.ℓ+1]
    kf = kind == 1 ? E.gens[gi].kf : E.stors[gi].kf
    kold = vec(MB.nodevals(s.k, lk, E.kspecs, kf))[i]
    Ut = E.kspecs[kf].B.Ut
    for q in MB.nzrange(Ut, i)
        s.k[lk.off[kf]+MB.rowvals(Ut)[q]] += (knew - kold) * MB.nonzeros(Ut)[q]
    end
    return s
end

"Weather change chain and z0 index of coordinate q of level ℓ (0: z0)."
function weather_chain(W, s, ℓ, q, L)
    iz = 0
    if ℓ == 0
        lay = W.lays[1]
        n0_ = MB.n0(W)
        o, τ0 = (q - 1) % n0_ + 1, (q - 1) ÷ n0_ + 1
        f, rows = 1, 0
        while rows + lay.m[f] < o
            rows += lay.m[f]; f += 1
        end
        Dm = [zeros(lay.m[ff], 1) for ff in eachindex(W.specs)]
        Dm[f][o-rows, 1] = 1.0
        d0 = MB.WDelta(0, τ0, τ0, Dm)
        iz = lay.off[f] + (τ0 - 1) * lay.m[f] + (o - rows)
    else
        f, part, i, bb, j = MB.innovation_coord(W, ℓ, q)
        d0 = MB.wdelta_innovation(W, s.m, ℓ, f, part, i, bb, j)
    end
    return MB.wdelta_chain(W, s.m, d0, L), iz
end

"Apply a weather coordinate change t to the state (as weather_coord_move! does)."
function apply_weather!(s, W, ch, ℓ, q, iz, t)
    if ℓ == 0
        s.z0[iz] += t
        s.xs[1] === s.z0 || (s.xs[1][iz] += t)
        MB.apply_wdelta!(s, W, ch[2:end], t)
    else
        s.ws[ℓ][q] += t
        MB.apply_wdelta!(s, W, ch, t)
    end
    return s
end

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    cfg = MB.load_config(joinpath(@__DIR__, "..", "configs", "haar_poc.toml")); cfg.verbose = false
    cfg.use_bias = true; cfg.bias_warmstart = 20.0; cfg.bias_monotone = true; cfg.measure = "physical"
    P, sL, info = MB.setup(cfg; rng=MersenneTwister(cfg.seed))

    @testset "monotone-price bias and its tables" begin
        b = P.bias
        nb = MB.nθ(b)
        φ, d1, d2 = zeros(nb), zeros(nb), zeros(nb)
        λ = Float64[]
        for u in range(-3.0, 3.0; length=601)
            MB.bias_basis!(φ, d1, d2, b, u, u - 0.3)
            push!(λ, -(dot(b.θ, d1) + dot(b.θ, d2)) / (P.βv * b.Dbar))
        end
        @test all(diff(λ) .>= -1e-6)
        @test minimum(λ) >= -P.E.c_dump - 1e-6
        @test maximum(λ) <= P.E.c_shed + 1e-6
        # firm capacity has no negative value: θ_{u₂} ≤ 0, so F₂ is non-increasing (its table too)
        n1_ = length(b.τ1)
        @test b.θ[2+n1_] <= 1e-9
        @test all(diff(b.tab2.F) .<= 1e-9 * maximum(abs, b.tab2.F))
        # the tables equal the analytic ramp sums at the grid points, are close between them,
        # continue linearly beyond the grid, and their secant slopes are non-increasing
        # (F₁, F₂ concave: the implied price of the tabulated ĝ is monotone)
        n1 = length(b.τ1)
        for (tb, θl, θr, τ) in ((b.tab1, b.θ[1], b.θ[2:1+n1], b.τ1), (b.tab2, b.θ[2+n1], b.θ[3+n1:end], b.τ2))
            grid = [tb.u0 + (j - 1) * tb.h for j in eachindex(tb.F)]
            @test maximum(abs(MB.tabval(tb, u) - MB.ramp_analytic(θl, θr, τ, b.s, u)) for u in grid) <=
                  1e-12 * maximum(abs, tb.F) + 1e-12
            mid = grid[1:end-1] .+ 0.37 * tb.h
            scale = sum(abs, θr) / (4 * b.s) * tb.h^2 / 8      # interpolation error bound of a ramp sum
            @test maximum(abs(MB.tabval(tb, u) - MB.ramp_analytic(θl, θr, τ, b.s, u)) for u in mid) <= scale + 1e-12
            sl = diff(tb.F) ./ tb.h
            @test all(diff(sl) .<= 1e-9 * maximum(abs, sl) + 1e-12)
            uL, uR = grid[1] - 2.0, grid[end] + 3.0
            @test MB.tabval(tb, uL) ≈ tb.F[1] + sl[1] * (uL - grid[1])
            @test MB.tabval(tb, uR) ≈ tb.F[end] + sl[end] * (uR - grid[end])
            @test abs(MB.tabval(tb, uR) - MB.ramp_analytic(θl, θr, τ, b.s, uR)) <= 1e-6 * (1 + abs(MB.tabval(tb, uR)))
        end
        # changing θ rebuilds the tables (lazily if θ is set by hand)
        θs = copy(b.θ)
        b.θ[1] += 1.0
        t1, _ = MB.tables(b)
        @test t1.F[1] ≈ MB.ramp_analytic(b.θ[1], b.θ[2:1+n1], b.τ1, b.s, t1.u0)
        b.θ .= θs
        MB.tables(b)
    end

    L = sL.ℓ
    E, W, b = P.E, P.W, P.bias
    lk0, lkL = P.lk[1], P.lk[L+1]

    @testset "lift of the bias to the finest level" begin
        rng = MersenneTwister(21)
        b.a .= 0.5 .* randn(rng, length(b.a)); b.c .= 0.1 .* abs.(randn(rng, length(b.c)))
        a0, c0, θ0, ks0, D0 = copy(b.a), copy(b.c), copy(b.θ), copy(b.kscale), b.Dbar
        s0 = MB.restrict_state(P, sL, 0)
        zc0 = MB.clamped_kcoeffs(E, lk0, s0.k)[1]
        h0 = MB.hhat(b, zc0)[1]
        MB.lift_bias!(P, L; rng=MersenneTwister(5), nsteps=2048)
        @test b.ℓb == L && b.Δt == Float64(P.levels[end][2])
        @test b.θ == θ0 && b.Dbar == D0
        for f in eachindex(E.kspecs)
            m0 = lk0.m[f]
            @test MB.fieldview(b.a, lkL, E.kspecs, f)[1:m0, 1] == MB.fieldview(a0, lk0, E.kspecs, f)[:, 1]
            @test MB.fieldview(b.c, lkL, E.kspecs, f)[1:m0, 1] == MB.fieldview(c0, lk0, E.kspecs, f)[:, 1]
            @test all(iszero, MB.fieldview(b.a, lkL, E.kspecs, f)[m0+1:end, 1])
            @test all(iszero, MB.fieldview(b.c, lkL, E.kspecs, f)[m0+1:end, 1])
            @test MB.fieldview(b.kscale, lkL, E.kspecs, f)[1:m0, 1] ≈ MB.fieldview(ks0, lk0, E.kspecs, f)[:, 1]
        end
        @test b.nobs == 0 && all(iszero, b.GtG) && all(iszero, b.Gts) && size(b.hk, 1) == lkL.n
        @test length(b.sD) == 2048 && size(b.sA[1], 1) == 2048
        @test b.wstep ≈ b.Δt * lkL.T / 2048
        # ĥ of capacities without details beyond level 0 is unchanged by the lift
        kL = MB.prolong_to(s0.k, P.lk, P.trk, E.kspecs, 0, L)
        @test MB.hhat(b, MB.clamped_kcoeffs(E, lkL, kL)[1])[1] ≈ h0 rtol = 1e-10
        # the evaluation is now at the finest level and needs a finest-level state
        @test_throws ErrorException MB.bias_term(P, s0)
        # a bias of realistic size for the remaining tests (monotone fit kept)
        b.θ .*= 3.0
        MB.rebuild_tables!(b)
    end

    s = MB.evaluate!(P, copy(sL))
    # ĥ with every branch near the state: concave coordinates (c < 0) with their vertex close to the
    # current clamped coefficient (capacity moves cross it into the flat part), convex ones elsewhere,
    # and data ranges around the current coefficient (moves leave them on both sides)
    let rng = MersenneTwister(22), κs = MB.clamped_kcoeffs(E, lkL, s.k)[1] ./ b.kscale
        for j in eachindex(b.a)
            if isodd(j)
                b.c[j] = -(0.5 + rand(rng)) * 2.0
                b.a[j] = -b.c[j] * (κs[j] + 0.05 * randn(rng) * (abs(κs[j]) + 0.1))
            else
                b.c[j] = 0.5 * rand(rng); b.a[j] = 2.0 * randn(rng)
            end
            w = 0.1 * (abs(κs[j]) + 0.1)
            b.κlo[j], b.κhi[j] = κs[j] - w * rand(rng), κs[j] + w * rand(rng)
        end
    end

    @testset "saturating ĥ and the exact measure term" begin
        # hsat: the quadratic below the vertex, flat (C¹) beyond it when c < 0
        for (a, c) in ((2.0, -1.0), (-1.5, -0.5), (1.0, 0.7), (0.3, 0.0))
            for κ in range(-6.0, 6.0; length=121)
                c < 0 && κ > -a / c ? (@test MB.hsat(a, c, κ) ≈ -a^2 / (2c)) : (@test MB.hsat(a, c, κ) ≈ a * κ + 0.5 * c * κ^2)
                h = 1e-6
                @test MB.hsat_d(a, c, κ) ≈ (MB.hsat(a, c, κ + h) - MB.hsat(a, c, κ - h)) / 2h atol = 1e-5
            end
            c < 0 && @test MB.hsat_d(a, c, -a / c) == 0.0
        end
        # hcoord: hsat inside the data range, outside continued with the edge slope clipped so that it
        # never pushes away from the data; continuous at the edges; unbounded range = hsat
        for (a, c, lo, hi) in ((2.0, -1.0, -1.0, 1.0), (2.0, -1.0, 0.5, 3.0), (-1.0, 0.5, -2.0, 1.0), (1.0, 0.0, -0.5, 0.5),
                               (-0.7, -0.3, 0.2, 0.2))
            for κ in range(-6.0, 6.0; length=121)
                v = MB.hcoord(a, c, lo, hi, κ)
                if lo <= κ <= hi
                    @test v ≈ MB.hsat(a, c, κ)
                elseif κ > hi
                    @test v ≈ MB.hsat(a, c, hi) + max(MB.hsat_d(a, c, hi), 0.0) * (κ - hi)
                    @test MB.hcoord_d(a, c, lo, hi, κ) >= 0
                else
                    @test v ≈ MB.hsat(a, c, lo) + min(MB.hsat_d(a, c, lo), 0.0) * (κ - lo)
                    @test MB.hcoord_d(a, c, lo, hi, κ) <= 0
                end
                @test MB.hcoord(a, c, -Inf, Inf, κ) == MB.hsat(a, c, κ)
                h = 1e-7
                abs(κ - lo) > 1e-3 && abs(κ - hi) > 1e-3 &&
                    @test MB.hcoord_d(a, c, lo, hi, κ) ≈ (MB.hcoord(a, c, lo, hi, κ + h) - MB.hcoord(a, c, lo, hi, κ - h)) / 2h atol = 1e-5
            end
            @test MB.hcoord(a, c, lo, hi, hi + 1e-12) ≈ MB.hcoord(a, c, lo, hi, hi) atol = 1e-9
            @test MB.hcoord(a, c, lo, hi, lo - 1e-12) ≈ MB.hcoord(a, c, lo, hi, lo) atol = 1e-9
        end
        # b̂ carries (1 − r) log J exactly: the global value equals its parts with log_measure
        lv = P.lv[L+1]
        Xn = MB.weather_nodes(W, L, s.xs[end])
        g, gm = MB.ghat(b, E, lkL, s.k, Xn), MB.ghat_mean(b, E, lkL, s.k)
        h = MB.hhat(b, MB.clamped_kcoeffs(E, lkL, s.k)[1])[1]
        @test MB.log_measure(E, lkL, lv, s.k) != 0
        @test MB.bias_term(P, s) ≈ g - b.ratio * gm + (1 - b.ratio) * (h + MB.log_measure(E, lkL, lv, s.k)) rtol = 1e-12
        BC = MB.BiasCache(P, s)
        @test BC.lj ≈ MB.log_measure(E, lkL, lv, s.k) rtol = 1e-14
        @test count(j -> b.c[j] < 0 && BC.zc[j] / b.kscale[j] > -b.a[j] / b.c[j], eachindex(b.a)) > 0
        # ĥ regresses the score without the exact measure term: y = s_k − ∂ₖlog J − ∂ₖĝ
        Xb, sx, sk, kb = MB.bias_scores(P, s)
        PB = MB.PostBias()
        PB.GtG, PB.Gts, PB.hk = zeros(MB.nθ(b), MB.nθ(b)), zeros(MB.nθ(b)), MB.hk_sums(lkL.n)
        MB.bias_observe!(b, E, W, lkL, kb, Xb, sx, sk; into=PB)
        _, gk = MB.bias_design(b, E, W, lkL, kb, Xb)
        @test PB.hk[:, 4] ≈ (sk .- MB.log_measure_grad(E, lkL, lv, kb) .- gk) .* b.kscale
        κobs = MB.clamped_kcoeffs(E, lkL, kb)[1] ./ b.kscale
        @test PB.hk[:, 6] == κobs && PB.hk[:, 7] == κobs     # the data range of ĥ: one observation
    end

    @testset "bias cache: capacity moves of every unit kind" begin
        rng = MersenneTwister(23)
        BC = MB.BiasCache(P, s)
        @test MB.bias_value(BC, b) == MB.bias_term(P, s)
        kinds = Dict(:renewable => [(1, gi, i) for (gi, g) in enumerate(E.gens) if g.kf > 0 && g.carrier != "gas" for i in eachindex(g.bus)],
                     :gas => [(1, gi, i) for (gi, g) in enumerate(E.gens) if g.kf > 0 && g.carrier == "gas" for i in eachindex(g.bus)],
                     :storage => [(2, si, i) for (si, st) in enumerate(E.stors) if st.kf > 0 for i in eachindex(st.bus)])
        sc = copy(s)
        for (kname, units) in kinds
            isempty(units) && continue
            err, errc, Δmax = 0.0, 0.0, 0.0
            for trial in 1:40
                kind, gi, i = rand(rng, units)
                kf = kind == 1 ? E.gens[gi].kf : E.stors[gi].kf
                k0 = BC.kraw[kf][i]
                knew = trial % 5 == 0 ? E.kspecs[kf].hi[i] * (1 + rand(rng)) : max(k0, 50.0) * exp(0.7 * randn(rng))
                Δ = MB.bias_delta_k(BC, b, E, lkL, kind, gi, i, knew)
                s1 = set_node_capacity!(P, copy(sc), kind, gi, i, knew)
                Δg = MB.bias_term(P, s1) - MB.bias_term(P, sc)
                err = max(err, abs(Δ - Δg)); Δmax = max(Δmax, abs(Δg))
                if trial % 2 == 0
                    MB.bias_delta_k(BC, b, E, lkL, kind, gi, i, knew; commit=true)
                    sc = s1
                    errc = max(errc, abs(MB.bias_value(BC, b) - MB.bias_term(P, sc)))
                end
            end
            @test Δmax > 1e-3
            @test err <= 1e-9 * (1 + Δmax)
            @test errc <= 1e-9 * (1 + abs(MB.bias_term(P, sc)))
        end
        # the cache after the commits equals a rebuild
        BC2 = MB.BiasCache(P, sc)
        @test maximum(abs.(BC.U1x .- BC2.U1x)) < 1e-12 && maximum(abs.(BC.U1s .- BC2.U1s)) < 1e-12
        @test maximum(abs.(BC.zc .- BC2.zc)) < 1e-6 && abs(BC.firm - BC2.firm) < 1e-6
        @test maximum(maximum(abs.(BC.Kw[f] .- BC2.Kw[f]); init=0.0) for f in eachindex(BC.Kw)) < 1e-6
    end

    @testset "bias cache: fine weather moves" begin
        rng = MersenneTwister(29)
        s1 = copy(s)
        LM = MB.LocalModel(P, L); LC = MB.LocalCost(P, LM, s1); BC = MB.BiasCache(P, s1)
        span = 1; for ℓ in 1:L; span = (span + 2) * W.trs[ℓ].r; end
        bm = MB.BlockMove(MB.BlockPool(P, span).wss, 0, Int[], P, LM, LC, Float64[], Float64[])
        jac = NTuple{5,Float64}[]
        err, errc, Δmax, nmv = 0.0, 0.0, 0.0, 0
        for trial in 1:80
            ℓ = rand(rng, 0:L)
            q = rand(rng, 1:(ℓ == 0 ? W.lays[1].n : W.trs[ℓ].n))
            ch, iz = weather_chain(W, s1, ℓ, q, L)
            MB.prep_weather_move!(bm, P, LM, LC, ch, jac) === nothing && continue
            t = 0.5 * randn(rng)
            Δ = MB.bias_delta_weather(BC, b, E, bm, t)
            s2 = apply_weather!(copy(s1), W, ch, ℓ, q, iz, t)
            Δg = MB.bias_term(P, s2) - MB.bias_term(P, s1)
            err = max(err, abs(Δ - Δg)); Δmax = max(Δmax, abs(Δg)); nmv += 1
            if trial % 2 == 0
                MB.commit_block!(bm, s1.v, t)
                MB.bias_delta_weather(BC, b, E, bm, t; commit=true)
                MB.bias_resum!(BC, b)
                apply_weather!(s1, W, ch, ℓ, q, iz, t)
                errc = max(errc, abs(MB.bias_value(BC, b) - MB.bias_term(P, s1)))
            end
        end
        @test nmv > 40 && Δmax > 1e-3
        @test err <= 1e-9 * (1 + Δmax)
        @test errc <= 1e-9 * (1 + abs(MB.bias_term(P, s1)))
        BC2 = MB.BiasCache(P, s1)
        @test maximum(abs.(BC.U1x .- BC2.U1x)) < 1e-12 && maximum(abs.(BC.Dx .- BC2.Dx)) < 1e-6
        @test maximum(maximum(abs.(BC.At[gi] .- BC2.At[gi]); init=0.0) for gi in eachindex(BC.At)) < 1e-12
    end

    @testset "capacity and weather sweeps with the bias cache" begin
        LM = MB.LocalModel(P, L)
        s1 = copy(s)
        BC = MB.BiasCache(P, s1)
        s1, nk = MB.kscale_sweep!(MersenneTwister(3), P, LM, s1; BC=BC)
        s1, np = MB.kpeak_sweep!(MersenneTwister(4), P, LM, s1; BC=BC)
        s1, st = MB.fine_weather_sweep!(MersenneTwister(5), P, LM, s1; frac=0.01, frac0=0.01, BC=BC)
        @test nk > 0 && np > 0 && sum(st[:, 2]) > 0
        @test abs(MB.bias_value(BC, b) - MB.bias_term(P, s1)) <= 1e-9 * (1 + abs(MB.bias_term(P, s1)))
        L1 = s1.L
        @test MB.evaluate!(P, copy(s1)).L ≈ L1
        BC2 = MB.BiasCache(P, s1)
        @test maximum(abs.(BC.U1x .- BC2.U1x)) < 1e-12 && maximum(abs.(BC.zc .- BC2.zc)) < 1e-6
    end

    @testset "scores and design against finite differences" begin
        rng = MersenneTwister(31)
        lv = P.lv[L+1]
        # ∂ₖ log J (physical measure) against differences of log_measure
        g = MB.log_measure_grad(E, lkL, lv, s.k)
        @test maximum(abs, g) > 0
        nok = 0
        idx = rand(rng, findall(!iszero, g), 15)
        for i in idx
            h = 1e-3
            kp, km = copy(s.k), copy(s.k); kp[i] += h; km[i] -= h
            fd = (MB.log_measure(E, lkL, lv, kp) - MB.log_measure(E, lkL, lv, km)) / 2h
            nok += isapprox(fd, g[i]; rtol=1e-5)
        end
        @test nok >= 14
        # the design (∇ĝ = Jz θ in the weather state coordinates) and ∂ₖĝ against the analytic ĝ
        Xn = MB.weather_nodes(W, L, s.xs[end])
        Jz, gk = MB.bias_design(b, E, W, lkL, s.k, Xn)
        gz = Jz * b.θ
        # the state coordinates (z0, ξ₁, …, ξ_L) and their inverse map to the anomalies of level L
        function coords(x)
            parts = Vector{Vector{Float64}}()
            for l in L:-1:1
                x, ξ = MB.split_state(W.trs[l], W.lays[l], W.lays[l+1], W.specs, x)
                pushfirst!(parts, ξ)
            end
            return vcat(x, parts...)
        end
        function anomalies(c)
            o = W.lays[1].n
            x = c[1:o]
            for l in 1:L
                n = W.trs[l].n
                x = MB.merge_state(W.trs[l], W.lays[l], W.lays[l+1], W.specs, x, c[o+1:o+n])
                o += n
            end
            return x
        end
        c0 = coords(s.xs[end])
        @test anomalies(c0) ≈ s.xs[end]
        gA(c) = ghat_analytic(b, E, lkL, s.k, MB.weather_nodes(W, L, anomalies(c)))
        nok, ntry = 0, 0
        nz0 = W.lays[1].n
        cand = findall(x -> abs(x) > 1e-8, gz)
        for i in vcat(rand(rng, filter(<=(nz0), cand), 6), rand(rng, filter(>(nz0), cand), 14))
            h = 1e-4
            cp, cm = copy(c0), copy(c0); cp[i] += h; cm[i] -= h
            fd = (gA(cp) - gA(cm)) / 2h
            ntry += 1
            nok += isapprox(fd, gz[i]; rtol=1e-3, atol=1e-6)
        end
        @test ntry >= 10 && nok >= ntry - 1
        nok, ntry = 0, 0
        for i in rand(rng, findall(!iszero, gk), 12)
            h = 1e-2
            kp, km = copy(s.k), copy(s.k); kp[i] += h; km[i] -= h
            fd = (ghat_analytic(b, E, lkL, kp, Xn) - ghat_analytic(b, E, lkL, km, Xn)) / 2h
            ntry += 1
            nok += isapprox(fd, gk[i]; rtol=1e-3, atol=1e-6)
        end
        @test ntry >= 10 && nok >= ntry - 1
        # the tabulated ĝ against the analytic ĝ
        @test isapprox(MB.ghat(b, E, lkL, s.k, Xn), ghat_analytic(b, E, lkL, s.k, Xn); rtol=1e-6)
        # ∂ĥ against differences of ĥ
        zc = MB.clamped_kcoeffs(E, lkL, s.k)[1]
        _, gh = MB.hhat(b, zc)
        nok = 0
        for i in rand(rng, eachindex(zc), 10)
            h = 1e-2
            zp, zm = copy(zc), copy(zc); zp[i] += h; zm[i] -= h
            nok += isapprox((MB.hhat(b, zp)[1] - MB.hhat(b, zm)[1]) / 2h, gh[i]; rtol=1e-6, atol=1e-10)
        end
        @test nok == 10
        # the capacity score contains the measure term: s_k = −β_v ∂ₖC + ∂ₖ log J
        _, sx, sk, kb = MB.bias_scores(P, s)
        _, (_, gko, _, _) = MB.energy_cost(E, lkL, lv, s.k, s.v, Xn; grad=true)
        @test sk ≈ -P.βv .* gko .+ g
        @test length(sx) == W.lays[L+1].n && kb == s.k
    end

    @testset "refit safeguards: minimum observations and trust region" begin
        @test !MB.refit_due(MB.OnlineBias(; nmin=16), 8) && MB.refit_due(MB.OnlineBias(; nmin=16), 16) &&
              MB.refit_due(MB.OnlineBias(; nmin=16), 32) && !MB.refit_due(MB.OnlineBias(; nmin=16), 24)
        @test MB.refit_due(MB.OnlineBias(), 4)                    # the default floor of 4
        mono = (P.βv, E.c_shed, E.c_dump)
        feasible(θ) = (n1 = length(b.τ1); ramps = vcat(θ[2:1+n1], θ[3+n1:end]);
                       all(ramps .<= 1e-9) && θ[2+n1] <= 1e-9 && θ[1] + θ[2+n1] <= P.βv * b.Dbar * E.c_dump * (1 + 1e-9) &&
                       θ[1] + θ[2+n1] + sum(ramps) >= -P.βv * b.Dbar * E.c_shed * (1 + 1e-9))
        b0 = deepcopy(b)
        b.θ ./= 3.0                                # the monotone LP warm-start fit (feasible)
        MB.rebuild_tables!(b)
        @test feasible(b.θ) && b.nobs == 0
        # minimum observation count: no refit before 16 observations, the lifted fit stays
        OB = MB.OnlineBias(; nmin=16, trust=1e-3, nrecent=8)
        θl, al, cl = copy(b.θ), copy(b.a), copy(b.c)
        ups = Bool[]
        s1 = copy(s)
        for it in 1:16
            s1 = copy(s)                           # states with other capacities: other b̂, other scores
            s1.k .*= 1 .+ 0.05 .* randn(MersenneTwister(it), length(s1.k))
            MB.evaluate!(P, s1)
            _, upd = MB.online_bias_step!(P, s1, OB; monotone=mono)
            push!(ups, upd)
        end
        @test !any(ups[1:15]) && ups[16] && OB.updates == [16]
        obs = findall(>(0), b.hk[:, 1])                             # the data ranges of ĥ come with the refit
        @test !isempty(obs) && b.κlo[obs] == b.hk[obs, 6] && b.κhi[obs] == b.hk[obs, 7] && all(b.κlo[obs] .<= b.κhi[obs])
        @test length(OB.recent) == 8
        # trust region: a convex combination of the lifted and the new fit, within the bound, feasible
        λ = OB.steps[1]
        @test 0 <= λ < 1
        old = (θl, al, cl)
        bnew = deepcopy(b); MB.bias_update!(bnew; monotone=mono)              # the full step (same sums)
        @test b.θ ≈ (1 - λ) .* θl .+ λ .* bnew.θ && b.a ≈ (1 - λ) .* al .+ λ .* bnew.a && b.c ≈ (1 - λ) .* cl .+ λ .* bnew.c
        @test feasible(bnew.θ) && feasible(b.θ)
        vals(bb) = [MB.bias_from_record(bb, E, lkL, r...) for r in OB.recent]
        bl = deepcopy(b); MB.set_coeffs!(bl, old, old, 0.0)
        v0 = vals(bl); d = vals(b) .- v0
        bound = 1e-3 * max(maximum(v0) - minimum(v0), Float64(lkL.n))
        @test maximum(abs.(d .- mean(d))) <= bound * (1 + 1e-9)
        dfull = vals(bnew) .- v0
        @test maximum(abs.(dfull .- mean(dfull))) > bound            # the full step would have left the region
        # a wide trust region takes the full step
        bw = deepcopy(bl); bw.GtG .= b.GtG; bw.Gts .= b.Gts; bw.hk .= b.hk; bw.nobs = b.nobs
        MB.bias_update!(bw; monotone=mono)
        @test MB.trust_step!(bw, E, lkL, OB.recent, old, 1e12) == 1.0 && bw.θ ≈ bnew.θ
        # the cache with the stepped coefficients still equals the global evaluation
        BC = MB.BiasCache(P, s1)
        @test MB.bias_value(BC, b) ≈ MB.bias_term(P, s1) rtol = 1e-12
        # back to the bias of the other tests
        for f in fieldnames(MB.BiasModel)
            setfield!(b, f, getfield(b0, f))
        end
    end

    @testset "post-correction records" begin
        PB = MB.PostBias()
        s1 = copy(s)
        MB.post_record!(PB, P, s1, [0.0])
        @test PB.nobs == 1 && size(PB.GtG) == (MB.nθ(b), MB.nθ(b))
        @test isapprox(MB.bias_from_record(b, E, lkL, PB.U1[1], PB.kc[1], PB.zc[1]), PB.bused[1]; rtol=1e-12)
        @test length(PB.U1[1]) == lkL.T                    # features per finest step, not the raw weather
    end

    @testset "Metropolis bias correction of a fine weather move (quadrature)" begin
        # Repeated moves of one innovation coordinate without compensation (a straight line in
        # (w, v)): the chain of draws must sample p(t) ∝ exp(−β_v ΔC(t) − prior(t) − Δb̂(t)) on
        # the line through the start, computed by quadrature from the start state's move.
        rng = MersenneTwister(37)
        LM = MB.LocalModel(P, L)
        best = nothing
        for trial in 1:400                       # an availability coordinate with a wide line and a bias change
            ℓ = L
            q = rand(rng, 1:W.trs[ℓ].n)
            f, part, i, bb, j = MB.innovation_coord(W, ℓ, q)
            (f > 1 && i == 1) || continue
            s1 = copy(s); LC = MB.LocalCost(P, LM, s1); BC = MB.BiasCache(P, s1)
            ch, _ = weather_chain(W, s1, ℓ, q, L)
            bm = MB.BlockMove(MB.BlockPool(P, 40).wss, 0, Int[], P, LM, LC, Float64[], Float64[])
            MB.prep_weather_move!(bm, P, LM, LC, ch, NTuple{5,Float64}[]; comp=false) === nothing && continue
            grid = range(-4.0, 4.0; length=4001)
            lp = [-P.βv * MB.mv_eval!(bm, x) - 0.5 * x^2 - s1.ws[ℓ][q] * x for x in grid]
            w = exp.(lp .- maximum(lp)); cdf = cumsum(w) ./ sum(w)
            lo, hi = grid[searchsortedfirst(cdf, 0.01)], grid[searchsortedfirst(cdf, 0.99)]
            hi - lo > 0.3 || continue
            Δb = abs(MB.bias_delta_weather(BC, b, E, bm, hi) - MB.bias_delta_weather(BC, b, E, bm, lo))
            Δb > 1e-3 || continue
            best = (ℓ, q, Δb, lo, hi)
            break
        end
        @test best !== nothing
        ℓ, q, Δb, lo, hi = best
        θs = copy(b.θ)
        b.θ .*= 2.0 / Δb                          # a bias change of about 2 nats across the line
        MB.rebuild_tables!(b)
        s1 = copy(s); LC = MB.LocalCost(P, LM, s1); BC = MB.BiasCache(P, s1)
        ch, _ = weather_chain(W, s1, ℓ, q, L)
        bm0 = MB.BlockMove(MB.BlockPool(P, 40).wss, 0, Int[], P, LM, LC, Float64[], Float64[])
        MB.prep_weather_move!(bm0, P, LM, LC, ch, NTuple{5,Float64}[]; comp=false)
        w0 = s1.ws[ℓ][q]
        grid = range(lo - 3 * (hi - lo), hi + 3 * (hi - lo); length=8001)
        lp = [-P.βv * MB.mv_eval!(bm0, x) - 0.5 * (w0 + x)^2 - MB.bias_delta_weather(BC, b, E, bm0, x) for x in grid]
        lp0 = [-P.βv * MB.mv_eval!(bm0, x) - 0.5 * (w0 + x)^2 for x in grid]
        cdfof(lp) = (w = exp.(lp .- maximum(lp)); cumsum(w) ./ sum(w))
        cdf, cdf0 = cdfof(lp), cdfof(lp0)
        bm0 = nothing
        # the chain of moves (bias cache and caches follow the accepted moves)
        LC = MB.LocalCost(P, LM, s1)
        BC = MB.BiasCache(P, s1)
        bm = MB.BlockMove(MB.BlockPool(P, 40).wss, 0, Int[], P, LM, LC, Float64[], Float64[])
        jac, sp = NTuple{5,Float64}[], Float64[]
        n = 20000
        ts = zeros(n)
        nacc = 0
        for it in 1:n
            t = MB.weather_coord_move!(rng, P, LM, LC, s1, bm, jac, sp, ℓ, q; comp=false, BC=BC)
            nacc += t != 0.0
            ts[it] = s1.ws[ℓ][q] - w0
        end
        Fq(x, c) = c[clamp(searchsortedlast(grid, x), 1, length(grid))]
        ks = maximum(abs(mean(ts .<= x) - Fq(x, cdf)) for x in quantile(ts, 0.02:0.02:0.98))
        ks0 = maximum(abs(mean(ts .<= x) - Fq(x, cdf0)) for x in quantile(ts, 0.02:0.02:0.98))
        println("  weather MH with the bias: acceptance $(round(nacc / n; digits=3)), KS $(round(ks; digits=4)) ",
                "(against the bias-free conditional $(round(ks0; digits=4)))")
        @test ks < 0.03
        @test ks0 > 2 * ks                       # the test can tell the bias correction
        b.θ .= θs
        MB.rebuild_tables!(b)
    end
end
