using Test, Random, LinearAlgebra, Statistics, SparseArrays
using ESMSampling
const M = ESMSampling

@testset "nested Laplace bases" begin
    rng = MersenneTwister(1)
    N = 12
    edges = [(i, i + 1, 1.0 + rand(rng)) for i in 1:N-1]
    push!(edges, (1, 7, 2.0))
    U, λ = M.laplace_modes(N, edges)
    @test issorted(λ)
    @test U' * U ≈ I
    sup = [2, 3, 5, 8, 11]
    B = M.nested_basis(U, sup)
    @test B.U' * B.U ≈ I
    @test B.rank_at[end] == length(sup)
    @test issorted(B.rank_at)
    # nestedness: the level-M basis spans the restriction of the first M modes
    for Mm in 1:N
        r = B.rank_at[Mm]
        P = B.U[:, 1:r] * B.U[:, 1:r]'
        @test norm(P * U[sup, 1:Mm] - U[sup, 1:Mm]) < 1e-8
    end
    Z = M.component_indicators(N, [e[1] for e in edges], [e[2] for e in edges])
    Bθ = M.nested_basis(U, collect(1:N); deflate=Z)
    @test M.nmodes(Bθ) == N - 1
    @test norm(Bθ.U' * ones(N)) < 1e-8
end

@testset "level transfer" begin
    rng = MersenneTwister(2)
    N = 6
    U, _ = M.laplace_modes(N, [(i, i + 1, 1.0) for i in 1:N-1])
    specs = [M.FieldSpec(:a, M.STEP, M.nested_basis(U, collect(1:N)), zeros(N), ones(N)),
             M.FieldSpec(:e, M.INSTANT, M.nested_basis(U, [1, 3, 4]), zeros(3), ones(3)),
             M.FieldSpec(:k, M.KFIELD, M.nested_basis(U, [2, 5]), zeros(2), fill(5.0, 2))]
    for (lc, lf) in [((2, 24), (6, 24)), ((6, 24), (6, 6)), ((2, 24), (4, 8))]
        Lc, Lf = M.Layout(specs, lc...), M.Layout(specs, lf...)
        tr = M.Transfer(specs, Lc, Lf)
        z = randn(rng, Lf.n)
        zc, ξ = M.split_state(tr, Lc, Lf, specs, z)
        @test M.merge_state(tr, Lc, Lf, specs, zc, ξ) ≈ z
        @test length(ξ) + Lc.n == Lf.n
        # block averages of step fields commute with the restriction
        Yf = M.nodevals(z, Lf, specs, 1)
        Yc = M.nodevals(zc, Lc, specs, 1)
        r = tr.r
        mc = Lc.m[1]
        Pc = specs[1].B.U[:, 1:mc] * specs[1].B.U[:, 1:mc]'
        @test Pc * hcat([mean(Yf[:, (j-1)*r+1:j*r]; dims=2) for j in 1:Lc.T]...) ≈ Yc
    end
end

@testset "reflective HMC on a polytope" begin
    # truncated standard Gaussian on {x ≥ 0, x₁ + x₂ ≤ 1, x₂ - x₃ ≤ 0.5} in 3D
    rng = MersenneTwister(3)
    n = 3
    A = sparse([1.0 1.0 0.0; 0.0 1.0 -1.0])
    C = M.Constraints(n, zeros(n), fill(Inf, n), A, sparse(A'), [1.0, 0.5])
    pot(q) = (0.5 * sum(abs2, q .- 0.3), q .- 0.3)
    st = M.HMCState(0.3, 10, ones(n))
    ws = M.BilliardWS(C)
    q = [0.1, 0.1, 0.1]
    xs = Vector{Vector{Float64}}()
    for it in 1:40000
        M.hmc_step!(rng, q, pot, C, st, ws)
        @test M.isfeasible(C, q; tol=1e-9)
        it > 1000 && push!(xs, copy(q))
    end
    # reference by rejection sampling
    ref = Vector{Vector{Float64}}()
    while length(ref) < 200000
        x = 0.3 .+ randn(rng, n)
        all(x .>= 0) && x[1] + x[2] <= 1 && x[2] - x[3] <= 0.5 && push!(ref, x)
    end
    @test norm(mean(xs) - mean(ref)) < 0.02
    @test norm(var(xs) - var(ref)) < 0.02
    @test st.nacc / st.ntot > 0.5
end

@testset "FFBS samples the regime posterior" begin
    rng = MersenneTwister(4)
    K, T = 2, 4
    ll = randn(rng, K, T)
    Ptr = [0.8 0.2; 0.3 0.7]
    π0 = [0.6, 0.4]
    paths = vec(collect(Iterators.product(fill(1:K, T)...)))
    w = [π0[p[1]] * exp(ll[p[1], 1]) * prod(Ptr[p[t-1], p[t]] * exp(ll[p[t], t]) for t in 2:T) for p in paths]
    w ./= sum(w)
    cnt = Dict(p => 0 for p in paths)
    nS = 100000
    for _ in 1:nS
        cnt[Tuple(M.ffbs(rng, ll, Ptr, π0))] += 1
    end
    @test maximum(abs(cnt[p] / nS - w[i]) for (i, p) in enumerate(paths)) < 0.01
    γ, _, _ = M.forward_backward(ll, Ptr, π0)
    @test γ[1, 1] ≈ sum(w[i] for (i, p) in enumerate(paths) if p[1] == 1)
end

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    @testset "energy model on the PyPSA-Eur data" begin
        rng = MersenneTwister(5)
        sys = M.load_system(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
        E = M.build_energy_model(sys; regularized=false)   # (A1) concerns the hard polytope
        levels = [(4, 24), (40, 24), (40, 6)]
        lk = [M.Layout(E.kspecs, a, b) for (a, b) in levels]
        lv = [M.Layout(E.vspecs, a, b) for (a, b) in levels]
        trk = [M.Transfer(E.kspecs, lk[j], lk[j+1]) for j in 1:2]
        trv = [M.Transfer(E.vspecs, lv[j], lv[j+1]) for j in 1:2]
        cons = [M.build_constraints(E, lk[j], lv[j]) for j in 1:3]
        # (A1): a feasible finest state restricts to feasible coarse states
        c = M.interior_center(E, lv[3])
        vals = [clamp.(x .+ 0.3 .* (rand(rng, size(x)...) .- 0.5), 0.0, 1.0) for x in c]
        vals[E.fθ] .= 0.0
        vals[E.flink] .= 0.0
        for s in E.stors   # storage idle at constant charge level satisfies the dynamics
            vals[s.fp] .= 0.5; vals[s.fm] .= 0.0; vals[s.fe] .= rand(rng, size(vals[s.fe], 1))
        end
        kv = [reshape(s.hi .* rand(rng, length(s.hi)), :, 1) for s in E.kspecs]
        zk = M.to_modes(lk[3], E.kspecs, kv)
        zv = M.to_modes(lv[3], E.vspecs, vals)
        @test M.isfeasible(cons[3], vcat(zk, zv))
        for ℓ in 1:-1:0
            @test M.isfeasible(cons[ℓ+1], vcat(M.restrict_to(zk, lk, trk, E.kspecs, 2, ℓ),
                                                M.restrict_to(zv, lv, trv, E.vspecs, 2, ℓ)))
        end
        # gradient of the operational cost at a spatially coarse level
        zk0 = M.restrict_to(zk, lk, trk, E.kspecs, 2, 0)
        zv0 = M.restrict_to(zv, lv, trv, E.vspecs, 2, 0) .+ 0.01 .* randn(rng, lv[1].n)
        Xn = [repeat(mean(Float64.(sys.weather[s.name][2][:, :, 1]); dims=2), 1, lv[1].T) for s in E.wspecs]
        p, (gkc, gko, gv, gX) = M.energy_cost(E, lk[1], lv[1], zk0, zv0, Xn; grad=true)
        for i in rand(rng, 1:length(zv0), 10)
            h = 1e-4 * max(1, abs(zv0[i]))   # op ~ 1e12 €: smaller steps drown in round-off
            zp, zm = copy(zv0), copy(zv0)
            zp[i] += h; zm[i] -= h
            fd = (M.energy_cost(E, lk[1], lv[1], zk0, zp, Xn).op - M.energy_cost(E, lk[1], lv[1], zk0, zm, Xn).op) / 2h
            # the cost is piecewise linear: a difference quotient may straddle a kink
            @test isapprox(fd, gv[i]; rtol=0.05, atol=1e-6 * abs(p.op) / length(zv0))
        end
    end
end

@testset "MLDA core on a three-level Gaussian toy" begin
    # level 2 target: correlated Gaussian on (x, y, u); u is a pCN mode with
    # reference N(0,1), so L₂ excludes that factor. Coarser levels are deliberately
    # wrong approximations; MLDA must still be exact on the finest level.
    rng = MersenneTwister(6)
    μ = [0.5, -1.0, 0.3]
    Σ = [1.0 0.6 0.3; 0.6 1.5 -0.4; 0.3 -0.4 0.8]
    Σi = inv(Σ)
    logN3(z) = -0.5 * dot(z .- μ, Σi * (z .- μ))
    L = [z -> -0.5 * (z[1] - 0.2)^2 / 1.5^2,                                  # level 0: x
         z -> -0.5 * ((z[1] - 0.4)^2 + (z[2] + 0.8)^2 / 2 - 0.5 * z[1] * z[2]),  # level 1: (x, y)
         z -> logN3(z) + 0.5 * z[3]^2]                                         # level 2
    mk(ℓ, z) = (ℓ=ℓ, z=z, L=L[ℓ+1](z))
    function base(rng, s)
        zn = s.z .+ 0.8 .* randn(rng, 1)
        sn = mk(0, zn)
        return rand(rng) < exp(sn.L - s.L) ? sn : s
    end
    function step(rng, s)
        s.ℓ == 0 && return base(rng, s)
        ℓ = s.ℓ
        prop = ℓ == 1 ? ((r, y) -> y .+ 0.7 .* randn(r, 1)) :
                        ((r, u) -> sqrt(1 - 0.5^2) .* u .+ 0.5 .* randn(r, 1))
        ops = (split=s_ -> (mk(ℓ - 1, s_.z[1:end-1]), s_.z[end:end]),
               step=step, propose=prop,
               assemble=(sc, f) -> mk(ℓ, vcat(sc.z, f)),
               logtarget=s_ -> s_.L, record=false)
        return M.mlda_core(rng, s, 3, ops)[1]
    end
    s = mk(2, [0.0, 0.0, 0.0])
    xs = zeros(3, 200000)
    for i in axes(xs, 2)
        s = step(rng, s)
        xs[:, i] = s.z
    end
    xs = xs[:, 10001:end]
    @test norm(vec(mean(xs; dims=2)) - μ) < 0.05
    @test norm(cov(xs') - Σ) < 0.08
end

@testset "graph Haar basis" begin
    labels = ["a", "a", "a", "b", "b", "c", "c", "c", "c"]
    U, counts = M.haar_vectors([labels, collect(1:length(labels))])
    @test U' * U ≈ I
    @test counts == [3, 9]
    # the first counts[1] vectors span the cell-constant fields
    P = U[:, 1:3] * U[:, 1:3]'
    for c in unique(labels)
        v = Float64.(labels .== c)
        @test P * v ≈ v
    end
    # every detail vector of the node level lives in one cell and has zero sum there
    for j in 4:9
        sup = findall(x -> abs(x) > 1e-12, U[:, j])
        @test length(unique(labels[sup])) == 1
        @test abs(sum(U[:, j])) < 1e-12
    end
end
include("test_local.jl")
include("test_flow.jl")
include("test_bias.jl")
include("test_model.jl")
include("test_convergence.jl")
