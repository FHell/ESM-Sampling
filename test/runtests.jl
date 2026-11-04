using Test, Random, LinearAlgebra, Statistics, SparseArrays
using ESMSampling
const M = ESMSampling

@testset "level transfer" begin
    rng = MersenneTwister(2)
    N = 6
    parts = [ones(Int, N), [1, 1, 1, 2, 2, 2], collect(1:N)]
    edges = [(i, i + 1, 1.0) for i in 1:N-1]
    _, gcounts, _ = M.haar_basis(parts; edges=edges)
    hb(sup) = M.haar_field_basis(parts, gcounts, sup; edges=edges)
    specs = [M.FieldSpec(:a, M.STEP, hb(collect(1:N)), zeros(N), ones(N)),
             M.FieldSpec(:e, M.INSTANT, hb([1, 3, 4]), zeros(3), ones(3)),
             M.FieldSpec(:k, M.KFIELD, hb([2, 5]), zeros(2), fill(5.0, 2))]
    for (lc, lf) in [((2, 24), (6, 24)), ((6, 24), (6, 6)), ((1, 24), (2, 8))]
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
        E = M.build_energy_model(sys)
        levels = [(10, 24), (40, 24), (40, 6)]
        lk = [M.Layout(E.kspecs, a, b) for (a, b) in levels]
        lv = [M.Layout(E.vspecs, a, b) for (a, b) in levels]
        trk = [M.Transfer(E.kspecs, lk[j], lk[j+1]) for j in 1:2]
        trv = [M.Transfer(E.vspecs, lv[j], lv[j+1]) for j in 1:2]
        c = M.interior_center(E, lv[3])
        vals = [clamp.(x .+ 0.3 .* (rand(rng, size(x)...) .- 0.5), 0.0, 1.0) for x in c]
        vals[E.fθ] .= 0.0
        vals[E.flink] .= 0.0
        kv = [reshape(s.hi .* rand(rng, length(s.hi)), :, 1) for s in E.kspecs]
        zk = M.to_modes(lk[3], E.kspecs, kv)
        zv = M.to_modes(lv[3], E.vspecs, vals)
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
include("test_kop.jl")
include("test_kbox.jl")
include("test_wblock.jl")
include("test_moves.jl")
include("test_replicas.jl")
include("test_model.jl")
include("test_convergence.jl")
include("test_hpc.jl")
