# Convergence diagnostics (src/convergence.jl) on synthetic chains with known behaviour.
using Test, Random, Statistics
import ESMSampling as ML

@testset "convergence diagnostics on synthetic chains" begin
    rng = MersenneTwister(1)
    ar(n, φ) = (x = zeros(n); x[1] = randn(rng) / sqrt(1 - φ^2); for t in 2:n; x[t] = φ * x[t-1] + randn(rng); end; x)
    iid = ML.convergence_summary([randn(rng, 1000) for _ in 1:4])
    @test iid.pass && iid.rhat < 1.01 && 3000 < iid.ess_bulk < 5000
    a9 = ML.convergence_summary([ar(2000, 0.9) for _ in 1:4])   # expected ESS ≈ 8000 · 0.1/1.9 ≈ 420
    @test 250 < a9.ess_bulk < 700
    @test ML.convergence_summary([randn(rng, 1000) .+ (k == 1 ? 0.5 : 0.0) for k in 1:4]).rhat > 1.01     # location
    @test ML.convergence_summary([randn(rng, 1000) .* (k == 1 ? 2.0 : 1.0) for k in 1:4]).rhat > 1.01     # scale (folded R̂)
    @test !ML.convergence_summary([cumsum(randn(rng, 1000)) for _ in 1:4]).pass                            # random walk
    # warmup detection: chains start apart and relax to a common distribution after ≈ 100 steps
    tr = [[vcat(collect(range(3k, 0; length=100)) .+ 0.1 .* randn(rng, 100), randn(rng, 300)) for k in 1:4]]
    w = ML.warmup_end(tr; every=10)
    @test 50 <= w <= 160
end
