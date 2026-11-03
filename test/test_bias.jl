# Monotone-price bias fit: the implied price of the residual load is non-decreasing and
# bounded by [−c_dump, c_shed] (docs/setup_overview.md §10).
using Test, Random, LinearAlgebra
using ESMSampling
const MB = ESMSampling

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    @testset "monotone-price bias" begin
        cfg = MB.load_config(joinpath(@__DIR__, "..", "configs", "haar_poc.toml")); cfg.verbose = false
        cfg.use_bias = true; cfg.bias_warmstart = 20.0; cfg.bias_monotone = true
        P, sL, info = MB.setup(cfg; rng=MersenneTwister(cfg.seed))
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
    end
end
