# Phase-1 acceptance (docs/reactant_design.md §8): the device cost equals energy_cost
# (all model-completion terms on), the residual equals LocalCost.r, on Julia arrays
# and traced by Reactant.
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice
cfg = ML.load_config(joinpath(@__DIR__, "..", "..", "configs", "haar_poc.toml")); cfg.verbose = false
cfg.data = joinpath(@__DIR__, "..", "..", "data", "processed", "model.h5")
cfg.out = joinpath(@__DIR__, "..", "..", "results", "haar_poc.h5")
cfg.storage = "soc"; cfg.renewables = "load"; cfg.measure = "physical"; cfg.init = "heuristic"
cfg.losses = true; cfg.thermal = "soft"; cfg.link_losses = true; cfg.uc = ["gas", "nuclear"]; cfg.brownfield = true
P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
rng = MersenneTwister(3)
L = sL.ℓ
ss = ML.State[]
for c in 1:3
    s = copy(sL)
    c > 1 && (s.v .+= 0.02 .* c .* randn(rng, length(s.v)); s.k .*= 1 .+ 0.05 .* randn(rng, length(s.k)))
    push!(ss, ML.evaluate!(P, s))
end
M, ix = D.pack(P, L)
S = D.to_device(P, M, ix, ss)
r, parts = D.cost_parts(M, S)
@testset "device cost = energy_cost (Julia arrays)" begin
    for (c, s) in enumerate(ss)
        @test isapprox(parts.op[c], s.parts.op; rtol=1e-11)
        @test isapprox(parts.capex[c], s.parts.capex; rtol=1e-11)
        @test isapprox(parts.co2[c], s.parts.co2; rtol=1e-11)
        @test isapprox(parts.shed[c], s.parts.shed; rtol=1e-11, atol=1e-6)
        LC = ML.LocalCost(P, ML.LocalModel(P, L), s)
        @test maximum(abs.(r[c, :, :]' .- LC.r)) < 1e-8
        @printf("chain %d: op %.6e vs %.6e (rel %.1e), capex rel %.1e\n", c, parts.op[c], s.parts.op,
                abs(parts.op[c] / s.parts.op - 1), abs(parts.capex[c] / s.parts.capex - 1))
    end
end
@testset "round trip State → device → State" begin
    for (c, s) in enumerate(ss)
        s2 = copy(s); s2.v .= 0.0          # renewable capacities are host-side: keep k
        D.to_host!(P, M, ix, S, c, s2)
        @test maximum(abs.(s2.v .- s.v)) < 1e-10
        @test isapprox(s2.parts.op, s.parts.op; rtol=1e-12)
    end
end
if get(ENV, "DEVICE_REACTANT", "1") == "1"

    @testset "device cost traced by Reactant" begin
        Mr, Sr = Reactant.to_rarray(M), Reactant.to_rarray(S)   # the model is a traced argument (no constant folding)
        tc = @elapsed f = @compile D.cost_parts(Mr, Sr)
        rd, pd = f(Mr, Sr)
        @printf("compile %.1f s\n", tc)
        @test maximum(abs.(Array(rd) .- r)) < 1e-8
        @test maximum(abs.(Array(pd.op) ./ parts.op .- 1)) < 1e-12
        @test maximum(abs.(Array(pd.capex) ./ parts.capex .- 1)) < 1e-12
    end
end
