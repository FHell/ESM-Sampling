using ESMDevice, ESMSampling, Random, Test
const ML = ESMSampling
const D = ESMDevice
include(joinpath(@__DIR__, "setup_all_terms.jl"))
S = D.to_device(P, M, ix, [sL, sL])
S.kg .*= 1 .+ 0.3 .* randn(MersenneTwister(1), size(S.kg)...)      # capacities off the start, some outside the boxes
cpH = D.chain_params(M, S)
cpD = D.chain_params_dev(D.KParams(M), S.kg, S.ks, cpH.cls, Val(2))
@testset "device chain parameters = host chain_params" begin
    for f in fieldnames(typeof(cpH))
        @test getfield(cpD, f) == getfield(cpH, f)
    end
end
