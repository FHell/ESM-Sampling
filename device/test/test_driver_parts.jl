using ESMDevice, ESMSampling, Random, Test, Printf
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
@testset "device proxy class = CPU frame_injections > 0" begin
    S2 = D.to_device(P, M, ix, [sL])
    cls = D.proxy_class(S2.dnet, D.load_shares(P, [sL]))
    p = ML.frame_injections(P, L, sL)                       # N × T
    @test cls[1, :] == vec(Float64.(permutedims(p) .> 0))   # column τ + (n − 1) T
    @printf("surplus share %.3f\n", sum(cls) / length(cls))
end
