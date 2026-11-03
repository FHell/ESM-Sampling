# Device bias change of a capacity move against differences of the CPU bias_term (no randomness).
using ESMDevice, ESMSampling, Random, Test, Printf
const ML = ESMSampling
const D = ESMDevice
include(joinpath(@__DIR__, "setup_all_terms.jl"))
Pb = ML.with_bias(P, true)
b = Pb.bias
rng = MersenneTwister(4)
b.θ .= 0.05 .* randn(rng, length(b.θ))                      # a non-trivial bias
b.a .= 0.5 .* randn(rng, length(b.a)); b.c .= 0.1 .* abs.(randn(rng, length(b.c)))
BD, BS, gitem, sitem, ritem = D.pack_bias(Pb, ix, [sL])
E, lk = Pb.E, Pb.lk[L+1]
b0 = ML.bias_term(Pb, ML.k_at0(Pb, L, sL.k), sL.z0)
errs = Float64[]
for trial in 1:30
    kind = rand(rng, (:gen, :stor, :ren))
    isgen = kind == :gen
    q = kind == :gen ? rand(rng, findall(>(0), gitem)) : kind == :stor ? rand(rng, findall(>(0), sitem)) : rand(rng, eachindex(ritem))
    gi, i = kind == :gen ? ix.gidx[q] : kind == :stor ? ix.sidx[q] : D.ren_units(Pb)[q]
    kf = kind == :stor ? E.stors[gi].kf : E.gens[gi].kf
    δ = 200.0 * randn(rng)
    s1 = copy(sL)
    Ut = E.kspecs[kf].B.Ut
    for qq in ML.nzrange(Ut, i)
        s1.k[lk.off[kf]+ML.rowvals(Ut)[qq]] += δ * ML.nonzeros(Ut)[qq]
    end
    Δcpu = ML.bias_term(Pb, ML.k_at0(Pb, L, s1.k), s1.z0) - b0
    Δdev, _ = D.delta_bias(BD, BS, kind == :gen ? gitem[q] : kind == :stor ? sitem[q] : ritem[q], [δ])
    push!(errs, abs(Δdev[1] - Δcpu))
    trial <= 5 && @printf("  %-4s unit %d δ %8.1f MW: Δb̂ CPU %.6e  device %.6e\n", string(kind), q, δ, Δcpu, Δdev[1])
end
@printf("max |Δb̂ device − CPU| = %.2e nats (|b̂| ≈ %.2e)\n", maximum(errs), abs(b0))
@testset "device bias change = CPU bias_term difference" begin
    @test maximum(errs) < 1e-8 * max(1.0, abs(b0))
end
