# Device bias change of a capacity move against the CPU bias (no randomness): the bias lifted to
# the chains' level, the same proposals on both sides; (1) the device code on Julia arrays against
# the CPU's incremental bias_delta_k and the bias_term difference, and the aggregates after a
# commit against a fresh pack; (2) the same code compiled by Reactant (CPU backend) against (1).
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice
include(joinpath(@__DIR__, "setup_all_terms.jl"))
Pb = ML.with_bias(P, true)
b = Pb.bias
ML.lift_bias!(Pb, L; rng=MersenneTwister(5), nsteps=1024)
rng = MersenneTwister(4)
b.θ .= 500.0 .* randn(rng, length(b.θ))                     # a non-trivial bias (Δb̂ of nats per proposal)
b.a .= 0.5 .* randn(rng, length(b.a)); b.c .= 0.1 .* abs.(randn(rng, length(b.c)))
ML.rebuild_tables!(b)
C = 2
ss = [ML.evaluate!(Pb, copy(sL)) for _ in 1:C]
ss[2].k .*= 1.1; ML.evaluate!(Pb, ss[2])
BD, BS, gitem, sitem, ritem = D.pack_bias(Pb, ix, ss)
E, lk = Pb.E, Pb.lk[L+1]
BCs = [ML.BiasCache(Pb, s) for s in ss]
# proposals: unit kind, unit, δ per chain
props = Tuple{Symbol,Int,Vector{Float64}}[]
for trial in 1:30
    kind = (:gen, :stor, :ren)[mod1(trial, 3)]
    q = kind == :gen ? rand(rng, findall(>(0), gitem)) : kind == :stor ? rand(rng, findall(>(0), sitem)) : rand(rng, eachindex(ritem))
    push!(props, (kind, q, 200.0 .* randn(rng, C)))
end
unitof(kind, q) = kind == :gen ? (1, ix.gidx[q]...) : kind == :stor ? (2, ix.sidx[q]...) : (1, D.ren_units(Pb)[q]...)
itemof(kind, q) = kind == :gen ? gitem[q] : kind == :stor ? sitem[q] : ritem[q]
errs, errg, mags = Float64[], Float64[], Float64[]
for (trial, (kind, q, δ)) in enumerate(props)
    knd, gi, i = unitof(kind, q)
    kf = knd == 1 ? E.gens[gi].kf : E.stors[gi].kf
    Δdev, _ = D.delta_bias(BD, BS, itemof(kind, q), δ)
    for c in 1:C
        k0 = BCs[c].kraw[kf][i]
        Δcpu = ML.bias_delta_k(BCs[c], b, E, lk, knd, gi, i, k0 + δ[c])
        sm = copy(ss[c])
        Ut = E.kspecs[kf].B.Ut
        for qq in ML.nzrange(Ut, i)
            sm.k[lk.off[kf]+ML.rowvals(Ut)[qq]] += δ[c] * ML.nonzeros(Ut)[qq]
        end
        Δglob = ML.bias_term(Pb, sm) - ML.bias_term(Pb, ss[c])
        push!(errs, abs(Δdev[c] - Δcpu)); push!(errg, abs(Δdev[c] - Δglob)); push!(mags, abs(Δcpu))
        trial <= 4 && c == 1 && @printf("  %-4s unit %3d δ %8.1f MW: Δb̂ CPU %.6e  device %.6e\n", string(kind), q, δ[c], Δcpu, Δdev[c])
    end
end
@printf("max |Δb̂ device − CPU cache| = %.2e, − global difference = %.2e nats (max |Δb̂| %.2e)\n", maximum(errs), maximum(errg), maximum(mags))
# a commit: the aggregates after accepted proposals equal a fresh pack of the moved CPU states
function commit_all(BD, BS, props, itemof)
    for (kind, q, δ) in props
        _, new = D.delta_bias(BD, BS, itemof(kind, q), δ)
        BS = D.commit_bias!(BS, new, [true, false])               # chain 1 accepts, chain 2 rejects
    end
    return BS
end
BSa = commit_all(BD, deepcopy(BS), props[1:6], itemof)
s1 = copy(ss[1])
for (kind, q, δ) in props[1:6]
    knd, gi, i = unitof(kind, q)
    kf = knd == 1 ? E.gens[gi].kf : E.stors[gi].kf
    Ut = E.kspecs[kf].B.Ut
    for qq in ML.nzrange(Ut, i)
        s1.k[lk.off[kf]+ML.rowvals(Ut)[qq]] += δ[1] * ML.nonzeros(Ut)[qq]
    end
end
_, BSc, _, _, _ = D.pack_bias(Pb, ix, [ML.evaluate!(Pb, s1), ss[2]])
errpack = maximum(maximum(abs.(x .- y)) for (x, y) in zip(D.bias_fields(BSa), D.bias_fields(BSc)))
@printf("aggregates after commits − fresh pack: %.2e\n", errpack)
@testset "device bias change = CPU bias (Julia arrays)" begin
    @test maximum(mags) > 1e-2
    @test maximum(errs) < 1e-9 * max(1.0, maximum(mags))
    @test maximum(errg) < 1e-8 * max(1.0, abs(ML.bias_term(Pb, ss[1])))
    @test maximum(abs.(BSa.u1x .- BSc.u1x)) < 1e-12 && maximum(abs.(BSa.u2s .- BSc.u2s)) < 1e-12
    @test maximum(abs.(BSa.gx .- BSc.gx)) < 1e-9 * maximum(abs, BSc.gx) && maximum(abs.(BSa.zc .- BSc.zc)) < 1e-6
end
# (2) compiled by Reactant on the CPU backend: the same proposals, against the array version
Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
R = Reactant.to_rarray
BDr, BSr = R(BD), R(BS)
dev_delta(BD, BS, ι, δ) = D.delta_bias(BD, BS, ι, δ)
RN(x) = Reactant.ConcreteRNumber(x)              # a traced item index (a plain Int would be compiled in)
tc = @elapsed fd = @compile dev_delta(BDr, BSr, RN(itemof(props[1][1], props[1][2])), R(props[1][3]))
errr = Float64[]
for (kind, q, δ) in props
    Δa, newa = D.delta_bias(BD, BS, itemof(kind, q), δ)
    Δr, newr = fd(BDr, BSr, RN(itemof(kind, q)), R(δ))
    push!(errr, maximum(abs.(Array(Δr) .- Δa)))
    push!(errr, maximum(maximum(abs.(Array(x) .- y) ./ max.(1.0, abs.(y))) for (x, y) in zip(newr, newa)))
end
@printf("Reactant (%s): compile %.1f s, max |compiled − arrays| = %.2e\n", get(ENV, "REACTANT_BACKEND", "cpu"), tc, maximum(errr))
@testset "device bias change compiled by Reactant = arrays" begin
    @test maximum(errr) < 1e-9 * max(1.0, maximum(mags))
end
