# Renewable capacity moves on the device against the CPU reference (mock random numbers, α = 1).
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice
include(joinpath(@__DIR__, "mockrng.jl"))
include(joinpath(@__DIR__, "setup_all_terms.jl"))
Pb = ML.with_bias(P, true)
rng = MersenneTwister(6)
Pb.bias.θ .= 0.05 .* randn(rng, length(Pb.bias.θ))
Pb.bias.a .= 0.5 .* randn(rng, length(Pb.bias.a)); Pb.bias.c .= 0.1 .* abs.(randn(rng, length(Pb.bias.c)))
sb = ML.evaluate!(Pb, copy(sL))
C, nprop, σ = 2, 2, 0.3
units = D.ren_units(Pb)
nr = length(units)
Z, U = randn(rng, C, nr, nprop), rand(rng, C, nr, nprop)
cpu, LCs = ML.State[], ML.LocalCost[]
for c in 1:C
    s = copy(sb); LM = ML.LocalModel(Pb, L); LC = ML.LocalCost(Pb, LM, s)
    s, _ = ML.kscale_sweep!(MockRNG(vec(permutedims(Z[c, :, :])), vec(permutedims(U[c, :, :]))), Pb, LM, s;
                            LC=LC, mh=true, σ=σ, nprop=nprop, order=:fixed, kinds=:renewables)
    push!(cpu, s); push!(LCs, LC)
end
S = D.to_device(Pb, M, ix, [sb for _ in 1:C])
RP, RS = D.pack_ren(Pb, M, [sb for _ in 1:C])
BD, BS, _, _, ritem = D.pack_bias(Pb, ix, [sb for _ in 1:C])
function dev(RP, RS, r, dnet, cren, capren, Z, U, rq, BD, BS, ritem, G)
    RS, r, dnet, cren, capren, BS = D.kren_mh_loop(RP, RS, r, dnet, cren, capren, Z, U, rq; σ=σ, βk=Pb.βk, βv=Pb.βv,
                                                   bias=(BD, BS, ritem))
    return RS, r, dnet, cren, capren, BS, D.peak_sets_dev(dnet, G, Val(max(1, round(Int, 0.05 * size(dnet, 2)))))
end
function compare(out, label)
    RSo, ro, dno, cro, capo, BSo, inSo = map(x -> x isa D.RenState || x isa D.BiasState ? x : Array(x), out)
    @testset "$label" begin
        LMc = ML.LocalModel(Pb, L)
        for c in 1:C
            Sc = D.to_device(Pb, M, ix, ML.State[cpu[c]])
            _, RSc = D.pack_ren(Pb, M, ML.State[cpu[c]])
            ekr = maximum(abs.(Array(RSo.kr)[c, :] .- RSc.kr[1, :]) ./ max.(1.0, abs.(RSc.kr[1, :])))
            er = maximum(abs.(ro[c, :, :]' .- LCs[c].r)); edn = maximum(abs.(dno[c, :, :] .- Sc.dnet[1, :, :]))
            ecr = abs(cro[c] - Sc.cren[1]) / abs(Sc.cren[1]); ecap = abs(capo[c] - Sc.capren[1]) / abs(Sc.capren[1])
            @printf("  chain %d: k_ren %.1e, r %.1e MW, dnet %.1e MW, cren %.1e, capren %.1e (rel)\n", c, ekr, er, edn, ecr, ecap)
            @test ekr < 1e-12 && er < 1e-6 && edn < 1e-6 && ecr < 1e-12 && ecap < 1e-12
            # peak sets of the gas units after the moves = CPU peak_set on the CPU state
            LCc = ML.LocalCost(Pb, LMc, cpu[c]); Xn = ML.weather_nodes(Pb.W, L, cpu[c].xs[end])
            nbad = 0
            for (q, (gi, i)) in enumerate(ix.gidx)
                Sset = Set(ML.peak_set(Pb, LMc, LCc, Xn, Pb.E.gens[gi].bus[i]))
                nbad += Set(findall(inSo[c, :, q])) != Sset
            end
            @test nbad == 0
        end
        _, BSc, _, _, _ = D.pack_bias(Pb, ix, cpu)
        @test maximum(abs.(Array(BSo.u1x) .- BSc.u1x)) < 1e-9 && maximum(abs.(Array(BSo.zc) .- BSc.zc)) < 1e-6
    end
end
moved(RSo) = [count(Array(RSo.kr)[c, :] .!= RS.kr[c, :]) for c in 1:C]
RS1 = deepcopy(RS); BS1 = deepcopy(BS)
out = dev(RP, RS1, D.rebuild_r(M, S), copy(S.dnet), copy(S.cren), copy(S.capren), Z, U, collect(1:nr), BD, BS1, ritem, M.G)
@printf("moved renewable units per chain: %s of %d\n", string(moved(out[1])), nr)
compare(out, "renewable kscale (Julia arrays) = CPU reference")
Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
args = map(Reactant.to_rarray, (RP, deepcopy(RS), D.rebuild_r(M, S), copy(S.dnet), copy(S.cren), copy(S.capren), Z, U,
                                collect(1:nr), BD, deepcopy(BS), ritem, M.G))
tc = @elapsed f = @compile dev(args...)
@printf("Reactant (%s): compile %.1f s\n", get(ENV, "REACTANT_BACKEND", "cpu"), tc)
compare(f(args...), "renewable kscale compiled by Reactant = CPU reference")
