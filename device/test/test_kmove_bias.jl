# α = 1: device capacity moves with the bias against the CPU reference, mock random numbers.
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice
include(joinpath(@__DIR__, "mockrng.jl"))
include(joinpath(@__DIR__, "setup_all_terms.jl"))
Pb = ML.with_bias(P, true)
rng = MersenneTwister(4)
Pb.bias.θ .= 0.05 .* randn(rng, length(Pb.bias.θ))
Pb.bias.a .= 0.5 .* randn(rng, length(Pb.bias.a)); Pb.bias.c .= 0.1 .* abs.(randn(rng, length(Pb.bias.c)))
sb = ML.evaluate!(Pb, copy(sL))
units = D.kmove_units(M)
gq = [q for (k, q) in units if k == 1]; sq = [q for (k, q) in units if k == 2]
C, nprop, σ = 2, 2, 0.3
Z, U = randn(rng, C, length(units), nprop), rand(rng, C, length(units), nprop)
Zp, Up = randn(rng, C, length(gq), nprop), rand(rng, C, length(gq), nprop)
S0 = D.to_device(Pb, M, ix, [sb])
σk = 0.05 .* max.(S0.kg[1, gq], 100.0)
LM = ML.LocalModel(Pb, L); LC = ML.LocalCost(Pb, LM, sb)
Xn = ML.weather_nodes(Pb.W, L, sb.xs[end])
inS = falses(C, LM.T, length(ix.gidx))
for (q, (gi, i)) in enumerate(ix.gidx)
    inS[:, ML.peak_set(Pb, LM, LC, Xn, Pb.E.gens[gi].bus[i]), q] .= true
end
# CPU: kscale then kpeak, per chain stream
cpu, LCs = ML.State[], ML.LocalCost[]
for c in 1:C
    s = copy(sb); LMc = ML.LocalModel(Pb, L); LCc = ML.LocalCost(Pb, LMc, s)
    s, _ = ML.kscale_sweep!(MockRNG(vec(permutedims(Z[c, :, :])), vec(permutedims(U[c, :, :]))), Pb, LMc, s;
                            LC=LCc, mh=true, σ=σ, nprop=nprop, order=:fixed, kinds=:dispatch)
    s, _ = ML.kpeak_sweep!(MockRNG(vec(permutedims(Zp[c, :, :])), vec(permutedims(Up[c, :, :]))), Pb, LMc, s;
                           LC=LCc, mh=true, σk=σk, nprop=nprop, order=:fixed)
    push!(cpu, s); push!(LCs, LCc)
end
function compare(S, r, label)
    @testset "$label" begin
        for c in 1:C
            Sc = D.to_device(Pb, M, ix, ML.State[cpu[c]])
            for (fld, a, b) in (("kg", Array(S.kg)[c, :], Sc.kg[1, :]), ("ks", Array(S.ks)[c, :], Sc.ks[1, :]),
                                ("ω", Array(S.ω)[c, :, :], Sc.ω[1, :, :]), ("e", Array(S.e)[c, :, :], Sc.e[1, :, :]))
                err = maximum(abs.(a .- b) ./ max.(1.0, abs.(b)))
                @printf("  chain %d %-3s max rel diff %.2e\n", c, fld, err)
                @test err < 1e-12
            end
            @test maximum(abs.(Array(r)[c, :, :]' .- LCs[c].r)) < 1e-6
        end
    end
end
function device_run(KP, S, r, Z, U, gq, sq, Zp, Up, σk, inS, bias)
    S, r, BS = D.kscale_mh_loop(KP, S, r, Z, U, gq, sq; σ=σ, βk=Pb.βk, βv=Pb.βv, physical=true, bias=bias)
    BD, _, gi, si = bias
    S, r, BS = D.kpeak_mh_loop(KP, S, r, Zp, Up, gq, σk, inS; βk=Pb.βk, βv=Pb.βv, physical=true, bias=(BD, BS, gi, si))
    return S, r, BS
end
bias = D.pack_bias(Pb, ix, [sb for _ in 1:C])[1:4]
S = D.to_device(Pb, M, ix, [sb for _ in 1:C])
S, r, BS = device_run(D.KParams(M), S, D.rebuild_r(M, S), Z, U, gq, sq, Zp, Up, σk, inS, bias)
@printf("moved: kg %s of %d units\n", string([count(Array(S.kg)[c, :] .!= S0.kg[1, :]) for c in 1:C]), length(gq))
compare(S, r, "α = 1 kscale + kpeak (Julia arrays) = CPU reference")
# the aggregates after the moves equal a fresh pack from the CPU states
_, BSc, _, _, _ = D.pack_bias(Pb, ix, cpu)
@test maximum(abs.(BS.u2x .- BSc.u2x)) < 1e-9 && maximum(abs.(BS.zc .- BSc.zc)) < 1e-6
Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
S3 = D.to_device(Pb, M, ix, [sb for _ in 1:C])
args = map(Reactant.to_rarray, (D.KParams(M), S3, D.rebuild_r(M, S3), Z, U, gq, sq, Zp, Up, σk, inS, D.pack_bias(Pb, ix, [sb for _ in 1:C])[1:4]))
tc = @elapsed f = @compile device_run(args...)
(So, ro, _) = f(args...)
@printf("Reactant (%s): compile %.1f s\n", get(ENV, "REACTANT_BACKEND", "cpu"), tc)
compare(So, ro, "α = 1 kscale + kpeak compiled by Reactant = CPU reference")
