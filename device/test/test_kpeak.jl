# Device peak-set capacity moves against the CPU reference with mock random numbers.
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice
include(joinpath(@__DIR__, "mockrng.jl"))
include(joinpath(@__DIR__, "setup_all_terms.jl"))     # P, sL, L, M, ix (40 nodes, all model terms, α = 0)

gq = [q for (k, q) in D.kmove_units(M) if k == 1]
nu, nprop, C = length(gq), 2, 2
rng = MersenneTwister(9)
Z, U = randn(rng, C, nu, nprop), rand(rng, C, nu, nprop)
S0 = D.to_device(P, M, ix, [sL])
σk = 0.05 .* max.(S0.kg[1, gq], 100.0)                 # a fixed per-unit scale
# peak sets on the host (CPU `peak_set`), identical for both chains here
LM = ML.LocalModel(P, L); LC = ML.LocalCost(P, LM, sL)
Xn = ML.weather_nodes(P.W, L, sL.xs[end])
inS = falses(C, LM.T, length(ix.gidx))
for (q, (gi, i)) in enumerate(ix.gidx)
    inS[:, ML.peak_set(P, LM, LC, Xn, P.E.gens[gi].bus[i]), q] .= true
end
cpu, LCs = ML.State[], ML.LocalCost[]
for c in 1:C
    s = copy(sL); LMc = ML.LocalModel(P, L); LCc = ML.LocalCost(P, LMc, s)
    s, _ = ML.kpeak_sweep!(MockRNG(vec(permutedims(Z[c, :, :])), vec(permutedims(U[c, :, :]))), P, LMc, s;
                           LC=LCc, mh=true, σk=σk, nprop=nprop, order=:fixed)
    push!(cpu, s); push!(LCs, LCc)
end
function compare(S, r, label)
    @testset "$label" begin
        for c in 1:C
            Sc = D.to_device(P, M, ix, ML.State[cpu[c]])
            for (fld, a, b) in (("kg", Array(S.kg)[c, :], Sc.kg[1, :]), ("ω", Array(S.ω)[c, :, :], Sc.ω[1, :, :]),
                                ("uc", Array(S.uc)[c, :, :], Sc.uc[1, :, :]))
                err = maximum(abs.(a .- b) ./ max.(1.0, abs.(b)))
                @printf("  chain %d %-3s max rel diff %.2e\n", c, fld, err)
                @test err < 1e-12
            end
            rerr = maximum(abs.(Array(r)[c, :, :]' .- LCs[c].r))
            @printf("  chain %d r   max abs diff %.2e MW\n", c, rerr)
            @test rerr < 1e-6
        end
    end
end
S = D.to_device(P, M, ix, [sL for _ in 1:C])
S, r = D.kpeak_mh_loop(D.KParams(M), S, D.rebuild_r(M, S), Z, U, gq, σk, inS; βk=P.βk, βv=P.βv, physical=true)
@printf("moved units per chain: %s of %d\n", string([count(Array(S.kg)[c, gq] .!= S0.kg[1, gq]) for c in 1:C]), nu)
compare(S, r, "kpeak MH (Julia arrays) = CPU reference")
Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
S3 = D.to_device(P, M, ix, [sL for _ in 1:C]); r3 = D.rebuild_r(M, S3)
args = map(Reactant.to_rarray, (D.KParams(M), S3, r3, Z, U, gq, σk, inS))
kf(a...) = D.kpeak_mh_loop(a...; βk=P.βk, βv=P.βv, physical=true)
tc = @elapsed f = @compile kf(args...)
tr = @elapsed (So, ro) = f(args...)
@printf("Reactant (%s): compile %.1f s, run %.3f s\n", get(ENV, "REACTANT_BACKEND", "cpu"), tc, tr)
compare(So, ro, "kpeak MH compiled by Reactant = CPU reference")
