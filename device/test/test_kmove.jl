# Device capacity moves against the CPU reference with mock random numbers (no statistics):
# the same proposal normals and uniforms drive `kscale_sweep!(…; mh=true)` on the CPU and
# `kscale_mh!` on the device (Julia arrays, then Reactant); the states must agree.
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice

"An RNG that returns pre-drawn normals (randn) and uniforms (rand) in order."
mutable struct MockRNG <: Random.AbstractRNG
    z::Vector{Float64}
    u::Vector{Float64}
    iz::Int
    iu::Int
end
MockRNG(z, u) = MockRNG(z, u, 0, 0)
Random.randn(r::MockRNG) = (r.iz += 1; r.z[r.iz])
Base.rand(r::MockRNG) = (r.iu += 1; r.u[r.iu])

cfg = ML.load_config(joinpath(@__DIR__, "..", "..", "configs", "haar_poc.toml")); cfg.verbose = false
cfg.data = joinpath(@__DIR__, "..", "..", "data", "processed", "model.h5")
cfg.out = joinpath(@__DIR__, "..", "..", "results", "haar_poc.h5")
cfg.storage = "soc"; cfg.renewables = "load"; cfg.measure = "physical"; cfg.init = "heuristic"
cfg.losses = true; cfg.thermal = "soft"; cfg.link_losses = true; cfg.uc = ["gas", "nuclear"]; cfg.use_bias = false
P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
L = sL.ℓ
M, ix = D.pack(P, L)
units = D.kmove_units(M)
nu, nprop, C = length(units), 2, 2
σ = 0.3
rng = MersenneTwister(5)
Z = randn(rng, C, nu, nprop)
U = rand(rng, C, nu, nprop)

# CPU reference, one run per chain stream
cpu = ML.State[]
LCs = ML.LocalCost[]
for c in 1:C
    s = copy(sL)
    LM = ML.LocalModel(P, L)
    LC = ML.LocalCost(P, LM, s)
    zs = vec(permutedims(Z[c, :, :]))             # unit-major: (unit 1: z₁, z₂), (unit 2: …)
    us = vec(permutedims(U[c, :, :]))
    s, nm = ML.kscale_sweep!(MockRNG(zs, us), P, LM, s; LC=LC, mh=true, σ=σ, nprop=nprop, order=:fixed, kinds=:dispatch)
    push!(cpu, s); push!(LCs, LC)
end

function compare(S, r, label)
    @testset "$label" begin
        for c in 1:C
            Sh = ML.State[cpu[c]]
            Sc = D.to_device(P, M, ix, Sh)                     # CPU result in device layout
            for (fld, a, b) in (("kg", Array(S.kg)[c, :], Sc.kg[1, :]), ("ks", Array(S.ks)[c, :], Sc.ks[1, :]),
                                ("ω", Array(S.ω)[c, :, :], Sc.ω[1, :, :]), ("uc", Array(S.uc)[c, :, :], Sc.uc[1, :, :]),
                                ("e", Array(S.e)[c, :, :], Sc.e[1, :, :]))
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

# device, Julia arrays
S = D.to_device(P, M, ix, [sL for _ in 1:C])
r0 = D.rebuild_r(M, S)
S, r, nacc = D.kscale_mh!(M, S, r0, Z, U; units=units, σ=σ, βk=P.βk, βv=P.βv, physical=true)
@printf("accepted moves per chain: %s of %d units\n", string(Int.(nacc)), nu)
compare(S, r, "kscale MH: device (Julia arrays) = CPU reference")

Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))   # CPU for fast compiles; "gpu" for timings
# the device loop on Julia arrays (same code path as traced), then compiled by Reactant
gq = [q for (k, q) in units if k == 1]; sq = [q for (k, q) in units if k == 2]
S2 = D.to_device(P, M, ix, [sL for _ in 1:C])
S2, r2 = D.kscale_mh_loop(D.KParams(M), S2, D.rebuild_r(M, S2), Z, U, gq, sq; σ=σ, βk=P.βk, βv=P.βv, physical=true)
compare(S2, r2, "kscale MH device loop (Julia arrays) = CPU reference")
S3 = D.to_device(P, M, ix, [sL for _ in 1:C])
r3 = D.rebuild_r(M, S3)
Mr, Sr, rr = Reactant.to_rarray(D.KParams(M)), Reactant.to_rarray(S3), Reactant.to_rarray(r3)
Zr, Ur, gqr, sqr = Reactant.to_rarray(Z), Reactant.to_rarray(U), Reactant.to_rarray(gq), Reactant.to_rarray(sq)
kfun(M_, S_, r_, Z_, U_, g_, s_) = D.kscale_mh_loop(M_, S_, r_, Z_, U_, g_, s_; σ=σ, βk=P.βk, βv=P.βv, physical=true)
tc = @elapsed f = @compile kfun(Mr, Sr, rr, Zr, Ur, gqr, sqr)
tr = @elapsed (So, ro) = f(Mr, Sr, rr, Zr, Ur, gqr, sqr)
@printf("Reactant (%s): compile %.1f s, run %.3f s (%d units × %d proposals × %d chains)\n", get(ENV, "REACTANT_BACKEND", "cpu"), tc, tr, nu, nprop, C)
compare(So, ro, "kscale MH compiled by Reactant = CPU reference")
