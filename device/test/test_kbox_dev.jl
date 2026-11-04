# Device capacity moves at the capacity box against the CPU reference with mock random
# numbers (no statistics): from a start with the raw capacities of every second
# dispatchable and storage unit 1.6 × above their box, the same proposals drive
# `kscale_sweep!(…; mh=true)` / `kpeak_sweep!(…; mh=true)` on the CPU and `kscale_mh_loop` /
# `kpeak_mh_loop` on the device (Julia arrays, then Reactant on the CPU backend); the states
# must agree, and the raw capacities must move back toward the box.
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice
include(joinpath(@__DIR__, "mockrng.jl"))
include(joinpath(@__DIR__, "setup_all_terms.jl"))     # P, sL, L, M, ix (40 nodes, all model terms, α = 0)

E = P.E
s0 = copy(sL)
lk = P.lk[L+1]
"Raw node capacity of unit i of field kf set to knew."
function set_raw!(s, kf, i, knew)
    Ut = E.kspecs[kf].B.Ut
    Δ = knew - ML.knodes(P, L, s.k)[kf][i]
    for q in ML.nzrange(Ut, i)
        s.k[lk.off[kf]+ML.rowvals(Ut)[q]] += Δ * ML.nonzeros(Ut)[q]
    end
end
abovef = Int[]
for (kind, items) in ((1, [(g.kf, i) for g in E.gens if g.kf > 0 && g.vf > 0 for i in eachindex(g.bus)]),
                      (2, [(st.kf, i) for st in E.stors if st.kf > 0 for i in eachindex(st.bus)]))
    for (j, (kf, i)) in enumerate(items)
        hi = E.kspecs[kf].hi[i]
        (isodd(j) && isfinite(hi) && hi > 100) || continue
        set_raw!(s0, kf, i, 1.6hi); push!(abovef, kf)
    end
end
s0 = ML.evaluate!(P, s0)
excess(s) = sum(sum(max.(ML.knodes(P, L, s.k)[kf] .- E.kspecs[kf].hi, 0.0)) for kf in unique(abovef))
@printf("raw excess above the box at the start: %.1f GW\n", excess(s0) / 1e3)

units = D.kmove_units(M)
gq = [q for (k, q) in units if k == 1]; sq = [q for (k, q) in units if k == 2]
C, nprop = 2, 2
rng = MersenneTwister(15)
Zs, Us = randn(rng, C, length(units), nprop), rand(rng, C, length(units), nprop)
Zp, Up = randn(rng, C, length(gq), nprop), rand(rng, C, length(gq), nprop)
σ = 0.3
σk = 0.3 .* max.(D.to_device(P, M, ix, [s0]).kg[1, gq], 100.0)
LM = ML.LocalModel(P, L); LC0 = ML.LocalCost(P, LM, s0)
Xn = ML.weather_nodes(P.W, L, s0.xs[end])
inS = falses(C, LM.T, length(ix.gidx))
for (q, (gi, i)) in enumerate(ix.gidx)
    inS[:, ML.peak_set(P, LM, LC0, Xn, E.gens[gi].bus[i]), q] .= true
end

# CPU reference: kscale (dispatchable then storage, fixed order) then kpeak, per chain stream
cpu1, cpu2, LCs = ML.State[], ML.State[], ML.LocalCost[]
for c in 1:C
    s = copy(s0); LMc = ML.LocalModel(P, L); LCc = ML.LocalCost(P, LMc, s)
    s, _ = ML.kscale_sweep!(MockRNG(vec(permutedims(Zs[c, :, :])), vec(permutedims(Us[c, :, :]))), P, LMc, s;
                            LC=LCc, mh=true, σ=σ, nprop=nprop, order=:fixed, kinds=:dispatch)
    push!(cpu1, copy(s))
    s, _ = ML.kpeak_sweep!(MockRNG(vec(permutedims(Zp[c, :, :])), vec(permutedims(Up[c, :, :]))), P, LMc, s;
                           LC=LCc, mh=true, σk=σk, nprop=nprop, order=:fixed)
    push!(cpu2, s); push!(LCs, LCc)
    @printf("chain %d: excess after kscale %.1f GW, after kpeak %.1f GW\n", c, excess(cpu1[c]) / 1e3, excess(s) / 1e3)
end
@test all(excess(s) < 0.8 * excess(s0) for s in cpu2)

function compare(S, r, ref, label; LCr=nothing)
    @testset "$label" begin
        for c in 1:C
            Sc = D.to_device(P, M, ix, ML.State[ref[c]])
            for (fld, a, b) in (("kg", Array(S.kg)[c, :], Sc.kg[1, :]), ("ks", Array(S.ks)[c, :], Sc.ks[1, :]),
                                ("ω", Array(S.ω)[c, :, :], Sc.ω[1, :, :]), ("uc", Array(S.uc)[c, :, :], Sc.uc[1, :, :]),
                                ("e", Array(S.e)[c, :, :], Sc.e[1, :, :]))
                err = maximum(abs.(a .- b) ./ max.(1.0, abs.(b)))
                @printf("  chain %d %-3s max rel diff %.2e\n", c, fld, err)
                @test err < 1e-12
            end
            if LCr !== nothing
                rerr = maximum(abs.(Array(r)[c, :, :]' .- LCr[c].r))
                @printf("  chain %d r   max abs diff %.2e MW\n", c, rerr)
                @test rerr < 1e-6
            end
        end
    end
end

# (1) device code on Julia arrays
S = D.to_device(P, M, ix, [s0 for _ in 1:C])
S, r = D.kscale_mh_loop(D.KParams(M), S, D.rebuild_r(M, S), Zs, Us, gq, sq; σ=σ, βk=P.βk, βv=P.βv, physical=true)
compare(S, r, cpu1, "kscale at the box: device loop (Julia arrays) = CPU reference")
S, r = D.kpeak_mh_loop(D.KParams(M), S, r, Zp, Up, gq, σk, inS; βk=P.βk, βv=P.βv, physical=true)
compare(S, r, cpu2, "kscale + kpeak at the box: device loop (Julia arrays) = CPU reference"; LCr=LCs)

# (2) the same code compiled by Reactant on the CPU backend
Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
S3 = D.to_device(P, M, ix, [s0 for _ in 1:C]); r3 = D.rebuild_r(M, S3)
KPr = Reactant.to_rarray(D.KParams(M))
a1 = map(Reactant.to_rarray, (S3, r3, Zs, Us, gq, sq))
kf1(K_, S_, r_, Z_, U_, g_, s_) = D.kscale_mh_loop(K_, S_, r_, Z_, U_, g_, s_; σ=σ, βk=P.βk, βv=P.βv, physical=true)
tc1 = @elapsed f1 = @compile kf1(KPr, a1...)
So, ro = f1(KPr, a1...)
compare(So, ro, cpu1, "kscale at the box compiled by Reactant = CPU reference")
a2 = map(Reactant.to_rarray, (Zp, Up, gq, σk, inS))
kf2(K_, S_, r_, Z_, U_, g_, σ_, i_) = D.kpeak_mh_loop(K_, S_, r_, Z_, U_, g_, σ_, i_; βk=P.βk, βv=P.βv, physical=true)
tc2 = @elapsed f2 = @compile kf2(KPr, So, ro, a2...)
So2, ro2 = f2(KPr, So, ro, a2...)
@printf("Reactant compile: kscale %.1f s, kpeak %.1f s\n", tc1, tc2)
compare(So2, ro2, cpu2, "kscale + kpeak at the box compiled by Reactant = CPU reference"; LCr=LCs)
