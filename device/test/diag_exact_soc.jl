# Diagnostic for test (v) (CPU only, ≈ 10 min): is the storage/dump shift of the SoC coordinate
# moves a device bug, a CPU-reference equilibration problem or a CPU-kernel bias?
#   E  the device soc schedule emulated on the CPU (cpu_move + sample_line! + commit_block!)
#   A  the CPU plain kernel continued from the same equilibrium starts (400/800/1200 sweeps)
#   B  the CPU plain kernel continued from the states after E (does the shift relax back?)
# Paired changes against the starts, mean ± s.e. over chains.
# Statistical many-chain run: off by default (test strategy: deterministic comparisons with mock
# uniforms, test_mock.jl). RUN_STAT=1 runs it.
get(ENV, "RUN_STAT", "0") == "1" || (println("skipped (set RUN_STAT=1)"); exit(0))
using ESMDevice, ESMSampling, Random, Printf, Statistics
const D = ESMDevice
const ML = ESMSampling
ENV["EXACT_SKIP"] = "1"
include(joinpath(@__DIR__, "..", "..", "test", "test_exact.jl"))
include(joinpath(@__DIR__, "helpers.jl"))

const NC = parse(Int, get(ENV, "NCHAINS", "64"))
global P_, s0 = exact_setup()
P = P_
LM = MX.LocalModel(P, 1)
M, ix = D.pack(P, 1)
pl = D.plan(M, D.to_device(P, M, ix, [s0]), P.βv)
β = P.βv
plain = [(:unit, 0.3), (:pair, 0.25), (:xpair, 0.15), (:potential, 0.1), (:cellpair, 0.1), (:cell, 0.1), (:spair, 0.3)]
rng = MersenneTwister(12)
ref(x, r=rng) = MX.local_sweep!(r, P, LM, x; extra=4.0, families=plain)[1]
x = copy(s0); for _ in 1:300; global x = ref(x); end
starts = MX.State[]
for i in 1:NC; for _ in 1:8; global x = ref(x); end; push!(starts, copy(x)); end
q(s) = [s.parts.op, s.parts.storage, s.parts.shed, s.parts.dump]
names = ("op", "storage", "shed", "dump")
report(tag, d) = (print(rpad(tag, 44)); for j in 1:4; @printf("  %s %.3g ± %.2g (z %4.1f)", names[j], mean(d[:, j]), std(d[:, j]) / sqrt(NC), mean(d[:, j]) / (std(d[:, j]) / sqrt(NC))); end; println())

function cpu_emulate!(s, rounds, fam, nsw, r)
    LC = MX.LocalCost(P, LM, s)
    for _ in 1:nsw, rr in rounds, site in rr
        bm = cpu_move(P, LM, LC, ix, fam, site)
        bm === nothing && continue
        MX.commit_block!(bm, s.v, MX.sample_line!(r, bm, β, Float64[]))
    end
    return MX.evaluate!(P, s)
end

q0 = reduce(hcat, q.(starts))'
dA = [zeros(NC, 4) for _ in 1:3]
dE = zeros(NC, 4); dB = zeros(NC, 4)
for i in 1:NC
    r = MersenneTwister(5000 + i)
    y = copy(starts[i])
    for blk in 1:3
        for _ in 1:400; y = ref(y, r); end
        dA[blk][i, :] .= q(y) .- q0[i, :]
    end
    z = cpu_emulate!(copy(starts[i]), pl.sched[:soc], :soc, 200, MersenneTwister(9000 + i))
    dE[i, :] .= q(z) .- q0[i, :]
    for _ in 1:400; z = ref(z, r); end
    dB[i, :] .= q(z) .- q0[i, :]
end
report("E: device soc schedule (CPU), 200 sweeps", dE)
for blk in 1:3; report("A: CPU plain kernel, $(400blk) sweeps", dA[blk]); end
report("B: E, then CPU plain kernel 400 sweeps", dB)
