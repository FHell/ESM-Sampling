# Replica Langevin run on chains40 (ledger E24): M weather replicas sharing k, weather by staggered
# pCN bursts with score gating, Langevin capacity step with the lag probe adapting the step per
# carrier. From the repository root:
#   nice -n 19 julia --project=. -t 4 scripts/replicas/run_replicas.jl <low|high> <nsteps> <out.jls> [seed] [λ] [state.jls] [ηw]
# With a state file the run continues from its replicas and random streams (the tilt
# demonstration; with ηw given, at that fixed step for every carrier, no adaptation), otherwise it
# starts from the heuristic capacities ×0.5 (low) or ×2 (high) with M fresh prior weather draws.
include(joinpath(@__DIR__, "..", "langevin", "setup40.jl"))
include(joinpath(@__DIR__, "replicas.jl"))

kind, nsteps, out = ARGS[1], parse(Int, ARGS[2]), ARGS[3]
seed = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : (kind == "low" ? 1 : 2)
λ = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.0
from = length(ARGS) >= 6 ? ARGS[6] : ""
M = 4
κs = [500.0, 1000.0, 200.0, 300.0, 100.0, 100.0]          # onwind, solar, offwind, gas, battery, H2 (MW)
kref = ML.knodes(P, sL.ℓ, sL.k)                            # reference capacities of the weather features (heuristic)
# declared tilt functional: annual mean residual load at the reference capacities, standardised
# by its prior mean and sd (2000 exact draws, results/replicas/prior_test.jls)
const μΦ, σΦ = 55.4811, 3.5384
Φres(x) = (wfeatures(P, x, kref).resmean - μΦ) / σΦ
F = VarFactors(P.W)
wrngs = [MersenneTwister(1000seed + m) for m in 1:M]
rng = MersenneTwister(10seed + 3)
prng = MersenneTwister(10seed + 5)

t0 = time()
if isempty(from)
    s0 = ML.evaluate!(P, ML.chain_start(MersenneTwister(seed), P, sL, kind, false))
    LC0r = copy(ML.LocalCost(P, LM, s0).r)
    reps = ML.State[]
    for m in 1:M                                           # fresh prior weather per replica, compensated, 40 sweeps at fixed k
        mm, z0, ws = ML.sample_weather(wrngs[m], P.W, s0.ℓ)
        s = compensate!(P, LM, LC0r, ML.make_state(P, s0.ℓ, mm, z0, ws, s0.k, s0.v))
        for _ in 1:40
            s = lgv_op!(rng, P, S.loc, LM, s)
        end
        push!(reps, s)
        @printf("replica %d warm: op %.4e shed %.3g MWh (%.0f s)\n", m, s.parts.op, s.parts.shed, time() - t0); flush(stdout)
    end
    ctl = RepCtl(; ηw=fill(5e-4, 6), κs, clip=0.02, box=true, burst_every=40, φ=0.3, gate=25, burst_regime=0.5,
                 regmode=:window, regB=30, λ, Φ=Φres, probe_every=20, probe_len=30, probe_tail=15, probe_win=10, adapt=true)
    trace = NamedTuple[]; probes = NamedTuple[]; hist = nothing; since = nothing; it0 = 0
else
    D = deserialize(from)
    reps, trace0 = D.reps, D.trace
    ctl = RepCtl(; merge(D.ctl, (; λ, Φ=Φres))...)
    wrngs, rng, prng = D.wrngs, D.rng, D.prng            # the same streams as the run it continues
    trace = NamedTuple[]; probes = NamedTuple[]; hist = D.hist; since = D.since; it0 = trace0[end].it
    if length(ARGS) >= 7
        ctl.ηw .= parse(Float64, ARGS[7]); ctl.adapt = false; ctl.probe_every = 0
    end
end
chunk = 50
done = 0
while done < nsteps
    n = min(chunk, nsteps - done)
    global reps, hist, since = rep_run!(rng, wrngs, prng, P, S, reps, n, ctl; trace, probes, it0=it0 + done, log_every=10,
                                        label="$kind λ=$λ", F, kref, hist, since)
    global done += n
    serialize(out, (; kind, seed, λ, M, reps, trace, probes, ctl=(; (f => getfield(ctl, f) for f in fieldnames(RepCtl) if f != :Φ)...),
                    hist, since, wrngs, rng, prng, t=time() - t0))
end
@printf("done: %d steps in %.0f s\n", nsteps, time() - t0)
