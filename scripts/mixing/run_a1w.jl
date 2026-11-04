# α = 1 at fixed capacities (the heuristic start; the level-0 bias of run_a1.jl, frozen; the
# utilization measure; own weather draws): operation and fine weather moves only, so that the
# capacity part of the bias plays no role. Records the weather traces; the end states are the
# input of scripts/mixing/run_a1_weather.jl. Usage:
#   nice -n 19 julia --project=. -t 4 scripts/mixing/run_a1w.jl [nsweeps]
include(joinpath(@__DIR__, "driver.jl"))
using Serialization
nsw = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
cfg, P0, sL, info = mix_setup("configs/chains40.toml"; measure="utilization")
P1 = deserialize("results/mixing/a1u_problem.jls")
chains = [ML.chain_start(MersenneTwister(cfg.seed + 7 * c), P1, sL, "heuristic", true) for c in 1:cfg.nchains]
foreach(s -> ML.evaluate!(P1, s), chains)
rec = run_mix(cfg, P1, sL; nsweeps=nsw, out="results/mixing/a1w.h5", label="a1w", every=10, chains=chains, capacity=false)
save_rec("results/mixing/a1w.h5", rec, nsw)
serialize("results/mixing/a1w_chains.jls", rec["chains"])
println("done, wall ", rec["wall"])
