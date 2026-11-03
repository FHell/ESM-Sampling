# Profile of the local sweep (1 thread): flat profile of the hottest frames.
# Usage: nice -n 19 julia -t 1 --project=. scripts/prof_kernel.jl [data] [N] [fams: flow|local|atoms]
using ESMSampling, Random, Profile, Printf
const M = ESMSampling
data = length(ARGS) > 0 ? ARGS[1] : "data/processed/model.h5"
N = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 40
kind = length(ARGS) > 2 ? ARGS[3] : "flow"
cfg = M.load_config("configs/flow.toml"); cfg.verbose = false
cfg.data = data; cfg.levels = [(10, 24), (N, 24)]; cfg.init = "heuristic"; cfg.flow_theta = "haar"; cfg.out = "results/bench.h5"
P, s, info = M.setup(cfg; rng=MersenneTwister(1))
LM = M.LocalModel(P, 1); FF = M.FlowFrames(P, 1, s)
wsets = [M.MoveWS(P) for _ in 1:2Threads.maxthreadid()]
rng = MersenneTwister(3)
fams, mult = M.flow_families(P.E)
sw(x) = kind == "flow" ? M.local_sweep!(rng, P, LM, x; wsets=wsets, families=fams, extra=4.0 * mult, frames=FF, atoms=false)[1] :
        kind == "atoms" ? M.local_sweep!(rng, P, LM, x; wsets=wsets, extra=0.0)[1] :
        M.local_sweep!(rng, P, LM, x; wsets=wsets, atoms=false)[1]
s = sw(s)
Profile.clear()
Profile.init(n=10^7, delay=0.0005)
@profile (s = sw(s))
Profile.print(format=:flat, sortedby=:count, mincount=40, C=false, noisefloor=1.0)
