# Mixing of the operation at fixed capacities and weather (chains40, 40 nodes daily, local kernel):
# integrated autocorrelation times of the quantities the samplers use, and of slow space-time
# directions of the storage. Operation sweeps exactly as in the fine step (local sweep, storage
# multiscale sweep with probability 1/2); no capacity or weather moves, so the bias plays no role.
#
# Usage: nice -n 19 julia --project=. -t 4 opmix.jl <chains.jls> <chain index> <nsweeps> <out.jls> [kernel] [nmoves] [nx] [sx]
# kernel "base" (default, ledger E17): local sweep + storage multiscale sweep (probability 1/2);
# "slong": the same plus the long-range storage sweep every sweep (slong, slevel; nmoves plateaus
# per unit, default 32; ledger E21); "sx": the slong kernel plus the storage pairs (ledger E27):
# nx interval pairs sxlong per unit in the long-range sweep (default 8) and the single-instant
# pair sxpair at rate sx in the local sweep (default 0.1).
using Random, Statistics, Printf, Serialization
import ESMSampling as ML

chainsfile, ci, nsw, out = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4]
kernel = length(ARGS) >= 5 ? ARGS[5] : "base"
nlong = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 32
nx = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 8
sx = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 0.1
kernel == "sx" || (nx = 0; sx = 0.0)
cfg = ML.load_config("configs/chains40.toml"); cfg.verbose = false
opmix_start(P, sL) = deserialize(chainsfile)[ci]
include(joinpath(@__DIR__, "opmix_lib.jl"))

rng = MersenneTwister(1000 + ci)
rec = new_record(kernel, nlong, nx, sx)
t0 = time()
for i in 1:nsw
    a = time()
    global s
    s, lst = opmix_sweep!(rng, s, kernel, nlong, nx, sx)
    q = record!(rec, s, time() - a, lst)
    if i % 50 == 0
        @printf("chain %d sweep %d: opex %.4e shed %.3e  [%.0f s, %.2f s/sweep]\n", ci, i, q[iop], q[ish], time() - t0,
                mean(rec["time"][end-49:end])); flush(stdout)
        serialize(out, rec)
    end
end
serialize(out, rec)
println("done ", time() - t0)
