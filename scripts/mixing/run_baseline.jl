# Baseline: the chains40 screening (α = 0, local kernel, fine weather 0.05) for many more sweeps,
# instrumented (scripts/mixing/driver.jl). Usage:
#   nice -n 19 julia --project=. -t 4 scripts/mixing/run_baseline.jl [nsweeps] [out]
include(joinpath(@__DIR__, "driver.jl"))
nsw = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 600
out = length(ARGS) >= 2 ? ARGS[2] : "results/mixing/baseline.h5"
cfg, P, sL, info = mix_setup("configs/chains40.toml")
rec = run_mix(cfg, P, sL; nsweeps=nsw, out=out, label="baseline", every=25)
save_rec(out, rec, nsw)
using Serialization
serialize(replace(out, ".h5" => "_chains.jls"), rec["chains"])
println("done, wall ", rec["wall"])
