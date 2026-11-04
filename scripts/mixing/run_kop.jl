# The baseline chains (scripts/mixing/run_baseline.jl) plus one kop sweep per fine step
# (scripts/mixing/kop.jl): every renewable and storage unit once, and every such carrier
# once with a shared scale, in random order. Usage:
#   nice -n 19 julia --project=. -t 4 scripts/mixing/run_kop.jl [nsweeps] [out] [mode]
include(joinpath(@__DIR__, "driver.jl"))
include(joinpath(@__DIR__, "kop.jl"))
nsw = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 400
out = length(ARGS) >= 2 ? ARGS[2] : "results/mixing/kop.h5"
mode = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :both
cfg, P, sL, info = mix_setup("configs/chains40.toml")
extras = [("kop", (rng, P, LM, s) -> kop_sweep!(rng, P, LM, s; mode=mode)[1])]
rec = run_mix(cfg, P, sL; nsweeps=nsw, out=out, label="kop-$mode", every=25, extras=extras)
save_rec(out, rec, nsw)
using Serialization
serialize(replace(out, ".h5" => "_chains.jls"), rec["chains"])
println("done, wall ", rec["wall"])
