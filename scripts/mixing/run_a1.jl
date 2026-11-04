# α = 1 with the bias learned in a shortened level-0 warmup and then frozen (a fixed target), the
# utilization measure, own weather draws: the baseline kernel and the kernel with kop, from the
# same starts, in one process (one warmup). Usage:
#   nice -n 19 julia --project=. -t 4 scripts/mixing/run_a1.jl [nsweeps] [warmup0]
include(joinpath(@__DIR__, "driver.jl"))
include(joinpath(@__DIR__, "kop.jl"))
using Serialization
nsw = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 150
w0 = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300
cfg, P, sL, info = mix_setup("configs/chains40.toml"; use_bias=true, measure="utilization", warmup0=w0)
serialize("results/mixing/a1u_problem.jls", P)
for (name, extras) in (("a1u_kop", [("kop", (rng, P, LM, s) -> kop_sweep!(rng, P, LM, s; mode=:both)[1])]),
                       ("a1u_base", Tuple{String,Function}[]))
    rec = run_mix(cfg, P, sL; nsweeps=nsw, out="results/mixing/$name.h5", label=name, every=25, extras=extras)
    save_rec("results/mixing/$name.h5", rec, nsw)
    serialize("results/mixing/$(name)_chains.jls", rec["chains"])
    println("done $name, wall ", rec["wall"]); flush(stdout)
end
