# Test of review proposal P1 (capacities by stochastic-gradient Langevin at β_k from the sampled
# scores, no learned bias): two gas configurations of the same weather year (the `low` chain's end
# state of docs/mixing_diagnosis.md §10, and the same state with gas × 2.5) must move toward the
# same gas total under exp(−β_k [K + F_x(k)]). Usage:
#   nice -n 19 julia --project=. -t 4 scripts/review/run_sgld.jl <config> <chains.jls> <outdir> [nsweeps] [eta]
include(joinpath(@__DIR__, "..", "mixing", "common.jl"))
include(joinpath(@__DIR__, "analyze_bias.jl"))
include(joinpath(@__DIR__, "sgld_gas.jl"))
using Serialization

path, chainsfile, outdir = ARGS[1], ARGS[2], ARGS[3]
nsweeps = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 150
η = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1e4
mkpath(outdir)
cfg = ML.load_config(path)
cfg.use_bias = false                      # no bias: the target of k is exp(−β_k [K + F_x(k)]) for the chain's year
rng = MersenneTwister(cfg.seed)
P, sL, info = ML.setup(cfg; rng=rng)
chains = deserialize(chainsfile)
s0 = ML.evaluate!(P, chains[1])
E = P.E; lk = P.lk[end]
gasf = findfirst(sp -> String(sp.name) == "k_gas", E.kspecs)
U = ML.cols(E.kspecs[gasf].B, lk.m[gasf])
k0 = U * vec(ML.fieldview(s0.k, lk, E.kspecs, gasf))
@printf("start: gas %.1f GW, capex %.3e, op %.3e, shed %.3g\n", sum(k0) / 1e3, s0.parts.capex, s0.parts.op, s0.parts.shed)
sA = copy(s0)
sB = set_capacities_fixed_output!(P, copy(s0), gasf, min.(2.5 .* k0, E.kspecs[gasf].hi))
@printf("B start: gas %.1f GW, capex %.3e, op %.3e, shed %.3g\n", sum(U * vec(ML.fieldview(sB.k, lk, E.kspecs, gasf))) / 1e3, sB.parts.capex, sB.parts.op, sB.parts.shed)
out = Dict{String,Any}()
for (lab, s) in (("A_low", sA), ("B_high", sB))
    println("== run $lab: $nsweeps sweeps, η = $η"); flush(stdout)
    S = ML.new_sampler(P, s, cfg)
    t = time()
    tr, send = sgld_run!(MersenneTwister(lab == "A_low" ? 21 : 22), P, S, s, gasf, nsweeps; η=η, log_every=10)
    @printf("run %s done in %.0f s: final gas %.1f GW\n", lab, time() - t, tr[end].total / 1e3); flush(stdout)
    out[lab] = (; trace=tr, state=send)
    serialize(joinpath(outdir, "sgld_$lab.jls"), out[lab])
end
