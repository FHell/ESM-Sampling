# Reference LP of a configuration (validation, ledger campaign H2; docs/system/validation.md §2):
# solves (or reads from the cache) the zero-temperature LP at the configuration's reference level
# with HiGHS and writes a small summary next to the run's output: objective, capex, opex, the
# capacity per carrier and per country, and the free-coordinate counts.
#
#   julia --project=. -t <n> scripts/lp_reference.jl <config.toml>
# (on the cluster through hpc/run.sh in `script` mode, e.g. hpc/jobs/build.sbatch)
using ESMSampling, Random, Printf, TOML
const ML = ESMSampling

cfg = ML.load_config(ARGS[1])
cfg.init = "lp"
t0 = time()
clock = ML.StageClock(dirname(cfg.out))    # stages.csv (hpc/cost_table.jl)
P, s, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
ML.stage!(clock, "setup and reference LP")   # setup solves the LP (init = lp; cached)
lp = info.lp
E = P.E
countries = sort(unique(E.sys.country))
cap = Dict{String,Any}()
for (f, sp) in enumerate(E.kspecs)
    bus = sp.B.support
    kv = lp.k[f]
    byc = Dict(c => sum((kv[i] for i in eachindex(bus) if E.sys.country[bus[i]] == c); init=0.0) for c in countries)
    cap[String(sp.name)] = Dict("total_MW" => sum(kv), "by_country_MW" => byc)
end
gitcommit = try readchomp(`git rev-parse HEAD`) catch; "unknown" end
summary = Dict(
    "commit" => get(ENV, "ESM_COMMIT", gitcommit), "run_id" => get(ENV, "ESM_RUN_ID", ""),
    "data_sha" => get(ENV, "ESM_DATA_SHA", ""),
    "config" => ARGS[1], "data" => cfg.data, "levels" => cfg.levels,
    "objective_eur_per_a" => lp.objective, "capex_eur_per_a" => lp.capex,
    "opex_eur_per_a" => lp.objective - lp.capex,
    "n_v_free" => lp.n_v_free, "n_v_total" => lp.n_v_total, "n_k_free" => lp.n_k_free, "n_k_total" => lp.n_k_total,
    "capacity" => cap, "seconds" => time() - t0)
path = joinpath(dirname(cfg.out), "summary_lp.toml")
open(io -> TOML.print(io, summary), path, "w")
@printf("reference LP: objective %.4e €/a, capex %.4e €/a; summary %s\n", lp.objective, lp.capex, path)
