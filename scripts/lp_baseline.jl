# LP baselines on the historical weather years (validation, campaign H2; docs/system/validation.md
# §2.7): the capacity-expansion LP of a configuration at its reference level (the first spatially
# complete level of `levels`), for one historical year or stacked over several (shared capacities,
# operation per year with storage cyclic over each year, equal year weights), optionally followed
# by an MGA sweep (minimum and maximum total capacity per carrier at cost ≤ (1 + ε) C*).
#
#   julia --project=. -t <n> scripts/lp_baseline.jl <config.toml> [key=value ...]
#
#   years=2013 | 2001-2005 | 2003,2010,2013 | all   historical years of the data file (default 2013)
#   eps=0.01,0.02,0.05,0.1                          MGA cost slacks (default: none, optimum only)
#   carriers=all | onwind,solar,...                 MGA carriers: k fields, with or without "k_" (default all)
#   time_limit=<s>                                  HiGHS time limit per LP (default none)
#
# On the cluster through hpc/run.sh in `script` mode (a line of campaigns/H2/list.txt). Writes next
# to the run's output (dirname(cfg.out)):
#   summary_lp.toml      the optimum: costs, per-year statistics, capacities per carrier and country,
#                        LP size, times, memory
#   years.csv            per year: weight, opex, shed, dump, curtailment, its cost in the sampler's
#                        model, CO₂, energy per carrier
#   capacity.csv         the optimum's capacity per carrier and country (MW)
#   mga.csv              one row per MGA solve (written as they finish; a requeued job skips them)
#   mga_capacity.csv     capacity per carrier and country at every MGA extreme
#   summary_mga.toml     the envelopes: per ε and carrier, min, optimum, max (MW)
# Exit code 0, or 75 if the stop file appeared between MGA solves (the sweep resumes on restart).
using ESMSampling, JuMP, Printf, TOML, Dates
const ML = ESMSampling
get(ENV, "ESM_IGNORE_SIGTERM", "") == "1" && ML.ignore_sigterm()

cfg = ML.load_config(ARGS[1])
opts = Dict{String,String}()
for a in ARGS[2:end]
    occursin('=', a) || error("argument $a: expected key=value")
    k, v = split(a, '='; limit=2)
    k in ("years", "eps", "carriers", "time_limit") || error("unknown argument $k")
    opts[k] = v
end
out = dirname(cfg.out)
mkpath(out)
t0 = time()
sys = ML.load_system(cfg.data)
E = ML.energy_model(cfg, sys)
ref = ML.check_levels(cfg.levels, sys.N)
lv = ML.Layout(E.vspecs, cfg.levels[ref+1]...)

function parse_years(s, all)
    s == "all" && return copy(all)
    ys = Int[]
    for p in split(s, ',')
        if occursin('-', p)
            a, b = parse.(Int, split(p, '-'))
            append!(ys, a:b)
        else
            push!(ys, parse(Int, p))
        end
    end
    return ys
end
years = parse_years(get(opts, "years", "2013"), sys.years)
eps = haskey(opts, "eps") ? parse.(Float64, split(opts["eps"], ',')) : Float64[]
fieldnames_ = [String(s.name) for s in E.kspecs]
fields = if get(opts, "carriers", "all") == "all"
    collect(eachindex(E.kspecs))
else
    [let n = startswith(c, "k_") ? c : "k_" * c
         f = findfirst(==(n), fieldnames_)
         f === nothing && error("carrier $c: the k fields are $fieldnames_")
         f
     end for c in split(opts["carriers"], ',')]
end
tl = haskey(opts, "time_limit") ? parse(Float64, opts["time_limit"]) : Inf
ignored = [k for (k, on) in (("losses", cfg.losses), ("thermal=soft", cfg.thermal == "soft")) if on]
isempty(ignored) || @warn "the LP has hard line limits and no line losses; ignored: $ignored"
println("LP baseline: $(sys.N) nodes, level $(cfg.levels[ref+1]) (T = $(lv.T)), years $years, ",
        isempty(eps) ? "optimum only" : "MGA ε = $eps on $(fieldnames_[fields])", "; HiGHS threads ",
        get(ENV, "ESM_LP_THREADS", "default"))
flush(stdout)

# ---- the optimum
tb = time()
res, lpm = ML.solve_lp_years(E, lv, years; time_limit=tl)
t_opt = time() - tb
C = res.objective
model = lpm.model
@printf("optimum: %.6e €/a (capex %.4e, mean opex %.4e), %d variables, %d constraints  [%.0f s]\n",
        C, res.capex, sum(res.weights .* res.opex), num_variables(model),
        num_constraints(model; count_variable_in_set_constraints=false), t_opt)
flush(stdout)

countries = sort(unique(sys.country))
capc = ML.capacity_by_country(E, res.k)
gitcommit = try readchomp(`git rev-parse HEAD`) catch; "unknown" end
statkeys = sort(unique(reduce(vcat, [collect(keys(s)) for s in res.stats])))
summary = Dict{String,Any}(
    "commit" => get(ENV, "ESM_COMMIT", gitcommit), "run_id" => get(ENV, "ESM_RUN_ID", ""),
    "data_sha" => get(ENV, "ESM_DATA_SHA", ""), "config" => ARGS[1], "data" => cfg.data,
    "args" => join(ARGS[2:end], " "), "nodes" => sys.N, "level" => collect(cfg.levels[ref+1]), "T" => lv.T,
    "years" => years, "weights" => res.weights, "status" => res.status,
    "objective_eur_per_a" => C, "capex_eur_per_a" => res.capex,
    "opex_eur_per_a" => sum(res.weights .* res.opex), "opex_by_year_eur" => res.opex,
    "sampler_cost_eur_per_a" => C + sum(res.weights .* [s["curtail_cost_eur"] + s["cycling_cost_eur"] for s in res.stats]),
    "ignored_model_terms" => ignored,
    "mean_stats" => Dict(k => sum(res.weights .* [get(s, k, 0.0) for s in res.stats]) for k in statkeys),
    "n_k_free" => res.sols[1].n_k_free, "n_k_total" => res.sols[1].n_k_total,
    "n_v_free" => [s.n_v_free for s in res.sols], "n_v_total" => [s.n_v_total for s in res.sols],
    "lp_variables" => num_variables(model),
    "lp_constraints" => num_constraints(model; count_variable_in_set_constraints=false),
    "lp_threads" => get(ENV, "ESM_LP_THREADS", ""), "julia_threads" => Threads.nthreads(),
    "seconds_setup" => tb - t0, "seconds_optimum" => t_opt, "maxrss_MB" => Sys.maxrss() / 2^20,
    "capacity" => Dict(n => Dict("total_MW" => sum(res.k[f]), "by_country_MW" => capc[n]) for (f, n) in enumerate(fieldnames_)))
open(io -> TOML.print(io, summary), joinpath(out, "summary_lp.toml"), "w")
open(joinpath(out, "years.csv"), "w") do io
    println(io, join(vcat(["year", "weight", "opex_eur"], statkeys), ","))
    for (y, yr) in enumerate(years)
        println(io, join(vcat([string(yr), string(res.weights[y]), string(res.opex[y])],
                              [string(get(res.stats[y], k, 0.0)) for k in statkeys]), ","))
    end
end
open(joinpath(out, "capacity.csv"), "w") do io
    println(io, "carrier,country,MW")
    for n in fieldnames_, c in countries
        println(io, "$n,$c,$(capc[n][c])")
    end
end
isempty(eps) && (println("summary $(joinpath(out, "summary_lp.toml"))"); exit(0))

# ---- MGA sweep (resumable: finished solves are rows of mga.csv)
mpath, cpath = joinpath(out, "mga.csv"), joinpath(out, "mga_capacity.csv")
done = Set{Tuple{Float64,Int,Symbol}}()
rows = Dict{Tuple{Float64,Int,Symbol},Vector{String}}()
if isfile(mpath)
    for l in readlines(mpath)[2:end]
        c = split(l, ',')
        length(c) == 9 || continue                    # a line cut off by a kill
        f = findfirst(==(c[2]), fieldnames_)
        f === nothing && continue
        key = (parse(Float64, c[1]), f, Symbol(c[3]))
        push!(done, key); rows[key] = String.(c)
    end
    keep = isfile(cpath) ? filter(l -> begin
        c = split(l, ','); length(c) == 6 && (f = findfirst(==(c[2]), fieldnames_); f !== nothing &&
                                              (parse(Float64, c[1]), f, Symbol(c[3])) in done)
    end, readlines(cpath)[2:end]) : String[]
    open(io -> (println(io, "eps,target,sense,carrier,country,MW"); foreach(l -> println(io, l), keep)), cpath, "w")
    open(io -> (println(io, "eps,target,sense,status,attempts,cost_eur,cost_rel_excess,total_MW,seconds");
                foreach(k -> println(io, join(rows[k], ",")), sort(collect(done)))), mpath, "w")
    println("MGA: resuming, $(length(done)) solves done")
else
    open(io -> println(io, "eps,target,sense,status,attempts,cost_eur,cost_rel_excess,total_MW,seconds"), mpath, "w")
    open(io -> println(io, "eps,target,sense,carrier,country,MW"), cpath, "w")
end
stop = cfg.stop_file
function record(ε, f, sense, r)
    open(cpath, "a") do io
        cc = ML.capacity_by_country(E, r.k)
        for n in fieldnames_, c in countries
            println(io, "$ε,$(fieldnames_[f]),$sense,$n,$c,$(cc[n][c])")
        end
    end
    row = [string(ε), fieldnames_[f], string(sense), r.status, string(r.attempts), string(r.cost),
           string(r.cost / C - 1), string(r.total), @sprintf("%.1f", r.seconds)]
    open(io -> println(io, join(row, ",")), mpath, "a")
    rows[(ε, f, sense)] = row
    @printf("  ε %.3f  %-10s %s  %-22s total %.4e MW (optimum %.4e)  cost +%.4f %%  [%.0f s]\n", ε, fieldnames_[f], sense,
            r.status, r.total, sum(res.k[f]), 100 * (r.cost / C - 1), r.seconds)
    flush(stdout)
    if !isempty(stop) && isfile(stop)
        rm(stop; force=true)
        println("stop file: exiting with 75 after $(length(rows)) MGA solves (the sweep resumes on restart)")
        exit(75)
    end
end
ML.mga_sweep!(lpm, E, C, eps, fields; done=done, on_result=record)

# ---- envelopes
env = Dict{String,Any}()
for ε in eps
    d = Dict{String,Any}()
    for f in fields
        mn, mx = get(rows, (ε, f, :min), nothing), get(rows, (ε, f, :max), nothing)
        d[fieldnames_[f]] = Dict("min_MW" => mn === nothing ? NaN : parse(Float64, mn[8]),
                                 "max_MW" => mx === nothing ? NaN : parse(Float64, mx[8]),
                                 "optimum_MW" => sum(res.k[f]),
                                 "status" => [mn === nothing ? "missing" : mn[4], mx === nothing ? "missing" : mx[4]])
    end
    env[@sprintf("eps_%g", ε)] = d
end
open(io -> TOML.print(io, Dict("objective_eur_per_a" => C, "years" => years, "eps" => eps,
                               "envelope" => env, "seconds_total" => time() - t0, "maxrss_MB" => Sys.maxrss() / 2^20)),
     joinpath(out, "summary_mga.toml"), "w")
println("MGA done: $(joinpath(out, "summary_mga.toml"))  [$(round(Int, time() - t0)) s]")
