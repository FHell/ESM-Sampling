# Operation mixing at fixed (k, x) as a cluster run (campaign H10; ledger E17, E21, E27): the
# protocol of opmix.jl through `hpc/run.sh` in `script` mode, preemptible and self-contained.
#
#   julia --project=. -t <n> scripts/mixing/opmix_hpc.jl <config.toml> <start> <burn-in> <sweeps> <kernel> [nmoves] [nx] [sx]
#
# The config is the run's (configs/chains40.toml as run by the wrapper: `out`, caches and the stop
# file set to the run directory). The start state is made in the run, from the config and its seed,
# with no input file: `start` = "heuristic" (the set-up's heuristic capacities and interior
# dispatch) or "lp" (the reference LP at the config's levels, solved with HiGHS or read from the
# content-addressed cache). From it, `burn-in` operation sweeps of the E21 kernel (slong, 32 moves
# per unit) on a random stream of their own bring the operation to the near-optimal regime; every
# kernel of a start therefore begins from the same state (its hash is in the summary). Then
# `sweeps` sweeps of the kernel: "base", "slong" (the E21 kernel), "sx" (slong plus the storage
# pairs: nx sxlong per unit, sxpair at rate sx). Outputs next to `out` (dirname(cfg.out)):
#   opmix.jls            the full record (as opmix.jl; stays in the archive);
#   opmix_snapshot.jls   state, record and random streams for a resume (removed at the end);
#   summary_opmix.toml   the run, the start and its burn-in, its cost per sweep, and per quantity
#                        the integrated autocorrelation time τ (sweeps, after a further 200),
#                        drift and sd: opex, shed, capacity scores, stored energy per carrier
#                        (total, regions, level and seasonal shape), the seasonal shape Q1 − Q3 per
#                        storage unit with ≥ 24 h, and the long-range sweep's move statistics;
#   summary_trace.csv    per sweep: time, opex, shed, stored energy per carrier, Q1 − Q3 per unit
#                        with ≥ 24 h (for equal-CPU comparisons across runs);
#   summary_burnin.csv   per burn-in sweep: time, opex.
# Preemption: with ESM_IGNORE_SIGTERM=1 SIGTERM is ignored; when the stop file exists at a sweep
# boundary (burn-in or main), the snapshot is written and the script exits with 75; a snapshot is
# also written every 10 minutes, and a run that finds one resumes from it (the same random
# streams: an exact continuation).
using Random, Statistics, Printf, Serialization, TOML
import ESMSampling as ML
get(ENV, "ESM_IGNORE_SIGTERM", "") == "1" && ML.ignore_sigterm()

cfgpath, start, nburn, nsw, kernel = ARGS[1], ARGS[2], parse(Int, ARGS[3]), parse(Int, ARGS[4]), ARGS[5]
nlong = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 32
nx = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 8
sx = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 0.1
kernel == "sx" || (nx = 0; sx = 0.0)
start in ("heuristic", "lp") || error("start = $start: \"heuristic\" or \"lp\"")
cfg = ML.load_config(cfgpath); cfg.verbose = false
cfg.init = start
ci = start == "heuristic" ? 1 : 2                    # (the random stream of the main phase, as opmix.jl's chain index)
outdir = dirname(abspath(cfg.out))
mkpath(outdir)
opmix_start(P, sL) = sL
clock = ML.StageClock(outdir)                 # stages.csv: wall and CPU time per stage (hpc/cost_table.jl)
t_setup = @elapsed include(joinpath(@__DIR__, "opmix_lib.jl"))
ML.stage!(clock, "setup")

# the run's progress: burn-in (E21 kernel, its own stream), then the main phase
snapfile = joinpath(outdir, "opmix_snapshot.jls")
function snapshot!(s, R)
    tmp = snapfile * ".tmp-$(getpid())"
    open(io -> serialize(io, (; s, R)), tmp, "w")
    mv(tmp, snapfile; force=true)
end
if isfile(snapfile)
    sn = deserialize(snapfile)
    s, R = ML.evaluate!(P, sn.s), sn.R
    @printf("resumed in the %s phase at sweep %d from %s\n", R[:phase], R[:phase] == :burnin ? R[:bi] : R[:i], snapfile); flush(stdout)
else
    R = Dict{Symbol,Any}(:phase => :burnin, :brng => MersenneTwister(cfg.seed + 4711), :bi => 0, :btime => Float64[],
                         :bop => Float64[], :op0 => s.parts.op, :rng => MersenneTwister(1000 + ci),
                         :rec => new_record(kernel, nlong, nx, sx), :i => 0, :start_hash => "")
end
stopfile = cfg.stop_file
tsnap = time()
"At a sweep boundary: snapshot and exit 75 if the stop file exists; a snapshot every 10 minutes."
function boundary!(s, R, i)
    if !isempty(stopfile) && isfile(stopfile)
        snapshot!(s, R)
        rm(stopfile; force=true)
        @printf("stopped in the %s phase at sweep %d (snapshot %s)\n", R[:phase], i, snapfile); flush(stdout)
        ML.stage!(clock, R[:phase] == :burnin ? "burn-in (stopped)" : "main (stopped)")
        exit(75)
    end
    if time() - tsnap > 600
        snapshot!(s, R)
        global tsnap = time()
    end
end
t0 = time()
if R[:phase] == :burnin
    for b in R[:bi]+1:nburn
        a = time()
        global s
        s, _ = opmix_sweep!(R[:brng], s, "slong", 32, 0, 0.0)
        push!(R[:btime], time() - a); push!(R[:bop], s.parts.op)
        R[:bi] = b
        b % 50 == 0 && (@printf("%s burn-in sweep %d: opex %.4e  [%.2f s/sweep]\n", start, b, s.parts.op,
                                mean(R[:btime][end-49:end])); flush(stdout))
        boundary!(s, R, b)
    end
    ML.stage!(clock, "burn-in")
    R[:phase] = :main
    R[:start_hash] = string(hash((s.k, s.v, s.z0)); base=16)
    @printf("%s start after %d burn-in sweeps: opex %.4e, state hash %s\n", start, nburn, s.parts.op, R[:start_hash]); flush(stdout)
end
rng, rec = R[:rng], R[:rec]
for i in R[:i]+1:nsw
    a = time()
    global s
    s, lst = opmix_sweep!(rng, s, kernel, nlong, nx, sx)
    q = record!(rec, s, time() - a, lst)
    R[:i] = i
    if i % 50 == 0
        @printf("%s %s sweep %d: opex %.4e shed %.3e  [%.0f s, %.2f s/sweep]\n", start, kernel, i, q[iop], q[ish],
                time() - t0, mean(rec["time"][end-49:end])); flush(stdout)
    end
    boundary!(s, R, i)
end
ML.stage!(clock, "main")
serialize(joinpath(outdir, "opmix.jls"), rec)

# --- summaries -------------------------------------------------------------------------------
"Integrated autocorrelation time, Sokal's automatic window (c = 5)."
function iat(x)
    n = length(x); y = x .- mean(x); v = sum(abs2, y) / n
    v == 0 && return NaN
    τ = 1.0
    for t in 1:n-1
        τ += 2 * sum(y[1:n-t] .* y[1+t:n]) / ((n - t) * v)
        t >= 5τ && return τ
    end
    return Inf
end
drift(x) = (n = length(x); (mean(x[end-n÷3+1:end]) - mean(x[1:n÷3])) / std(x))
stats(x) = Dict("tau" => iat(x), "drift" => drift(x), "sd" => std(x), "mean" => mean(x))
burn = min(200, length(rec["op"]) ÷ 5)
sel = burn+1:length(rec["op"])
summ = Dict{String,Any}("run_id" => get(ENV, "ESM_RUN_ID", ""), "commit" => get(ENV, "ESM_COMMIT", ""),
                        "start" => start, "seed" => cfg.seed, "data" => cfg.data, "data_sha" => get(ENV, "ESM_DATA_SHA", ""),
                        "setup_s" => t_setup, "burnin_sweeps" => nburn, "burnin_kernel" => "slong 32",
                        "burnin_s" => sum(R[:btime]; init=0.0), "opex_start" => R[:op0],
                        "opex_after_burnin" => isempty(R[:bop]) ? R[:op0] : R[:bop][end], "start_hash" => R[:start_hash],
                        "kernel" => kernel, "nmoves" => nlong, "nx" => nx, "sx" => sx,
                        "sweeps" => length(rec["op"]), "burn_in" => burn, "threads" => Threads.nthreads(),
                        "s_per_sweep_mean" => mean(rec["time"][2:end]), "s_per_sweep_median" => median(rec["time"]),
                        "opex_first" => rec["op"][1], "opex_burn" => rec["op"][max(burn, 1)], "opex_last" => rec["op"][end])
summ["opex"] = stats(rec["op"][sel]); summ["shed"] = stats(rec["shed"][sel])
Sc = reduce(hcat, rec["score"])
summ["score"] = Dict(nm => stats(Sc[j, sel]) for (j, nm) in enumerate(rec["knames"]))
stor = Dict{String,Any}()
for (si, sn) in enumerate(rec["snames"])
    tot = [sum(x[si, :, :]) for x in rec["soc"][sel]]
    maximum(abs, tot) < 1 && continue
    lev = reduce(hcat, [vec(mean(x[si, :, :]; dims=2)) for x in rec["soc"][sel]])
    shp = reduce(hcat, [vec(x[si, :, :] .- mean(x[si, :, :]; dims=2)) for x in rec["soc"][sel]])
    ok(M) = [i for i in axes(M, 1) if std(M[i, :]) > 1e-9 * max(1.0, maximum(abs, M))]
    τs(M) = [iat(M[i, :]) for i in ok(M)]
    stor[sn] = Dict("total" => stats(tot),
                    "level_tau_median" => median(τs(lev)), "level_tau_max" => maximum(τs(lev)),
                    "shape_tau_median" => median(τs(shp)), "shape_tau_max" => maximum(τs(shp)))
end
summ["storage"] = stor
units = [r for (r, (car, node, h)) in enumerate(rec["ustore"]) if h >= 24]
q13(r) = [x[r, 1] - x[r, 3] for x in rec["unitq"]]
uq = Dict{String,Any}()
for r in units
    car, node, h = rec["ustore"][r]
    x = q13(r)[sel]
    std(x) > 1e-9 * max(1.0, maximum(abs, x)) || continue
    uq["$(car) $(node)"] = stats(x)
end
summ["unit_q1_q3"] = uq
if !isempty(rec["lstats"])
    Ls = sum(rec["lstats"]) ./ length(rec["lstats"])
    rows = ("slong", "slevel", "sxlong")
    summ["long_sweep"] = Dict(rows[q] => Dict("moves" => Ls[q, 1], "acceptance" => Ls[q, 2] / max(Ls[q, 1], 1),
                                              "MWh_per_move" => Ls[q, 3] / max(Ls[q, 1], 1)) for q in 1:size(Ls, 1) if Ls[q, 1] > 0)
end
open(io -> TOML.print(io, summ; sorted=true), joinpath(outdir, "summary_opmix.toml"), "w")
open(joinpath(outdir, "summary_trace.csv"), "w") do io
    sn = [s_ for (si, s_) in enumerate(rec["snames"])]
    un = ["$(rec["ustore"][r][1]) $(rec["ustore"][r][2])" for r in units]
    println(io, join(vcat(["sweep", "time_s", "opex", "shed_MWh"], ["stored_MWh " * x for x in sn], ["q1_q3_MWh " * x for x in un]), ","))
    for i in eachindex(rec["op"])
        row = vcat([i, rec["time"][i], rec["op"][i], rec["shed"][i]], [sum(rec["soc"][i][si, :, :]) for si in eachindex(sn)],
                   [rec["unitq"][i][r, 1] - rec["unitq"][i][r, 3] for r in units])
        println(io, join((@sprintf("%.6g", x) for x in row), ","))
    end
end
open(joinpath(outdir, "summary_burnin.csv"), "w") do io
    println(io, "sweep,time_s,opex")
    for b in eachindex(R[:bop])
        println(io, b, ",", @sprintf("%.6g", R[:btime][b]), ",", @sprintf("%.6g", R[:bop][b]))
    end
end
rm(snapfile; force=true)
@printf("done: %d sweeps, %.3f s/sweep; summaries in %s\n", length(rec["op"]), summ["s_per_sweep_mean"], outdir)
