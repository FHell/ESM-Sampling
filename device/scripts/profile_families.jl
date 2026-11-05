# Per-family profile of the operation sweeps (Reactant's profiler), docs/system/device.md.
#
# Usage (from the repository root):
#   nice -n 19 julia --project=device -t 4 device/scripts/profile_families.jl <config> [outdir]
#   FAMILIES=fast (default: FAMILIES_FAST), plain (FAMILIES) or all (every device family, `device_families`; the dense angle
#   directions and the dispatch shapes only with ADAPT=1); NSWEEPS (default 2) sweeps per call;
#   FAMS=gen,soc restricts the profile to these families; BUDGET_MB (default 256) the memory budget of a round
#   (`DENSE_BUDGET`: rounds whose dense tensor exceeds it are split); JUMP = auto (default), none or all: the
#   line-sampler form of the compensated, cell and block families (`plan(…; jump)`); FLOAT=32 profiles the
#   Float32 build (`ESMDevice.F32`); ADAPT=1 sets up the host adaptation (direction sources of the dense
#   angle families `:tdir…`, learned block profiles and dispatch shapes).
#
# For every family alone: rounds per sweep, sites per round, real line moves per chain and
# sweep, compile time, run time per sweep and per line move, the five most expensive kernels.
using ESMSampling, ESMDevice, Reactant, Printf
const Prof = Reactant.Profiler
const D = get(ENV, "FLOAT", "64") == "32" ? ESMDevice.F32 : ESMDevice     # FLOAT=32: the Float32 build

cfg = ESMSampling.load_config(ARGS[1])
# default output: the run directory under hpc/run.sh (next to `out`), else results/profiles/
outdir = abspath(length(ARGS) > 1 ? ARGS[2] : haskey(ENV, "ESM_RUN_ID") ? dirname(abspath(cfg.out)) :
                 joinpath("results", "profiles", splitext(basename(ARGS[1]))[1] * "_families"))
mkpath(outdir)
nsw = parse(Int, get(ENV, "NSWEEPS", "2"))
fset = get(ENV, "FAMILIES", "fast")
budget = parse(Float64, get(ENV, "BUDGET_MB", "256")) * 2.0^20
jump = Symbol(get(ENV, "JUMP", "auto"))
sel = Symbol.(filter(!isempty, split(get(ENV, "FAMS", ""), ",")))     # FAMS=gen,soc: only these
X = D.device_setup(cfg; families=fset == "all" ? :all : fset == "plain" ? D.FAMILIES : D.FAMILIES_FAST, budget=budget, jump=jump,
                   only=isempty(sel) ? nothing : sel, adapt=get(ENV, "ADAPT", "0") == "1")
(; pl, st, cp, sp, kk, tabs, drng) = X
mv = D.moves_per_sweep(pl)
ns = Reactant.ConcreteRNumber(nsw)

io = open(joinpath(outdir, "report.txt"), "w")
say(s) = (println(s); println(io, s); flush(io))
say(@sprintf("per-family profile: %s, backend %s, %d chains, T = %d, N = %d, %d sweeps per call, jump = %s, %s",
             ARGS[1], cfg.device_backend, X.K, X.M.T, X.M.N, nsw, jump, D.FT))
say("slope-jump form: " * join(sort([string(f) * " (cap " * string(D.jumpcap(pl, f)) * ")" for f in pl.fams if D.jumpform(pl, f)]), ", "))
say(@sprintf("%-10s %7s %8s %9s %9s %10s %10s", "family", "rounds", "sites", "moves", "compile", "s/sweep", "µs/move"))
tot = 0.0
for f in pl.fams
    (isempty(sel) || f in sel) || continue
    fv = Val((f,))
    swf(st, cp, sp, k, tabs, ns, rng) = D.run_sweeps(st, cp, sp, k, tabs, ns, rng, fv)
    tc = @elapsed fc = D.ftcompile(() -> @compile sync=true donated_args=:none swf(st, cp, sp, kk, tabs, ns, drng))   # F32: no TF32
    r = Prof.profile_thunk_with_xprof(fc, st, cp, sp, kk, tabs, ns, drng; nrepeat=1, warmup=1,
                                      profile_dir=joinpath(outdir, string(f)))
    ts = r.profiling_result.runtime_ns / 1e9 / nsw
    global tot += ts
    nmv = X.K * mv[f]
    say(@sprintf("%-10s %7d %8d %9d %8.0fs %10.4f %10.3f", f, size(pl.dev[f].on, 2), size(pl.dev[f].on, 1), mv[f],
                 tc, ts, 1e6 * ts / max(nmv, 1)))
    ks = Prof.get_kernel_stats(r.xplane_file)
    rep = sort(ks.reports; by=x -> -x.total_duration_ns)
    for x in rep[1:min(5, length(rep))]
        say(@sprintf("    %-40s %6d × %9.3f ms = %8.3f ms per sweep, occupancy %3.0f %%, grid %s × block %s", x.name, x.occurrences,
                     x.total_duration_ns / 1e6 / max(x.occurrences, 1), x.total_duration_ns / 1e6 / nsw, x.occupancy_pct,
                     join(Int.(x.grid_dim), "×"), join(Int.(x.block_dim), "×")))
    end
end
say(@sprintf("sum of families: %.4f s per sweep (%d chains)", tot, X.K))
close(io)
