# Profile the two device programs with Reactant's profiler (docs/system/device.md, profiling).
#
# Usage (from the repository root):
#   nice -n 19 julia --project=device -t 4 device/scripts/profile_device.jl <config> [outdir]
#
# For the operation sweeps (`run_sweeps`, `device_sweeps` sweeps per call) and the capacity
# pass it prints the compile-time breakdown (tracing, MLIR pass pipeline, XLA), the mean run
# time per call, and the kernel and framework-op statistics. The xplane traces stay in
# `outdir` (default `results/profiles/<config name>/`) for xprof or Perfetto. Programs are
# compiled with `sync=true` and without input donation, so the profiler can call them
# repeatedly on the same inputs.
using ESMSampling, ESMDevice, Reactant, Printf
const Prof = Reactant.Profiler

cfg = ESMSampling.load_config(ARGS[1])
cfg.verbose = true
outdir = abspath(length(ARGS) > 1 ? ARGS[2] : joinpath("results", "profiles", splitext(basename(ARGS[1]))[1]))
mkpath(outdir)
nrep = parse(Int, get(ENV, "NREPEAT", "3"))

D = ESMDevice.device_setup(cfg)
(; sw, kp, kargs, st, cp, sp, kk, tabs, kgd, ksd, dn, crd, cap, BDd, BSd, RSd, drng, Z, U, Zp, Up, Zr, Ur, nsd) = D
programs = [("operation", sw, (st, cp, sp, kk, tabs, nsd, drng)),
            ("capacity", kp, kargs(st, kgd, ksd, dn, crd, cap, Z, U, Zp, Up, BDd, BSd, RSd, Zr, Ur))]

io = open(joinpath(outdir, "report.txt"), "w")
say(s) = (println(s); println(io, s); flush(io))
say(@sprintf("device profile: %s, backend %s, %d chains, T = %d, N = %d, %d operation sweeps per call",
             ARGS[1], cfg.device_backend, D.K, D.M.T, D.M.N, cfg.device_sweeps))
for (name, f, args) in programs
    fc, ct = Prof.profile_compile_timings(joinpath(outdir, "compile_" * name)) do
        @compile sync=true donated_args=:none f(args...)
    end
    say(@sprintf("%-9s compile %.1f s: tracing %.1f s, MLIR passes %.1f s, XLA %.1f s",
                 name, ct.total_compile_time_μs / 1e6, ct.tracing_time_μs / 1e6, ct.mlir_time_μs / 1e6, ct.xla_time_μs / 1e6))
    r = Prof.profile_thunk_with_xprof(fc, args...; nrepeat=nrep, warmup=1, profile_dir=joinpath(outdir, "run_" * name))
    say(@sprintf("%-9s run %.4f s per call (mean of %d)", name, r.profiling_result.runtime_ns / 1e9, nrep))
    ks = Prof.get_kernel_stats(r.xplane_file)
    if !isempty(ks.reports)
        say("---- kernels ($name)")
        Prof.print_kernel_report(ks; io=stdout); Prof.print_kernel_report(ks; io=io)
    end
    fw = Prof.get_framework_op_stats(r.xplane_file)
    if !isempty(fw)
        say("---- framework ops ($name)")
        Prof.print_framework_op_stats(fw; io=stdout); Prof.print_framework_op_stats(fw; io=io)
    end
    say(sprint(show, r.profiling_result))
end
close(io)
