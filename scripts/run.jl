# CPU driver: `nchains` finest-level chains with diagnostics (docs/system/sampler.md).
# Usage (from the repository root):
#   nice -n 19 julia --project=. -t 4 scripts/run.jl configs/complete40.toml
# Exit code: 0 finished; 75 stopped by the stop file after writing a snapshot (resume by
# running again with the same configuration); anything else is a failure. With
# ESM_IGNORE_SIGTERM=1 the process ignores SIGTERM (Slurm preemption: the job wrapper
# creates the stop file instead; docs/system/hpc.md §5, §7).
using ESMSampling
get(ENV, "ESM_IGNORE_SIGTERM", "") == "1" && ESMSampling.ignore_sigterm()
cfg = length(ARGS) > 0 ? load_config(ARGS[1]) : Config()
res = ESMSampling.run(cfg)
exit(res.exitcode)
