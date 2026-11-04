# Device driver (docs/system/device.md §6). Usage, from the repository root:
#   nice -n 19 julia --project=device -t 4 device/scripts/run_device.jl configs/device40.toml
# Exit code: 0 finished; 75 stopped by the stop file after writing a snapshot (resume by running
# again with the same configuration); anything else is a failure. With ESM_IGNORE_SIGTERM=1 the
# process ignores SIGTERM (Slurm preemption: the job wrapper creates the stop file instead;
# docs/system/hpc.md §5, §7).
using ESMSampling, ESMDevice
get(ENV, "ESM_IGNORE_SIGTERM", "") == "1" && ESMSampling.ignore_sigterm()
cfg = ESMSampling.load_config(ARGS[1])
res = ESMDevice.run_device(cfg)
exit(res.exitcode)
