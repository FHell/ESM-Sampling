# Device driver (docs/system/device.md §6). Usage, from the repository root:
#   nice -n 19 julia --project=device -t 4 device/scripts/run_device.jl configs/device40.toml
# Exit code: 0 finished; 75 stopped by the stop file after writing a snapshot (resume by running
# again with the same configuration); anything else is a failure. With ESM_IGNORE_SIGTERM=1 the
# process ignores SIGTERM (Slurm preemption: the job wrapper creates the stop file instead;
# docs/system/hpc.md §5, §7).
#   FAMILIES=all (default: the fast plain families), ADAPT=1 (host adaptation: learned modes, shapes and
#   coefficients, the dense angle families), FLOAT=32 (the Float32 build, `ESMDevice.F32`)
using ESMSampling, ESMDevice
get(ENV, "ESM_IGNORE_SIGTERM", "") == "1" && ESMSampling.ignore_sigterm()
cfg = ESMSampling.load_config(ARGS[1])
D = get(ENV, "FLOAT", "64") == "32" ? ESMDevice.F32 : ESMDevice
if haskey(ENV, "FAMILIES") || haskey(ENV, "ADAPT")
    fams = get(ENV, "FAMILIES", "fast") == "all" ? :all : D.FAMILIES_FAST
    res = D.run_device(cfg; families=fams, adapt=get(ENV, "ADAPT", "0") == "1")
else
    res = D.run_device(cfg)
end
exit(res.exitcode)
