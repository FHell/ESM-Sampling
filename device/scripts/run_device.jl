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
# a small text summary next to `out` (collected into git by hpc/collect.sh): the device timings
# of the run (the group `device_profile` of `out`) and the build
if res.exitcode == 0 && isfile(cfg.out)
    prof = ESMSampling.HDF5.h5open(cfg.out, "r") do f
        haskey(f, "device_profile") ? Dict(k => read(f["device_profile/$k"]) for k in keys(f["device_profile"])) : Dict()
    end
    open(joinpath(dirname(cfg.out), "summary_device.toml"), "w") do io
        ESMSampling.TOML.print(io, Dict("config" => ARGS[1], "float" => get(ENV, "FLOAT", "64"),
            "families" => get(ENV, "FAMILIES", "fast"), "adapt" => get(ENV, "ADAPT", "0"),
            "device_sampler" => cfg.device_sampler, "nchains" => cfg.nchains, "replicas" => cfg.replicas,
            "levels" => [collect(l) for l in cfg.levels], "device_sweeps" => cfg.device_sweeps,
            "run_id" => get(ENV, "ESM_RUN_ID", ""), "commit" => get(ENV, "ESM_COMMIT", ""),
            "device_profile_seconds" => Dict(k => Float64(v) for (k, v) in prof if v isa Real)))
    end
end
exit(res.exitcode)
