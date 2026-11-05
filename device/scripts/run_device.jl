# Device driver (docs/system/device.md §6). Usage, from the repository root:
#   nice -n 19 julia --project=device -t 4 device/scripts/run_device.jl configs/device40.toml
# Exit code: 0 finished; 75 stopped by the stop file after writing a snapshot (resume by running
# again with the same configuration); anything else is a failure. With ESM_IGNORE_SIGTERM=1 the
# process ignores SIGTERM (Slurm preemption: the job wrapper creates the stop file instead;
# docs/system/hpc.md §5, §7).
#   FAMILIES=all (every device family, `device_families`; default: the fast plain families), ADAPT=1 (host
#   adaptation: learned modes, shapes and coefficients, the dense angle families and the dispatch shapes), FLOAT=32 (the Float32 build, `ESMDevice.F32`), SYNC=0 (replica
#   driver: no synchronization after every device program, so the host work overlaps the device's;
#   the per-part timings then lump the device time into the parts that download)
using ESMSampling, ESMDevice
get(ENV, "ESM_IGNORE_SIGTERM", "") == "1" && ESMSampling.ignore_sigterm()
cfg = ESMSampling.load_config(ARGS[1])
D = get(ENV, "FLOAT", "64") == "32" ? ESMDevice.F32 : ESMDevice
sync = get(ENV, "SYNC", "1") != "0"
if haskey(ENV, "FAMILIES") || haskey(ENV, "ADAPT")
    fams = get(ENV, "FAMILIES", "fast") == "all" ? :all : D.FAMILIES_FAST
    res = D.run_device(cfg; families=fams, adapt=get(ENV, "ADAPT", "0") == "1", sync=sync)
else
    res = D.run_device(cfg; sync=sync)
end
# a small text summary next to `out` (collected into git by hpc/collect.sh): the device timings
# of the run (the group `device_profile` of `out`), the guard counts per family (`device/guard_*`:
# sites held at t = 0 by the root guard and by the slope-jump cap, totals over chains and host
# checkpoints, the largest count of one checkpoint, the cap rate per line move) and the build
"Guard counts of the run per family (docs/system/data.md §3), or an empty Dict."
function guard_summary(f)
    (haskey(f, "device") && haskey(f["device"], "guard_families")) || return Dict{String,Any}()
    fams, sites = read(f["device/guard_families"]), read(f["device/guard_sites"])
    root, cap, rsw = read(f["device/guard_root"]), read(f["device/guard_cap"]), Float64(read(f["device/guard_row_sweeps"]))
    perck(A, i) = size(A, 3) == 0 ? 0.0 : maximum(sum(A[i, :, j]) for j in axes(A, 3))
    return Dict{String,Any}(fams[i] => Dict{String,Any}("sites_per_row_sweep" => Int(sites[i]), "root" => sum(root[i, :, :]; init=0.0),
                                                        "cap" => sum(cap[i, :, :]; init=0.0), "cap_max_checkpoint" => perck(cap, i),
                                                        "root_max_checkpoint" => perck(root, i),
                                                        "cap_rate" => rsw * sites[i] > 0 ? sum(cap[i, :, :]; init=0.0) / (rsw * sites[i]) : 0.0)
                            for i in eachindex(fams))
end
if res.exitcode == 0 && isfile(cfg.out)
    prof, guards, rsw = ESMSampling.HDF5.h5open(cfg.out, "r") do f
        (haskey(f, "device_profile") ? Dict(k => read(f["device_profile/$k"]) for k in keys(f["device_profile"])) : Dict(),
         guard_summary(f),
         (haskey(f, "device") && haskey(f["device"], "guard_row_sweeps")) ? Float64(read(f["device/guard_row_sweeps"])) : 0.0)
    end
    open(joinpath(dirname(cfg.out), "summary_device.toml"), "w") do io
        ESMSampling.TOML.print(io, Dict("config" => ARGS[1], "float" => get(ENV, "FLOAT", "64"),
            "families" => get(ENV, "FAMILIES", "fast"), "adapt" => get(ENV, "ADAPT", "0"), "sync" => get(ENV, "SYNC", "1"),
            "device_sampler" => cfg.device_sampler, "nchains" => cfg.nchains, "replicas" => cfg.replicas,
            "levels" => [collect(l) for l in cfg.levels], "device_sweeps" => cfg.device_sweeps,
            "run_id" => get(ENV, "ESM_RUN_ID", ""), "commit" => get(ENV, "ESM_COMMIT", ""),
            "device_profile_seconds" => Dict(k => Float64(v) for (k, v) in prof if v isa Real),
            "guard_row_sweeps" => rsw, "guards" => guards))
    end
end
exit(res.exitcode)
