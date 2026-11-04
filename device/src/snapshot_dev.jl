# State snapshots and resume of the Metropolis device driver (docs/system/hpc.md §7): the
# contract of the CPU driver (src/checkpoint.jl). The replica driver writes the CPU replica
# driver's layout instead (`write_replica_snapshot`, replica_run.jl). At a host checkpoint the state is on the host already (it is
# downloaded for the monitored quantities), so a snapshot is the file write:
#
# * every `checkpoint_minutes` of wall time, and when `stop_file` exists (then the stop file is
#   removed and the driver returns exit code 75);
# * the layout of the CPU snapshots (phase `fine`, the counter `it` = host checkpoints done, the
#   states of every chain, their start kinds, the monitored traces), plus the device group
#   `device/` (the driver);
# * written under a temporary name, flushed and renamed over the previous snapshot;
# * resume: the states are read back (`read_snapshot`, evaluated under the current target), the
#   device arrays and programs are rebuilt from them, the random streams (host and Reactant's)
#   re-seeded from `seed` and the restart count, never a replay of the old streams.

"""
    write_device_snapshot(Ck, cfg, P, data, extra) -> path

`ML.write_snapshot` of `data` (phase, it, states, kinds, traces) plus the entries of `extra`
(a Dict of name => value) under `device/`, in one atomic write to `Ck.path`.
"""
function write_device_snapshot(Ck::ML.Checkpointer, cfg::ML.Config, P::ML.Problem, data::NamedTuple, extra::AbstractDict)
    t = time()
    quiet = deepcopy(cfg); quiet.verbose = false
    ML.atomic_write(Ck.path; sync=true) do tmp
        T = ML.Checkpointer(tmp, Ck.interval, "", Ck.last, Ck.nwritten, Ck.restarts, nothing)
        ML.write_snapshot(T, quiet, P, data)                      # the CPU layout (itself atomic, into tmp)
        ML.HDF5.h5open(tmp, "r+") do f
            for (k, v) in extra
                f["device/$k"] = v
            end
        end
    end
    Ck.last = time()
    Ck.nwritten += 1
    ML.logmsg(cfg, @sprintf("snapshot %s: %s  [%.1f MB, %.1fs]", Ck.path, ML.ckpt_where(data.phase, data.it, cfg),
                            filesize(Ck.path) / 1e6, Ck.last - t))
    return Ck.path
end

"""
    read_device_snapshot(path, cfg, P) -> NamedTuple or nothing

`ML.read_snapshot` (checks of the format, levels, seed and sizes; states rebuilt) plus the
`device/` entries as a Dict (`extra`).
"""
function read_device_snapshot(path::String, cfg::ML.Config, P::ML.Problem)
    snap = ML.read_snapshot(path, cfg, P)
    snap === nothing && return nothing
    extra = Dict{String,Any}()
    ML.HDF5.h5open(path, "r") do f
        if haskey(f, "device")
            for k in keys(f["device"])
                extra[k] = read(f["device/$k"])
            end
        end
    end
    return (; snap..., extra)
end

"""
    device_boundary!(Ck, cfg, P, it, make_data, make_extra; force=false) -> stop::Bool

At a host checkpoint (`it` done): with the stop file present, write a snapshot, remove the stop
file and return true (the driver then exits with code 75); else write one if `force` or
`checkpoint_minutes` have passed, and return false. `make_data()`, `make_extra()` build the
snapshot (called only when one is written).
"""
function device_boundary!(Ck::ML.Checkpointer, cfg::ML.Config, P::ML.Problem, it::Int, make_data, make_extra; force::Bool=false)
    ML.enabled(Ck) || return false
    Ck.hook === nothing || Ck.hook(:fine, it)
    stop = !isempty(Ck.stop_file) && isfile(Ck.stop_file)
    if stop || force || time() - Ck.last >= Ck.interval
        write_device_snapshot(Ck, cfg, P, (; phase=:fine, it, make_data()...), make_extra())
        if stop
            rm(Ck.stop_file; force=true)
            ML.logmsg(cfg, "stop file $(cfg.stop_file): stopped after the snapshot $(Ck.path) ($(ML.ckpt_where(:fine, it, cfg))); exit code $(ML.EXIT_CHECKPOINTED)")
            return true
        end
    end
    return false
end

"Seed of Reactant's RNG of a device run after `r` resumes (as `restart_seed`, a separate stream)."
device_rng_seed(cfg::ML.Config, r::Int) = UInt64[ML.restart_seed(cfg, r, 0), 0x9e3779b9]
