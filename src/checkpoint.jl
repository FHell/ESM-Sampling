# Durable files of a run (docs/system/hpc.md §4, §7): atomic writes, the shared cache with
# its read-only mirror, and the state snapshots of the CPU driver.
#
# * Every cache entry and every snapshot is written to a temporary name in its target
#   directory and renamed into place (rename(2) replaces atomically), so a concurrent reader
#   sees the complete file or none, and an interrupted writer leaves no entry behind.
# * Caches: `cache_dir` (env ESM_CACHE_DIR) or `<dirname(out)>/cache`; a miss is first
#   looked up in `cache_mirror` (env ESM_CACHE_MIRROR) and copied in, then computed.
# * Snapshots (`run_chains`, key `checkpoint`): the chain states and a few scalars of the
#   driver as plain HDF5 data, at sweep boundaries. A resumed run continues the phase and
#   the sweep counter from the saved states, with fresh random streams (seed and restart
#   count) and the kernels, the bias and the caches rebuilt: it is not a bitwise
#   continuation of the stopped run.

# ---------------------------------------------------------------- atomic writes

"A temporary name next to `path` (same directory, so that the rename stays on one file system)."
function tmp_name(path::AbstractString)
    tag = string(rand(RandomDevice(), UInt64); base=16)   # not the task's RNG: the run's streams stay untouched
    return joinpath(dirname(abspath(path)), "." * basename(path) * ".tmp-$(getpid())-$tag")
end

"""
    atomic_write(f, path; sync=false) -> path

Call `f(tmp)` to write the file under a temporary name `tmp` in the directory of `path`,
then rename it to `path` (an existing file is replaced atomically). If `f` throws, the
temporary file is removed and nothing appears at `path`. `sync = true` flushes the file to
disk before the rename.
"""
function atomic_write(f, path::AbstractString; sync::Bool=false)
    mkpath(dirname(abspath(path)))
    tmp = tmp_name(path)
    try
        f(tmp)
        isfile(tmp) || error("atomic_write: the writer produced no file for $path")
        if sync
            open(tmp, "r+") do io
                ccall(:fsync, Cint, (Cint,), fd(io)) == 0 || @warn "fsync failed on $tmp"
            end
        end
        Base.Filesystem.rename(tmp, path)
    catch
        rm(tmp; force=true)
        rethrow()
    end
    return path
end

"`atomic_write` for a function of an open stream: `f(io)`."
atomic_open(f, path::AbstractString; sync::Bool=false) = atomic_write(tmp -> open(f, tmp, "w"), path; sync=sync)

# ---------------------------------------------------------------- caches

"Value of a path key with its environment override (set and non-empty wins)."
env_override(key::String, val::String) = (v = get(ENV, key, ""); isempty(v) ? val : v)

"The cache directory of a run: env ESM_CACHE_DIR, else `cache_dir`, else `<dirname(out)>/cache`."
function cache_dir(cfg::Config)
    d = env_override("ESM_CACHE_DIR", cfg.cache_dir)
    return isempty(d) ? joinpath(dirname(cfg.out), "cache") : d
end

"The read-only cache mirror of a run: env ESM_CACHE_MIRROR, else `cache_mirror` (\"\": none)."
cache_mirror(cfg::Config) = env_override("ESM_CACHE_MIRROR", cfg.cache_mirror)

"""
    cache_lookup(path, mirror) -> path or nothing

The cache entry at `path` if it exists; else, if the mirror directory has an entry of the
same name, a copy of it at `path` (written atomically; if the cache cannot be written, the
mirror's file itself); else `nothing` (to be computed).
"""
function cache_lookup(path::AbstractString, mirror::AbstractString="")
    isfile(path) && return path
    isempty(mirror) && return nothing
    src = joinpath(mirror, basename(path))
    isfile(src) || return nothing
    try
        atomic_write(tmp -> cp(src, tmp; force=true), path)
        return path
    catch e
        e isa InterruptException && rethrow()
        @warn "cache: could not copy $src into $(dirname(path)) ($(sprint(showerror, e))); reading the mirror"
        return src
    end
end

"""
    cache_read(read, path, mirror) -> value or nothing

Read a cache entry (`read(file)`), from the cache or the mirror (`cache_lookup`); `nothing`
if there is none or it cannot be read (then it is recomputed and overwritten).
"""
function cache_read(rd, path::AbstractString, mirror::AbstractString="")
    p = cache_lookup(path, mirror)
    p === nothing && return nothing
    try
        return rd(p)
    catch e
        e isa InterruptException && rethrow()
        @warn "cache: unreadable entry $p ($(sprint(showerror, e))); recomputing"
        return nothing
    end
end

# ---------------------------------------------------------------- snapshots

"Format version of the snapshot files (bump on any change of their layout)."
const SNAPSHOT_VERSION = 1

"Exit code of a run that stopped after writing a snapshot (EX_TEMPFAIL: requeue and resume)."
const EXIT_CHECKPOINTED = 75

"Thrown at a sweep boundary after the stop snapshot has been written; caught by `run_chains`."
struct CheckpointStop <: Exception
    path::String
    where_::String
    states::Vector{State}
end

"""
Seed of a random stream of the run: `cfg.seed + 1000 c` for chain `c` (`c = 0`: the master
stream), plus `1_000_003 r` after the `r`-th resume (r = 0: the seeds of a fresh run).
"""
restart_seed(cfg::Config, r::Int, c::Int=0) = cfg.seed + 1000 * c + 1_000_003 * r

"""
Snapshot control of a run: the file, the interval, the stop file, the time of the last
write, and the number of resumes of the run so far. `boundary!` is called at every sweep
boundary.
"""
mutable struct Checkpointer
    path::String
    interval::Float64        # seconds
    stop_file::String
    last::Float64
    nwritten::Int
    restarts::Int
    hook::Any                # tests: called as hook(phase, it) at every boundary, before the stop check
end
Checkpointer(cfg::Config; hook=nothing) = Checkpointer(cfg.checkpoint, 60 * cfg.checkpoint_minutes, cfg.stop_file, time(), 0, 0, hook)

enabled(C::Checkpointer) = !isempty(C.path)

"Human-readable position of a snapshot (phase, sweep counter)."
function ckpt_where(phase::Symbol, it::Int, cfg::Config)
    phase == :level0 && return "level-0 warmup, iteration $it/$(cfg.warmup0)"
    nw, np = cfg.mlda_warmup, cfg.nsamples
    if phase == :replicas
        it == 0 && return "replicas initialised, before the first step"
        return it <= nw ? "replica warmup, step $it/$nw" : "replica production, step $(it - nw)/$np"
    end
    it == 0 && return "after the level-0 warmup, before the first fine sweep"
    it <= nw && return "fine warmup, sweep $it/$nw"
    return "production, sweep $(it - nw)/$np"
end

"""
    write_snapshot(C, cfg, P, data) -> path

Write the snapshot `data` atomically to `C.path` (layout in docs/system/data.md §5):
`phase` (`:level0` or `:fine`), the sweep counter `it` (sweeps completed in the phase), the
chain `states`, and optionally `kinds` (start kinds), `traces` (monitored quantities so
far), `co2_trace`, `sigma_scale` and `gibbs` (the level-0 step sizes and slice widths).
"""
function write_snapshot(C::Checkpointer, cfg::Config, P::Problem, data::NamedTuple)
    t = time()
    where_ = ckpt_where(data.phase, data.it, cfg)
    atomic_write(C.path; sync=true) do tmp
        h5open(tmp, "w") do f
            f["version"] = SNAPSHOT_VERSION
            f["phase"] = String(data.phase)
            f["it"] = data.it
            f["where"] = where_
            f["restarts"] = C.restarts
            f["written"] = t
            f["co2_price"] = P.E.co2[]
            f["config/seed"] = cfg.seed
            f["config/levels"] = reduce(hcat, [[a, b] for (a, b) in cfg.levels])
            f["nchains"] = length(data.states)
            for (c, s) in enumerate(data.states)
                g = "chains/$c"
                f["$g/level"] = s.ℓ
                f["$g/m"] = s.m
                f["$g/z0"] = s.z0
                for j in 1:s.ℓ
                    f["$g/w/$j"] = s.ws[j]
                end
                f["$g/k"] = s.k
                f["$g/v"] = s.v
                f["$g/logtarget"] = s.L
                haskey(data, :kinds) && (f["$g/start"] = data.kinds[c])
                if haskey(data, :traces) && !isempty(data.traces[c])
                    f["$g/Q"] = reduce(hcat, data.traces[c])
                end
            end
            haskey(data, :co2_trace) && !isempty(data.co2_trace) && (f["co2/price_trace"] = data.co2_trace)
            haskey(data, :sigma_scale) && (f["sampler/sigma_scale"] = data.sigma_scale)
            if haskey(data, :gibbs)
                f["sampler/gibbs/wv"], f["sampler/gibbs/wz"], f["sampler/gibbs/wk"] = data.gibbs.wv, data.gibbs.wz, data.gibbs.wk
            end
        end
    end
    C.last = time()
    C.nwritten += 1
    logmsg(cfg, @sprintf("snapshot %s: %s  [%.1f MB, %.1fs]", C.path, where_, filesize(C.path) / 1e6, C.last - t))
    return C.path
end

"""
    read_snapshot(path, cfg, P) -> NamedTuple or nothing

Read the snapshot at `path` (nothing if there is none): `phase`, `it`, `restarts` (resumes
before the run that wrote it), `co2`, the chain `states` (rebuilt with `make_state`, i.e.
evaluated under the current target), and `kinds`, `traces`, `co2_trace`, `sigma_scale`,
`gibbs` where present. Checks the format version, the levels, the seed and the sizes of
the state vectors against the configuration and the problem.
"""
function read_snapshot(path::String, cfg::Config, P::Problem)
    (isempty(path) || !isfile(path)) && return nothing
    h5open(path, "r") do f
        haskey(f, "sampler") && error("snapshot $path was written by the $(read(f["sampler"])) driver, not by run_chains")
        v = read(f["version"])
        v == SNAPSHOT_VERSION || error("snapshot $path: format version $v, the code reads $SNAPSHOT_VERSION")
        lv = read(f["config/levels"])
        lv == reduce(hcat, [[a, b] for (a, b) in cfg.levels]) || error("snapshot $path: levels $lv differ from the configuration $(cfg.levels)")
        read(f["config/seed"]) == cfg.seed || error("snapshot $path: seed $(read(f["config/seed"])), the configuration has $(cfg.seed)")
        K = read(f["nchains"])
        states = State[]
        kinds = String[]
        traces = Vector{Vector{Vector{Float64}}}()
        for c in 1:K
            g = f["chains/$c"]
            ℓ = read(g["level"])
            m, z0, k, vv = read(g["m"]), read(g["z0"]), read(g["k"]), read(g["v"])
            ws = [read(g["w/$j"]) for j in 1:ℓ]
            (length(k) == P.lk[ℓ+1].n && length(vv) == P.lv[ℓ+1].n && length(z0) == P.W.lays[1].n) ||
                error("snapshot $path: the state of chain $c does not fit the problem at level $ℓ")
            push!(states, make_state(P, ℓ, m, z0, ws, k, vv))
            haskey(g, "start") && push!(kinds, read(g["start"]))
            Q = haskey(g, "Q") ? read(g["Q"]) : zeros(0, 0)
            push!(traces, [Q[:, j] for j in 1:size(Q, 2)])
        end
        gibbs = haskey(f, "sampler/gibbs") ?
                (wv=read(f["sampler/gibbs/wv"]), wz=read(f["sampler/gibbs/wz"]), wk=read(f["sampler/gibbs/wk"])) : nothing
        return (; phase=Symbol(read(f["phase"])), it=read(f["it"]), restarts=read(f["restarts"]), where_=read(f["where"]),
                written=read(f["written"]), co2=read(f["co2_price"]), states, kinds, traces,
                co2_trace=haskey(f, "co2/price_trace") ? read(f["co2/price_trace"]) : Float64[],
                sigma_scale=haskey(f, "sampler/sigma_scale") ? read(f["sampler/sigma_scale"]) : nothing, gibbs)
    end
end

"""
    boundary!(C, cfg, P, phase, it, make_data; force=false)

At a sweep boundary (`it` sweeps of `phase` done): if the stop file exists, write a
snapshot, remove the stop file and throw `CheckpointStop`; else write one if `force` or
`checkpoint_minutes` have passed since the last write. `make_data()` builds the snapshot
data (called only when one is written; it has the field `states`), `writer` writes it
(`write_snapshot`; the replica driver: `write_replica_snapshot`).
"""
function boundary!(C::Checkpointer, cfg::Config, P::Problem, phase::Symbol, it::Int, make_data; force::Bool=false,
                   writer=write_snapshot)
    enabled(C) || return nothing
    C.hook === nothing || C.hook(phase, it)
    stop = !isempty(C.stop_file) && isfile(C.stop_file)
    if stop || force || time() - C.last >= C.interval
        data = make_data()
        writer(C, cfg, P, (; phase, it, data...))
        if stop
            rm(C.stop_file; force=true)    # consumed: the resumed run does not stop at once
            throw(CheckpointStop(C.path, ckpt_where(phase, it, cfg), [copy(x) for x in data.states]))
        end
    end
    return nothing
end

"""
    ignore_sigterm()

Make the process ignore SIGTERM (`ESM_IGNORE_SIGTERM=1` in scripts/run.jl). Julia blocks
SIGTERM in its threads and receives it with `sigwait` in a listener thread, so `SIG_IGN`
alone does not help (a blocked signal is never discarded): the calling thread (the main
thread, to which `kill` directs the signal) also unblocks it, and the kernel then discards
it. Slurm's preemption SIGTERM then leaves the driver running; the job wrapper creates the
stop file instead, and the driver writes a snapshot and exits with code 75 at the next sweep
boundary (Slurm's SIGKILL follows after the grace time).
"""
function ignore_sigterm()
    SIGTERM, SIG_IGN, SIG_UNBLOCK = 15, Ptr{Cvoid}(1), 1
    ccall(:signal, Ptr{Cvoid}, (Cint, Ptr{Cvoid}), SIGTERM, SIG_IGN) == Ptr{Cvoid}(-1) && error("signal(SIGTERM, SIG_IGN) failed")
    set = zeros(UInt64, 16)                                    # sigset_t (1024 bits)
    ccall(:sigemptyset, Cint, (Ptr{UInt64},), set)
    ccall(:sigaddset, Cint, (Ptr{UInt64}, Cint), set, SIGTERM)
    ccall(:pthread_sigmask, Cint, (Cint, Ptr{UInt64}, Ptr{Cvoid}), SIG_UNBLOCK, set, C_NULL) == 0 || error("pthread_sigmask failed")
    return nothing
end
