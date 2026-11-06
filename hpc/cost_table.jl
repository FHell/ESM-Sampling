# The cost table of a campaign (docs/system/hpc.md §6): wall time, allocated core-hours, consumed
# CPU hours, GPU-hours and energy, per run (all attempts, preempted ones included), per stage of
# a run (stages.csv: setup, LP optimum, every MGA solve, warmup, production, …) and for the
# campaign. Called by hpc/collect.sh after the copy; reads the collected run directories.
#
#   julia --startup-file=no hpc/cost_table.jl <campaign dir, e.g. campaigns/H2> [<out csv>]
#
# Columns of cost.csv: level (campaign | run | stage | device), run_id, stage, attempts, wall_h,
# alloc_core_h (allocated cores × wall: what fair share charges), cpu_h (CPU time used: sacct's
# TotalCPU per attempt, the process's user + system time per stage), gpu_h (GPUs × wall),
# energy_kwh (sacct, per attempt), status, note. An attempt without an end record (killed
# before the wrapper's end block) is completed from `sacct -j <job id> -D` where available.
using TOML, Printf

"Seconds of a Slurm duration ([D-]HH:MM:SS, MM:SS(.fff), HH:MM:SS(.fff)); NaN if empty."
function slurm_seconds(x)
    s = strip(string(x))
    isempty(s) && return NaN
    d = 0
    if occursin('-', s)
        a, s = split(s, '-'; limit=2)
        d = parse(Int, a)
    end
    p = parse.(Float64, split(s, ':'))
    sec = length(p) == 3 ? 3600p[1] + 60p[2] + p[3] : length(p) == 2 ? 60p[1] + p[2] : p[1]
    return 86400d + sec
end

"Number of GPUs of an attempt from its `gpus` field (a count or a device list)."
function ngpus(x)
    s = strip(string(x))
    isempty(s) && return 0
    n = tryparse(Int, s)
    n !== nothing && return n
    return length(split(s, ','))
end

"An attempt's end figures from sacct (by job id and start time), or nothing."
function sacct_attempt(jobid, start)
    (isempty(jobid) || Sys.which("sacct") === nothing) && return nothing
    out = try
        read(`sacct -j $jobid -n -P -X -D --format=Start,Elapsed,TotalCPU,ConsumedEnergyRaw,State`, String)
    catch
        return nothing
    end
    best, bd = nothing, Inf
    for l in split(out, '\n'; keepempty=false)
        c = split(l, '|')
        length(c) >= 5 || continue
        dt = abs(hash_time(c[1]) - hash_time(first(string(start), 19)))
        dt < bd && (best, bd = c, dt)
    end
    best === nothing && return nothing
    return (; elapsed=slurm_seconds(best[2]), cpu=slurm_seconds(best[3]), energy=something(tryparse(Float64, best[4]), NaN),
            state=best[5])
end
# seconds since an arbitrary origin of a "YYYY-MM-DDTHH:MM:SS" stamp (for nearest-start matching)
function hash_time(s)
    m = match(r"(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)", string(s))
    m === nothing && return Inf
    y, mo, d, h, mi, se = parse.(Int, m.captures)
    return ((((y * 12 + mo) * 31 + d) * 24 + h) * 60 + mi) * 60 + se
end

nanz(x) = isnan(x) ? 0.0 : x

function cost_rows(campdir)
    rows = Vector{Vector{String}}()
    runsdir = joinpath(campdir, "runs")
    tot = Dict("wall" => 0.0, "core" => 0.0, "cpu" => 0.0, "gpu" => 0.0, "energy" => 0.0)
    nruns = 0; natt = 0
    for run in sort(isdir(runsdir) ? readdir(runsdir) : String[])
        rt = joinpath(runsdir, run, "run.toml")
        isfile(rt) || continue
        atts = get(TOML.parsefile(rt), "attempt", Any[])
        nruns += 1
        wall = core = cpu = gpu = energy = 0.0
        ncpus = 1; ng = 0; status = ""; notes = String[]
        for a in atts
            natt += 1
            ncpus = Int(get(a, "cpus", 1)); ng = ngpus(get(a, "gpus", ""))
            el, cp, en = slurm_seconds(get(a, "elapsed", "")), slurm_seconds(get(a, "total_cpu", "")), tryparse(Float64, string(get(a, "energy_j", "")))
            en = something(en, NaN)
            status = string(get(a, "status", ""))
            if isnan(el)                                   # no end record: ask sacct
                sa = sacct_attempt(string(get(a, "job_id", "")), get(a, "start", ""))
                if sa === nothing
                    push!(notes, "attempt $(get(a, "restart_count", "?")) without end record")
                else
                    el, cp, en = sa.elapsed, sa.cpu, sa.energy
                    status = "sacct:" * sa.state
                end
            end
            wall += nanz(el); core += ncpus * nanz(el); cpu += nanz(cp); gpu += ng * nanz(el); energy += nanz(en)
        end
        push!(rows, ["run", run, "", string(length(atts)), @sprintf("%.4f", wall / 3600), @sprintf("%.4f", core / 3600),
                     @sprintf("%.4f", cpu / 3600), @sprintf("%.4f", gpu / 3600), @sprintf("%.4f", energy / 3.6e6), status,
                     join(notes, "; ")])
        tot["wall"] += wall; tot["core"] += core; tot["cpu"] += cpu; tot["gpu"] += gpu; tot["energy"] += energy
        # stages of the run, summed over attempts (a stage recorded in several attempts: re-done after a requeue)
        sc = joinpath(runsdir, run, "stages.csv")
        if isfile(sc)
            agg = Dict{String,Vector{Float64}}(); order = String[]; nat = Dict{String,Set{Int}}()
            for l in readlines(sc)[2:end]
                c = split(l, ',')
                length(c) >= 6 || continue
                nm = c[2]; w = parse(Float64, c[3]); u = parse(Float64, c[4]); k = parse(Int, c[5])
                haskey(agg, nm) || (agg[nm] = zeros(4); push!(order, nm); nat[nm] = Set{Int}())
                agg[nm] .+= [w, k * w, nanz(u), ng * w]
                push!(nat[nm], parse(Int, c[1]))
            end
            for nm in order
                v = agg[nm]
                push!(rows, ["stage", run, nm, string(length(nat[nm])), @sprintf("%.4f", v[1] / 3600), @sprintf("%.4f", v[2] / 3600),
                             @sprintf("%.4f", v[3] / 3600), @sprintf("%.4f", v[4] / 3600), "", "", ""])
            end
        end
        # device runs: compile times of the device programs (summary_device.toml, wall seconds)
        sd = joinpath(runsdir, run, "summary_device.toml")
        if isfile(sd)
            prof = get(TOML.parsefile(sd), "device_profile_seconds", Dict())
            for (k, v) in sort(collect(prof))
                startswith(k, "compile") || continue
                push!(rows, ["device", run, k, "", @sprintf("%.4f", v / 3600), @sprintf("%.4f", ncpus * v / 3600), "",
                             @sprintf("%.4f", ng * v / 3600), "", "", "wall of the device compile"])
            end
        end
    end
    pushfirst!(rows, ["campaign", basename(abspath(campdir)), "", string(natt), @sprintf("%.4f", tot["wall"] / 3600),
                      @sprintf("%.4f", tot["core"] / 3600), @sprintf("%.4f", tot["cpu"] / 3600), @sprintf("%.4f", tot["gpu"] / 3600),
                      @sprintf("%.4f", tot["energy"] / 3.6e6), "", "$nruns runs"])
    return rows
end

campdir = ARGS[1]
out = length(ARGS) > 1 ? ARGS[2] : joinpath(campdir, "cost.csv")
rows = cost_rows(campdir)
open(out, "w") do io
    println(io, "level,run_id,stage,attempts,wall_h,alloc_core_h,cpu_h,gpu_h,energy_kwh,status,note")
    for r in rows
        println(io, join(replace.(r, ',' => ';'), ","))
    end
end
println("cost table: $out ($(length(rows)) rows)")
