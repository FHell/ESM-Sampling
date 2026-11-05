# Wall and CPU time per stage of a run (docs/system/hpc.md §6): rows appended to `stages.csv`
# next to the run's output, read by hpc/cost_table.jl into the campaign's cost table.

"CPU seconds (user + system, all threads) of this process so far; NaN where /proc is missing."
function process_cpu_seconds()
    try
        s = read("/proc/self/stat", String)
        f = split(s[findlast(')', s)+2:end])          # fields from 3 (state) on
        tck = ccall(:sysconf, Clong, (Cint,), 2)        # _SC_CLK_TCK
        return (parse(Int, f[12]) + parse(Int, f[13])) / tck
    catch
        return NaN
    end
end

"""
    StageClock(dir) -> clock;  stage!(clock, name);  stage!(clock, name)…

Times the stages of a run: each `stage!(clock, name)` closes the stage that began at the
previous call (or at the clock's creation) under `name` and appends one row to
`dir/stages.csv`: `attempt, stage, wall_s, cpu_s, cpus, threads, start, end`. `attempt` is the
job attempt (`ESM_ATTEMPT`, set by hpc/run.sh; 0 outside it), `cpus` the allocated cores
(`SLURM_CPUS_PER_TASK`, else the machine's), `threads` Julia's threads. A run stopped in the
middle of a stage closes it with its partial time (call `stage!` on the way out).
"""
mutable struct StageClock
    path::String
    attempt::Int
    cpus::Int
    wall::Float64
    cpu::Float64
    start::String
end

function StageClock(dir::AbstractString)
    mkpath(dir)
    path = joinpath(dir, "stages.csv")
    isfile(path) || open(io -> println(io, "attempt,stage,wall_s,cpu_s,cpus,threads,start,end"), path, "w")
    attempt = parse(Int, get(ENV, "ESM_ATTEMPT", "0"))
    cpus = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", string(Sys.CPU_THREADS)))
    return StageClock(path, attempt, cpus, time(), process_cpu_seconds(), Libc.strftime("%Y-%m-%dT%H:%M:%S", time()))
end

function stage!(c::StageClock, name::AbstractString)
    w, u = time(), process_cpu_seconds()
    e = Libc.strftime("%Y-%m-%dT%H:%M:%S", time())
    nm = replace(name, ',' => ';')
    open(c.path, "a") do io
        @printf(io, "%d,%s,%.3f,%.3f,%d,%d,%s,%s\n", c.attempt, nm, w - c.wall, u - c.cpu, c.cpus, Threads.nthreads(), c.start, e)
    end
    c.wall, c.cpu, c.start = w, u, e
    return c
end
