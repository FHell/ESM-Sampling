# Where the compile time of one family's round goes (docs/system/device.md §7, "Debugging compile
# time"): the raw trace (no passes: op count, `stablehlo.while`, the largest tensors), then timed
# compiles (`Reactant.Profiler.@timed_compile`: tracing, MLIR passes, XLA). Model of the tests
# (40 nodes, no line losses or lossy links), a coarser year with H > 6 (T = 8760/H).
#   FAM=slong NSITE=2 H=24 STEPS=raw,none,full \
#     timeout --signal=KILL 900 nice -n 19 julia --project=device -t 4 device/scripts/compile_probe.jl
# PROFSEC=n: profile the raw trace for n seconds instead, print the flat profile and exit.
const T0 = time()
include(joinpath(@__DIR__, "..", "test", "setup40.jl"))
using Profile
Reactant.set_default_backend("cpu")
stamp(s) = (println("[$(round(Int, time() - T0)) s] ", s); flush(stdout))
f = Symbol(get(ENV, "FAM", "slong"))
nsite = parse(Int, get(ENV, "NSITE", "2"))
H = parse(Int, get(ENV, "H", "6"))
P, M, ix, S, ss = setup40(; C=2, losses=false, link_losses=false, levels=H == 6 ? nothing : [(10, H), (40, H)])
LM = ML.LocalModel(P, ss[1].ℓ)
D.set_sources!(P, LM, ix)
cls = zeros(2, M.T * M.N)
pl = D.plan(M, S, P.βv; families=(f,), cls=cls, bmax=Dict(f => nsite))
tabs = NamedTuple{(f,)}((D.device_table(pl, f, D.host_table(f, pl.tp, pl.sched[f][1:1])),))
B, R = size(tabs[f].on)
U = NamedTuple{(f,)}((rand(MersenneTwister(1), 2, B, 3, R),))
fv = Val((f,))
runU(st, cp, sp, k, tabs, U) = D.run_rounds(st, cp, sp, k, tabs, 1, nothing, U, fv)
args = map(Reactant.to_rarray, (pl.st, pl.cp, pl.sp, pl.k, tabs, U))
stamp("$f: T = $(M.T), B = $B")
if haskey(ENV, "PROFSEC")
    Profile.init(n=10^7, delay=0.02)
    Profile.start_timer()
    tk = Threads.@spawn Reactant.@code_hlo optimize = false runU(args...)
    t1 = time()
    while !istaskdone(tk) && time() - t1 < parse(Int, ENV["PROFSEC"]); sleep(5); end
    Profile.stop_timer()
    stamp("trace done: $(istaskdone(tk))")
    Profile.print(IOContext(stdout, :displaysize => (3000, 260)); format=:flat, sortedby=:count, mincount=300, noisefloor=1)
    exit(0)
end
for step in split(get(ENV, "STEPS", "raw,full"), ",")
    if step == "raw"
        t = @elapsed h = Reactant.@code_hlo optimize = false runU(args...)
        s = string(h)
        ops = Dict{String,Int}()
        for m in eachmatch(r"= (\"?[a-z_]+\.[a-z_]+)", s); ops[m.captures[1]] = get(ops, m.captures[1], 0) + 1; end
        big = maximum((prod(parse.(Int, split(m.captures[1], "x"))) for m in eachmatch(r"tensor<([0-9x]+)xf64>", s)); init=0)
        stamp("raw trace $(round(t; digits=1)) s, $(length(s) ÷ 1000) kB, $(sum(values(ops))) ops, while $(get(ops, "stablehlo.while", 0)), " *
              "largest f64 tensor $big; " * join(("$k $v" for (k, v) in sort(collect(ops); by=last, rev=true)[1:min(end, 8)]), ", "))
    elseif step == "none" || step == "full"
        opt = step == "none" ? :none : true
        t = @elapsed (c, tm) = Reactant.Profiler.@timed_compile optimize = opt runU(args...)
        stamp("timed compile optimize=$opt: $(round(t; digits=1)) s; $tm")
    end
end
