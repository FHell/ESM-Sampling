# The src kernel (`fine_local_step`) with and without `kop` (docs/mixing_diagnosis.md §4,
# docs/system/moves/capacity.md): chains from the `run_chains` starts, recorded per sweep
# (monitored quantities, node capacities, log target, b̂, wall time), HDF5 readable by
# scripts/mixing/analyze.jl (`load_rec`, `compare_runs`). Usage:
#   nice -n 19 julia --project=. -t 4 scripts/mixing/run_kop_ab.jl <name> <nsweeps> <kop> [key=value ...]
include(joinpath(@__DIR__, "common.jl"))

name = ARGS[1]
nsw = parse(Int, ARGS[2])
kop = parse(Bool, ARGS[3])
ov = Dict{Symbol,Any}(:kop => kop)
for a in ARGS[4:end]
    k, v = split(a, "=")
    T = fieldtype(ML.Config, Symbol(k))
    ov[Symbol(k)] = T <: AbstractString ? String(v) : parse(T, v)
end
cfg, P, sL, info = mix_setup("configs/chains40.toml"; ov...)
println("overrides: ", ov); flush(stdout)
kinds, chains, samplers, rngs = mix_chains(cfg, P, sL)
L = ML.nlevels(P) - 1
LM = ML.LocalModel(P, L)
names = ML.monitor_names(P)
K = length(chains)
kflat(s) = reduce(vcat, ML.knodes(P, s.ℓ, s.k))
Q = [zeros(length(names), nsw) for _ in 1:K]
Kn = [zeros(length(kflat(chains[1])), nsw) for _ in 1:K]
Ls = [zeros(nsw) for _ in 1:K]
Bs = [zeros(nsw) for _ in 1:K]
tstep = Float64[]
out = "results/mixing/$name.h5"
function save(upto)
    h5open(out, "w") do f
        f["names"] = names; f["kinds"] = kinds; f["upto"] = upto
        for c in 1:K
            f["chains/$c/Q"] = Q[c][:, 1:upto]; f["chains/$c/k"] = Kn[c][:, 1:upto]
            f["chains/$c/L"] = Ls[c][1:upto]; f["chains/$c/bias"] = Bs[c][1:upto]
            f["chains/$c/z0m"] = zeros(1, upto); f["chains/$c/z0s"] = zeros(1, upto); f["chains/$c/parts"] = zeros(1, upto)
        end
        f["time/step"] = tstep
    end
end
t0 = time()
for it in 1:nsw
    for c in 1:K
        a = time()
        chains[c] = ML.fine_local_step(rngs[c], P, chains[c], samplers[c])
        push!(tstep, time() - a)
        Q[c][:, it] .= ML.monitor_quantities(P, chains[c]; LM=LM)
        Kn[c][:, it] .= kflat(chains[c])
        Ls[c][it] = chains[c].L
        Bs[c][it] = ML.bias_term(P, chains[c])
    end
    if it % 10 == 0 || it == nsw
        @printf("[%s] sweep %d/%d  ktot %s  L %s  b̂ %s  [%.0fs]\n", name, it, nsw,
                join((@sprintf("%.4g", sum(Kn[c][:, it])) for c in 1:K), " "), join((@sprintf("%.6g", Ls[c][it]) for c in 1:K), " "),
                join((@sprintf("%.4g", Bs[c][it]) for c in 1:K), " "), time() - t0)
        flush(stdout)
        save(it)
    end
end
println("done, wall ", time() - t0)
