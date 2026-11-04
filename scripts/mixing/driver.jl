# Instrumented multi-chain driver for the mixing diagnosis (docs/mixing_diagnosis.md).
# Mirrors `run_chains` at α = 0 (same starts, same samplers, `fine_local_step` structure),
# records per sweep: node capacities, per-unit capacity changes by kscale and kpeak, the
# z0 row means and seasonal means, regimes, cost parts, the monitored quantities, and the
# wall time of each component. Extra (prototype) moves are passed as `extras`, a list of
# (name, f!) with f!(rng, P, LM, s, rec) -> s, run after the standard step.

include(joinpath(@__DIR__, "common.jl"))

"Seasonal means (4 blocks of the year) of every z0 row."
function z0_seasonal(P, z0)
    lay = P.W.lays[1]
    Z = ML.z0matrix(z0, lay, P.W.specs)
    T0 = size(Z, 2)
    edges = round.(Int, range(0, T0, length=5))
    return vcat([vec(mean(Z[:, edges[q]+1:edges[q+1]]; dims=2)) for q in 1:4]...)
end

z0_rowmeans(P, z0) = vec(mean(ML.z0matrix(z0, P.W.lays[1], P.W.specs); dims=2))

"Flatten node capacities."
kflat(P, s) = reduce(vcat, ML.knodes(P, s.ℓ, s.k))

"""
One fine step as `fine_local_step` (kernel :local), timed per component, with the capacity
changes of the kscale and kpeak passes recorded; then the extra moves.
"""
function step_timed(rng, P, s, S, extras; kscale_passes::Int=1, weather::Bool=true, capacity::Bool=true)
    K = S.loc
    LM = ML.local_model(K, P, s.ℓ)
    s = copy(s)
    tm = Dict{String,Float64}()
    dk = Dict{String,Vector{Float64}}()
    function kmoves(x)
        k0 = kflat(P, x)
        a = time()
        for _ in 1:kscale_passes
            x = ML.kscale_sweep!(rng, P, LM, x)[1]
        end
        b = time()
        k1 = kflat(P, x)
        x = ML.kpeak_sweep!(rng, P, LM, x)[1]
        c = time()
        k2 = kflat(P, x)
        tm["kscale"] = b - a; tm["kpeak"] = c - b
        dk["kscale"] = log.(max.(k1, 1e-9)) .- log.(max.(k0, 1e-9))
        dk["kpeak"] = log.(max.(k2, 1e-9)) .- log.(max.(k1, 1e-9))
        return x
    end
    function op(x)
        a = time()
        x, st = ML.local_sweep!(rng, P, LM, x; wsets=K.wsets, extra=K.extra)
        tm["local"] = time() - a
        return x
    end
    if !capacity
        s = op(s)
    elseif rand(rng) < 0.5
        s = op(s); s = kmoves(s)
    else
        s = kmoves(s); s = op(s)
    end
    if weather
        a = time()
        s = ML.maybe_weather!(rng, P, K, LM, s)
        tm["weather"] = time() - a
    end
    for (name, f!) in extras
        a = time()
        k0 = kflat(P, s)
        s = f!(rng, P, LM, s)
        tm[name] = time() - a
        dk[name] = log.(max.(kflat(P, s), 1e-9)) .- log.(max.(k0, 1e-9))
    end
    return s, tm, dk
end

"""
    run_mix(cfg, P, sL; nsweeps, extras, out, label, chains=nothing) -> rec

Run `cfg.nchains` chains from the `run_chains` starts (or the given `chains`), record
everything, write `out` (HDF5).
"""
function run_mix(cfg, P, sL; nsweeps::Int=200, extras=Tuple{String,Function}[], out::String="", label::String="run",
                 chains=nothing, seedoff::Int=0, kscale_passes::Int=1, weather::Bool=true, every::Int=50, capacity::Bool=true)
    kinds, ch0, samplers, rngs = mix_chains(cfg, P, sL)
    chains = chains === nothing ? ch0 : chains
    seedoff != 0 && (rngs = [MersenneTwister(cfg.seed + 1000 * c + seedoff) for c in eachindex(chains)])
    Kc = length(chains)
    L = ML.nlevels(P) - 1
    LM = ML.local_model(samplers[1].loc, P, L)
    names = ML.monitor_names(P)
    rec = Dict{String,Any}()
    rec["kinds"] = kinds
    rec["names"] = names
    rec["Q"] = [zeros(length(names), nsweeps) for _ in 1:Kc]
    nk = length(kflat(P, chains[1]))
    rec["k"] = [zeros(nk, nsweeps) for _ in 1:Kc]
    rec["z0m"] = [zeros(ML.n0(P.W), nsweeps) for _ in 1:Kc]
    rec["z0s"] = [zeros(4 * ML.n0(P.W), nsweeps) for _ in 1:Kc]
    rec["parts"] = [zeros(8, nsweeps) for _ in 1:Kc]
    rec["L"] = [zeros(nsweeps) for _ in 1:Kc]
    rec["time"] = Dict{String,Vector{Float64}}()
    rec["dk"] = Dict{String,Vector{Matrix{Float64}}}()
    t0 = time()
    for it in 1:nsweeps
        for c in 1:Kc
            s, tm, dk = step_timed(rngs[c], P, chains[c], samplers[c], extras; kscale_passes=kscale_passes, weather=weather,
                                   capacity=capacity)
            chains[c] = s
            for (n, v) in tm
                push!(get!(rec["time"], n, Float64[]), v)
            end
            for (n, v) in dk
                M = get!(rec["dk"], n, [zeros(nk, nsweeps) for _ in 1:Kc])
                M[c][:, it] .= v
            end
            rec["Q"][c][:, it] .= ML.monitor_quantities(P, s; LM=LM)
            rec["k"][c][:, it] .= kflat(P, s)
            rec["z0m"][c][:, it] .= z0_rowmeans(P, s.z0)
            rec["z0s"][c][:, it] .= z0_seasonal(P, s.z0)
            p = s.parts
            rec["parts"][c][:, it] .= [p.capex, p.op, p.gen, p.storage, p.shed, p.dump, p.penalty, p.co2]
            rec["L"][c][it] = s.L
        end
        if it % every == 0 || it == nsweeps
            tot = [sum(rec["k"][c][:, it]) for c in 1:Kc]
            @printf("[%s] sweep %d/%d  ktot %s  L %s  [%.0fs]\n", label, it, nsweeps,
                    join((@sprintf("%.4g", x) for x in tot), " "), join((@sprintf("%.6g", rec["L"][c][it]) for c in 1:Kc), " "), time() - t0)
            flush(stdout)
            isempty(out) || save_rec(out, rec, it)
        end
    end
    rec["chains"] = chains
    rec["wall"] = time() - t0
    return rec
end

function save_rec(out, rec, upto)
    mkpath(dirname(out))
    h5open(out, "w") do f
        f["names"] = rec["names"]
        f["kinds"] = rec["kinds"]
        f["upto"] = upto
        for c in eachindex(rec["Q"])
            f["chains/$c/Q"] = rec["Q"][c][:, 1:upto]
            f["chains/$c/k"] = rec["k"][c][:, 1:upto]
            f["chains/$c/z0m"] = rec["z0m"][c][:, 1:upto]
            f["chains/$c/z0s"] = rec["z0s"][c][:, 1:upto]
            f["chains/$c/parts"] = rec["parts"][c][:, 1:upto]
            f["chains/$c/L"] = rec["L"][c][1:upto]
            for (n, M) in rec["dk"]
                f["chains/$c/dk_$n"] = M[c][:, 1:upto]
            end
        end
        for (n, v) in rec["time"]
            f["time/$n"] = v
        end
    end
end
