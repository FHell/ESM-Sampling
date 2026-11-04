# Analysis helpers for the mixing runs (scripts/mixing/driver.jl outputs).
include(joinpath(@__DIR__, "common.jl"))

"Load a run file into a Dict."
function load_rec(path)
    h5open(path) do f
        K = length(keys(f["chains"]))
        d = Dict{String,Any}("names" => read(f["names"]), "kinds" => read(f["kinds"]), "upto" => read(f["upto"]))
        for key in ("Q", "k", "z0m", "z0s", "parts", "L")
            d[key] = [read(f["chains/$c/$key"]) for c in 1:K]
        end
        d["dk"] = Dict{String,Any}()
        for n in keys(f["chains/1"])
            startswith(n, "dk_") && (d["dk"][n[4:end]] = [read(f["chains/$c/$n"]) for c in 1:K])
        end
        d["time"] = Dict(n => read(f["time/$n"]) for n in keys(f["time"]))
        return d
    end
end

"Unit labels (carrier, node index) in the order of kflat."
function unit_labels(P)
    E = P.E
    lab = Tuple{String,Int}[]
    for sp in E.kspecs
        c = String(sp.name)[3:end]
        append!(lab, [(c, n) for n in sp.B.support])
    end
    return lab
end

"Rank-normalized R̂ and bulk ESS of a quantity over chains on a window."
function rhat_ess(xs::Vector{<:AbstractVector})
    return ML.rhat_rank(xs), ML.ess_bulk(xs)
end

"CPU seconds per sweep and chain from the time records."
cpu_per_sweep(d) = sum(sum(v) for v in values(d["time"])) / (length(d["L"]) * d["upto"])

"Per-carrier capacity move statistics and chain spread over the window w (sweeps)."
function capacity_table(d, lab, w; io=stdout)
    K = length(d["k"])
    up = d["upto"]
    for c in unique(first.(lab))
        ii = findall(l -> l[1] == c, lab)
        stats = Dict{String,Float64}()
        for (n, M) in d["dk"]
            x = reduce(vcat, [vec(abs.(M[q][ii, w])) for q in 1:K])
            stats[n] = median(x)
            stats[n*"_rms"] = sqrt(mean(abs2, x))
        end
        lk = [log.(max.(d["k"][q][ii, up], 1.0)) for q in 1:K]
        spread = median([std([lk[q][j] for q in 1:K]) for j in eachindex(ii)])
        tot = [vec(sum(d["k"][q][ii, :]; dims=1)) for q in 1:K]
        rh = ML.rhat_rank([tot[q][w] for q in 1:K])
        drift = mean([(log(max(tot[q][w[end]], 1)) - log(max(tot[q][w[1]], 1))) / length(w) for q in 1:K])
        @printf(io, "%-8s %s | spread sd(log k) %.3f | R̂(total) %.2f | drift %+.2e/sweep | totals %s\n", c,
                join([@sprintf("%s med %.1e rms %.1e", n, stats[n], stats[n*"_rms"]) for n in sort(collect(keys(d["dk"])))], "; "),
                spread, rh, drift, join([@sprintf("%.3g", tot[q][up]) for q in 1:K], " "))
    end
end

"R̂ and bulk ESS of every monitored quantity on the window w; prints the worst ones."
function rhat_table(d, w; nshow=25, io=stdout)
    K = length(d["Q"])
    names = d["names"]
    res = []
    for (i, n) in enumerate(names)
        xs = [d["Q"][q][i, w] for q in 1:K]
        all(x -> all(isfinite, x), xs) || continue
        std(reduce(vcat, xs)) == 0 && continue
        push!(res, (n, ML.rhat_rank(xs), ML.ess_bulk(xs)))
    end
    sort!(res; by=x -> -x[2])
    for r in res[1:min(nshow, end)]
        @printf(io, "  %-24s R̂ %6.2f  ESS %6.1f\n", r...)
    end
    return res
end

"Compact comparison of runs: CPU per chain-sweep, L spread, carrier totals and R̂ on the second half."
function compare_runs(names; io=stdout)
    labp = nothing
    for n in names
        f = "results/mixing/$n.h5"
        isfile(f) || (println(io, "$n: missing"); continue)
        d = load_rec(f)
        up = d["upto"]
        w = up÷2+1:up
        Lend = [d["L"][q][up] for q in eachindex(d["L"])]
        @printf(io, "%-16s sweeps %d  CPU/chain-sweep %.2f s  L end %s (spread %.2e)\n", n, up, cpu_per_sweep(d),
                join((@sprintf("%.4g", x) for x in Lend), " "), maximum(Lend) - minimum(Lend))
        nm = d["names"]
        for c in ["k_onwind", "k_solar", "k_offwind", "k_gas", "k_battery", "k_H2", "opex", "capex", "H2_soc_mean", "z0mean_load"]
            i = findfirst(==(c), nm)
            i === nothing && continue
            xs = [d["Q"][q][i, w] for q in eachindex(d["Q"])]
            @printf(io, "    %-12s R̂ %5.2f  ESS %5.1f  end %s\n", c, ML.rhat_rank(xs), ML.ess_bulk(xs),
                    join((@sprintf("%.3g", d["Q"][q][i, up]) for q in eachindex(d["Q"])), " "))
        end
    end
end
