# Ledger E21: operation mixing before (E17, results/opmix/) and after the long-range storage
# moves (results/smix/), at equal CPU time. For each chain: the cost per sweep, opex at equal
# CPU time, and integrated autocorrelation times and drifts (after a burn-in of 200 sweeps, as
# E17) of opex, shed, the capacity scores, the stored energy per carrier, per region, the leading
# PCs, and the split into the regional level (annual mean) and the seasonal shape (quarter
# minus annual mean).
#
# Ledger E27: the same with base and new runs of any kernel (base prefix "slong_", new "sx_"), plus
# the seasonal shape per storage unit (Q1 − Q3 of the stored energy) and the long-range sweep's
# move statistics.
#
# Usage: julia --project=. smix_analyze.jl <base dir> <new dir> [new file prefix] [burn-in] [base file prefix] [chains, e.g. 4,1] [cost ratio new/base]
using Serialization, Statistics, LinearAlgebra, Printf

base, new = ARGS[1], ARGS[2]
prefix = length(ARGS) >= 3 ? ARGS[3] : "slong_"
burn = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 200
bprefix = length(ARGS) >= 5 ? ARGS[5] : ""
chains_ = length(ARGS) >= 6 ? parse.(Int, split(ARGS[6], ",")) : [4, 1]
# cost ratio new/base per sweep from an interleaved benchmark: replaces the recorded times (which
# carry the machine load of the time of each run) by base 1 and new `ratio` per sweep (in base sweeps)
ratio = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : NaN
fixt!(r, c) = (isnan(ratio) || (r["time"] = fill(c, length(r["time"]))); r)

"Integrated autocorrelation time, Sokal's automatic window (c = 5)."
function iat(x)
    n = length(x); y = x .- mean(x); v = sum(abs2, y) / n
    v == 0 && return NaN
    τ = 1.0
    for t in 1:n-1
        ρ = sum(y[1:n-t] .* y[1+t:n]) / ((n - t) * v)
        τ += 2ρ
        t >= 5τ && return τ
    end
    return Inf
end
drift(x) = (n = length(x); a = x[1:n÷3]; b = x[end-n÷3+1:end]; (mean(b) - mean(a)) / std(x))

"The quantities of a run as (name, series) pairs, sweeps sel."
function series(r, sel)
    out = Pair{String,Vector{Float64}}[]
    push!(out, "opex" => r["op"][sel], "shed" => r["shed"][sel])
    Sc = reduce(hcat, r["score"])
    for (j, nm) in enumerate(r["knames"]); push!(out, "score Σ " * nm => Sc[j, sel]); end
    return out
end

function storage_summary(r, sel)
    soc = r["soc"][sel]
    res = Dict{String,Any}()
    for (si, sn) in enumerate(r["snames"])
        A = [sum(x[si, :, :]) for x in soc]
        maximum(abs, A) < 1 && continue
        R = reduce(hcat, [vec(sum(x[si, :, :]; dims=2)) for x in soc])
        okr = [i for i in axes(R, 1) if std(R[i, :]) > 1e-9 * max(1.0, maximum(abs, R))]
        τr = [iat(R[i, :]) for i in okr]
        Q = reduce(hcat, [vec(x[si, :, :]) for x in soc])
        X = Q .- mean(Q; dims=2)
        F = svd(X)
        pcs = [(100 * F.S[k]^2 / sum(F.S .^ 2), iat(vec(F.U[:, k]' * Q)), drift(vec(F.U[:, k]' * Q))) for k in 1:min(3, size(F.U, 2))]
        lev = reduce(hcat, [vec(mean(x[si, :, :]; dims=2)) for x in soc])
        shp = reduce(hcat, [vec(x[si, :, :] .- mean(x[si, :, :]; dims=2)) for x in soc])
        okl = [i for i in axes(lev, 1) if std(lev[i, :]) > 1e-9 * max(1.0, maximum(abs, lev))]
        oks = [i for i in axes(shp, 1) if std(shp[i, :]) > 1e-9 * max(1.0, maximum(abs, shp))]
        res[sn] = (total=(iat(A), drift(A), std(A)), region=(isempty(τr) ? NaN : median(τr), isempty(τr) ? NaN : maximum(τr)), pcs=pcs,
                   level=(median(iat(lev[i, :]) for i in okl), maximum(iat(lev[i, :]) for i in okl), median(std(lev[i, :]) for i in okl)),
                   shape=(median(iat(shp[i, :]) for i in oks), maximum(iat(shp[i, :]) for i in oks), median(std(shp[i, :]) for i in oks)))
    end
    return res
end

for c in chains_
    rb = deserialize(joinpath(base, bprefix * "chain$c.jls"))
    rn = deserialize(joinpath(new, prefix * "chain$c.jls"))
    fixt!(rb, 1.0); fixt!(rn, ratio)
    tb, tn = cumsum(rb["time"]), cumsum(rn["time"])
    nb, nn = length(tb), length(tn)
    neq = searchsortedlast(tn, tb[end])               # sweeps of the new run within the base run's CPU time
    @printf("\n=== chain %d (%s)\n", c, c == 4 ? "heuristic" : "low")
    @printf("  cost per sweep: base %.3f s, %s %.3f s (median %.3f / %.3f); base %d sweeps = %.0f s; new %d sweeps, %d within that time\n",
            mean(rb["time"][2:end]), rn["kernel"], mean(rn["time"][2:end]), median(rb["time"]), median(rn["time"]), nb, tb[end], nn, neq)
    for tt in (60.0, 150.0, 300.0, 600.0, tb[end])
        ib, in_ = searchsortedlast(tb, tt), searchsortedlast(tn, tt)
        (ib >= 1 && in_ >= 1) && @printf("  opex at %5.0f s CPU: base %.5e (sweep %d)  new %.5e (sweep %d)\n", tt, rb["op"][ib], ib, rn["op"][in_], in_)
    end
    @printf("  opex sweep 1 → 200 → end: base %.5e → %.5e → %.5e; new %.5e → %.5e → %.5e\n",
            rb["op"][1], rb["op"][burn], rb["op"][end], rn["op"][1], rn["op"][burn], rn["op"][end])
    selb = burn+1:nb
    burn_n = searchsortedfirst(tn, tb[burn])             # burn-in of the new run at the same CPU time
    for (lab, seln) in (("all sweeps after the burn-in", burn+1:nn), ("equal CPU time", burn_n+1:neq))
        println("  -- new run: ", lab, " (sweeps ", first(seln), "–", last(seln), "); τ in sweeps, [τ × s/sweep in s]")
        sb, sn = series(rb, selb), series(rn, seln)
        sb_t, sn_t = mean(rb["time"][2:end]), mean(rn["time"][2:end])
        for ((nm, xb), (_, xn)) in zip(sb, sn)
            @printf("    %-20s base τ %6.1f [%5.0f s] drift %+5.2f   new τ %6.1f [%5.0f s] drift %+5.2f   (mean %.4g → %.4g)\n",
                    nm, iat(xb), iat(xb) * sb_t, drift(xb), iat(xn), iat(xn) * sn_t, drift(xn), mean(xb), mean(xn))
        end
        Sb, Sn = storage_summary(rb, selb), storage_summary(rn, seln)
        for sn in rb["snames"]
            haskey(Sb, sn) && haskey(Sn, sn) || continue
            b, n = Sb[sn], Sn[sn]
            @printf("    %-8s total  base τ %6.1f drift %+5.2f sd %.3g | new τ %6.1f drift %+5.2f sd %.3g\n", sn, b.total..., n.total...)
            @printf("    %-8s region τ med/max base %.0f/%.0f | new %.0f/%.0f\n", "", b.region..., n.region...)
            @printf("    %-8s level  τ med/max base %.0f/%.0f sd %.3g | new %.0f/%.0f sd %.3g\n", "", b.level..., n.level...)
            @printf("    %-8s shape  τ med/max base %.0f/%.0f sd %.3g | new %.0f/%.0f sd %.3g\n", "", b.shape..., n.shape...)
            for k in eachindex(b.pcs)
                k <= length(n.pcs) || continue
                @printf("    %-8s PC%d base %4.1f%% τ %6.1f drift %+5.2f | new %4.1f%% τ %6.1f drift %+5.2f\n", "", k, b.pcs[k]..., n.pcs[k]...)
            end
        end
        # seasonal shape per unit: Q1 − Q3 of the stored energy (units with ≥ 24 h and a range)
        if haskey(rb, "unitq") && haskey(rn, "unitq") && !isempty(rb["unitq"])
            us = get(rn, "ustore", nothing)
            Ub, Un = rb["unitq"][selb], rn["unitq"][seln]
            byc = Dict{String,Vector{NTuple{4,Float64}}}()
            for r in axes(Ub[1], 1)
                us === nothing && continue
                car, node, h = us[r]
                h >= 24 || continue
                xb = [x[r, 1] - x[r, 3] for x in Ub]; xn = [x[r, 1] - x[r, 3] for x in Un]
                (std(xb) > 1e-9 * max(1.0, maximum(abs, xb)) && std(xn) > 1e-9 * max(1.0, maximum(abs, xn))) || continue
                push!(get!(byc, car, NTuple{4,Float64}[]), (iat(xb), drift(xb), iat(xn), drift(xn)))
                car == "hydro" && @printf("    hydro %-8s Q1−Q3 base τ %6.1f drift %+5.2f | new τ %6.1f drift %+5.2f\n", node, iat(xb), drift(xb), iat(xn), drift(xn))
            end
            for (car, v) in sort(collect(byc))
                fin(x) = filter(isfinite, x)
                @printf("    %-8s Q1−Q3 per unit (%d): τ med/max base %.0f/%.0f |drift| med %.2f | new %.0f/%.0f |drift| med %.2f\n", car, length(v),
                        median(fin(first.(v))), maximum(fin(first.(v))), median(abs.(getindex.(v, 2))),
                        median(fin(getindex.(v, 3))), maximum(fin(getindex.(v, 3))), median(abs.(getindex.(v, 4))))
            end
        end
    end
    for (lab, r) in (("base", rb), ("new", rn))
        haskey(r, "lstats") && !isempty(r["lstats"]) || continue
        S = sum(r["lstats"]) ./ length(r["lstats"])
        rows = size(S, 1) >= 3 ? ("slong", "slevel", "sxlong") : ("slong", "slevel")
        for (q, nm) in enumerate(rows)
            S[q, 1] > 0 && @printf("  %s %-7s per sweep: %.0f moves, acceptance %.3f, %.3g MWh per move\n", lab, nm, S[q, 1], S[q, 2] / S[q, 1], S[q, 3] / S[q, 1])
        end
    end
end
