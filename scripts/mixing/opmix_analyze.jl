using Serialization, Statistics, LinearAlgebra, Printf
S = ARGS[1]
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
"Drift: (mean of the last third − mean of the first third) / sd."
drift(x) = (n = length(x); a = x[1:n÷3]; b = x[end-n÷3+1:end]; (mean(b) - mean(a)) / std(x))
for c in (4, 1)
    r = deserialize(joinpath(S, "chain$c.jls"))
    n = length(r["op"]); b = 200
    @printf("\n=== chain %d (%s), %d sweeps, burn-in %d, %.2f s/sweep\n", c, c == 4 ? "heuristic" : "low", n, b, mean(r["time"][2:end]))
    show1(name, x) = @printf("  %-36s τ = %7.1f  drift %+5.2f sd   mean %.4g  sd %.3g\n", name, iat(x[b+1:end]), drift(x[b+1:end]), mean(x[b+1:end]), std(x[b+1:end]))
    @printf("  opex sweep 1 → 200 → 1000: %.4e → %.4e → %.4e\n", r["op"][1], r["op"][b], r["op"][end])
    show1("opex", r["op"]); show1("shed", r["shed"])
    Sc = reduce(hcat, r["score"])
    for (j, nm) in enumerate(r["knames"]); show1("score Σ " * nm, Sc[j, :]); end
    soc = r["soc"]
    for (si, sn) in enumerate(r["snames"])
        A = [sum(x[si, :, :]) for x in soc]
        maximum(abs, A) < 1 && continue
        show1("stored " * sn * " (all regions)", A)
        R = reduce(hcat, [vec(sum(x[si, :, :]; dims=2)) for x in soc])
        τr = [iat(R[i, b+1:end]) for i in axes(R, 1) if std(R[i, b+1:end]) > 1e-9 * max(1.0, maximum(abs, R))]
        isempty(τr) || @printf("  %-36s τ median %.1f max %.1f (%d regions)\n", sn * " per region", median(τr), maximum(τr), length(τr))
        Q = reduce(hcat, [vec(x[si, :, :]) for x in soc])
        X = Q[:, b+1:end]; X = X .- mean(X; dims=2)
        F = svd(X)
        for k in 1:min(3, size(F.U, 2))
            pc = vec(F.U[:, k]' * Q[:, b+1:end])
            Ur = reshape(F.U[:, k], :, 4)
            sysw = norm(sum(Ur; dims=1)) / (sqrt(size(Ur, 1)) * norm(Ur))   # 1: same in every region
            @printf("    PC%d %5.1f %% of var, regional uniformity %.2f: τ = %.1f, drift %+5.2f sd\n", k, 100 * F.S[k]^2 / sum(F.S .^ 2), sysw, iat(pc), drift(pc))
        end
        swing = reduce(hcat, [x[si, :, 1] .- x[si, :, 3] for x in soc])
        con = swing[:, b+1:end] .- mean(swing[:, b+1:end]; dims=1)
        τc = [iat(con[i, :]) for i in axes(con, 1) if std(con[i, :]) > 1e-9 * max(1.0, maximum(abs, swing))]
        isempty(τc) || @printf("  %-36s τ median %.1f max %.1f\n", sn * " inter-regional Q1−Q3 contrast", median(τc), maximum(τc))
    end
end
