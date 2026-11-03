# Leak of the localized re-routing templates (xloc): for every (n, m) pair the residual
# change outside the ball (Σ|Δr| over nodes ≠ n, m; the exact re-routing has 0) and the
# template sizes, for several ball radii.
# Usage: nice -n 19 julia -t 1 --project=. scripts/xloc_leak.jl [data]
using ESMSampling, Statistics, Printf
const M = ESMSampling
data = length(ARGS) > 0 ? ARGS[1] : "data/processed/model.h5"
sys = M.load_system(data)
GS = M.GridSolver(sys)
N = sys.N
nodelines = [Int[] for _ in 1:N]
for e in eachindex(sys.line_s)
    push!(nodelines[sys.line_bus0[e]], e); push!(nodelines[sys.line_bus1[e]], e)
end
println("N = $N, lines = $(length(sys.line_s))")
for rb in 0:4
    part, xnod, xval = M.xloc_directions(sys, GS, nodelines; rp=1, rb=rb)
    leaks, sizes, lines = Float64[], Int[], Int[]
    d = zeros(N)
    for n in 1:N, (j, m) in enumerate(part[n])
        fill!(d, 0.0)
        d[xnod[n][j]] .= xval[n][j]
        r = zeros(N)
        ls = Set{Int}()
        for i in xnod[n][j], e in nodelines[i]; push!(ls, e); end
        for e in ls
            a, b = sys.line_bus0[e], sys.line_bus1[e]
            f = sys.line_b[e] * (d[a] - d[b])
            r[a] += f; r[b] -= f
        end
        r[n] -= 1; r[m] += 1
        push!(leaks, sum(abs, r) / 2)
        push!(sizes, length(xnod[n][j])); push!(lines, length(ls))
    end
    @printf("rb = %d: pairs %d, nodes mean %.1f max %d, lines mean %.1f, leak mean %.4f median %.4f p90 %.4f max %.4f\n",
            rb, length(leaks), mean(sizes), maximum(sizes), mean(lines), mean(leaks), median(leaks), quantile(leaks, 0.9), maximum(leaks))
end
