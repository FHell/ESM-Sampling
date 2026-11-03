# Correctness of the padded batched sampler on plain CPU arrays: B copies of one
# move, empirical CDF vs quadrature of exp(−β ΔC(t)).
using Random, Statistics, Printf
include(joinpath(@__DIR__, "linesampler.jl"))

function cost(t, y0, a, ci, cv, κ, slot, S0, nc, cs, cd)
    c = 0.0
    S = copy(S0)
    for j in eachindex(y0)
        y = y0[j] + t * a[j]; yc = clamp01(y)
        c += ci[j] * yc + cv[j] * abs(y - yc)
        S[slot[j]] += κ[j] * (yc - clamp01(y0[j]))
    end
    return c + sum(nc[q] * (S[q] > 0 ? cs * S[q] / nc[q] : -cd * S[q] / nc[q]) for q in eachindex(S))
end

function main()
    rng = MersenneTwister(1)
    E, C, B = 4, 2, 200_000
    y0 = [0.3, 0.8, 0.5, 0.0]; a = [1.0, -0.5, 0.7, 0.0]
    ci = [2.0, 1.0, 0.5, 0.0]; cv = [30.0, 30.0, 30.0, 0.0]; κ = [1.0, -1.0, 2.0, 0.0]
    slot = [1, 1, 2, 1]; S0 = [0.2, -0.3]; nc = [1.0, 1.0]; cs, cd, β = 10.0, 0.5, parse(Float64, get(ENV, "BETA", "1.0"))
    Y = repeat(y0, B); S = repeat(S0, B)
    idx = reshape(collect(1:E*B), E, B); cidx = reshape(collect(1:C*B), C, B)
    rep(x) = repeat(x, 1, B)
    _, _, t = line_round(Y, S, idx, rep(slot), rep(a), rep(ci), rep(cv), rep(κ), cidx, rep(nc),
                         rand(rng, 1, B), rand(rng, 1, B), cs, cd, β)
    t = vec(t)
    grid = range(-60, 60; length=1_200_001)
    c0 = cost(0.0, y0, a, ci, cv, κ, slot, S0, nc, cs, cd)
    dens = [exp(-β * (cost(x, y0, a, ci, cv, κ, slot, S0, nc, cs, cd) - c0)) for x in grid]
    cdf = cumsum(dens); cdf ./= cdf[end]
    ts = sort(t)
    ks = maximum(abs(searchsortedlast(ts, x) / B - cdf[i]) for (i, x) in enumerate(grid) if i % 100 == 0)
    println(@sprintf("mean %.4f (quad %.4f), KS distance %.4f (B = %d; 1.36/√B = %.4f), range [%.2f, %.2f]",
                     mean(t), sum(grid .* dens) / sum(dens), ks, B, 1.36 / sqrt(B), extrema(t)...))
end
main()
