# GPU (Reactant/XLA) vs CPU timing of the padded batched line sampler and of the
# batched ADMM congestion projection, on synthetic data of realistic sizes.
# Usage: nice -n 19 julia -t 4 --project=external/reactant-bench external/reactant-bench/bench.jl [what]
using Reactant, Random, Printf, LinearAlgebra
include(joinpath(@__DIR__, "linesampler.jl"))

println("Reactant devices: ", Reactant.devices())
BLAS.set_num_threads(4)

function moves(rng, ::Type{T}, E, C, B) where {T}
    nY = 4 * E * B
    Y = T.(rand(rng, nY))
    S = T.(randn(rng, 4 * C * B))
    idx = reshape(randperm(rng, nY)[1:E*B], E, B)
    cidx = reshape(randperm(rng, 4 * C * B)[1:C*B], C, B)
    slot = rand(rng, 1:C, E, B)
    a = T.(randn(rng, E, B)); a[rand(rng, E, B) .< 0.25] .= 0       # some padding
    ci = T.(rand(rng, E, B)); cv = fill(T(30), E, B); κ = T.(randn(rng, E, B))
    nc = ones(T, C, B)
    u1 = T.(rand(rng, 1, B)); u2 = T.(rand(rng, 1, B))
    return (Y, S, idx, slot, a, ci, cv, κ, cidx, nc, u1, u2)
end

function bench_line(T, E, C, B; reps=20, cpu=true)
    rng = MersenneTwister(1)
    args = moves(rng, T, E, C, B)
    cs, cd, β = T(10), T(0.5), T(1)
    tcpu = NaN
    if cpu
        line_round(args..., cs, cd, β)
        n = max(1, min(reps, 2_000_000 ÷ B))
        tcpu = @elapsed for _ in 1:n
            line_round(args..., cs, cd, β)
        end
        tcpu /= n
    end
    rargs = map(Reactant.to_rarray, args)
    tc = @elapsed f = @compile line_round(rargs..., cs, cd, β)
    Y, S = rargs[1], rargs[2]
    Y, S, t = f(Y, S, rargs[3:end]..., cs, cd, β); Array(t)
    tg = @elapsed begin
        for _ in 1:reps
            Y, S, t = f(Y, S, rargs[3:end]..., cs, cd, β)
        end
        Array(t)
    end
    tg /= reps
    println(@sprintf("line %-7s E=%2d C=%2d B=%7d: compile %5.1f s | GPU %8.3f ms/round = %8.1f ns/move | CPU (1 thr, broadcast) %8.3f ms = %7.0f ns/move | ratio %5.1f",
                     T, E, C, B, tc, 1e3 * tg, 1e9 * tg / B, 1e3 * tcpu, 1e9 * tcpu / B, tcpu / tg))
    flush(stdout)
end

function admm(Minv, G, A, lo, hi, Pm, ρ, iters)
    Gp = transpose(G) * Pm
    X = Minv * Gp
    AX = A * X
    Y = clamp.(AX, lo, hi)
    Λ = AX .- Y
    @trace for _ in 1:iters
        X = Minv * (Gp .+ ρ .* (transpose(A) * (Y .- Λ)))
        AX = A * X
        Y = clamp.(AX .+ Λ, lo, hi)
        Λ = Λ .+ AX .- Y
    end
    return X, Y
end

function admm_cpu(Minv, G, A, lo, hi, Pm, ρ, iters)
    Gp = G' * Pm
    X = Minv * Gp
    AX = A * X
    Y = clamp.(AX, lo, hi)
    Λ = AX .- Y
    for _ in 1:iters
        X = Minv * (Gp .+ ρ .* (A' * (Y .- Λ)))
        AX = A * X
        Y = clamp.(AX .+ Λ, lo, hi)
        Λ = Λ .+ AX .- Y
    end
    return X, Y
end

function bench_admm(T, N, nl, nk, Tst; iters=400)
    rng = MersenneTwister(2)
    b0 = rand(rng, 1:N, nl); b1 = [mod1(x + rand(rng, 1:5), N) for x in b0]
    D = zeros(nl, N)
    for e in 1:nl
        D[e, b0[e]] += 1.0; D[e, b1[e]] -= 1.0
    end
    for n in 1:N-1          # make it connected
        D = vcat(D, (1:N .== n)' .- (1:N .== n + 1)')
    end
    nl = size(D, 1)
    L = D' * D
    Cm = zeros(N, nk); for l in 1:nk; Cm[rand(rng, 1:N), l] += 1; Cm[rand(rng, 1:N), l] -= 1; end
    G = hcat(L, Cm)
    A = [D zeros(nl, nk); zeros(nk, N) Matrix(1.0I, nk, nk)]
    M = G' * G + (A' * A) + 1e-6I
    Minv = inv(Symmetric(M))
    lo = fill(-1.0, nl + nk); hi = fill(1.0, nl + nk)
    Pm = randn(rng, N, Tst)
    cv(x) = T.(x)
    a = (cv(Matrix(Minv)), cv(G), cv(A), cv(lo), cv(hi), cv(Pm))
    admm_cpu(a..., T(1), 2)
    tcpu = @elapsed admm_cpu(a..., T(1), iters)
    ra = map(Reactant.to_rarray, a)
    it = Reactant.ConcreteRNumber(iters)
    tc = @elapsed f = @compile admm(ra..., T(1), it)
    X, Y = f(ra..., T(1), it); Array(X)
    tg = @elapsed begin
        X, Y = f(ra..., T(1), it); Array(X)
    end
    flops = iters * Tst * 2 * ((N + nk)^2 + 2 * (nl + nk) * (N + nk))
    println(@sprintf("ADMM %-7s N=%4d lines=%4d T=%5d, %d its: compile %5.1f s | GPU %7.3f s (%6.0f GFLOP/s) | CPU 4 thr BLAS %7.3f s (%6.0f GFLOP/s) | ratio %5.1f",
                     T, N, nl, Tst, iters, tc, tg, flops / tg / 1e9, tcpu, flops / tcpu / 1e9, tcpu / tg))
    flush(stdout)
end

what = length(ARGS) > 0 ? ARGS[1] : "all"
if what in ("all", "line")
    for T in (Float32, Float64), (E, C) in ((4, 4), (16, 8)), B in (182, 4096, 65536)
        (E == 16 && B > 16384 && T == Float64) && (B = 16384)
        bench_line(T, E, C, B; cpu=(B <= 65536))
    end
end
if what in ("all", "admm")
    for T in (Float32, Float64)
        bench_admm(T, 256, 480, 4, 365)
    end
end

# R rounds inside one compiled program (device loop): per-round latency without host dispatch
function many_rounds(Y, S, idx, slot, a, ci, cv, κ, cidx, nc, u1, u2, cs, cd, β, R)
    t = u1
    @trace for _ in 1:R
        Y, S, t = line_round(Y, S, idx, slot, a, ci, cv, κ, cidx, nc, u1, u2, cs, cd, β)
    end
    return Y, S, t
end

function bench_loop(T, E, C, B; R=100)
    rng = MersenneTwister(1)
    rargs = map(Reactant.to_rarray, moves(rng, T, E, C, B))
    cs, cd, β = T(10), T(0.5), T(1)
    Rr = Reactant.ConcreteRNumber(R)
    tc = @elapsed f = @compile many_rounds(rargs..., cs, cd, β, Rr)
    Y, S, t = f(rargs..., cs, cd, β, Rr); Array(t)
    tg = @elapsed begin
        Y, S, t = f(rargs..., cs, cd, β, Rr); Array(t)
    end
    println(@sprintf("loop %-7s E=%2d C=%2d B=%7d R=%d: compile %5.1f s | GPU %8.3f ms/round = %8.1f ns/move",
                     T, E, C, B, R, tc, 1e3 * tg / R, 1e9 * tg / R / B))
    flush(stdout)
end

if what == "loop"
    for T in (Float32, Float64), B in (182, 16384)
        bench_loop(T, 4, 4, B)
    end
end
