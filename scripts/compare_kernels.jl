# Compare level-0 base kernels (coordinate Gibbs vs reflective HMC) on the same
# regularized target, from the same start, for equal wall time. The bias is
# frozen at its LP warm start so that both chains target the same π₀.
#
# Usage: julia -t 8 --project=. scripts/compare_kernels.jl configs/quick.toml [seconds]
using ESMSampling, Random, Statistics, Printf, HDF5
const M = ESMSampling

"Effective sample size by Geyer's initial monotone sequence."
function ess(x::AbstractVector)
    n = length(x)
    n < 10 && return NaN
    y = x .- mean(x)
    v = sum(abs2, y) / n
    v == 0 && return NaN
    ρ(k) = sum(y[1:n-k] .* y[1+k:n]) / (n * v)
    s, prev = 0.0, Inf
    for k in 0:2:n-2
        Γ = ρ(k) + ρ(k + 1)
        Γ <= 0 && break
        Γ = min(Γ, prev)
        s += Γ
        prev = Γ
    end
    return n / max(2s - 1, 1.0)
end

function chain(kernel, P, s0, sL, cfg, seconds; rng)
    cfg.kernel = kernel
    S = M.new_sampler(P, sL, cfg)   # needs the finest-level state for its proposal scales
    s = M.evaluate!(P, copy(s0))
    S.adapt = true
    S.gibbs === nothing || (S.gibbs.adapt = true)
    t0 = time()
    ts, Ls, Qs = Float64[], Float64[], Vector{Vector{Float64}}()
    it = 0
    windows = cumsum([50, 100, 200, 400, 800])
    while time() - t0 < seconds
        s = M.base_step(rng, P, s, S)
        it += 1
        if kernel == "hmc"
            M.accumulate_mass!(S.hmc, vcat(s.k, s.v, s.z0))
            it in windows && M.update_mass!(S.hmc)
        end
        push!(ts, time() - t0); push!(Ls, s.L); push!(Qs, M.quantities(P, s))
    end
    return ts, Ls, reduce(hcat, Qs)
end

cfg = M.load_config(ARGS[1])
seconds = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 600.0
P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
P.bias.active = false
alpha0 = length(ARGS) > 2 && ARGS[3] == "alpha0"
alpha0 && (P = M.with_bias(P, false); println("target: α = 0 (no bias)"))
s0 = M.predescend(P, M.restrict_state(P, sL, 0), cfg)
names = M.quantity_names(P)
out = Dict{String,Any}()
for kernel in ("gibbs", "hmc")
    println("running $kernel for $(seconds)s ...")
    ts, Ls, Q = chain(kernel, P, s0, sL, cfg, seconds; rng=MersenneTwister(11))
    n = length(Ls)
    burn = n ÷ 2
    println(@sprintf("%s: %d steps (%.2f s/step); L %.4e → %.4e", kernel, n, ts[end] / n, Ls[1], Ls[end]))
    for (i, nm) in enumerate(names)
        x = Q[i, burn+1:end]
        e = ess(x)
        println(@sprintf("   %-18s mean %.4e  sd %.3e  ESS %6.1f  ESS/min %7.2f  drift(2nd half) %.3e",
                         nm, mean(x), std(x), e, e / ((ts[end] - ts[burn]) / 60), x[end] - x[1]))
    end
    out[kernel] = (ts, Ls, Q)
end
mkpath("results")
h5open(alpha0 ? "results/compare_kernels_alpha0.h5" : "results/compare_kernels.h5", "w") do f
    f["names"] = names
    for (k, (ts, Ls, Q)) in out
        f["$k/t"] = ts; f["$k/L"] = Ls; f["$k/Q"] = Q
    end
end
println("done")
