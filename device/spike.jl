# Phase-0 Reactant spike (docs/reactant_design.md §8, §9): does a device loop with
# loop-carried state, scatter-add into a (C, T, N) array, a sort, and a Philox RNG
# trace, compile and run? Every kernel is plain array code; it runs on Julia Arrays
# (reference) and traced by Reactant, and the results are compared.
using Reactant, Random, Printf, Statistics

# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

"""
R rounds: in round k, every chain draws uniforms (`U` given, or Reactant's RNG),
adds t·a at the padded node items idx[:, k] (scatter-add into X[c, τ, n]), sorts the
candidate points per move and keeps the smallest. Loop-carried: X, acc (and the RNG).
"""
function rounds_loop(X, idx, a, R, U, rng)
    C, T, N = size(X)
    B = size(idx, 1)
    Rs = size(idx, 2)                                         # static (R itself is traced)
    acc = X[:, 1, 1] .* 0.0
    # everything shape-dependent is built outside the device loop (captured sizes are traced inside)
    oh = Float64.(reshape(idx, B, 1, Rs) .== reshape(1:N, 1, N, 1))    # (B, N, R) one-hot of the padded items
    ar = reshape(a, 1, 1, B, Rs) .+ 0.0                       # materialized (a lazy reshape cannot be loop-captured)
    dims = Val((C, T, B, N))                                  # static shapes (a Val is not traced)
    @trace for k in 1:R
        u = U === nothing ? draw(rng, dims) : U[:, :, :, k]   # (C, T, B)
        t = (u .- 0.5) .* ar[:, :, :, k]
        X = X .+ contract(t, oh[:, :, k], dims)              # scatter-add as a contraction
        s = sort(u; dims=3)                                  # sort per (chain, step)
        acc = acc .+ s[:, 1, 1]
    end
    return X, acc
end

draw(rng, ::Val{D}) where {D} = rand(rng, Float64, (D[1], D[2], D[3]))

"(C, T, B) × (B, N) → (C, T, N) with static shapes."
contract(t, ohk, ::Val{D}) where {D} = reshape(reshape(t, D[1] * D[2], D[3]) * ohk, D[1], D[2], D[4])

"X[c, τ, idx[b]] += t[c, τ, b] (idx may repeat; padded entries point at a dummy node N)."
function scatter_add_nodes(X, idx, t)
    C, T, N = size(X)
    B = length(idx)
    # one-hot (B, N) from the traced indices, then a contraction: dense but traceable
    oh = Float64.(reshape(idx, B, 1) .== reshape(1:N, 1, N))
    upd = reshape(reshape(t, C * T, B) * oh, C, T, N)
    return X .+ upd
end

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
function main(; C=16, T=64, N=40, B=24, R=100)
    rng = MersenneTwister(1)
    X = randn(rng, C, T, N)
    idx = Int32.(rand(rng, 1:N, B, R))
    a = rand(rng, B, R)
    U = rand(rng, C, T, B, R)
    t_ref = @elapsed Xr, accr = rounds_loop(copy(X), idx, a, R, U, nothing)
    @printf("reference (Julia arrays, given uniforms): %.3f s\n", t_ref)
    Xd, idxd, ad, Ud = Reactant.to_rarray(X), Reactant.to_rarray(idx), Reactant.to_rarray(a), Reactant.to_rarray(U)
    Rd = Reactant.ConcreteRNumber(R)
    ok = true
    for variant in (:given, :rng)
        try
            rr = Reactant.ReactantRNG(Reactant.to_rarray(UInt64[1, 2]))
            Ux = variant == :given ? Ud : nothing
            tc = @elapsed f = @compile rounds_loop(Xd, idxd, ad, Rd, Ux, rr)
            f(Reactant.to_rarray(X), idxd, ad, Rd, Ux, rr)                   # warm-up on a separate copy
            Xd2 = Reactant.to_rarray(X)
            tr = @elapsed (Xo, acco) = f(Xd2, idxd, ad, Rd, Ux, rr)
            println("  input X modified by the call: ", maximum(abs.(Array(Xd2) .- X)) > 0)
            Xh, acch = Array(Xo), Array(acco)
            if variant == :given
                err = max(maximum(abs.(Xh .- Xr)), maximum(abs.(acch .- accr)))
                @printf("given uniforms: compile %.1f s, %d rounds %.4f s (%.1f µs/round), max error %.2e\n", tc, R, tr, 1e6tr / R, err)
                ok &= err < 1e-10
            else
                # with the device RNG: X − X0 = Σ (u − 1/2) a over the scatter; check the mean and spread of the increments
                d = Xh .- X
                @printf("device RNG: compile %.1f s, %d rounds %.4f s (%.1f µs/round); increments mean %.3f, sd %.3f (reference sd %.3f); acc/R = %.4f (E min of %d uniforms = %.4f)\n",
                        tc, R, tr, 1e6tr / R, mean(d), std(d), std(Xr .- X), mean(acch) / R, B, 1 / (B + 1))
                ok &= abs(mean(acch) / R - 1 / (B + 1)) < 0.01
            end
        catch err
            ok = false
            println("device loop ($variant) failed: ", sprint(showerror, err)[1:min(end, 1500)])
        end
    end
    println(ok ? "SPIKE OK" : "SPIKE FAILED")
end

main()
