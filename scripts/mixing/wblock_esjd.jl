# Gain of the joint z0 block moves (src/weather_blocks.jl) over the z0 coordinate moves per CPU
# second under the weather prior alone (docs/mixing_diagnosis.md §5.1, §9): expected-squared-
# jump rates per move of the annual and seasonal means of the field-mean rows
# (scripts/mixing/prior_spectral.jl), averaged over prior regime draws, times the measured
# cost per move of each family at the fine level (40 nodes daily, an equilibrated α = 0 state).
#   include("scripts/mixing/wblock_esjd.jl"); wblock_esjd(P, LM, s)
include(joinpath(@__DIR__, "prior_spectral.jl"))

"Prior-only rates per move: coordinate z0 moves and the block mixture (lengths with probabilities ∝ 1/B)."
function wblock_prior_rates(W; lengths=(16, 64, 365), ndraws=3)
    n0_, T0_ = ML.n0(W), W.lays[1].T
    V = ML.z0_block_directions(W)
    fr = [z0row(W, f, 1) for f in eachindex(W.specs)]
    funcs = mean_functionals(W, fr)
    pB = [1 / B for B in lengths]; pB ./= sum(pB)
    rc = zeros(length(funcs)); rb = zeros(length(funcs), length(lengths))
    for d in 1:ndraws
        m, _, _ = ML.sample_weather(MersenneTwister(100 + d), W, 0)
        Λ = var_precision(W, m); F = cholesky(Symmetric(Λ))
        Dc = family_dirs(W, (:single,))
        Db = [family_dirs(W, (:vblock, B, V)) for B in lengths]
        for (q, (_, a)) in enumerate(funcs)
            rc[q] += esjd_rate(Λ, F, a, Dc)[1] / ndraws
            for j in eachindex(lengths)
                rb[q, j] += esjd_rate(Λ, F, a, Db[j])[1] / ndraws
            end
        end
    end
    return (; names=first.(funcs), rc, rb, pB, lengths)
end

"Measured cost per move (s) of the z0 coordinate moves and of block moves of each length at state s."
function wblock_costs(P, LM, s; lengths=(16, 64, 365), nmoves=(20, 12, 4), seed=3)
    W = P.W
    T0 = W.lays[1].T
    pools = Any[]
    x = deepcopy(s)
    ML.fine_weather_sweep!(MersenneTwister(seed), P, LM, x; frac=0.0, frac0=0.05, pools=pools)   # compile
    tz = @elapsed for i in 1:5
        x = ML.fine_weather_sweep!(MersenneTwister(seed + i), P, LM, x; frac=0.0, frac0=0.05, pools=pools)[1]
    end
    nz = round(Int, 0.05 * W.lays[1].n)
    cz = tz / 5 / nz
    V = ML.z0_block_directions(W)
    r = MersenneTwister(seed)
    cb = Float64[]; acc = Float64[]
    for (B, nm) in zip(lengths, nmoves)
        x1 = deepcopy(s); LC = ML.LocalCost(P, LM, x1)
        bm = ML.BlockMove(ML.BlockPool(P, ML.z0_block_span(W, s.ℓ, B)).wss, 0, Int[], P, LM, LC, Float64[], Float64[])
        jac, sp = NTuple{5,Float64}[], Float64[]
        ML.z0_block_move!(r, P, LM, LC, x1, bm, jac, sp, 1, min(B, T0), V[:, 1])                    # compile
        ts = Float64[]
        t = @elapsed for _ in 1:nm
            a = rand(r, 1:T0)
            push!(ts, ML.z0_block_move!(r, P, LM, LC, x1, bm, jac, sp, a, min(a + B - 1, T0), V[:, rand(r, 1:size(V, 2))]))
        end
        push!(cb, t / nm); push!(acc, count(!=(0.0), ts) / nm)
    end
    return (; cz, cb, acc, nz)
end

function wblock_esjd(P, LM, s; kw...)
    R = wblock_prior_rates(P.W; kw...)
    C = wblock_costs(P, LM, s)
    cmix = sum(R.pB .* C.cb)
    @printf("costs: z0 coordinate move %.3f ms (%d per sweep); block moves %s ms (accepted %s); mixture %.1f ms per move\n",
            1000C.cz, C.nz, join([@sprintf("%.0f", 1000c) for c in C.cb], "/"), join([@sprintf("%.2f", a) for a in C.acc], "/"), 1000cmix)
    @printf("%-22s %12s %12s %12s %12s %8s\n", "functional", "coord/move", "block/move", "coord/CPU s", "block/CPU s", "gain")
    for q in eachindex(R.names)
        rbm = sum(R.pB .* R.rb[q, :])
        gc, gb = R.rc[q] / C.cz, rbm / cmix
        @printf("%-22s %12.2e %12.2e %12.2e %12.2e %8.1f\n", R.names[q], R.rc[q], rbm, gc, gb, gb / gc)
    end
    return R, C
end
