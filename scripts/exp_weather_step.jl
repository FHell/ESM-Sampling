# Fast-block relaxation after a weather move (40 nodes × daily, Haar, α = 0): equilibrate
# from the LP, raise the network-mean load by +3σ for days 10–23, and measure the time
# until the operational cost is back within 10 % of the jump, and the chain's change in
# mean cost, for coordinate Gibbs and the node-level kernel. Baseline: warm LP 1.1 s,
# LP Δopex 2.95·10⁸ € (scripts/exp_lp_warm.jl).
# Usage: julia -t 8 --project=. scripts/exp_weather_step.jl [equilibration_s] [after_s]
using ESMSampling, Random, Statistics, Printf
const M = ESMSampling

function main()
    teq = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 240.0
    taft = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 180.0
    cfg = M.load_config("configs/haar_poc.toml"); cfg.verbose = false
    rng = MersenneTwister(13)
    P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    E, W = P.E, P.W
    ℓ = 1
    s1 = M.restrict_state(P, sL, ℓ)
    PS = M.PairSet(E)
    nv = length(E.vspecs)
    lw = W.lays[ℓ+1]
    idx = [lw.off[1] + (τ - 1) * lw.m[1] + 1 for τ in 10:23]
    for kind in (:coord, :node)
        G = M.GibbsState(P)
        widths = [50.0, 50.0, 50.0, 50.0]; st = zeros(4, 2)
        it = 0
        function step!(s)
            it += 1
            if kind == :coord || it % 5 == 1
                ctx = M.SweepCtx(P, s.k; ℓ=ℓ)
                for c in shuffle(rng, collect(1:M.NCOLOURS))
                    M.update_colour!(rng, P, G, ctx, s.v, s.z0, s.m, c; ℓ=ℓ, x=s.xs[end])
                end
                M.evaluate!(P, s)
                if it <= 40 && it % 5 == 0
                    for f in 1:nv
                        G.nmove[f] > 0 && (G.wv[f] = max(3 * G.smove[f] / G.nmove[f], 1e-8))
                    end
                    fill!(G.nmove, 0.0); fill!(G.smove, 0.0)
                end
            end
            if kind == :node
                s = M.pair_sweep!(rng, P, G, s, PS, widths, st; nmoves=6 * E.sys.N, p_same=0.5, p_single=0.4, p_theta=0.2)
                if it <= 40 && it % 5 == 0
                    for fam in 1:4
                        st[fam, 1] > 0 && (widths[fam] = max(3 * st[fam, 2] / st[fam, 1], 1e-3))
                    end
                    st .= 0
                end
            end
            return s
        end
        s = copy(s1)
        t0 = time(); ops = Float64[]
        while time() - t0 < teq
            s = step!(s); push!(ops, s.parts.op)
        end
        base = mean(ops[end-length(ops)÷3+1:end])
        sp = copy(s); sp.xs = copy.(s.xs); sp.xs[end][idx] .+= 3.0; M.evaluate!(P, sp)
        jump0 = sp.parts.op
        t0 = time(); ts, opa = Float64[0.0], Float64[jump0]
        while time() - t0 < taft
            sp = step!(sp); push!(ts, time() - t0); push!(opa, sp.parts.op)
        end
        after = mean(opa[end-length(opa)÷3+1:end])
        τ10 = ts[something(findfirst(<=(after + 0.1 * (jump0 - after)), opa), length(ts))]
        println(@sprintf("%-6s equilibrated opex %.4e; after the move %.4e → relaxed %.4e in %.0f s (to 10 %% of the jump); chain Δopex %.3e (LP 2.95e8)",
                         kind, base, jump0, after, τ10, after - base))
        flush(stdout)
    end
end
main()
