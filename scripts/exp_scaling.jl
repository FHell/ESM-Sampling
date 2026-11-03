# Convergence of the fast block vs finest time resolution (40 nodes, Haar, α = 0,
# capacities from the daily LP, one weather draw). For each finest step h:
#   reference: operational LP at that level and fixed capacities (IPM, relative
#   interior) → C*_h and the pinned count n_h, predicted equilibrium C*_h + n_h/β_v;
#   chains (coordinate Gibbs, coordinate Gibbs + pair moves) from (a) the level's own
#   LP solution (2 % interior push) and (b) the daily LP prolonged in time.
# Usage: julia -t 8 --project=. scripts/exp_scaling.jl [sweeps] [hours...]
using ESMSampling, Random, Statistics, Printf, HDF5
const M = ESMSampling

function main()
    nsw = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 60
    hours = length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [24, 12, 6]
    cfg = M.load_config(get(ENV, "SCALING_CONFIG", "configs/haar_poc.toml"))
    cfg.verbose = false
    haskey(ENV, "SCALING_CONFIG") || (cfg.levels = [(10, 24), (40, 24), (40, 12), (40, 6), (40, 3)])
    rng = MersenneTwister(9)
    P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    E, W = P.E, P.W
    βv = P.βv
    PS = M.PairSet(E)
    nv = length(E.vspecs)
    kLP = info.lp.k
    println("== fast-block convergence vs finest time step ($(E.sys.N) nodes, $(nsw) sweeps per chain) ==")
    for h in hours
        ℓ = findfirst(l -> l == (E.sys.N, h), P.levels) - 1
        lv, lk = P.lv[ℓ+1], P.lk[ℓ+1]
        s_pro = M.restrict_state(P, sL, ℓ)                 # daily LP, prolonged in time to level ℓ
        Xn = M.weather_nodes(W, ℓ, s_pro.xs[end])
        Wd = [map(v -> M.datamap(E.wspecs[f], v), Xn[f]) for f in eachindex(Xn)]
        tlp = @elapsed lp = M.solve_lp(E, lk, lv, Wd; kfixed=kLP)
        Cop = lp.objective - lp.capex
        npin = lp.n_v_total - lp.n_v_free
        target = Cop + npin / βv
        center = M.interior_center(E, lv)
        vlp = M.to_modes(lv, E.vspecs, [0.98 .* lp.v[f] .+ 0.02 .* center[f] for f in eachindex(center)])
        s_lp = copy(s_pro); s_lp.v = vlp; M.evaluate!(P, s_lp)
        println(@sprintf("\n h = %2d h: T = %d, %d coordinates; LP %.0f s, C*_op %.4e, pinned %d → predicted %.4e (excess %.2e); start opex: own LP %.4e, prolonged daily %.4e",
                         h, lv.T, length(s_pro.v), tlp, Cop, npin, target, npin / βv, s_lp.parts.op, s_pro.parts.op))
        for (sname, s0, kind) in (("own LP", s_lp, :coord), ("own LP", s_lp, :node), ("prolonged", s_pro, :node))
            s = copy(s0)
            G = M.GibbsState(P)
            widths = [50.0, 50.0, 50.0, 50.0]; st = zeros(4, 2)
            ops = Float64[]
            t = @elapsed for it in 1:nsw
                if kind == :coord || it % 5 == 1
                    ctx = M.SweepCtx(P, s.k; ℓ=ℓ)
                    for c in shuffle(rng, collect(1:M.NCOLOURS))
                        M.update_colour!(rng, P, G, ctx, s.v, s.z0, s.m, c; ℓ=ℓ, x=s.xs[end])
                    end
                    M.evaluate!(P, s)
                end
                if it <= 30 && it % 5 == 0
                    for f in 1:nv
                        G.nmove[f] > 0 && (G.wv[f] = max(3 * G.smove[f] / G.nmove[f], 1e-8))
                    end
                    fill!(G.nmove, 0.0); fill!(G.smove, 0.0)
                end
                if kind == :node
                    s = M.pair_sweep!(rng, P, G, s, PS, widths, st; nmoves=6 * E.sys.N, p_same=0.5, p_single=0.4, p_theta=0.2)
                    if it <= 30 && it % 5 == 0
                        for fam in 1:4
                            st[fam, 1] > 0 && (widths[fam] = max(3 * st[fam, 2] / st[fam, 1], 1e-3))
                        end
                        st .= 0
                    end
                end
                push!(ops, s.parts.op)
            end
            ex(x) = (x - Cop) / (npin / βv)
            i3 = findfirst(x -> ex(x) < 3, ops)
            println(@sprintf("   %-10s %-12s excess/predicted after 1, %d/4, %d/2, %d sweeps: %7.2f %7.2f %7.2f %7.2f   first ≤3×: %4s   %.1f s/sweep",
                             sname, kind, nsw, nsw, nsw, ex(ops[1]), ex(ops[nsw÷4]), ex(ops[nsw÷2]), ex(ops[end]),
                             i3 === nothing ? "—" : string(i3), t / nsw))
            flush(stdout)
        end
    end
end

main()
