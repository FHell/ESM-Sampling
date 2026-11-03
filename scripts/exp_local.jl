# Convergence of the sparse local kernel (src/local.jl: Haar atoms + unit/pair/potential/
# cell moves, exact line conditionals) vs finest time resolution, same protocol as
# scripts/exp_scaling.jl: capacities from the daily LP, one weather draw, reference =
# operational LP at the level; chains from the level's own LP (2 % interior push) and
# from the daily LP prolonged in time. Prints excess over the LP optimum relative to the
# equipartition prediction n_pinned/β_v and (in brackets) to (n_pinned + N·T)/β_v,
# which also counts the balance kinks.
# Usage: julia -t 8 --project=. scripts/exp_local.jl [sweeps] [hours...]
using ESMSampling, Random, Statistics, Printf
const M = ESMSampling

function main()
    nsw = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 100
    hours = length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [24, 12, 6]
    cfg = M.load_config(get(ENV, "SCALING_CONFIG", "configs/haar_poc.toml"))
    cfg.verbose = false
    haskey(ENV, "SCALING_CONFIG") || (cfg.levels = [(10, 24), (40, 24), (40, 12), (40, 6), (40, 3)])
    rng = MersenneTwister(9)
    P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    E = P.E
    βv = P.βv
    kLP = info.lp.k
    wsets = [M.MoveWS(P) for _ in 1:Threads.maxthreadid()]
    println("== local kernel: convergence vs finest time step ($(E.sys.N) nodes, $nsw sweeps, $(Threads.nthreads()) threads) ==")
    for h in hours
        ℓ = findfirst(l -> l == (E.sys.N, h), P.levels) - 1
        lv, lk = P.lv[ℓ+1], P.lk[ℓ+1]
        s_pro = M.restrict_state(P, sL, ℓ)
        Xn = M.weather_nodes(P.W, ℓ, s_pro.xs[end])
        Wd = [map(v -> M.datamap(E.wspecs[f], v), Xn[f]) for f in eachindex(Xn)]
        tlp = @elapsed lp = M.solve_lp(E, lk, lv, Wd; kfixed=kLP)
        Cop = lp.objective - lp.capex
        npin = lp.n_v_total - lp.n_v_free
        nbal = E.sys.N * lv.T
        center = M.interior_center(E, lv)
        vlp = M.to_modes(lv, E.vspecs, [0.98 .* lp.v[f] .+ 0.02 .* center[f] for f in eachindex(center)])
        s_lp = copy(s_pro); s_lp.v = vlp; M.evaluate!(P, s_lp)
        println(@sprintf("\n h = %2d h: T = %d, %d coordinates; LP %.0f s, C*_op %.4e, pinned %d, balance kinks %d",
                         h, lv.T, length(s_pro.v), tlp, Cop, npin, nbal))
        LM = M.LocalModel(P, ℓ)
        for (sname, s0) in (("own LP", s_lp), ("prolonged", s_pro))
            s = copy(s0)
            ops = Float64[]
            ts = Float64[]
            t0 = time()
            for it in 1:nsw
                s, _ = M.local_sweep!(rng, P, LM, s; wsets=wsets)
                push!(ops, s.parts.op); push!(ts, time() - t0)
            end
            ex(x) = (x - Cop) / (npin / βv)
            ex2(x) = (x - Cop) / ((npin + nbal) / βv)
            i3 = findfirst(x -> ex(x) < 3, ops)
            tail = mean(ops[end-nsw÷4+1:end])
            println(@sprintf("   %-10s excess/pred after 1, %d/4, %d/2, %d sweeps: %7.2f %7.2f %7.2f %7.2f [%5.2f]  last-quarter mean %5.2f [%5.2f]  first ≤3×: %4s  %.2f s/sweep",
                             sname, nsw, nsw, nsw, ex(ops[1]), ex(ops[nsw÷4]), ex(ops[nsw÷2]), ex(ops[end]), ex2(ops[end]),
                             ex(tail), ex2(tail), i3 === nothing ? "—" : string(i3), ts[end] / nsw))
            flush(stdout)
        end
    end
end

main()
