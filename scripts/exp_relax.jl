# Relaxation of the fast block (40 nodes × daily, α = 0, Haar hierarchy, weather
# and capacities fixed at the LP's) — docs/alternatives.md §9:
#
#   R1  plain coordinate Gibbs vs coordinate Gibbs + temporal multigrid
#       directions on the coarse spatial modes, from a far (30 %-push) start,
#       equal wall time;
#   R2  equilibration from the LP start with the better kernel;
#   R3  relaxation after a slow weather move (+3σ network-mean load for 14 days),
#       and the chain's change in mean cost;
#   R4  baseline: operational LP at fixed k, cold (IPM + crossover) and
#       warm-started dual simplex after the same load change.
#
# Usage: julia -t 8 --project=. scripts/exp_relax.jl [seconds_per_run]
using ESMSampling, Random, Statistics, Printf, HDF5, LinearAlgebra, JuMP
const M = ESMSampling

function main()
    secs = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 600.0
    cfg = M.load_config("configs/haar_poc.toml")
    cfg.verbose = false
    rng = MersenneTwister(5)
    P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    E, W = P.E, P.W
    lp = info.lp
    βv = P.βv
    s1 = M.restrict_state(P, sL, 1)
    lv1 = P.lv[2]
    nv = length(E.vspecs)
    Cop_lp = lp.objective - lp.capex
    npinned = lp.n_v_total - lp.n_v_free
    target = Cop_lp + npinned / βv
    println(@sprintf("LP opex %.4e, predicted equilibrium opex %.4e (excess %.3e)", Cop_lp, target, npinned / βv))
    lengths = [2, 4, 8, 16, 32, 64, 128, 256]

    function sweep!(rng, G, s)
        ctx = M.SweepCtx(P, s.k; ℓ=1)
        for c in shuffle(rng, collect(1:M.NCOLOURS))
            M.update_colour!(rng, P, G, ctx, s.v, s.z0, s.m, c; ℓ=1, x=s.xs[end])
        end
        return M.evaluate!(P, s)
    end

    # one chain for a fixed wall time; kernel :plain or :mg
    function run_chain(s0, kernel, seconds; G=M.GibbsState(P), widths=nothing, stats=nothing, label="")
        s = copy(s0)
        widths === nothing && (widths = [G.wv[f] * sqrt(L) for f in 1:nv, L in lengths])
        stats === nothing && (stats = zeros(nv, length(lengths), 2))
        t0 = time()
        ts, ops, sheds = Float64[0.0], Float64[s.parts.op], Float64[s.parts.shed]
        it = 0
        while time() - t0 < seconds
            it += 1
            s = sweep!(rng, G, s)
            if it <= 40 && it % 5 == 0          # coordinate slice widths
                for f in 1:nv
                    G.nmove[f] > 0 && (G.wv[f] = max(3 * G.smove[f] / G.nmove[f], 1e-8))
                end
                fill!(G.nmove, 0.0); fill!(G.smove, 0.0)
            end
            if kernel == :mg
                s = M.temporal_multigrid_sweep!(rng, P, G, s, widths, stats; lengths=lengths)
                if it <= 20                      # direction widths
                    for f in 1:nv, iL in eachindex(lengths)
                        stats[f, iL, 1] > 0 && (widths[f, iL] = max(3 * stats[f, iL, 2] / stats[f, iL, 1], 1e-8))
                    end
                    stats .= 0
                end
            end
            push!(ts, time() - t0); push!(ops, s.parts.op); push!(sheds, s.parts.shed)
            if it % 10 == 0
                println(@sprintf("  %-14s it %4d  t %5.0fs  opex %.4e  shed %.3e MWh", label, it, ts[end], ops[end], sheds[end]))
                flush(stdout)
            end
        end
        return s, (; ts, ops, sheds), G, widths
    end

    first_time_below(ts, ops, thr) = (i = findfirst(<=(thr), ops); i === nothing ? NaN : ts[i])

    # ---------------- R1 ----------------
    zc = M.to_modes(lv1, E.vspecs, M.interior_center(E, lv1))
    spush = copy(s1); spush.v = 0.7 .* s1.v .+ 0.3 .* zc; M.evaluate!(P, spush)
    println("\n== R1: far start (30 % push), equal wall time $(secs) s ==")
    _, trP, _, _ = run_chain(spush, :plain, secs; label="plain")
    _, trM, _, _ = run_chain(spush, :mg, secs; label="multigrid")
    for (name, tr) in (("plain", trP), ("multigrid", trM))
        println(@sprintf("  %-10s %4d its, final opex %.4e (%.2f× target); time to ≤ 2× target %.0f s, to ≤ 1.1× target %.0f s",
                         name, length(tr.ts) - 1, tr.ops[end], tr.ops[end] / target,
                         first_time_below(tr.ts, tr.ops, 2target), first_time_below(tr.ts, tr.ops, 1.1target)))
    end

    # ---------------- R2 ----------------
    println("\n== R2: LP start, multigrid kernel ==")
    seq, trE, Geq, weq = run_chain(s1, :mg, secs; label="LP start, mg")
    h = length(trE.ops) ÷ 3
    base = mean(trE.ops[end-h+1:end]); basesd = std(trE.ops[end-h+1:end])
    println(@sprintf("  final third: opex %.4e ± %.2e (target %.4e, excess %.2f× predicted)", base, basesd, target, (base - Cop_lp) / (npinned / βv)))

    # ---------------- R3 ----------------
    println("\n== R3: +3σ network-mean load for days 10–23, relaxation ==")
    lw1 = W.lays[2]
    days = 10:23
    idx = [lw1.off[1] + (τ - 1) * lw1.m[1] + 1 for τ in days]
    sp = copy(seq)
    sp.xs = copy.(seq.xs)
    sp.xs[end][idx] .+= 3.0
    M.evaluate!(P, sp)
    println(@sprintf("  immediately after the move: opex %.4e (Δ %.3e)", sp.parts.op, sp.parts.op - trE.ops[end]))
    srel, trR, _, _ = run_chain(sp, :mg, secs / 2; G=Geq, widths=weq, label="after move")
    hr = length(trR.ops) ÷ 3
    after = mean(trR.ops[end-hr+1:end])
    jump = trR.ops[1] - after
    τrel = first_time_below(trR.ts, trR.ops, after + 0.1 * jump)
    println(@sprintf("  relaxed opex %.4e; chain Δopex %.4e; time to within 10 %% of the jump %.0f s", after, after - base, τrel))

    # ---------------- R4 ----------------
    println("\n== R4: operational LP at fixed capacities ==")
    Xn0 = M.weather_nodes(W, 1, seq.xs[end]); Wd0 = [map(v -> M.datamap(E.wspecs[f], v), Xn0[f]) for f in eachindex(Xn0)]
    Xn1 = M.weather_nodes(W, 1, sp.xs[end]); Wd1 = [map(v -> M.datamap(E.wspecs[f], v), Xn1[f]) for f in eachindex(Xn1)]
    lk1 = P.lk[2]
    tcold = @elapsed (sol0, model, balc) = M.solve_lp(E, lk1, lv1, Wd0; kfixed=lp.k, crossover=true, return_model=true)
    obj0 = objective_value(model)
    for n in axes(balc, 1), t in axes(balc, 2)
        Wd1[1][n, t] != Wd0[1][n, t] && set_normalized_rhs(balc[n, t], -Wd1[1][n, t])
    end
    set_attribute(model, "solver", "simplex")
    twarm = @elapsed optimize!(model)
    obj1 = objective_value(model)
    println(@sprintf("  cold IPM+crossover %.0f s (opex %.4e); warm dual simplex after the load change %.1f s; LP Δopex %.4e",
                     tcold, obj0 - sol0.capex, twarm, obj1 - obj0))
    println(@sprintf("  chain Δopex / LP Δopex = %.3f", (after - base) / (obj1 - obj0)))

    h5open("results/exp_relax.h5", "w") do f
        for (name, tr) in (("R1_plain", trP), ("R1_mg", trM), ("R2_mg", trE), ("R3_after_move", trR))
            f["$name/t"] = tr.ts; f["$name/opex"] = tr.ops; f["$name/shed"] = tr.sheds
        end
        f["target"] = target; f["lp_opex"] = Cop_lp
        f["R4/t_cold"] = tcold; f["R4/t_warm"] = twarm; f["R4/lp_dopex"] = obj1 - obj0; f["R3/chain_dopex"] = after - base
    end
    println("wrote results/exp_relax.h5")
end

main()
