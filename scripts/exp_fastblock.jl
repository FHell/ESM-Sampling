# Small experiments for docs/alternatives.md §5 (40 nodes, α = 0, Haar hierarchy):
#
#   E1  fast block at fixed (x, k): coordinate Gibbs on v at nodes × daily from the
#       LP solution; operational cost vs C*_op + n_pinned/β_v; two starts.
#   E2  mean force on the coarse weather coefficients from the fast chain: Monte
#       Carlo error, autocorrelation, agreement with the LP-dual (zero-temperature) force.
#   E3  extensivity of the MLDA discrepancy: log α for coarse moves of n coordinates
#       (coarse-conditional slice moves, fine details fixed), n = 1 … full sweep.
#
# Usage: julia -t 8 --project=. scripts/exp_fastblock.jl [sweeps]
using ESMSampling, Random, Statistics, Printf, HDF5, LinearAlgebra
const M = ESMSampling

function main()
    nsweeps = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 120
    nadapt = 30
    cfg = M.load_config("configs/haar_poc.toml")
    cfg.verbose = false
    rng = MersenneTwister(3)
    P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    E, W = P.E, P.W
    lp = info.lp
    βv = P.βv
    s1 = M.restrict_state(P, sL, 1)          # nodes × daily: the LP's level and weather
    Cop_lp = lp.objective - lp.capex
    npinned = lp.n_v_total - lp.n_v_free
    println(@sprintf("LP: opex %.4e €/a, pinned coordinates %d → predicted excess n/β_v = %.3e €", Cop_lp, npinned, npinned / βv))
    println(@sprintf("level-1 state from the LP (interior push 2%%): opex %.4e, shed %.3e MWh, penalty %.3e", s1.parts.op, s1.parts.shed, s1.parts.penalty))

    lv1, lw0, lw1 = P.lv[2], W.lays[1], W.lays[2]

    # score −β_v ∇_x C on the level-0 weather coordinates (adjoint restriction of the level-1 score)
    function coarse_score(P, s)
        Xn = M.weather_nodes(P.W, 1, s.xs[end])
        _, (_, _, _, gX) = M.energy_cost(P.E, P.lk[2], P.lv[2], s.k, s.v, Xn; grad=true)
        g1 = M.weather_grad_to_x(P.W, 1, [-P.βv .* g for g in gX])
        g0, _ = M.split_state(P.W.trs[1], P.W.lays[1], P.W.lays[2], P.W.specs, g1)
        return g0 .* P.W.trs[1].r
    end

    # zero-temperature reference: LP duals at the same weather and capacities
    glp1 = M.weather_grad_to_x(W, 1, [-βv .* g for g in lp.dF_dX])
    glp0 = M.split_state(W.trs[1], lw0, lw1, W.specs, glp1)[1] .* W.trs[1].r
    # record the first (network-mean) coordinate of every weather field, all days
    rec = vcat([lw0.off[f] .+ (0:lw0.T-1) .* lw0.m[f] .+ 1 for f in eachindex(W.specs)]...)

    function sweep!(rng, P, G, s)
        ctx = M.SweepCtx(P, s.k; ℓ=1)
        for c in shuffle(rng, collect(1:M.NCOLOURS))
            M.update_colour!(rng, P, G, ctx, s.v, s.z0, s.m, c; ℓ=1, x=s.xs[end])
        end
        return M.evaluate!(P, s)
    end

    function adapt_widths!(G, nv)
        for f in 1:nv
            G.nmove[f] > 0 || continue
            G.wv[f] = max(3 * G.smove[f] / G.nmove[f], 1e-8)
        end
        fill!(G.nmove, 0.0); fill!(G.smove, 0.0)
    end

    # ---------------- E1 / E2 ----------------
    zc = M.to_modes(lv1, E.vspecs, M.interior_center(E, lv1))
    starts = Dict("lp" => copy(s1), "pushed" => M.evaluate!(P, (s = copy(s1); s.v = 0.7 .* s1.v .+ 0.3 .* zc; s)))
    traces = Dict{String,Any}()
    finals = Dict{String,Any}()
    gstates = Dict{String,Any}()
    nv = length(E.vspecs)
    for (name, s0) in sort(collect(starts); by=first)
        G = M.GibbsState(P)
        s = copy(s0)
        ops, sheds, pens, Ls, ts = Float64[], Float64[], Float64[], Float64[], Float64[]
        F = zeros(length(rec), 0)
        t0 = time()
        for it in 1:nsweeps
            s = sweep!(rng, P, G, s)
            it <= nadapt && it % 5 == 0 && adapt_widths!(G, nv)
            push!(ops, s.parts.op); push!(sheds, s.parts.shed); push!(pens, s.parts.penalty); push!(Ls, s.L); push!(ts, time() - t0)
            F = hcat(F, coarse_score(P, s)[rec])
            if it % 20 == 0
                println(@sprintf("  E1 %-6s sweep %4d  opex %.4e  shed %.3e MWh  penalty %.3e  [%.0fs]", name, it, s.parts.op, s.parts.shed, s.parts.penalty, ts[end]))
                flush(stdout)
            end
        end
        traces[name] = (; ops, sheds, pens, Ls, ts, F)
        finals[name] = s
        gstates[name] = G
    end

    println("\n== E1: operational cost at fixed (x, k) ==")
    for (name, tr) in sort(collect(traces); by=first)
        h = length(tr.ops) ÷ 2
        q1, q2 = mean(tr.ops[h+1:h+h÷2]), mean(tr.ops[h+h÷2+1:end])
        println(@sprintf("  %-6s start %.4e → mean 2nd half %.4e (quarters %.4e, %.4e); excess over LP %.3e (predicted %.3e); %.1f s/sweep",
                         name, tr.ops[1], mean(tr.ops[h+1:end]), q1, q2, mean(tr.ops[h+1:end]) - Cop_lp, npinned / βv, tr.ts[end] / length(tr.ts)))
    end

    # batch-means standard error and integrated autocorrelation time estimate
    function bm(x; nb=10)
        n = length(x) ÷ nb * nb
        b = [mean(x[(i-1)*(n÷nb)+1:i*(n÷nb)]) for i in 1:nb]
        se = std(b) / sqrt(nb)
        τ = (n ÷ nb) * var(b) / max(var(x[1:n]), 1e-300)
        return se, τ
    end

    println("\n== E2: mean force on coarse weather coordinates (score −β_v∇_xC, nats per σ) ==")
    for (name, tr) in sort(collect(traces); by=first)
        Fh = tr.F[:, size(tr.F, 2)÷2+1:end]
        μ = vec(mean(Fh; dims=2))
        stats = [bm(Fh[i, :]) for i in axes(Fh, 1)]
        se = first.(stats); τs = last.(stats)
        lpv = glp0[rec]
        for (f, sname) in enumerate(W.specs)
            r = (f-1)*lw0.T+1:f*lw0.T
            c = cor(μ[r], lpv[r])
            println(@sprintf("  %-6s %-8s |mean force| %.3e, MC s.e./|force| %.3f, τ_int ≈ %.1f sweeps, corr with LP force %.3f, ‖chain − LP‖/‖LP‖ %.3f",
                             name, sname.name, mean(abs.(μ[r])), median(se[r] ./ max.(abs.(μ[r]), 1e-12)), median(τs[r]), c,
                             norm(μ[r] .- lpv[r]) / max(norm(lpv[r]), 1e-12)))
        end
    end
    μA = vec(mean(traces["lp"].F[:, end÷2+1:end]; dims=2)); μB = vec(mean(traces["pushed"].F[:, end÷2+1:end]; dims=2))
    println(@sprintf("  two starts: ‖μ_lp − μ_pushed‖/‖μ_lp‖ = %.3f", norm(μA .- μB) / norm(μA)))

    # ---------------- E3 ----------------
    println("\n== E3: MLDA discrepancy vs number of moved coarse coordinates ==")
    s = finals["lp"]
    sc, (w, ξk, ξv) = M.split_level(P, s)
    lv0 = P.lv[1]
    ws_ = M.StepWS(P)
    # coarse slice widths from the fine widths scaled by √(cell size): rough, the slice sampler is exact for any width
    G0 = M.GibbsState(P)
    G0.wv .= 2.0 .* gstates["lp"].wv     # cells hold several nodes: coarse coordinates are wider

    # coarse-conditional slice moves of n random level-0 v coordinates, then the fine DA log α
    function block_move(rng, n)
        ψc = copy(sc)
        ctx0 = M.SweepCtx(P, ψc.k; ℓ=0)
        coords = randperm(rng, lv0.n)[1:n]
        moved = 0.0
        for i in coords
            f = findlast(o -> o < i, lv0.off)
            τ = (i - lv0.off[f] - 1) ÷ lv0.m[f] + 1
            τm = mod(τ - 2, lv0.T) + 1
            term(t) = M.step_term(P, 0, ctx0, ws_, ψc.v, ψc.z0, ψc.z0, ψc.m, t)
            lf = E.vspecs[f].kind == M.INSTANT ? (() -> -(term(τm) + term(τ))) : (() -> -term(τ))
            moved += M.slice_coord!(rng, ψc.v, i, lf, G0.wv[f])
        end
        M.evaluate!(P, ψc)
        ψ = M.assemble(P, ψc, w, ξk, ξv)
        return (ψ.L - s.L) - (ψc.L - sc.L), moved / n
    end

    function full_sweep_move(rng)
        ψc = copy(sc)
        ctx0 = M.SweepCtx(P, ψc.k; ℓ=0)
        for c in shuffle(rng, collect(1:M.NCOLOURS))
            M.update_colour!(rng, P, G0, ctx0, ψc.v, ψc.z0, ψc.m, c; ℓ=0, x=ψc.z0, weather=false)
        end
        M.evaluate!(P, ψc)
        ψ = M.assemble(P, ψc, w, ξk, ξv)
        return (ψ.L - s.L) - (ψc.L - sc.L)
    end

    e3 = Dict{Int,Vector{Float64}}()
    for (n, reps) in ((1, 200), (4, 60), (16, 40), (64, 20), (256, 10), (1024, 6))
        la = Float64[]
        for _ in 1:reps
            push!(la, block_move(rng, n)[1])
        end
        e3[n] = la
        println(@sprintf("  n = %5d coords: log α median %.3e, mean %.3e, sd %.3e, mean acceptance %.3f",
                         n, median(la), mean(la), std(la), mean(min.(1.0, exp.(la)))))
        flush(stdout)
    end
    lafull = [full_sweep_move(rng) for _ in 1:4]
    println(@sprintf("  full coarse sweep (%d coords): log α = %s", lv0.n, string(round.(lafull; sigdigits=3))))

    h5open("results/exp_fastblock.h5", "w") do f
        for (name, tr) in traces
            f["E1/$name/opex"] = tr.ops; f["E1/$name/shed"] = tr.sheds; f["E1/$name/penalty"] = tr.pens
            f["E1/$name/L"] = tr.Ls; f["E1/$name/t"] = tr.ts; f["E2/$name/force"] = tr.F
        end
        f["E2/lp_force"] = glp0[rec]
        f["E1/lp_opex"] = Cop_lp; f["E1/npinned"] = npinned; f["beta_v"] = βv
        for (n, la) in e3
            f["E3/logalpha_n$n"] = la
        end
        f["E3/logalpha_fullsweep"] = lafull
    end
    println("wrote results/exp_fastblock.h5")

end

main()
