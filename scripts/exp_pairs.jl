# Balance-preserving pair moves in the energy model (40 nodes × daily, Haar,
# α = 0, weather and capacities fixed at the LP's).
# Usage: julia -t 8 --project=. scripts/exp_pairs.jl [seconds_per_run] [levels-config]
using ESMSampling, Random, Statistics, Printf, HDF5, LinearAlgebra
const M = ESMSampling

const KINDS = length(ARGS) > 2 ? Symbol.(split(ARGS[3], ",")) : [:coord, :coord_same, :coord_pairs, :pairs_only]

function main()
    secs = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 300.0
    cfg = M.load_config(length(ARGS) > 1 ? ARGS[2] : "configs/haar_poc.toml")
    cfg.verbose = false
    rng = MersenneTwister(7)
    P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    E, W, lp = P.E, P.W, info.lp
    βv = P.βv
    ℓ = findfirst(l -> l[1] >= E.sys.N, P.levels) - 1          # first spatially complete level (the LP's)
    s1 = M.restrict_state(P, sL, ℓ)
    lv = P.lv[ℓ+1]
    nv = length(E.vspecs)
    Cop = lp.objective - lp.capex
    npin = lp.n_v_total - lp.n_v_free
    target = Cop + npin / βv
    println(@sprintf("level %d (%s): LP opex %.4e, predicted equilibrium %.4e (excess %.3e), %d coordinates",
                     ℓ, string(P.levels[ℓ+1]), Cop, target, npin / βv, length(s1.v)))
    PS = M.PairSet(E)

    # ---- check: a pair move at an interior point leaves the balance unchanged
    let s = copy(s1), G = M.GibbsState(P)
        zc = M.to_modes(lv, E.vspecs, M.interior_center(E, lv))
        s.v = 0.5 .* s.v .+ 0.5 .* zc; M.evaluate!(P, s)
        p0 = s.parts
        st = zeros(2, 2)
        # deterministic: tiny widths so moves are small and stay interior
        M.pair_sweep!(rng, P, G, s, PS, [1e-3, 1e-3], st; nmoves=5)
        p1 = s.parts
        println(@sprintf("check: shed %.6e → %.6e, dump %.6e → %.6e after %d pair moves (should be unchanged)",
                         p0.shed, p1.shed, p0.dump, p1.dump, round(Int, sum(st[:, 1]))))
    end

    function coord_sweep!(rng, G, s)
        ctx = M.SweepCtx(P, s.k; ℓ=ℓ)
        for c in shuffle(rng, collect(1:M.NCOLOURS))
            M.update_colour!(rng, P, G, ctx, s.v, s.z0, s.m, c; ℓ=ℓ, x=s.xs[end])
        end
        return M.evaluate!(P, s)
    end

    function run_chain(s0, kind, seconds; label="")
        s = copy(s0)
        G = M.GibbsState(P)
        widths = [50.0, 50.0, 50.0, 50.0]
        st = zeros(4, 2)
        swidth = Ref(50.0); sst = zeros(2)
        t0 = time()
        ts, ops = Float64[0.0], Float64[s.parts.op]
        it = 0
        while time() - t0 < seconds
            it += 1
            if !(kind in (:pairs_only, :node_only, :node_storage)) || it % 5 == 1   # node_only keeps an occasional coordinate sweep for the state of charge
                s = coord_sweep!(rng, G, s)
                if it <= 40 && it % 5 == 0
                    for f in 1:nv
                        G.nmove[f] > 0 && (G.wv[f] = max(3 * G.smove[f] / G.nmove[f], 1e-8))
                    end
                    fill!(G.nmove, 0.0); fill!(G.smove, 0.0)
                end
            end
            if kind != :coord
                psame = kind == :coord_same ? 1.0 : 0.5
                node = kind in (:coord_node, :node_only, :node_storage)
                s = M.pair_sweep!(rng, P, G, s, PS, widths, st; nmoves=(node ? 6 : 2) * E.sys.N, p_same=psame,
                                  iweight=kind == :coord_wpairs ? 1.0 : 0.0,
                                  p_single=node ? 0.4 : 0.0, p_theta=node ? 0.2 : 0.0)
                if it <= 40 && it % 5 == 0
                    for fam in 1:4
                        st[fam, 1] > 0 && (widths[fam] = max(3 * st[fam, 2] / st[fam, 1], 1e-3))
                    end
                    st .= 0
                end
            end
            if kind in (:coord_pairs_storage, :node_storage)
                s = M.storage_shift_sweep!(rng, P, G, s, PS, swidth, sst; wmax=14, nmoves=40)
                if it <= 40 && it % 5 == 0 && sst[1] > 0
                    swidth[] = max(3 * sst[2] / sst[1], 1e-3); sst .= 0
                end
            end
            push!(ts, time() - t0); push!(ops, s.parts.op)
            if it % 10 == 0
                println(@sprintf("  %-22s it %4d t %5.0fs opex %.4e (%.3f× target) shed %.2e", label, it, ts[end], ops[end], ops[end] / target, s.parts.shed))
                flush(stdout)
            end
        end
        return s, (; ts, ops), widths
    end

    zc = M.to_modes(lv, E.vspecs, M.interior_center(E, lv))
    spush = copy(s1); spush.v = 0.7 .* s1.v .+ 0.3 .* zc; M.evaluate!(P, spush)
    results = Dict{String,Any}()
    for (startname, s0) in (("LP", s1), ("push30", spush))
        for kind in KINDS
            _, tr, w = run_chain(s0, kind, secs; label="$startname/$kind")
            h = length(tr.ops) ÷ 3
            fin = mean(tr.ops[end-h+1:end])
            println(@sprintf("RESULT %-7s %-12s its %4d  final-third opex %.4e = %.3f× target (excess %.2f× predicted)  widths %s",
                             startname, kind, length(tr.ts) - 1, fin, fin / target, (fin - Cop) / (npin / βv), string(round.(w; sigdigits=3))))
            results["$startname/$kind"] = tr
        end
    end
    out = "results/exp_pairs_$(P.levels[ℓ+1][2])h_$(join(KINDS, "-")).h5"
    h5open(out, "w") do f
        for (k, tr) in results
            f["$k/t"] = tr.ts; f["$k/opex"] = tr.ops
        end
        f["target"] = target; f["lp_opex"] = Cop
    end
    println("wrote $out")
end

main()
