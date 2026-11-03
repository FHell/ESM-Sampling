# Efficiency of the cross-node pair moves: global xpair vs localized xloc (several ball
# radii). Per family: µs per move, acceptance, mean t² (MW², t is the injection shift),
# mean entries/lines/breakpoints, and t²/µs. Single thread, after a short relaxation.
# Usage: nice -n 19 julia -t 4 --project=. scripts/xloc_eff.jl [data] [N] [nrelax] [nmoves]
using ESMSampling, Random, Statistics, Printf
const M = ESMSampling
data = length(ARGS) > 0 ? ARGS[1] : "data/processed/model.h5"
N = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 40
nrelax = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 5
nmv = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 20000
cfg = M.load_config("configs/flow.toml"); cfg.verbose = false
cfg.data = data; cfg.levels = [(10, 24), (N, 24)]; cfg.init = "heuristic"; cfg.flow_theta = "haar"; cfg.out = "results/bench.h5"
P, s, info = M.setup(cfg; rng=MersenneTwister(1))
E = P.E
LM = M.LocalModel(P, 1)
FF = M.FlowFrames(P, 1, s)
rng = MersenneTwister(3)
wsets = [M.MoveWS(P) for _ in 1:2Threads.maxthreadid()]
fams, mult = M.flow_families(E)
t0 = time()
for _ in 1:nrelax
    global s = M.local_sweep!(rng, P, LM, s; wsets=wsets, families=fams, extra=4.0 * mult, frames=FF)[1]
end
@printf("relaxed %d sweeps in %.0f s, op = %.6e\n", nrelax, time() - t0, s.parts.op)

function runfam(rng, P, LM, LC, s, fam, nmv)
    ws, wa = M.MoveWS(P), M.MoveWS(P)
    T = LM.T
    bm = M.BlockMove([wa, ws], 2, [1, 2], P, LM, LC, Float64[], Float64[])
    jac = NTuple{5,Float64}[]
    n = 0; acc = 0; t2 = 0.0; ne = 0; nl = 0; nb = 0; ta = Float64[]
    el = 0.0
    while n < nmv
        τ = rand(rng, 1:T); τm = mod(τ - 2, T) + 1
        bm.τs[1] = τm; bm.τs[2] = τ
        tt = time_ns()
        M.clear!(ws); M.clear!(wa)
        ok = fam == :xlocc ? M.prep_xloc!(rng, ws, P, LM, LC, τ, jac) : M.prep_random!(rng, ws, P, LM, LC, fam, τ)
        if !ok
            el += (time_ns() - tt) / 1e3
            continue
        end
        M.finalize!(ws, P, LM, LC, τ)
        isempty(ws.ent) && continue
        M.finalize!(wa, P, LM, LC, τm)
        M.link_soc!(wa, ws, LM)
        empty!(bm.bps); append!(bm.bps, wa.bps); append!(bm.bps, ws.bps)
        t = M.sample_line!(rng, bm, P.βv, ws.secprev)
        (fam == :xlocc && t != 0.0 && !isempty(jac) && rand(rng) >= exp(min(0.0, M.logjac(jac, t)))) && (t = 0.0)
        M.commit_block!(bm, s.v, t)
        el += (time_ns() - tt) / 1e3
        n += 1; acc += t != 0.0; t2 += t^2; push!(ta, abs(t))
        ne += length(ws.ent); nl += length(ws.lines); nb += length(bm.bps)
    end
    return el / n, acc / n, t2 / n, ne / n, nl / n, nb / n, median(ta)
end

LC = M.LocalCost(P, LM, s)
runfam(rng, P, LM, LC, s, :xpair, 200)            # compile
@printf("%-12s %9s %8s %12s %10s %8s %8s %8s %12s %12s\n", "family", "µs/move", "accept", "E[t²] MW²", "med|t| MW", "entries", "lines", "bps", "E[t²]/µs", "med²/µs")
for (name, fam, rb) in (("xpair", :xpair, 2), ("xloc rb=0", :xloc, 0), ("xloc rb=1", :xloc, 1), ("xloc rb=2", :xloc, 2),
                        ("xloc rb=3", :xloc, 3), ("xloc rb=4", :xloc, 4),
                        ("xlocc rb=0", :xlocc, 0), ("xlocc rb=1", :xlocc, 1), ("xlocc rb=2", :xlocc, 2), ("xlocc rb=3", :xlocc, 3))
    LMx = fam != :xpair ? M.LocalModel(P, 1; xr_ball=rb) : LM
    runfam(rng, P, LMx, LC, s, fam, 200)
    us, a, t2, ne, nl, nb, md = runfam(rng, P, LMx, LC, s, fam, nmv)
    @printf("%-12s %9.2f %8.3f %12.4g %10.4g %8.1f %8.1f %8.1f %12.4g %12.4g\n", name, us, a, t2, md, ne, nl, nb, t2 / us, md^2 / us)
end
