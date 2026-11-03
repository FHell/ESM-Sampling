# Phase-2 throughput (docs/reactant_design.md §8 (vii)): line moves per second of the plain device
# families on this machine's GPU (Reactant/XLA), and of the CPU plain kernel for reference.
#   MODEL=40   40 nodes × 6 h (configs/haar_poc.toml, level 40×6h), all model terms on
#   MODEL=256  256 nodes × daily (configs/flow256.toml, level 256×24h), all model terms on
#   NCHAINS (default 4), NSWEEPS (default 5), PERFAM=1 also times every family alone,
#   GPU=0 skips the device timing, CPU=1 times CPU local_sweep! (plain families, extra = 4) on chain 1.
# Run: cd device && nice -n 19 julia --project=. -t 8 scripts/throughput.jl
using ESMDevice, ESMSampling, Random, Printf, Reactant
using Reactant: to_rarray, ConcreteRNumber
const ML = ESMSampling
Reactant.set_default_backend(get(ENV, "DEVICE_BACKEND", "gpu"))     # timings: the GPU
const D = ESMDevice

const MODEL = get(ENV, "MODEL", "40")
const C = parse(Int, get(ENV, "NCHAINS", "4"))
const NSW = parse(Int, get(ENV, "NSWEEPS", "5"))
root = joinpath(@__DIR__, "..", "..")

function setup_model()
    if MODEL == "40"
        cfg = ML.load_config(joinpath(root, "configs", "haar_poc.toml"))
        cfg.data = joinpath(root, "data", "processed", "model.h5")
    else
        cfg = ML.load_config(joinpath(root, "configs", "flow256.toml"))
        cfg.data = joinpath(root, "data", "processed", "model256.h5")
    end
    cfg.verbose = false
    cfg.out = joinpath(root, "results", "device_throughput.h5")
    cfg.storage = "soc"; cfg.renewables = "load"; cfg.measure = "physical"; cfg.init = "heuristic"
    cfg.losses = true; cfg.thermal = "soft"; cfg.link_losses = true; cfg.uc = ["gas", "nuclear"]; cfg.brownfield = true
    P, sL, _ = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
    rng = MersenneTwister(3)
    ss = ML.State[]
    for c in 1:C
        s = copy(sL)
        c > 1 && (s.v .+= 0.02 .* randn(rng, length(s.v)))
        push!(ss, ML.evaluate!(P, s))
    end
    M, ix = D.pack(P, sL.ℓ)
    return P, M, ix, D.to_device(P, M, ix, ss), ss
end

tset = @elapsed (P, M, ix, S, ss) = setup_model()
pl = D.plan(M, S, P.βv; bmax=Dict(:theta => parse(Int, get(ENV, "BTHETA", "4096"))), bdefault=parse(Int, get(ENV, "BDEF", "32768")))
mv = D.moves_per_sweep(pl)
@printf("model %s: N = %d, T = %d, %d dispatchable, %d storage, %d links, %d lines; C = %d chains (setup %.0f s)\n",
        MODEL, M.N, M.T, length(ix.gidx), length(ix.sidx), length(M.kcap), length(M.lb), C, tset)
println("rounds per sweep ", NamedTuple{pl.fams}(Tuple(size(pl.dev[f].on, 2) for f in pl.fams)),
        "; line moves per chain and sweep ", mv, " = ", sum(mv))
cpR, spR, kR, tabR = to_rarray(pl.cp), to_rarray(pl.sp), to_rarray(pl.k), to_rarray(pl.dev)

function timeit(fams)
    fv = Val(fams)
    rng = Reactant.ReactantRNG(to_rarray(UInt64[1, 2]))
    sw(st, cp, sp, k, tabs, ns, rng) = D.run_sweeps(st, cp, sp, k, tabs, ns, rng, fv)
    st = to_rarray(pl.st)
    tc = @elapsed f = @compile sw(st, cpR, spR, kR, tabR, ConcreteRNumber(1), rng)
    st = f(st, cpR, spR, kR, tabR, ConcreteRNumber(1), rng); Array(st.r)          # warm-up
    t = @elapsed (st = f(st, cpR, spR, kR, tabR, ConcreteRNumber(NSW), rng); Array(st.r))
    n = C * NSW * sum(mv[f] for f in fams)
    return tc, t / NSW, n / t
end

if get(ENV, "GPU", "1") == "1"
tc, ts, rate = timeit(pl.fams)
@printf("GPU, all plain families: compile %.0f s, %.3f s per sweep (%d chains), %.3g line moves/s, %.3g chain-sweeps/s\n",
        tc, ts, C, rate, C / ts)
if get(ENV, "PERFAM", "0") == "1"
    for f in pl.fams
        tcf, tsf, ratef = timeit((f,))
        @printf("  %-6s %3d rounds × %6d sites: %.3f s per sweep, %.3g line moves/s (compile %.0f s)\n",
                f, size(pl.dev[f].on, 2), size(pl.dev[f].on, 1), tsf, ratef, tcf)
    end
end
end
if get(ENV, "CPU", "1") == "1"
    LM = ML.LocalModel(P, ss[1].ℓ)
    s = copy(ss[1])
    s, _ = ML.local_sweep!(MersenneTwister(1), P, LM, s; extra=4.0)               # warm-up
    t0 = time()
    s, stt = ML.local_sweep!(MersenneTwister(2), P, LM, s; extra=4.0)
    t = time() - t0
    @printf("CPU local_sweep! (default plain families, extra = 4, %d threads): %.2f s per sweep, %d line moves, %.3g line moves/s\n",
            Threads.nthreads(), t, sum(stt[:, 1]), sum(stt[:, 1]) / t)
end
