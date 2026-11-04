# Reproduce the α = 1 run of docs/mixing_diagnosis.md §10 (scripts/mixing/run_a1_full.jl, kop = false)
# step by step with the same seeds, and keep what that run did not write: the lifted and the
# frozen bias (all fields), the coefficients after every refit, every observation of the online
# regression (scores, capacity coordinates, node weather at the bias level), the chains after the
# warmup and after production, and the post-correction records. Usage:
#   nice -n 19 julia --project=. -t 4 scripts/review/reproduce_a1_bias.jl <config> <outdir>
include(joinpath(@__DIR__, "..", "mixing", "common.jl"))
using Serialization

path, outdir = ARGS[1], ARGS[2]
mkpath(outdir)
t00 = time()
cfg, P, sL, info = mix_setup(path)
@printf("setup and level-0 warmup: %.0f s; bias at level %d\n", time() - t00, P.bias.ℓb); flush(stdout)
const bfields = (:θ, :a, :c, :κlo, :κhi, :kscale, :Dbar, :GtG, :Gts, :hk, :nobs, :ℓb, :Δt, :sD, :sA, :wstep, :nsamp, :tabθ)
snap(b) = NamedTuple{bfields}(Tuple(deepcopy(getfield(b, f)) for f in bfields))
blift = snap(P.bias)
serialize(joinpath(outdir, "bias_lift.jls"), blift)
L = ML.nlevels(P) - 1
kinds, chains, samplers, rngs = mix_chains(cfg, P, sL)
serialize(joinpath(outdir, "chains_start.jls"), deepcopy(chains))
K = length(chains)
nwarm, nprod = cfg.mlda_warmup, cfg.nsamples
OB = ML.OnlineBias(; every=cfg.bias_online_every, freeze=K * nwarm, nmin=cfg.bias_online_min, trust=cfg.bias_trust)
PB = ML.PostBias()
names = ML.monitor_names(P)
LM = ML.LocalModel(P, L)
mono = ML.bias_monotone_args(P, cfg)
kflat(s) = reduce(vcat, ML.knodes(P, s.ℓ, s.k))

"The fine step of `fine_local_step` (local kernel) as replicated in run_a1_full.jl (same RNG use)."
function step_instr(rng, P, s, S)
    Kk = S.loc
    LMk = ML.local_model(Kk, P, s.ℓ)
    s = copy(s)
    BC = ML.BiasCache(P, s)
    function kmoves(x)
        x = ML.kscale_sweep!(rng, P, LMk, x; BC=BC)[1]
        x = ML.kpeak_sweep!(rng, P, LMk, x; BC=BC)[1]
        if Kk.kop
            x, nm = ML.kop_sweep!(rng, P, LMk, x; BC=BC)
        end
        return x
    end
    function op(x)
        x, _ = ML.local_sweep!(rng, P, LMk, x; wsets=Kk.wsets, extra=Kk.extra)
        if rand(rng) < 0.5 && LMk.complete[P.E.fθ]
            isempty(Kk.pools) && append!(Kk.pools, [ML.BlockPool(P, 140) for _ in 1:Threads.maxthreadid()])
            x, _ = ML.storage_ms_sweep!(rng, P, LMk, x; pools=Kk.pools)
        end
        return x
    end
    if rand(rng) < 0.5
        s = op(s); s = kmoves(s)
    else
        s = kmoves(s); s = op(s)
    end
    if Kk.weather > 0 && s.ℓ >= 1 && LMk.complete[P.E.fθ]
        s, _ = ML.fine_weather_sweep!(rng, P, LMk, s; frac=Kk.weather, frac0=Kk.weather, pools=Kk.wpools, BC=BC, zblocks=0.0)
        nb = Kk.wblocks > 0 ? floor(Int, Kk.wblocks) + (rand(rng) < Kk.wblocks - floor(Kk.wblocks)) : 0
        for _ in 1:nb
            s, _ = ML.z0_block_sweep!(rng, P, LMk, s, 1; BC=BC, pool=Kk.wpools[1], maxspan=Kk.wblockspan)
        end
    end
    return s
end

Q = [zeros(length(names), nwarm + nprod) for _ in 1:K]
bhat = [zeros(nwarm + nprod) for _ in 1:K]
obs = NamedTuple[]              # every observation of the online regression: (sweep, chain, sx, sk, kb, Xb)
hist = NamedTuple[]             # coefficients after every refit
refits = NamedTuple[]
t0 = time()
for it in 1:nwarm+nprod
    for c in 1:K
        chains[c] = step_instr(rngs[c], P, chains[c], samplers[c])
        if it <= nwarm
            Xb, sx, sk, kb = ML.bias_scores(P, chains[c])
            push!(obs, (; sweep=it, chain=c, sx=sx, sk=sk, kb=copy(kb), Xb=deepcopy(Xb), b=ML.bias_term(P, chains[c]),
                        parts=chains[c].parts))
            old = ML.bias_coeffs(P.bias)
            bch0 = [ML.bias_term(P, x) for x in chains]
            chains[c], upd = ML.online_bias_step!(P, chains[c], OB; monotone=mono)
            if upd
                bch1 = [ML.bias_term(P, x) for x in chains]
                push!(refits, (; sweep=it, chain=c, nobs=OB.nobs, λ=OB.steps[end], dchains=bch1 .- bch0))
                push!(hist, (; sweep=it, nobs=OB.nobs, λ=OB.steps[end], old=old, bias=snap(P.bias)))
                @printf("  refit after %d observations (sweep %d): λ %.3f, Δb̂ at the chains %s\n", OB.nobs, it, OB.steps[end],
                        join((@sprintf("%.3g", x) for x in bch1 .- bch0), " "))
                flush(stdout)
                foreach(x -> ML.evaluate!(P, x), chains)
            end
        end
        q = ML.monitor_quantities(P, chains[c]; LM=LM)
        Q[c][:, it] .= q
        bhat[c][it] = ML.bias_term(P, chains[c])
        it > nwarm && ML.post_record!(PB, P, chains[c], q)
    end
    if it == nwarm
        serialize(joinpath(outdir, "bias_frozen.jls"), snap(P.bias))
        serialize(joinpath(outdir, "bias_frozen_struct.jls"), deepcopy(P.bias))
        serialize(joinpath(outdir, "chains_warm.jls"), deepcopy(chains))
        serialize(joinpath(outdir, "obs.jls"), obs)
        serialize(joinpath(outdir, "hist.jls"), (; hist, refits, updates=OB.updates, steps=OB.steps))
        @printf("warmup done [%.0fs]; bias frozen and written\n", time() - t0); flush(stdout)
    end
    if it % 5 == 0 || it == nwarm + nprod
        ik = [findfirst(==(String(sp.name)), names) for sp in P.E.kspecs]
        @printf("[sweep %3d] k totals (GW) per chain: %s | b̂ %s [%.0fs]\n", it,
                join((join((@sprintf("%.0f", Q[c][i, it] / 1e3) for i in ik), "/") for c in 1:K), "  "),
                join((@sprintf("%.4g", bhat[c][it]) for c in 1:K), " "), time() - t0)
        flush(stdout)
    end
end
serialize(joinpath(outdir, "chains_end.jls"), deepcopy(chains))
serialize(joinpath(outdir, "post_records.jls"), PB)
serialize(joinpath(outdir, "traces.jls"), (; names, kinds, Q, bhat, nwarm, nprod))
pc = ML.post_correct(P, PB; monotone=mono)
@printf("post-correction: weight ESS %.2f of %d, log-weight spread %.3g (sd %.3g)\n", pc.ess, length(pc.w), maximum(pc.logw) - minimum(pc.logw), std(pc.logw))
serialize(joinpath(outdir, "bias_post.jls"), snap(P.bias))
serialize(joinpath(outdir, "post_correct.jls"), (; w=pc.w, logw=pc.logw, ess=pc.ess, btilde=pc.btilde))
@printf("done, wall %.0f s\n", time() - t00)
