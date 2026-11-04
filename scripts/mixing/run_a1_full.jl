# α = 1 assessment of the full stack (docs/mixing_diagnosis.md §10): `run_chains` mirrored step by
# step (level-0 warmup and lift, online refits with the trust region during the warmup,
# post-correction), with the fine step of the `local` kernel replicated from `fine_local_step`
# so that the kop pass and the z0 block moves can be instrumented. Recorded per sweep and chain:
# the monitored quantities, b̂, the kop moves (count, per-unit change of log k), the z0 block
# moves (count, accepted, |t|); per refit: Δb̂ at the chains and over all visited states, the
# trust step; at the end the post-correction weights. Usage:
#   nice -n 19 julia --project=. -t 4 scripts/mixing/run_a1_full.jl <config> <out.h5>
include(joinpath(@__DIR__, "common.jl"))
using Serialization

path, out = ARGS[1], ARGS[2]
t00 = time()
cfg, P, sL, info = mix_setup(path)                      # setup, level-0 warmup (learning), lift, as run_chains
@printf("setup and level-0 warmup: %.0f s; bias at level %d\n", time() - t00, P.bias.ℓb); flush(stdout)
L = ML.nlevels(P) - 1
kinds, chains, samplers, rngs = mix_chains(cfg, P, sL)
K = length(chains)
nwarm, nprod = cfg.mlda_warmup, cfg.nsamples
OB = ML.OnlineBias(; every=cfg.bias_online_every, freeze=K * nwarm, nmin=cfg.bias_online_min, trust=cfg.bias_trust)
PB = ML.PostBias()
names = ML.monitor_names(P)
LM = ML.LocalModel(P, L)
mono = ML.bias_monotone_args(P, cfg)
kflat(s) = reduce(vcat, ML.knodes(P, s.ℓ, s.k))

"The fine step of `fine_local_step` (local kernel), with the kop pass and the z0 block moves instrumented."
function step_instr(rng, P, s, S)
    Kk = S.loc
    LMk = ML.local_model(Kk, P, s.ℓ)
    s = copy(s)
    BC = ML.BiasCache(P, s)
    st = Dict{String,Any}()
    function kmoves(x)
        x = ML.kscale_sweep!(rng, P, LMk, x; BC=BC)[1]
        x = ML.kpeak_sweep!(rng, P, LMk, x; BC=BC)[1]
        if Kk.kop
            k0 = kflat(x)
            a = time()
            x, nm = ML.kop_sweep!(rng, P, LMk, x; BC=BC)
            st["kop_time"] = time() - a
            st["kop_moved"] = nm
            st["kop_dlogk"] = log.(max.(kflat(x), 1e-9)) .- log.(max.(k0, 1e-9))
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
        bt = Float64[]; bB = Int[]; bacc = 0
        a = time()
        for _ in 1:nb
            z0 = copy(s.z0)
            s, (n, moved) = ML.z0_block_sweep!(rng, P, LMk, s, 1; BC=BC, pool=Kk.wpools[1], maxspan=Kk.wblockspan)
            Δ = ML.z0matrix(s.z0 .- z0, P.W.lays[1], P.W.specs)
            B = count(j -> any(!iszero, view(Δ, :, j)), axes(Δ, 2))
            bacc += moved
            moved > 0 && (push!(bt, norm(Δ) / sqrt(B)); push!(bB, B))
        end
        st["blk_time"] = time() - a
        st["blk_n"] = nb; st["blk_acc"] = bacc; st["blk_t"] = bt; st["blk_B"] = bB
    end
    return s, st
end

rec = Dict{String,Any}("Q" => [zeros(length(names), nwarm + nprod) for _ in 1:K], "b" => [zeros(nwarm + nprod) for _ in 1:K],
                       "L" => [zeros(nwarm + nprod) for _ in 1:K], "tstep" => zeros(K, nwarm + nprod))
kopn = zeros(K, nwarm + nprod); kopd = [Float64[] for _ in 1:K]
blk = [Tuple{Int,Int,Float64,Int}[] for _ in 1:K]          # (sweep, B, |t|, accepted 1) per accepted move
blkn = zeros(Int, K, nwarm + nprod); blka = zeros(Int, K, nwarm + nprod)
visited = Any[]                                               # (ℓ, k, x, z0) of every chain after every warmup sweep
refits = NamedTuple[]
t0 = time()
for it in 1:nwarm+nprod
    for c in 1:K
        a = time()
        chains[c], st = step_instr(rngs[c], P, chains[c], samplers[c])
        rec["tstep"][c, it] = time() - a
        if haskey(st, "kop_moved")
            kopn[c, it] = st["kop_moved"]
            append!(kopd[c], filter(!=(0.0), st["kop_dlogk"]))
        end
        if haskey(st, "blk_n")
            blkn[c, it] = st["blk_n"]; blka[c, it] = st["blk_acc"]
            for (t, B) in zip(st["blk_t"], st["blk_B"]); push!(blk[c], (it, B, t, 1)); end
        end
        if it <= nwarm
            old = ML.bias_coeffs(P.bias)
            bch0 = [ML.bias_term(P, x) for x in chains]
            chains[c], upd = ML.online_bias_step!(P, chains[c], OB; monotone=mono)
            if upd
                new = ML.bias_coeffs(P.bias)
                bch1 = [ML.bias_term(P, x) for x in chains]
                bv1 = [ML.bias_term(P, v...) for v in visited]
                ML.set_coeffs!(P.bias, old, new, 0.0)
                bv0 = [ML.bias_term(P, v...) for v in visited]
                ML.set_coeffs!(P.bias, old, new, 1.0)
                dv = bv1 .- bv0
                r = (; sweep=it, chain=c, nobs=OB.nobs, λ=OB.steps[end], dchains=bch1 .- bch0,
                     dvis_mean=isempty(dv) ? NaN : mean(dv), dvis_dev=isempty(dv) ? NaN : maximum(abs.(dv .- mean(dv))),
                     dvis_sd=isempty(dv) ? NaN : std(dv), nvis=length(dv), spread_old=isempty(bv0) ? NaN : maximum(bv0) - minimum(bv0))
                push!(refits, r)
                @printf("  refit after %d observations (sweep %d): λ %.3f, Δb̂ at the chains %s, over %d visited states mean %.3g dev %.3g (old spread %.3g)\n",
                        r.nobs, it, r.λ, join((@sprintf("%.3g", x) for x in r.dchains), " "), r.nvis, r.dvis_mean, r.dvis_dev, r.spread_old)
                flush(stdout)
                foreach(x -> ML.evaluate!(P, x), chains)      # a refit changes the target of every chain (in place)
            end
        end
        q = ML.monitor_quantities(P, chains[c]; LM=LM)
        rec["Q"][c][:, it] .= q
        rec["b"][c][it] = ML.bias_term(P, chains[c])
        rec["L"][c][it] = chains[c].L
        it > nwarm && ML.post_record!(PB, P, chains[c], q)
        it <= nwarm && push!(visited, (chains[c].ℓ, copy(chains[c].k), copy(chains[c].xs[end]), copy(chains[c].z0)))
    end
    if it % 5 == 0 || it == nwarm + nprod
        ik = [findfirst(==(String(sp.name)), names) for sp in P.E.kspecs]
        @printf("[sweep %3d] k totals (GW) per chain: %s | b̂ %s [%.0fs]\n", it,
                join((join((@sprintf("%.0f", rec["Q"][c][i, it] / 1e3) for i in ik), "/") for c in 1:K), "  "),
                join((@sprintf("%.4g", rec["b"][c][it]) for c in 1:K), " "), time() - t0)
        flush(stdout)
    end
end
pc = ML.post_correct(P, PB; monotone=mono)
@printf("post-correction: weight ESS %.2f of %d, log-weight spread %.3g (sd %.3g)\n", pc.ess, length(pc.w), maximum(pc.logw) - minimum(pc.logw), std(pc.logw))
h5open(out, "w") do f
    f["names"] = names; f["kinds"] = kinds; f["nwarm"] = nwarm; f["nprod"] = nprod
    for c in 1:K
        f["chains/$c/Q"] = rec["Q"][c]; f["chains/$c/b"] = rec["b"][c]; f["chains/$c/L"] = rec["L"][c]
        f["chains/$c/kopd"] = kopd[c]
        f["chains/$c/blk"] = isempty(blk[c]) ? zeros(0, 4) : permutedims(reduce(hcat, [[x...] for x in blk[c]]))
    end
    f["tstep"] = rec["tstep"]; f["kopn"] = kopn; f["blkn"] = blkn; f["blka"] = blka
    f["post/w"] = pc.w; f["post/logw"] = pc.logw; f["post/ess"] = pc.ess
    if !isempty(refits)
        f["refits/sweep"] = [r.sweep for r in refits]; f["refits/nobs"] = [r.nobs for r in refits]; f["refits/lambda"] = [r.λ for r in refits]
        f["refits/dchains"] = reduce(hcat, [r.dchains for r in refits]); f["refits/dvis_mean"] = [r.dvis_mean for r in refits]
        f["refits/dvis_dev"] = [r.dvis_dev for r in refits]; f["refits/dvis_sd"] = [r.dvis_sd for r in refits]
        f["refits/spread_old"] = [r.spread_old for r in refits]
    end
end
serialize(replace(out, ".h5" => "_chains.jls"), chains)
@printf("done, wall %.0f s\n", time() - t00)
