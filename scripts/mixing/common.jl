# Common setup for the mixing diagnosis (docs/mixing_diagnosis.md).
# include("scripts/mixing/common.jl") from the repository root, in a session started with
# `nice -n 19 julia --project=. -t 4`.
import ESMSampling as ML
using Random, LinearAlgebra, SparseArrays, Statistics, Printf, HDF5

"Set up the problem of a configuration file (as `run_chains` does, without running chains)."
function mix_setup(path::String; overrides...)
    cfg = ML.load_config(path)
    for (k, v) in overrides
        setfield!(cfg, k, v)
    end
    rng = MersenneTwister(cfg.seed)
    P, sL, info = ML.setup(cfg; rng=rng)
    if P.use_bias                                  # level-0 warmup and bias learning, as in `run_chains`
        S0 = ML.new_sampler(P, sL, cfg)
        s0 = ML.predescend(P, ML.restrict_state(P, sL, 0), cfg)
        s0, _, _, _ = ML.run_level0!(rng, P, s0, S0, cfg.warmup0, cfg; learn=true, label="warmup")
        P.bias.active = false
    end
    return cfg, P, sL, info
end

"Chain starts and samplers as in `run_chains` (α = 0 path)."
function mix_chains(cfg, P, sL)
    K = cfg.nchains
    kinds = [cfg.chain_starts[mod1(c, length(cfg.chain_starts))] for c in 1:K]
    newweather = cfg.fine_weather > 0
    chains = [ML.chain_start(MersenneTwister(cfg.seed + 7 * c), P, sL, kinds[c], newweather) for c in 1:K]
    samplers = [ML.new_sampler(P, chains[c], cfg) for c in 1:K]
    rngs = [MersenneTwister(cfg.seed + 1000 * c) for c in 1:K]
    return kinds, chains, samplers, rngs
end

"Node capacities per k field."
knodes(P, s) = ML.knodes(P, s.ℓ, s.k)

"Carrier totals of the node capacities."
ktot(P, s) = [sum(k) for k in knodes(P, s)]
