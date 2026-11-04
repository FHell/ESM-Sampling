# Shared setup of the phase-2 tests: the 40-node model with all model terms on (as test_cost.jl),
# C chains perturbed from the heuristic start.
using ESMDevice, ESMSampling, Random, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice

function setup40(; C=3, data=joinpath(@__DIR__, "..", "..", "data", "processed", "model.h5"), seed=3, pert=0.02,
                 losses=true, link_losses=true, uc=["gas", "nuclear"], levels=nothing)
    cfg = ML.load_config(joinpath(@__DIR__, "..", "..", "configs", "haar_poc.toml")); cfg.verbose = false
    cfg.data = data
    levels === nothing || (cfg.levels = levels)            # e.g. [(10, 120), (40, 120)]: a short year for compile probes
    cfg.out = joinpath(@__DIR__, "..", "..", "results", "haar_poc.h5")
    cfg.storage = "soc"; cfg.renewables = "load"; cfg.measure = "physical"; cfg.init = "heuristic"
    cfg.losses = losses; cfg.thermal = "soft"; cfg.link_losses = link_losses; cfg.uc = uc; cfg.brownfield = true
    P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
    rng = MersenneTwister(seed)
    ss = ML.State[]
    for c in 1:C
        s = copy(sL)
        c > 1 && (s.v .+= pert .* c .* randn(rng, length(s.v)); s.k .*= 1 .+ 0.05 .* randn(rng, length(s.k)))
        push!(ss, ML.evaluate!(P, s))
    end
    M, ix = D.pack(P, sL.ℓ)
    S = D.to_device(P, M, ix, ss)
    return P, M, ix, S, ss
end
