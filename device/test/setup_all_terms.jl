# 40-node model with all model-completion terms, α = 0 (shared setup of the device tests)
cfg = ML.load_config(joinpath(@__DIR__, "..", "..", "configs", "haar_poc.toml")); cfg.verbose = false
cfg.data = joinpath(@__DIR__, "..", "..", "data", "processed", "model.h5")
cfg.out = joinpath(@__DIR__, "..", "..", "results", "haar_poc.h5")
cfg.storage = "soc"; cfg.renewables = "load"; cfg.measure = "physical"; cfg.init = "heuristic"
cfg.losses = true; cfg.thermal = "soft"; cfg.link_losses = true; cfg.uc = ["gas", "nuclear"]; cfg.use_bias = false
P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
L = sL.ℓ
M, ix = D.pack(P, L)
