# Session setup of the Langevin and weather-refresh measurements: the chains40 problem
# (40 nodes, daily, physical measure) without bias, from the repository root:
#   include("scripts/langevin/setup40.jl")
include(joinpath(@__DIR__, "..", "mixing", "common.jl"))
include(joinpath(@__DIR__, "langevin.jl"))
using Serialization
cfg = ML.load_config(joinpath(@__DIR__, "..", "..", "configs", "chains40.toml"))
cfg.use_bias = false
cfg.verbose = false
P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
E = P.E
S = ML.new_sampler(P, sL, cfg)
LM = ML.local_model(S.loc, P, sL.ℓ)
names_k = [String(sp.name) for sp in E.kspecs]
