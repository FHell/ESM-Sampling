# Usage: julia --project=. scripts/run.jl configs/demo.toml
using ESMSampling
cfg = length(ARGS) > 0 ? load_config(ARGS[1]) : Config()
ESMSampling.run(cfg)
