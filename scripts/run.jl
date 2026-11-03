# CPU driver: `nchains` finest-level chains with diagnostics (docs/system/sampler.md).
# Usage (from the repository root):
#   nice -n 19 julia --project=. -t 4 scripts/run.jl configs/complete40.toml
using ESMSampling
cfg = length(ARGS) > 0 ? load_config(ARGS[1]) : Config()
ESMSampling.run(cfg)
