# ESM-Sampling

Near-optimal MCMC sampling of an energy-system model built from PyPSA-Eur: weather,
capacities and dispatch are sampled from a two-temperature ensemble: weather replicas from
their prior, the shared capacities by a Langevin step on the replica-averaged sampled score
(`sampler = "replicas"`; the learned bias of the older chains driver is retired). `src/` is the Julia package `ESMSampling` (CPU reference), `device/` the GPU port
`ESMDevice` (Reactant.jl).

The documentation starts at [docs/README.md](docs/README.md); the system is described in
`docs/system/`.

## Quickstart

```bash
# data (data/processed/model.h5: 40 nodes, 22 weather years; docs/system/model.md §5)
uv sync
scripts/download_data.sh
uv run python scripts/prepare_data.py

# Julia environment and tests
julia --project=. -e 'using Pkg; Pkg.instantiate()'
nice -n 19 julia --project=. -t 4 test/runtests.jl
nice -n 19 julia --project=. -t 4 test/test_exact.jl

# a run of the CPU driver (4 chains, all model terms, α = 1; output in results/)
nice -n 19 julia --project=. -t 4 scripts/run.jl configs/complete40.toml

# a device run (Reactant; device_backend = "cpu" or "gpu")
nice -n 19 julia --project=device -t 4 device/scripts/run_device.jl configs/device40.toml
```

Run from the repository root (configuration paths are relative to it). The configuration
keys are listed in [docs/system/config.md](docs/system/config.md), the outputs in
[docs/system/data.md](docs/system/data.md), the tests in
[docs/system/testing.md](docs/system/testing.md).
