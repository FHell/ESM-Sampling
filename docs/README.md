# Documentation

The documents in `docs/system/` describe the system as it is built, or its intended final
shape where it is not built yet, with "Status: missing / deviation" notes where the code
differs. Start with the ensemble and the sampler; the architecture document maps everything
onto the code. Agent and session rules are in [../CLAUDE.md](../CLAUDE.md).

| document | content |
|---|---|
| [system/ensemble.md](system/ensemble.md) | the sampled measure: two temperatures, perfect-foresight dispatch per weather year, the learned bias, the reference measure, regularized constraints, zero-temperature limits |
| [system/model.md](system/model.md) | the energy model: fields, bases and levels, every cost term, the optional features, the data pipeline from PyPSA-Eur and the weather sources |
| [system/weather_model.md](system/weather_model.md) | the multilevel weather model: anomalies, regime-switching VAR, detail models, non-centered coordinates, data map, fit and cache |
| [system/sampler.md](system/sampler.md) | the sampler: exact line moves, invariance and compensation rules, sweeps, capacity and weather moves, bias learning and post-correction, adaptation, multi-chain runs |
| [system/moves/README.md](system/moves/README.md) | index of the move families, with CPU and device availability |
| [system/moves/coordinates.md](system/moves/coordinates.md) | atoms, single units, node angles, commitment |
| [system/moves/pairs.md](system/moves/pairs.md) | same-node and cross-node pairs, localized re-routing, storage–generator pairs |
| [system/moves/cells.md](system/moves/cells.md) | cell and cell-pair moves (coarse spatial directions) |
| [system/moves/transport.md](system/moves/transport.md) | transport moves in space and their compensation variants |
| [system/moves/storage.md](system/moves/storage.md) | transport in time: single-instant and dyadic-block storage moves |
| [system/moves/learned.md](system/moves/learned.md) | directions and shapes learned by the adaptive kernel |
| [system/moves/capacity.md](system/moves/capacity.md) | scaling and peak-set capacity moves, level-0 capacity moves |
| [system/moves/weather.md](system/moves/weather.md) | weather moves at the fine level and in the level-0 chain |
| [system/architecture.md](system/architecture.md) | packages, files, types, call graph of each driver, caches, threading, outputs, tests, legacy code |
| [system/device.md](system/device.md) | the GPU port: scope, state, exact padded sampler, rounds, capacity pass, host loop, status |
| [system/config.md](system/config.md) | every configuration key with its default, and the values that define the system |
| [system/data.md](system/data.md) | input files and their HDF5 layout, output layouts of each driver, caches |
| [system/validation.md](system/validation.md) | convergence criteria and the comparison strategy with PyPSA-Eur |
| [system/testing.md](system/testing.md) | exactness tests on the CPU and the verification of the device port |
| [system/roadmap.md](system/roadmap.md) | open problems, missing pieces and planned experiments |
| [system/glossary.md](system/glossary.md) | symbols and terms |

The other files in `docs/` are working notes from the development (designs, experiments,
plans). They are being archived and are not a description of the system; where they and
`docs/system/` disagree, `docs/system/` and the code win. `docs/reactant_phase3.md` records the
current phase of the device port.
