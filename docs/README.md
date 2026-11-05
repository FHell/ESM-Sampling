# Documentation

The documents in `docs/system/` describe the system as it is built, or its intended final
shape where it is not built yet, with "Status: missing / deviation" notes where the code
differs. Start with the ensemble and the sampler; the architecture document maps everything
onto the code. Agent and session rules are in [../CLAUDE.md](../CLAUDE.md).

| document | content |
|---|---|
| [system/ensemble.md](system/ensemble.md) | the sampled measure: two temperatures, perfect-foresight dispatch per weather year, how it is sampled (weather replicas from the prior, a Langevin capacity step on the replica-averaged score, the stress tilt), the reference measure, regularized constraints, zero-temperature limits |
| [system/model.md](system/model.md) | the energy model: fields, bases and levels, every cost term, the optional features, the data pipeline from PyPSA-Eur and the weather sources |
| [system/weather_model.md](system/weather_model.md) | the multilevel weather model: anomalies, regime-switching VAR, detail models, non-centered coordinates, data map, fit and cache |
| [system/sampler.md](system/sampler.md) | the sampler: exact line moves, invariance and compensation rules, sweeps, capacity and weather moves, bias learning and post-correction (retired), adaptation, multi-chain runs, the replica–Langevin sampler (§11) |
| [system/moves/README.md](system/moves/README.md) | index of the move families, with CPU and device availability |
| [system/moves/coordinates.md](system/moves/coordinates.md) | atoms, single units, node angles, commitment |
| [system/moves/pairs.md](system/moves/pairs.md) | same-node and cross-node pairs, localized re-routing, storage–generator pairs |
| [system/moves/cells.md](system/moves/cells.md) | cell and cell-pair moves (coarse spatial directions) |
| [system/moves/transport.md](system/moves/transport.md) | transport moves in space and their compensation variants |
| [system/moves/storage.md](system/moves/storage.md) | transport in time: single-instant and dyadic-block storage moves |
| [system/moves/learned.md](system/moves/learned.md) | directions and shapes learned by the adaptive kernel |
| [system/moves/capacity.md](system/moves/capacity.md) | scaling and peak-set capacity moves, level-0 capacity moves |
| [system/moves/weather.md](system/moves/weather.md) | weather moves at the fine level and in the level-0 chain |
| [system/moves/adding_a_move.md](system/moves/adding_a_move.md) | checklist for adding a move family, on the CPU and on the device |
| [system/architecture.md](system/architecture.md) | packages, files, types, call graph of each driver, caches, threading, outputs, tests |
| [system/device.md](system/device.md) | the GPU port: scope, state, exact padded sampler, rounds, capacity pass, host loop, status |
| [system/config.md](system/config.md) | every configuration key with its default, and the values that define the system |
| [system/data.md](system/data.md) | input files and their HDF5 layout, output layouts of each driver, caches |
| [system/data_sources.md](system/data_sources.md) | provenance: PyPSA-Eur networks and build configuration, NASA POWER, OPSD load, ERA5 retrieval and processing |
| [system/performance.md](system/performance.md) | measured cost per move and per sweep, scaling, device throughput, memory, estimates |
| [system/formulations.md](system/formulations.md) | sampling-friendly forms of components not yet modelled (ramping, line expansion, sector coupling, CO₂ caps, …) |
| [system/validation.md](system/validation.md) | convergence criteria and the comparison strategy with PyPSA-Eur |
| [system/testing.md](system/testing.md) | exactness tests on the CPU and the verification of the device port |
| [system/hpc.md](system/hpc.md) | runs on the PIK HPC cluster: environment, storage layout, caches, job types, checkpoints, run records |
| [system/roadmap.md](system/roadmap.md) | open problems, missing pieces and planned experiments |
| [system/glossary.md](system/glossary.md) | symbols and terms |

The other files are working notes, not a description of the system; where they and
`docs/system/` disagree, `docs/system/` and the code win:

- [worklog.md](worklog.md): the work log by date and the open to-dos not yet tackled.
- [retired.md](retired.md): approaches tried and retired, each with the reason.
- [campaigns.md](campaigns.md): the measurement campaigns (validation, baselines, scaling,
  production runs on the cluster), with how to run them and their results.
- [experiments.md](experiments.md): the ledger of experiments run and planned, with the
  question, setup, result and record of each.
- [archive/](archive/README.md): development notes superseded by `docs/system/` (designs,
  experiments, plans, earlier phases), with an index of what replaced each.
- [reactant_design.md](reactant_design.md): the design of the device port (its sections are
  cited from the device code); [reactant_phase3.md](reactant_phase3.md): the current phase of
  the port.
- [replica_sampler.md](replica_sampler.md): the prototype of the replica Langevin sampler (M
  weather replicas sharing the capacities, prior-reversible weather process, lag probe, stress
  tilt), the prior-invariance argument and the measurements (2026-11-04); the sampler itself is
  now `src/replicas.jl` ([system/sampler.md](system/sampler.md) §11).
- [device_review.md](device_review.md): a fresh-eyes review of the Reactant port (2026-11-04):
  the slong compile cause, the schedule drawn once per run, Float32 and driver findings.
- [mixing_diagnosis.md](mixing_diagnosis.md): the diagnosis of the sampler's mixing at 40 nodes
  (2026-11-04) and the prototypes and measurements behind `kop` and the z0 block moves.
