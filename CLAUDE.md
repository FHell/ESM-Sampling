# Working notes for agents and sessions

ESM-Sampling is a near-optimal MCMC sampler for an energy-system model built from PyPSA-Eur:
weather, capacities and dispatch are sampled jointly from a two-temperature ensemble with a
learned bias. `src/` is the Julia package `ESMSampling` (CPU reference), `device/` the GPU port
`ESMDevice` (Reactant.jl).

## Read first

1. [docs/README.md](docs/README.md), the index.
2. The `docs/system/` documents relevant to the task: [ensemble.md](docs/system/ensemble.md)
   and [sampler.md](docs/system/sampler.md) for anything that touches the target or a move,
   [architecture.md](docs/system/architecture.md) to find the code, the move document of the
   family you work on, [device.md](docs/system/device.md) and
   [testing.md](docs/system/testing.md) for device work.

The other files in `docs/` are development notes, not a description of the system.

## Keep the documents current

- `docs/system/` describes the system as built, or its intended final shape, with
  "Status: missing" or "Status: deviation" notes where the code is not there yet. It does not
  record how the code got there: no history, no superseded designs, no experiment logs.
- Update the relevant document in the same work item as the code change: a new move family
  gets its section in `docs/system/moves/` and a row in `moves/README.md`; a new config key goes
  into [config.md](docs/system/config.md); a new output into [data.md](docs/system/data.md); a
  removed limitation removes its status note.
- Record every experiment (planned, running, done) in the ledger
  [docs/experiments.md](docs/experiments.md): question, setup, one-line result, where the
  record lives.
- Personal or session notes (memory files, scratch notes, final reports) do not replace these
  documents.
- Code wins over documents. If you find a disagreement, fix the document (or the code, if the
  document states the intended design) and say so in your report.

## Machine rules

- The user works on this machine in parallel. Run Julia at `nice -n 19` with at most 4 threads
  (`julia -t 4`) unless told otherwise; keep other heavy jobs (PyPSA-Eur builds) niced and
  below about 6 cores; do not run several heavy jobs at once.
- Kill processes carefully: select PIDs with `ps -eo pid,args | grep "[p]attern"`; never use
  `pkill -f` or `pgrep -f` with a pattern that also matches your own shell command.
- Keep everything inside the project folder (data in `data/`, outputs in `results/`, scratch in
  your scratchpad or `results/`). Install no new external binaries.

## Testing rules

- CPU: exactness tests (`test/runtests.jl`, `test/test_exact.jl`): deterministic consistency
  checks (incremental against global cost, caches, gradients, invariants) and line draws
  against quadrature. A new move family needs at least the incremental-cost and cache checks
  and, if it is not a plain fixed-direction move, an invariance check of its curve.
- Device work, in this order: (1) the device code on plain Julia arrays against the CPU
  reference, with the same prepared moves and the same mock random numbers on both sides
  (`MockRNG` on the CPU, input arrays on the device), compared exactly or to round-off;
  (2) the same code compiled by Reactant on the CPU backend
  (`Reactant.set_default_backend("cpu")`), which must equal (1); (3) the GPU only for timings.
  No statistical tests (KS, z-tests, many-chain runs) for the port. Device code uses
  Reactant's own RNG (`ReactantRNG`).
- Work in one persistent Julia session (the Julia MCP server or a REPL): each fresh process
  costs minutes of start-up and compilation. Compile only once the array version passes.

## Where things are

- Roadmap and open problems: [docs/system/roadmap.md](docs/system/roadmap.md).
- System configurations: `configs/complete40.toml` (all model terms, adaptive kernel),
  `configs/chains40.toml` (multi-chain screening), `configs/device40.toml` (device run). Many
  `Config` defaults are legacy values; see [config.md](docs/system/config.md) §1.
- Run from the repository root (configuration paths are relative to it):
  `nice -n 19 julia --project=. -t 4 scripts/run.jl <config>`; device:
  `nice -n 19 julia --project=device -t 4 device/scripts/run_device.jl configs/device40.toml`.
- Data: `scripts/download_data.sh`, then `uv run python scripts/prepare_data.py`
  ([model.md](docs/system/model.md) §5).
