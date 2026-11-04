# Archived development notes

Designs, experiments and plans from the development of the sampler. They are not a
description of the system: [../system/](../system/) and the code win wherever they disagree,
and several of them describe designs that were later removed (MLDA, HMC, Gibbs blocks, hard
walls, the LP-linearized formulation).

| note | content | superseded by |
|---|---|---|
| [setup_overview.md](setup_overview.md) | the sampler as it stood before the legacy removal | [sampler.md](../system/sampler.md), [architecture.md](../system/architecture.md) |
| [mathematical_formulation.md](mathematical_formulation.md) | the first formulation of the joint weather × decision problem | [ensemble.md](../system/ensemble.md), [model.md](../system/model.md) |
| [physical_formulation.md](physical_formulation.md), [sampling_formulations.md](sampling_formulations.md) | the physical formulation and the sampling-friendly component forms (§12 onward: components not yet modelled) | [model.md](../system/model.md), [formulations.md](../system/formulations.md) |
| [implementation.md](implementation.md), [local_kernel.md](local_kernel.md) | algorithms of the first implementation, sparse bases and the local exact kernel | [sampler.md](../system/sampler.md), [moves/](../system/moves/README.md) |
| [flow_hierarchy.md](flow_hierarchy.md) | the flow-based hierarchical design, with its derivations | [sampler.md](../system/sampler.md), [moves/transport.md](../system/moves/transport.md) |
| [model_completion.md](model_completion.md) | the completion of the model (losses, links, commitment) | [model.md](../system/model.md) |
| [alternatives.md](alternatives.md), [toy_mlda.md](toy_mlda.md), [toy_results.md](toy_results.md) | alternatives after the first reviews; the toy model on which MLDA fails | [roadmap.md](../system/roadmap.md) |
| [plan.md](plan.md), [todo.md](todo.md), [validation_plan.md](validation_plan.md) | plans and to-do lists of the earlier phases | [roadmap.md](../system/roadmap.md), [validation.md](../system/validation.md) |
| [overnight_results.md](overnight_results.md) | results of an earlier overnight run | — |
| [perf_notes.md](perf_notes.md) | CPU performance work | [performance.md](../system/performance.md) |
| [reactant_performance.md](reactant_performance.md) | what a Reactant port could gain; §4 on Reactant's probabilistic-programming layer (`Reactant.ProbProg`) is still useful background for the device port | [performance.md](../system/performance.md) (including the H100 estimate), [device.md](../system/device.md) |
| [reactant_phase2.md](reactant_phase2.md) | phase 2 of the device port (term batch, dense sampler, plain rounds) | [device.md](../system/device.md) |
| [full_resolution_run.md](full_resolution_run.md) | the prepared 256-node hourly ERA5 run; its result sections were never filled | [roadmap.md](../system/roadmap.md) §6, [data_sources.md](../system/data_sources.md) |
