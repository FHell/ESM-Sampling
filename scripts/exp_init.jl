# LP-free initialization of the finest level by prolongation (docs/flow_hierarchy.md §7)
# against the LP-based start: equilibrate level 0, prolong level by level, relax each
# level with fixed-level sweeps; compare with the same number of sweeps from the LP start.
# Usage: julia -t 8 --project=. scripts/exp_init.jl [config] [level0_its] [sweeps_per_level]
using ESMSampling, Random, Statistics, Printf
const M = ESMSampling

function main()
    cfgpath = length(ARGS) > 0 ? ARGS[1] : "configs/flow.toml"
    n0 = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 10
    nsw = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 20
    cfg = M.load_config(cfgpath); cfg.verbose = false
    P, sL, info = M.setup(cfg; rng=MersenneTwister(cfg.seed))
    L = M.nlevels(P) - 1
    S = M.new_sampler(P, sL, cfg)
    rng = MersenneTwister(2)
    fmt(s) = @sprintf("total %.4e (capex %.4e, op %.4e, shed %.2e MWh)", s.parts.capex + s.parts.op, s.parts.capex, s.parts.op, s.parts.shed)
    s0 = M.restrict_state(P, sL, 0)
    t = @elapsed for it in 1:n0
        s0 = M.base_step(rng, P, s0, S)
    end
    println("level 0 after $n0 its [$(round(t)) s]: ", fmt(s0))
    t = @elapsed sP = M.init_by_prolongation(rng, P, S, s0, sL; nsweeps=fill(nsw, L))
    println("prolonged + $nsw sweeps per level [$(round(t)) s]: ", fmt(sP))
    s = sL
    t = @elapsed for _ in 1:nsw
        s = M.fine_local_step(rng, P, s, S)
    end
    println("LP start + $nsw finest sweeps [$(round(t)) s]: ", fmt(s))
    println("LP objective (reference level, joint): ", @sprintf("%.4e", info.lp.objective))
end
main()
