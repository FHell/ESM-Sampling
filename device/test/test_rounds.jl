# Phase-2 acceptance (docs/reactant_design.md §8, adapted), part 2: the compiled sweep.
#   (vi)  one sweep of all plain families on Julia arrays and traced by Reactant (CPU backend by default) with the
#         same given uniforms: identical to round-off
#   (iii) many sweeps in one compiled program (device loops, Reactant's RNG): the residual cache r
#         against rebuild_r ≤ 1e-8 MW, and cost_parts against the CPU energy_cost of the downloaded
#         state (to_host!)
# Run: cd device && nice -n 19 julia --project=. -t 4 test/test_rounds.jl   (≈ 15 min: two compiles
# of ≈ 4 min, one Julia-array sweep of ≈ 2 min)
include("setup40.jl")
using Test
using Reactant: to_rarray, ConcreteRNumber
# work rules: traced runs are checked on Reactant's CPU backend (no CUDA stage, no 4 GB limit);
# DEVICE_BACKEND=gpu runs them on the GPU
Reactant.set_default_backend(get(ENV, "DEVICE_BACKEND", "cpu"))

const NSW = parse(Int, get(ENV, "NSWEEPS", "10"))
C = parse(Int, get(ENV, "NCHAINS", "2"))
P, M, ix, S, ss = setup40(; C=C)
pl = D.plan(M, S, P.βv; bmax=Dict(:theta => 4096))
fv = Val(pl.fams)
cpR, spR, kR, tabR = to_rarray(pl.cp), to_rarray(pl.sp), to_rarray(pl.k), to_rarray(pl.dev)
host(o) = D.FlatState(map(Array, (o.θ, o.ω, o.uc, o.e, o.u, o.r))...)
mv = D.moves_per_sweep(pl)
println("plain families ", pl.fams, "; line moves per chain and sweep ", mv, " = ", sum(mv))

@testset "(vi) Julia arrays and Reactant: same uniforms, same result" begin
    U = NamedTuple{pl.fams}(Tuple(rand(MersenneTwister(11), C, size(pl.dev[f].on, 1), 2, size(pl.dev[f].on, 2)) for f in pl.fams))
    sweepU(st, cp, sp, k, tabs, U) = D.run_rounds(st, cp, sp, k, tabs, 1, nothing, U, fv)
    tc = @elapsed f = @compile sweepU(to_rarray(pl.st), cpR, spR, kR, tabR, to_rarray(U))
    oR = host(f(to_rarray(pl.st), cpR, spR, kR, tabR, to_rarray(U)))
    tj = @elapsed oJ = D.run_rounds(pl.st, pl.cp, pl.sp, pl.k, pl.dev, 1, nothing, U, fv)
    @printf("  compile %.0f s, Julia-array sweep %.0f s\n", tc, tj)
    for fld in (:θ, :ω, :uc, :e, :u, :r)
        a, b = getfield(oR, fld), getfield(oJ, fld)
        d = maximum(abs.(a .- b)); sc = max(maximum(abs.(b)), 1.0)
        @printf("  %-2s max |reactant − julia| = %.2e (max |x| %.2e, relative %.1e)\n", fld, d, sc, d / sc)
        @test d <= 1e-10 * sc
    end
end

@testset "(iii) $NSW sweeps on the device: r cache and cost" begin
    rng = Reactant.ReactantRNG(to_rarray(UInt64[7, 3]))
    sw(st, cp, sp, k, tabs, ns, rng) = D.run_sweeps(st, cp, sp, k, tabs, ns, rng, fv)
    st0 = to_rarray(pl.st)
    tc = @elapsed f = @compile sw(st0, cpR, spR, kR, tabR, ConcreteRNumber(1), rng)
    host(f(to_rarray(pl.st), cpR, spR, kR, tabR, ConcreteRNumber(1), rng))           # warm-up
    t1 = @elapsed o = host(f(to_rarray(pl.st), cpR, spR, kR, tabR, ConcreteRNumber(NSW), rng))
    nm = C * sum(mv) * NSW
    @printf("  compile %.0f s; %d sweeps × %d chains in %.1f s: %.2f s/sweep, %.3g line moves/s\n", tc, NSW, C, t1, t1 / NSW, nm / t1)
    S1 = deepcopy(S)
    rc = D.unflat!(S1, o)
    rb, parts = D.cost_parts(M, S1)
    drift = maximum(abs.(rb .- rc))
    @printf("  r cache vs rebuild_r after %d sweeps (no rebuild in between): %.2e MW (max |r| %.2e)\n",
            NSW, drift, maximum(abs.(rb)))
    @test drift <= 1e-8
    p0 = D.cost_parts(M, S)[2]
    for c in 1:C
        s = D.to_host!(P, M, ix, S1, c, copy(ss[c]))
        rel = abs(parts.op[c] / s.parts.op - 1)
        @printf("  chain %d: op %.6e → %.6e; device cost_parts vs CPU energy_cost of the downloaded state: rel %.1e (shed %.3e vs %.3e)\n",
                c, p0.op[c], parts.op[c], rel, parts.shed[c], s.parts.shed)
        @test rel < 1e-11
        @test isapprox(parts.shed[c], s.parts.shed; rtol=1e-10, atol=1e-6)
        @test isapprox(parts.co2[c], s.parts.co2; rtol=1e-10)
    end
end
