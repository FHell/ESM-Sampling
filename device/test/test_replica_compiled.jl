# The four device programs of the replica driver compiled by Reactant against the same code on
# plain Julia arrays (docs/system/testing.md, device order, step 2): the operation sweep (with
# given uniforms, `run_rounds`), the score accumulation, the Langevin step with the dispatch
# following (with given normals, `lgv_pass`) and the weather upload, on the inputs of a replica
# run after a few array steps; then a short compiled run of the driver itself.
#
# Compiles only with ESM_DEVICE_COMPILE=1 (several minutes; CPU backend unless REACTANT_BACKEND
# is set); without it, the array side alone runs as a check of the inputs:
#   ESM_DEVICE_COMPILE=1 nice -n 19 julia --project=device -t 8 device/test/test_replica_compiled.jl
# The array side against the CPU is test_replica_driver.jl.
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice

const ROOT = joinpath(@__DIR__, "..", "..")
const COMPILE = get(ENV, "ESM_DEVICE_COMPILE", "") == "1"
const FAMS = Tuple(Symbol.(split(get(ENV, "ESM_DEVICE_FAMILIES", "gen0,gen,uc,pair0,pair,soc,link,theta3,theta5,theta"), ",")))
cfg = ML.load_config(joinpath(ROOT, "hpc", "configs", "device_smoke.toml"))
cfg.data = joinpath(ROOT, "data", "processed", "model.h5")
cfg.out = joinpath(ROOT, "results", "test_replica_compiled", "out.h5"); mkpath(dirname(cfg.out))
cfg.checkpoint = ""; cfg.stop_file = ""; cfg.verbose = false
cfg.mlda_warmup, cfg.nsamples, cfg.device_sweeps = 4, 4, 1

# ---------------------------------------------------------------- inputs: a replica run on arrays after 3 steps
RD = D.replica_device_setup(cfg; families=FAMS, compile=false)
for it in 1:3
    D.replica_device_step!(RD, it)
end
dev, P = RD.dev, RD.P
C, nch = RD.nch * RD.Mr, RD.nch
fv = Val(Tuple(f for f in FAMS if haskey(dev[:tabs], f)))
dims, ph, comp = RD.dims, Val(P.E.measure == :physical), Val(RD.ctl.comp)
kw = (; rr=P.βk / P.βv, βk=P.βk, clip=RD.ctl.clip, κmin=RD.ctl.κmin, box=RD.ctl.box)
rng = MersenneTwister(77)
U = NamedTuple{keys(dev[:tabs])}(Tuple(rand(rng, C, size(t.on, 1), 3, size(t.on, 2)) for t in values(dev[:tabs])))
ξ = (randn(rng, nch, size(dev[:kg], 2)), randn(rng, nch, size(dev[:ks], 2)), randn(rng, nch, size(dev[:RS].kr, 2)))
W = Float64[isodd(c) ? 1.0 : 0.0 for c in 1:C]                       # one active replica per chain (M = 2)
reps = [s for R in RD.sets for s in R.reps]
for (c, R) in enumerate(RD.sets)                                      # new weather for the first replica of every chain
    ML.replica_weather!(R, 1, P, RD.LM, RD.F, RD.ctlw, 4)
end
D0, c0, A = D.weather_arrays(P, RD.M, [s for R in RD.sets for s in R.reps])
chg = Float64[mod1(c, RD.Mr) == 1 for c in 1:C]

op(st, cp, sp, k, tabs, U) = D.run_rounds(st, cp, sp, k, tabs, 1, nothing, U, fv)
sa(st, kg, ks, RS, KP, RP, acc, wt) = D.score_acc(st, kg, ks, RS, KP, RP, acc, wt, dims, ph; βv=P.βv, βk=P.βk)
lg(st, kg, ks, RS, dn, cr, cap, KP, RP, LP, acc, W, ξg, ξs, ξr, wsh, rg, sg) =
    D.lgv_pass(st, kg, ks, RS, dn, cr, cap, KP, RP, LP, acc, W, ξg, ξs, ξr, wsh, rg, sg, dims; kw...)
wu(st, kg, ks, RS, dn, cr, KP, RP, D0, c0, A, chg, wsh) = D.weather_update(st, kg, ks, RS, dn, cr, KP, RP, D0, c0, A, chg, wsh, dims, comp)
acc1 = sa(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:KP], dev[:RP], dev[:acc0](), 1.0)
args = (; op=(dev[:st], dev[:cp], dev[:sp], dev[:kk], dev[:tabs], U),
        sa=(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:KP], dev[:RP], dev[:acc0](), 1.0),
        lg=(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:cap], dev[:KP], dev[:RP], dev[:LP], acc1, W, ξ..., dev[:wsh],
            dev[:rg], dev[:sg]),
        wu=(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:KP], dev[:RP], D0, c0, A, chg, dev[:wsh]))
fns = (; op, sa, lg, wu)
outs = NamedTuple{keys(args)}(Tuple(fns[k](deepcopy(args[k])...) for k in keys(args)))
println("array side: ", join(keys(outs), ", "), " ran")

"Largest relative difference of two outputs (arrays and structs of arrays)."
function maxrel(a, b)
    a isa AbstractArray && return maximum(abs.(Array(a) .- Array(b)) ./ max.(1.0, abs.(Array(b))); init=0.0)
    a isa Number && return abs(a - b) / max(1.0, abs(b))
    a isa Union{Tuple,NamedTuple} && return maximum(maxrel(x, y) for (x, y) in zip(values(a), values(b)); init=0.0)
    return maximum(maxrel(getfield(a, f), getfield(b, f)) for f in fieldnames(typeof(a)); init=0.0)
end

if COMPILE
    Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
    R = Reactant.to_rarray
    @testset "replica programs compiled by Reactant = arrays" begin
        for k in keys(args)
            a = map(R, deepcopy(args[k]))
            tc = @elapsed f = @compile fns[k](a...)
            d = maxrel(f(a...), outs[k])
            @printf("  %s: compile %.0f s, max relative |compiled − arrays| %.2e\n", k, tc, d)
            @test d < 1e-9
        end
    end
    @testset "the replica driver compiled: a short run" begin
        res = D.run_replicas_device(cfg; families=FAMS, compile=true)
        @test res.exitcode == 0
        @test all(all(isfinite, s.k) && all(isfinite, s.v) for R in res.sets for s in R.reps)
    end
else
    println("ESM_DEVICE_COMPILE is not 1: the compiled half is skipped")
end
