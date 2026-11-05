# The device programs of the replica driver compiled by Reactant against the same code on plain
# Julia arrays (docs/system/testing.md, device order, step 2): the operation sweep (with given
# uniforms, `run_rounds`), the score accumulation, the Langevin step with the dispatch following
# (with given normals, `lgv_pass`; the probe slots frozen), the row-wise weather upload
# (`weather_rows`), the row copy of a probe start (`copy_rows`) and the residual rebuild
# (`rebuild_flat`), on the inputs of a replica run after a few array steps; the Float32 build's
# contractions compiled at HIGHEST precision (no TF32); then a short compiled run of the driver
# itself (and of its Float32 build with ESM_DEVICE_F32=1).
#
# Compiles only with ESM_DEVICE_COMPILE=1 (several minutes; CPU backend unless REACTANT_BACKEND
# is set); without it, the array side alone runs as a check of the inputs:
#   ESM_DEVICE_COMPILE=1 nice -n 19 julia --project=device -t 8 device/test/test_replica_compiled.jl
# ESM_DEVICE_FAMILIES (comma-separated) sets the operation families (default: the plain ones).
# The array side against the CPU is test_replica_driver.jl.
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice

const ROOT = joinpath(@__DIR__, "..", "..")
const COMPILE = get(ENV, "ESM_DEVICE_COMPILE", "") == "1"
const FAMS = Tuple(Symbol.(split(get(ENV, "ESM_DEVICE_FAMILIES", "gen0,gen,uc,pair0,pair,soc,link,theta3,theta5,theta"), ",")))
function compcfg(name)
    cfg = ML.load_config(joinpath(ROOT, "hpc", "configs", "device_smoke.toml"))
    cfg.data = joinpath(ROOT, "data", "processed", "model.h5")
    cfg.out = joinpath(ROOT, "results", "test_replica_compiled", name, "out.h5"); mkpath(dirname(cfg.out))
    cfg.checkpoint = ""; cfg.stop_file = ""; cfg.verbose = false
    cfg.mlda_warmup, cfg.nsamples, cfg.device_sweeps = 4, 4, 1
    return cfg
end
cfg = compcfg("programs")

# ---------------------------------------------------------------- inputs: a replica run on arrays after 3 steps
RD = D.replica_device_setup(cfg; families=FAMS, compile=false)
for it in 1:3
    D.replica_device_step!(RD, it)
end
dev, P = RD.dev, RD.P
C, nch = RD.nch * RD.Mp, RD.nch
fv = Val(Tuple(f for f in FAMS if haskey(dev[:tabs], f)))
dims, ph, comp = RD.dims, Val(P.E.measure == :physical), Val(RD.ctl.comp)
@assert RD.Mp == RD.Mr + 1 && typeof(dims).parameters[1][8] == 1                            # with a probe slot per chain (frozen in lgv_pass)
kw = (; rr=P.βk / P.βv, βk=P.βk, clip=RD.ctl.clip, κmin=RD.ctl.κmin, box=RD.ctl.box)
rng = MersenneTwister(77)
U = NamedTuple{keys(dev[:tabs])}(Tuple(rand(rng, C, size(t.on, 1), 3, size(t.on, 2)) for t in values(dev[:tabs])))
ξ = (randn(rng, nch, size(dev[:kg], 2)), randn(rng, nch, size(dev[:ks], 2)), randn(rng, nch, size(dev[:RS].kr, 2)))
W = zeros(C)
W[[D.slotrow(RD, c, 1) for c in 1:nch]] .= 1.0                       # one active replica per chain, the probe slot 0
for (c, R) in enumerate(RD.sets)                                      # new weather for the first replica of every chain
    ML.replica_weather!(R, 1, P, RD.LM, RD.F, RD.ctlw, 4)
end
rows = [D.slotrow(RD, c, 1) for c in 1:nch]
D0, c0, A = D.weather_arrays(P, RD.M, [RD.sets[c].reps[1] for c in 1:nch])
pos, chg = ones(Int, C), zeros(C)
pos[rows] .= 1:nch; chg[rows] .= 1.0
src = collect(1:C); src[[D.slotrow(RD, c, RD.Mp) for c in 1:nch]] .= [D.slotrow(RD, c, 2) for c in 1:nch]   # replica 2 into the probe slot

op(st, cp, sp, k, tabs, U) = D.run_rounds(st, cp, sp, k, tabs, 1, nothing, U, fv)
sa(st, kg, ks, RS, KP, RP, acc, wt) = D.score_acc(st, kg, ks, RS, KP, RP, acc, wt, dims, ph; βv=P.βv, βk=P.βk)
lg(st, kg, ks, RS, dn, cr, cap, KP, RP, LP, acc, W, ξg, ξs, ξr, wsh, rg, sg) =
    D.lgv_pass(st, kg, ks, RS, dn, cr, cap, KP, RP, LP, acc, W, ξg, ξs, ξr, wsh, rg, sg, dims; kw...)
wu(st, kg, ks, RS, dn, cr, KP, RP, D0, c0, A, pos, chg, wsh) = D.weather_rows(st, kg, ks, RS, dn, cr, KP, RP, D0, c0, A, pos, chg, wsh, dims, comp)
pc(st, cp, kg, ks, RS, dn, cr, cap, src) = D.copy_rows(st, cp, kg, ks, RS, dn, cr, cap, src)
rb(st, kg, ks, dn, cr, cap, M) = D.rebuild_flat(st, kg, ks, dn, cr, cap, M, D.rebuild_dims(C, RD.M, RD.ix))
acc1 = sa(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:KP], dev[:RP], dev[:acc0], 1.0)
args = (; op=(dev[:st], dev[:cp], dev[:sp], dev[:kk], dev[:tabs], U),
        sa=(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:KP], dev[:RP], dev[:acc0], 1.0),
        lg=(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:cap], dev[:KP], dev[:RP], dev[:LP], acc1, W, ξ..., dev[:wsh],
            dev[:rg], dev[:sg]),
        wu=(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:KP], dev[:RP], D0, c0, A, pos, chg, dev[:wsh]),
        pc=(dev[:st], dev[:cp], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:cap], src),
        rb=(dev[:st], dev[:kg], dev[:ks], dev[:dn], dev[:cr], dev[:cap], RD.M))
fns = (; op, sa, lg, wu, pc, rb)
outs = NamedTuple{keys(args)}(Tuple(fns[k](deepcopy(args[k])...) for k in keys(args)))
println("array side: ", join(keys(outs), ", "), " ran")
# the probe slots are frozen by the Langevin step (arrays): k, utilizations, residual and net load kept
let o = outs.lg, p = [D.slotrow(RD, c, RD.Mp) for c in 1:nch]
    @test o[2][p, :] == dev[:kg][p, :] && o[1].ω[p, :] == dev[:st].ω[p, :] && o[1].r[p, :] == dev[:st].r[p, :] && o[6][p, :, :] == dev[:dn][p, :, :]
    @test o[2][rows, :] != dev[:kg][rows, :]
end

"Largest relative difference of two outputs (arrays and structs of arrays)."
function maxrel(a, b)
    a isa AbstractArray && return maximum(abs.(Array(a) .- Array(b)) ./ max.(1.0, abs.(Array(b))); init=0.0)
    a isa Number && return abs(a - b) / max(1.0, abs(b))
    a isa Union{Tuple,NamedTuple} && return maximum(maxrel(x, y) for (x, y) in zip(values(a), values(b)); init=0.0)
    return maximum(maxrel(getfield(a, f), getfield(b, f)) for f in fieldnames(typeof(a)); init=0.0)
end
"Compile program k by Reactant and compare it with the arrays; returns (compile seconds, max relative difference)."
function compile_check(k)
    a = map(Reactant.to_rarray, deepcopy(args[k]))
    tc = @elapsed f = D.ftcompile(() -> @compile fns[k](a...))
    d = maxrel(f(a...), outs[k])
    @printf("  %s: compile %.0f s, max relative |compiled − arrays| %.2e\n", k, tc, d)
    return tc, d
end
"dot_general precisions in the HLO of a Float32 one-hot contraction (`tmul`) traced through `ftcompile` and without it."
function f32_precisions()
    F = D.F32
    x, y = Reactant.to_rarray(rand(Float32, 4, 3, 5)), Reactant.to_rarray(rand(Float32, 5, 2))
    pr(s) = unique(m.captures[1] for m in eachmatch(r"precision = \[(\w+), \w+\]", s))
    return pr(string(F.ftcompile(() -> @code_hlo F.tmul(x, y)))), pr(string(@code_hlo F.tmul(x, y)))
end

if COMPILE
    Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
    @testset "replica programs compiled by Reactant = arrays" begin
        for k in keys(args)
            tc, d = compile_check(k)
            @test d < 1e-9
        end
    end
    @testset "Float32 build: contractions at HIGHEST precision" begin
        p, p0 = f32_precisions()
        println("  dot_general precision: ftcompile $p, default $p0")
        @test p == ["HIGHEST"] && p0 == ["DEFAULT"]
    end
    "A run with device probes: step 3 has a candidate (test_replica_driver.jl B); they end and adapt in the warmup."
    function probecfg(name)
        c = compcfg(name)
        c.replica_probe_every = 3
        c.mlda_warmup, c.nsamples = 8, 4
        return c
    end
    @testset "the replica driver compiled: a short run (device probes, sync = false)" begin
        cr = probecfg("run")
        res = D.run_replicas_device(cr; families=FAMS, compile=true, sync=false)
        @test res.exitcode == 0
        @test all(all(isfinite, s.k) && all(isfinite, s.v) for R in res.sets for s in R.reps)
        ML.HDF5.h5open(cr.out) do f
            dr = read(f["replicas/1/residual_drift"])
            @printf("  residual drift per checkpoint: max %.2e MW\n", maximum(dr))
            @test maximum(dr) < 1e-6
            pit = read(f["replicas/1/probe/it"])
            println("  probes (step, replica, adapted): ", [Int.(pit[:, j]) for j in axes(pit, 2)])
            @test size(pit, 2) >= 1 && any(pit[3, :] .== 1)
        end
        LPh = D.lgv_params(res.P, res.RD.M, res.RD.ix; κs=res.RD.ctl.κs, ηw=[R.ηw for R in res.sets])
        @test Array(res.RD.dev[:LP].ηg) == LPh.ηg                # the adapted steps uploaded
    end
    if get(ENV, "ESM_DEVICE_F32", "") == "1"
        @testset "the replica driver compiled, Float32 build: a short run" begin
            cr = probecfg("run32")
            res = D.F32.run_replicas_device(cr; families=FAMS, compile=true)
            @test res.exitcode == 0
            @test all(all(isfinite, s.k) && all(isfinite, s.v) for R in res.sets for s in R.reps)
            ML.HDF5.h5open(cr.out) do f
                dr = read(f["replicas/1/residual_drift"])
                @printf("  Float32 residual drift per checkpoint (rebuilt at each): %s MW\n", string(round.(vec(maximum(dr; dims=1)); sigdigits=2)))
                @test all(isfinite, dr)
            end
        end
    end
else
    println("ESM_DEVICE_COMPILE is not 1: the compiled half is skipped")
end
