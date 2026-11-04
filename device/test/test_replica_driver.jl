# The replica driver of the device (device/src/replica_run.jl) on plain Julia arrays against the
# CPU replica sampler (src/replicas.jl; docs/system/testing.md, device order, step 1):
#   A. the weather paths of all replicas over a run equal those of `run_replicas` bitwise (the
#      weather never reads k or v; same streams), with the gate schedule (since, window);
#   B. one step after the operation sweeps: the downloaded scores against `capacity_scores`, the
#      gates (`update_gate!`) and drift weights, the Langevin step with the same normals against
#      `langevin_capacities`, the dispatch following against `follow_capacities!`; a probe step
#      (`lag_probe`, `adapt_step!`) and the per-chain steps uploaded;
#   C. snapshot, stop (exit 75) and resume in the CPU driver's layout.
# The compiled programs against these arrays: test_replica_compiled.jl (ESM_DEVICE_COMPILE=1).
#   julia --project=device -t 8 device/test/test_replica_driver.jl
using ESMDevice, ESMSampling, Random, Test, Printf
const ML = ESMSampling
const D = ESMDevice
include(joinpath(@__DIR__, "mockrng.jl"))
Random.randn(r::MockRNG, n::Integer) = [randn(r) for _ in 1:n]

const ROOT = joinpath(@__DIR__, "..", "..")
const OUTD = joinpath(ROOT, "results", "test_replica_driver")
const FAMS = (:gen0, :soc)                 # the operation families do not enter the comparisons
mkpath(OUTD)
function testcfg(name)
    cfg = ML.load_config(joinpath(ROOT, "hpc", "configs", "device_smoke.toml"))
    cfg.data = joinpath(ROOT, "data", "processed", "model.h5")
    cfg.out = joinpath(OUTD, name, "out.h5"); mkpath(dirname(cfg.out))
    cfg.checkpoint = ""; cfg.stop_file = ""; cfg.verbose = false
    cfg.mlda_warmup, cfg.nsamples, cfg.device_sweeps = 4, 4, 1      # a host checkpoint every step (8 trace entries, as the CPU)
    return cfg
end
rel(a, b) = maximum(abs.(a .- b) ./ max.(1.0, abs.(b)); init=0.0)

# ---------------------------------------------------------------- A. weather paths = run_replicas
@testset "weather paths and gate schedule = run_replicas" begin
    rc = ML.run_replicas(testcfg("cpu"))
    rd = D.run_replicas_device(testcfg("dev"); families=FAMS, compile=false)
    @test rc.exitcode == 0 && rd.exitcode == 0
    for c in eachindex(rc.sets), j in eachindex(rc.sets[c].reps)
        a, b = rc.sets[c].reps[j], rd.sets[c].reps[j]
        @test a.m == b.m && a.z0 == b.z0 && a.ws == b.ws
    end
    @test all(rc.sets[c].since == rd.sets[c].since && rc.sets[c].window == rd.sets[c].window for c in eachindex(rc.sets))
end

# ---------------------------------------------------------------- B. one step against the CPU functions
cfg = testcfg("step")
cfg.replica_probe_every = 3                  # replica 1 is active at steps 1–3, so step 3 has a probe candidate
RD = D.replica_device_setup(cfg; families=FAMS, compile=false)
P, ctl, LM = RD.P, RD.ctl, RD.LM
units = ML.capacity_units(P, LM)
nch, Mr = RD.nch, RD.Mr
rr = P.βk / P.βv
"The normals of `lgv_pass_rng` from a copy of the Langevin stream, per chain in the CPU's order (k field, node)."
function cpu_normals(RD, c, ηw)
    lr = copy(RD.dev[:lrng])
    Ug, Us, Ur = size(RD.dev[:kg], 2), size(RD.dev[:ks], 2), size(RD.dev[:RS].kr, 2)
    ξ = (randn(lr, Float64, nch, Ug), randn(lr, Float64, nch, Us), randn(lr, Float64, nch, Ur))
    z = Float64[]
    for kf in unique(u[4] for u in units)
        ηw[kf] == 0 && continue
        append!(z, [ξ[b][c, q] for (b, q) in RD.um[kf]])
    end
    return z
end
"One device step in its parts, checked against the CPU functions on the downloaded states."
function check_step(RD, it)
    moved, tw = D.replica_device_weather!(RD, it)
    acc = D.replica_device_operation!(RD)
    D.replica_download!(RD)
    pre = deepcopy(RD.sets)                                   # states after the operation, gates before the update
    zs = [cpu_normals(RD, c, pre[c].ηw) for c in 1:nch]
    recs = D.replica_device_capacity!(RD, it, acc, moved, tw)
    D.replica_download!(RD)
    e = Dict(:score => 0.0, :weights => 0.0, :k => 0.0, :ω => 0.0, :r => 0.0, :e => 0.0)
    gates_ok = true
    for c in 1:nch
        R = pre[c]
        scs = [ML.capacity_scores(P, s) for s in R.reps]
        fk = scs[1][2]
        Kraw = ML.knodes(P, R.reps[1].ℓ, R.reps[1].k)
        mask = ML.interior_nodes(P, Kraw)
        fkm = ML.masked_means(fk, mask)
        for j in 1:Mr, f in eachindex(fk)
            e[:score] = max(e[:score], maximum(abs.(recs[c].sks[j][f] .- scs[j][1][f]) ./ max.(abs.(scs[j][1][f]), 1e-3 .* fk[f] ./ rr); init=0.0))
        end
        for j in 1:Mr
            ML.update_gate!(R, j, ML.masked_means([rr .* x for x in scs[j][1]], mask) ./ fkm, ctl)
        end
        gates_ok &= R.active == RD.sets[c].active
        Φs = ctl.Φ === nothing ? zeros(Mr) : [ctl.Φ(P, s.xs[end]) for s in R.reps]
        wts = ML.drift_weights(ctl, Φs, R.active)
        e[:weights] = max(e[:weights], maximum(abs.(wts .- recs[c].wts)))
        sk̄ = [sum(wts[j] .* scs[j][1][f] for j in 1:Mr) for f in eachindex(fk)]
        knew, _ = ML.langevin_capacities(MockRNG(zs[c], Float64[]), P, units, Kraw, sk̄, fk; ηw=R.ηw, κs=ctl.κs, κmin=ctl.κmin,
                                         clip=ctl.clip, box=ctl.box)
        cpu = [ML.follow_capacities!(P, LM, copy(s), units, Kraw, knew; κmin=ctl.κmin)[1] for s in R.reps]
        devs = RD.sets[c].reps
        Sc, Sd = D.to_device(P, RD.M, RD.ix, cpu), D.to_device(P, RD.M, RD.ix, devs)
        e[:k] = max(e[:k], rel(Sd.kg, Sc.kg), rel(Sd.ks, Sc.ks), maximum(rel(a.k, b.k) for (a, b) in zip(devs, cpu)))
        e[:ω] = max(e[:ω], maximum(abs.(Sd.ω .- Sc.ω)), maximum(abs.(Sd.uc .- Sc.uc)))
        e[:e] = max(e[:e], maximum(abs.(Sd.e .- Sc.e)))
        e[:r] = max(e[:r], maximum(abs.(D.rebuild_r(RD.M, Sd) .- D.rebuild_r(RD.M, Sc))))
    end
    return e, gates_ok, recs
end
okstep(e) = e[:score] < 1e-9 && e[:weights] < 1e-14 && e[:k] < 1e-12 && e[:ω] < 1e-9 && e[:e] == 0 && e[:r] < 1e-6
@testset "one step on arrays = capacity_scores, update_gate!, drift_weights, langevin_capacities, follow_capacities!" begin
    # step 1 unchecked: the "high" starts put capacities exactly on their box bound, where the strict
    # inside test of the device (node values) and of the CPU (coefficients, one round trip) can
    # disagree by one ulp; the Langevin step moves them off the bound
    D.replica_device_step!(RD, 1)
    for it in 2:ctl.probe_every
        e, gok, recs = check_step(RD, it)
        @printf("  step %d: %s, gates %s, active %s\n", it, string(e), gok, string([count(R.active) for R in RD.sets]))
        @test okstep(e) && gok
        if it == ctl.probe_every                              # probe step: a probe ran, the uploaded steps are the sets'
            @test any(haskey(r, :probe) for r in recs)
            LPh = D.lgv_params(P, RD.M, RD.ix; κs=ctl.κs, ηw=[R.ηw for R in RD.sets])
            @test RD.dev[:LP].ηg == LPh.ηg && RD.dev[:LP].ηs == LPh.ηs && RD.dev[:LP].ηr == LPh.ηr
            println("  probe step: η_w per chain ", [round.(R.ηw; sigdigits=3) for R in RD.sets])
        end
    end
end

# ---------------------------------------------------------------- C. snapshot, stop, resume
@testset "snapshot, stop (exit 75) and resume in the CPU layout" begin
    cfg = testcfg("snap")
    cfg.checkpoint = joinpath(dirname(cfg.out), "snapshot.h5"); cfg.stop_file = joinpath(dirname(cfg.out), "STOP")
    rm(cfg.checkpoint; force=true); rm(cfg.stop_file; force=true)
    hook(phase, it) = it == 4 && touch(cfg.stop_file)
    r1 = D.run_replicas_device(cfg; families=FAMS, compile=false, hook=hook)
    @test r1.exitcode == ML.EXIT_CHECKPOINTED && !isfile(cfg.stop_file)
    sn = ML.read_replica_snapshot(cfg.checkpoint, cfg, r1.P)
    @test sn.it == 4 && length(sn.sets) == cfg.nchains && all(length(s.reps) == cfg.replicas for s in sn.sets)
    @test all(sn.sets[c].reps[j].m == r1.sets[c].reps[j].m && sn.sets[c].reps[j].k ≈ r1.sets[c].reps[j].k
              for c in 1:cfg.nchains, j in 1:cfg.replicas)
    r2 = D.run_replicas_device(cfg; families=FAMS, compile=false)
    @test r2.exitcode == 0 && r2.restarts == 1
    ML.HDF5.h5open(cfg.out) do f
        @test size(read(f["chains/1/Q"]), 2) == (cfg.mlda_warmup + cfg.nsamples) ÷ cfg.device_sweeps
        @test read(f["run/restarts"]) == 1 && read(f["run/resumed_at_sweep"]) == 4
    end
end
