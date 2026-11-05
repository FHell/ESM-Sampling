# The replica driver of the device (device/src/replica_run.jl) on plain Julia arrays against the
# CPU replica sampler (src/replicas.jl; docs/system/testing.md, device order, step 1):
#   A. the weather paths of all replicas over a run equal those of `run_replicas` bitwise (the
#      weather never reads k or v; same streams), with the gate schedule (since, window); the
#      residual drift recorded at the host checkpoints;
#   B. per step after the operation sweeps: the downloaded scores against `capacity_scores`, the
#      gates (`update_gate!`) and drift weights, the Langevin step with the same normals against
#      `langevin_capacities`, the dispatch following against `follow_capacities!`; the device probe:
#      the candidate copied into the probe slot exactly, the slot frozen in (k, x) while the chain's
#      k moves, its scores per step against `capacity_scores` of its state, the probe's record
#      (gap, equilibrated and in-flight scores, costs) against `lag_probe`'s formulas on those
#      scores, the adaptation and the per-chain steps uploaded; the residual rebuild against
#      `rebuild_r`/`cost_parts`; the row-wise weather upload against the full one (bitwise);
#   C. snapshot, stop (exit 75) and resume in the CPU driver's layout;
#   D. the Float32 build uploads Float32 only (`ftconv`) and runs a step.
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
"Largest relative difference of two vectors with NaN entries (NaN only where both are NaN)."
relnan(a, b) = (isnan.(a) == isnan.(b) || return Inf; f = .!isnan.(b); rel(a[f], b[f]))

# ---------------------------------------------------------------- A. weather paths = run_replicas
@testset "weather paths and gate schedule = run_replicas" begin
    rc = ML.run_replicas(testcfg("cpu"))
    cd = testcfg("dev")
    rd = D.run_replicas_device(cd; families=FAMS, compile=false)
    @test rc.exitcode == 0 && rd.exitcode == 0
    for c in eachindex(rc.sets), j in eachindex(rc.sets[c].reps)
        a, b = rc.sets[c].reps[j], rd.sets[c].reps[j]
        @test a.m == b.m && a.z0 == b.z0 && a.ws == b.ws
    end
    @test all(rc.sets[c].since == rd.sets[c].since && rc.sets[c].window == rd.sets[c].window for c in eachindex(rc.sets))
    ML.HDF5.h5open(cd.out) do f                  # the residual drift at every host checkpoint (FP64: round-off)
        dr = read(f["replicas/1/residual_drift"])
        @printf("  residual drift per checkpoint (chain 1): max %.2e MW\n", maximum(dr))
        @test size(dr) == (cd.replicas, (cd.mlda_warmup + cd.nsamples) ÷ cd.device_sweeps) && all(isfinite, dr) && maximum(dr) < 1e-6
        @test read(f["replicas/probe_slots"]) == 1 && haskey(f, "device_profile/rebuild")
        nck = (cd.mlda_warmup + cd.nsamples) ÷ cd.device_sweeps      # guard counts per family, chain and host checkpoint
        @test read(f["device/guard_families"]) == ["gen0", "soc"]
        @test size(read(f["device/guard_root"])) == (2, length(rd.sets), nck) && size(read(f["device/guard_cap"])) == (2, length(rd.sets), nck)
        @test size(read(f["replicas/1/guard_root"])) == (2, nck) && all(isfinite, read(f["device/guard_root"]))
        @test read(f["device/guard_sites"]) == collect(values(D.moves_per_sweep(rd.RD.dev[:pl])))
    end
end

# ---------------------------------------------------------------- B. one step against the CPU functions
cfg = testcfg("step")
cfg.replica_probe_every = 3                  # replica 1 is active at steps 1–3, so step 3 has a probe candidate
cfg.mlda_warmup = 8                          # the probe of step 3 ends at step 3 + probe_len = 7, in the warmup: it adapts
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
"Capacities and weather of device row b (frozen in a probe slot)."
rowvals(RD, b) = (kg=RD.dev[:kg][b, :], ks=RD.dev[:ks][b, :], kr=RD.dev[:RS].kr[b, :], A=RD.dev[:RS].A[b, :, :], dn=RD.dev[:dn][b, :, :],
                  cr=RD.dev[:cr][b], cap=RD.dev[:cap][b], wsh=RD.wsh[b, :])
"Every per-row device array equal in rows p and b."
rows_equal(RD, p, b) = all(getfield(RD.dev[:st], f)[p, :] == getfield(RD.dev[:st], f)[b, :] for f in fieldnames(typeof(RD.dev[:st]))) &&
                       all(getfield(RD.dev[:cp], f)[p, :] == getfield(RD.dev[:cp], f)[b, :] for f in fieldnames(typeof(RD.dev[:cp]))) &&
                       rowvals(RD, p) == rowvals(RD, b)
"Host State of device row b after a download (its weather from `s`: the probe slot's is that of the copied replica)."
function row_state(RD, b, s)
    s = copy(s)
    D.ren_to_host!(P, RD.dev[:RS].kr, b, s)
    return D.to_host!(P, RD.M, RD.ix, RD.Sh, b, s)
end
# the probe of every chain as seen from the CPU: template (the copied replica), start values, in-flight window, slot scores
PRB = Dict{Int,Any}()
"One device step in its parts, checked against the CPU functions on the downloaded states."
function check_step(RD, it)
    moved, tw = D.replica_device_weather!(RD, it)
    acc = D.replica_device_operation!(RD)
    D.replica_download!(RD)
    pre = deepcopy(RD.sets)                                   # states after the operation, gates before the update
    zs = [cpu_normals(RD, c, pre[c].ηw) for c in 1:nch]
    psc = Dict{Int,Any}()                                     # the probe slots' scores (CPU) of this step
    for (c, pb) in PRB
        sp = row_state(RD, D.slotrow(RD, c, RD.Mp), pb.tmpl)
        sc = ML.capacity_scores(P, sp)
        psc[c] = (sk=[rr .* x for x in sc[1]], fk=sc[2], s=sp)
        push!(pb.cpu, psc[c].sk)
    end
    recs = D.replica_device_capacity!(RD, it, acc, moved, tw)
    D.replica_download!(RD)
    e = Dict(:score => 0.0, :weights => 0.0, :k => 0.0, :kc => 0.0, :ω => 0.0, :r => 0.0, :e => 0.0, :probe => 0.0)
    gates_ok, probe_ok = true, true
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
        e[:k] = max(e[:k], rel(Sd.kg, Sc.kg), rel(Sd.ks, Sc.ks))
        # the coefficient vectors relative to their scale (per entry, near-zero Haar coefficients carry the
        # cancellation of their node values: up to 2e-12 relative after a few steps)
        e[:kc] = max(e[:kc], maximum(maximum(abs.(a.k .- b.k)) / max(1.0, maximum(abs.(b.k))) for (a, b) in zip(devs, cpu)))
        e[:ω] = max(e[:ω], maximum(abs.(Sd.ω .- Sc.ω)), maximum(abs.(Sd.uc .- Sc.uc)))
        e[:e] = max(e[:e], maximum(abs.(Sd.e .- Sc.e)))
        e[:r] = max(e[:r], maximum(abs.(D.rebuild_r(RD.M, Sd) .- D.rebuild_r(RD.M, Sc))))
        # the device probe of chain c
        p = D.slotrow(RD, c, RD.Mp)
        if haskey(PRB, c)
            pb = PRB[c]
            restarted = RD.probes[c] !== nothing && RD.probes[c].it == it   # ended and the next one copied in this step
            restarted || (probe_ok &= rowvals(RD, p) == pb.v0)              # frozen in (k, x)
            pb.kmoved[] |= RD.dev[:kg][D.slotrow(RD, c, 1), :] != pb.v0.kg
            if haskey(recs[c], :probe)                              # ended: the record against lag_probe's formulas
                rec = recs[c].probe
                n, nt = D.probe_steps(ctl)
                probe_ok &= length(pb.cpu) == n
                eq = sum(pb.cpu[end-nt+1:end]) ./ nt
                mk = ML.interior_nodes(P, ML.knodes(P, pb.tmpl.ℓ, pb.tmpl.k))
                rs_eq, rs_in = ML.masked_means(eq, mk), ML.masked_means(sum(pb.hist) ./ length(pb.hist), mk)
                fkp = ML.masked_means(psc[c].fk, mk)
                e[:probe] = max(e[:probe], relnan(rec["rs_eq"], rs_eq), relnan(rec["rs_in"], rs_in), relnan(rec["fk"], fkp),
                                relnan(rec["gap"], (rs_in .- rs_eq) ./ fkp),
                                rel(rec["cost"], [pb.tmpl.parts.op, psc[c].s.parts.op, pb.tmpl.parts.shed, psc[c].s.parts.shed]))
                probe_ok &= rec["it"][1:2] == [pb.it, pb.j] && pb.kmoved[]
                println("  probe of chain $c ended at step $it: started $(pb.it), replica $(pb.j), adapted $(rec["it"][3]), ",
                        "gap ", round.(rec["gap"]; sigdigits=3))
                delete!(PRB, c)
            else                                                    # running: this step's scores against capacity_scores
                ps = RD.probes[c].sc[end]
                e[:probe] = max(e[:probe], maximum(rel(ps[f], psc[c].sk[f]) for f in eachindex(ps)))
            end
        end
        pb = RD.probes[c]
        if pb !== nothing && pb.it == it                            # started: the candidate copied into the slot
            b = D.slotrow(RD, c, pb.j)
            probe_ok &= rows_equal(RD, p, b)
            PRB[c] = (; tmpl=deepcopy(RD.sets[c].reps[pb.j]), v0=rowvals(RD, p), hist=deepcopy(pb.hist), it, j=pb.j,
                      cpu=Vector{Vector{Float64}}[], kmoved=Ref(false))
        end
    end
    return e, gates_ok, probe_ok, recs
end
okstep(e) = e[:score] < 1e-9 && e[:weights] < 1e-14 && e[:k] < 1e-12 && e[:kc] < 1e-12 && e[:ω] < 1e-9 && e[:e] == 0 && e[:r] < 1e-6 && e[:probe] < 1e-9
@testset "steps on arrays = capacity_scores, update_gate!, drift_weights, langevin_capacities, follow_capacities!; the device probe" begin
    # step 1 unchecked: the "high" starts put capacities exactly on their box bound, where the strict
    # inside test of the device (node values) and of the CPU (coefficients, one round trip) can
    # disagree by one ulp; the Langevin step moves them off the bound
    D.replica_device_step!(RD, 1)
    n, _ = D.probe_steps(ctl)
    nprobe = 0
    for it in 2:ctl.probe_every+n
        e, gok, pok, recs = check_step(RD, it)
        @printf("  step %d: %s, gates %s, probe %s, active %s\n", it, string(e), gok, pok, string([count(R.active) for R in RD.sets]))
        @test okstep(e) && gok && pok
        it == ctl.probe_every && @test !isempty(PRB)              # a probe started
        nprobe += count(r -> haskey(r, :probe), recs)
    end
    @test nprobe > 0                                              # a probe ended (and was checked)
    LPh = D.lgv_params(P, RD.M, RD.ix; κs=ctl.κs, ηw=[R.ηw for R in RD.sets])
    @test RD.dev[:LP].ηg == LPh.ηg && RD.dev[:LP].ηs == LPh.ηs && RD.dev[:LP].ηr == LPh.ηr
    @test any(R.ηw != ctl.ηw for R in RD.sets)                     # adapted (probe_avg = 1, in the warmup)
    println("  η_w per chain after the probes ", [round.(R.ηw; sigdigits=3) for R in RD.sets])
end

@testset "residual rebuild = rebuild_r and cost_parts of the downloaded state" begin
    D.replica_download!(RD)
    Sh = deepcopy(RD.Sh)
    Sh.dnet .= RD.dev[:dn]; Sh.cren .= RD.dev[:cr]; Sh.capren .= RD.dev[:cap]
    r0 = copy(RD.dev[:st].r)
    st2, dr, op, shed = RD.dev[:rb](RD.dev[:st], RD.dev[:kg], RD.dev[:ks], RD.dev[:dn], RD.dev[:cr], RD.dev[:cap])
    r, parts = D.cost_parts(RD.M, Sh)
    C = size(r0, 1)
    @test st2.r == reshape(r, C, :) && op == parts.op && shed == parts.shed
    @test dr == vec(maximum(abs.(r0 .- st2.r); dims=2)) && maximum(dr) < 1e-6
    @test st2.θ === RD.dev[:st].θ && RD.dev[:st].r == r0                      # the inputs unchanged
    @printf("  drift after %d steps: max %.2e MW\n", ctl.probe_every + D.probe_steps(ctl)[1], maximum(dr))
end

@testset "row-wise weather upload = the full upload (bitwise)" begin
    dev = RD.dev
    C = RD.nch * RD.Mp
    reps = [s for R in RD.sets for s in R.reps]
    for R in RD.sets                                           # new weather for replica 2 of every chain (host step, no compensation)
        ML.replica_weather!(R, 2, P, RD.LM, RD.F, RD.ctlw, 2)
    end
    rows = [D.slotrow(RD, c, 2) for c in 1:RD.nch]
    full = [s for R in RD.sets for s in vcat(R.reps, R.reps[1:1])]        # every device row (the probe slot: any weather, unchanged)
    D0, c0, A = D.weather_arrays(P, RD.M, full)
    chg = zeros(C); chg[rows] .= 1
    dims, comp = RD.dims, Val(ctl.comp)
    a = D.weather_update(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:KP], dev[:RP], D0, c0, A, chg, dev[:wsh], dims, comp)
    ss = [RD.sets[c].reps[2] for c in 1:RD.nch]
    D0m, c0m, Am = D.weather_arrays(P, RD.M, ss)
    pad(X) = (Y = zeros(RD.wb, size(X)[2:end]...); selectdim(Y, 1, 1:size(X, 1)) .= X; Y)
    pos = ones(Int, C); pos[rows] .= 1:length(rows)
    b = D.weather_rows(dev[:st], dev[:kg], dev[:ks], dev[:RS], dev[:dn], dev[:cr], dev[:KP], dev[:RP], pad(D0m), pad(c0m), pad(Am), pos, chg,
                       dev[:wsh], dims, comp)
    eqf(x, y) = x isa AbstractArray ? x == y : all(eqf(getfield(x, f), getfield(y, f)) for f in fieldnames(typeof(x)))
    @test all(eqf(x, y) for (x, y) in zip(a, b))
    @test RD.wb == RD.nch                                      # bursts staggered: one replica per chain and step
    @test D.load_shares(P, ss) == D.load_shares(P, full)[rows, :]
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
    @test r1.RD.fresh                                          # the snapshot was written from downloaded States
    r2 = D.run_replicas_device(cfg; families=FAMS, compile=false)
    @test r2.exitcode == 0 && r2.restarts == 1
    ML.HDF5.h5open(cfg.out) do f
        @test size(read(f["chains/1/Q"]), 2) == (cfg.mlda_warmup + cfg.nsamples) ÷ cfg.device_sweeps
        @test size(read(f["replicas/1/residual_drift"]), 2) == (cfg.mlda_warmup + cfg.nsamples) ÷ cfg.device_sweeps
        @test read(f["run/restarts"]) == 1 && read(f["run/resumed_at_sweep"]) == 4
    end
end

# ---------------------------------------------------------------- D. the Float32 build: Float32 uploads
"Element types of every floating-point array or number in x (recursively through structs, tuples and Dicts)."
function floattypes(x, acc=Set{Type}())
    if x isa AbstractArray{<:AbstractFloat} || x isa AbstractFloat
        push!(acc, eltype(x))
    elseif x isa AbstractArray{<:Number} || x isa Number
    elseif x isa AbstractDict || x isa Union{Tuple,NamedTuple}
        foreach(v -> floattypes(v, acc), values(x))
    elseif x isa AbstractArray
        foreach(v -> floattypes(v, acc), x)
    elseif isstructtype(typeof(x)) && !(x isa Function) && !(x isa AbstractRNG) && fieldcount(typeof(x)) > 0
        foreach(f -> floattypes(getfield(x, f), acc), fieldnames(typeof(x)))
    end
    return acc
end
@testset "Float32 build: every device input converted (ftconv)" begin
    F = D.F32
    cfg = testcfg("f32")
    R32 = F.replica_device_setup(cfg; families=FAMS, compile=false)
    dv = Dict(k => v for (k, v) in R32.dev if !(v isa Function) && k ∉ (:drng, :lrng))
    @test floattypes(dv) == Set([Float32])
    acc = F.replica_device_operation!(R32)                    # the step in its parts: the score sums stay Float32
    @test floattypes(acc) == Set([Float32])
    F.replica_device_capacity!(R32, 1, acc, [falses(R32.Mr) for _ in 1:R32.nch], 0.0)
    @test floattypes((R32.dev[:st], R32.dev[:kg], R32.dev[:RS], R32.dev[:cp], R32.dev[:dn], R32.dev[:cr], R32.dev[:cap])) == Set([Float32])
    @test all(isfinite, R32.dev[:kg]) && all(isfinite, R32.dev[:st].r)
    dr = F.replica_rebuild!(R32)
    @printf("  Float32 build: residual drift after one step %.2e MW (FP64 build ≈ 1e-11)\n", maximum(dr))
end
