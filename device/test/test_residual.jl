# The residual cache rebuild of the device drivers (device/src/residual.jl; docs/system/device.md
# §6.3; the replica driver's use of it: test_replica_driver.jl):
#   A. arrays: after a few operation sweeps on Julia arrays (the plan of `device_setup`, the
#      device40 configuration), `rebuild_flat` returns the residual and the costs of `cost_parts` on
#      the same node arrays (bitwise), the drift max |r_cache − r_rebuilt| (FP64: round-off), and
#      leaves its inputs unchanged; the same sweeps in the Float32 build, for its drift;
#   B. (ESM_DEVICE_COMPILE=1) the rebuild compiled against the arrays; a short compiled run of the
#      Metropolis driver, stopped by the stop file at the first checkpoint and resumed: the drift
#      per chain and checkpoint in `device/residual_drift`, continued across the resume, and the guard
#      counts per family, chain and checkpoint (`device/guard_root`, `guard_cap`) likewise.
#   ESM_DEVICE_COMPILE=1 nice -n 19 julia --project=device -t 8 device/test/test_residual.jl
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice

const ROOT = joinpath(@__DIR__, "..", "..")
const COMPILE = get(ENV, "ESM_DEVICE_COMPILE", "") == "1"
const OUTD = joinpath(ROOT, "results", "test_residual"); mkpath(OUTD)
function rcfg(name)
    cfg = ML.load_config(joinpath(ROOT, "configs", "device40.toml")); cfg.verbose = false
    cfg.data = joinpath(ROOT, "data", "processed", "model.h5")
    cfg.out = joinpath(OUTD, name, "out.h5"); mkpath(dirname(cfg.out))
    cfg.device_backend = "cpu"
    cfg.nchains, cfg.mlda_warmup, cfg.nsamples, cfg.device_sweeps = 2, 2, 2, 2
    cfg.checkpoint = ""; cfg.stop_file = ""
    return cfg
end
const NSW = 3

"Sweeps on Julia arrays from the plan of `device_setup` (module Dm), then the rebuild; returns (rebuild output, cost_parts of the swept state, swept FlatState, setup)."
function swept(Dm, cfg)
    X = Dm.device_setup(cfg)
    pl, S = X.pl, deepcopy(X.S)
    st = Dm.run_sweeps(pl.st, pl.cp, pl.sp, pl.k, pl.dev, NSW, MersenneTwister(5), Val(pl.fams))
    Dm.unflat!(S, st)
    out = Dm.rebuild_flat(st, S.kg, S.ks, S.dnet, S.cren, S.capren, X.M, Dm.rebuild_dims(X.K, X.M, X.ix))
    return out, Dm.cost_parts(X.M, S), st, X
end

# ---------------------------------------------------------------- A. arrays
cfg = rcfg("arrays")
(st2, dr, op, shed), (r, parts), st, X = swept(D, cfg)
@testset "rebuild_flat = cost_parts on the node arrays (arrays)" begin
    C = size(st.r, 1)
    @test st2.r == reshape(r, C, :) && op == parts.op && shed == parts.shed
    @test dr == vec(maximum(abs.(st.r .- st2.r); dims=2)) && maximum(dr) < 1e-6
    @test st2.θ === st.θ && st2.ω === st.ω                    # the node arrays passed through, unchanged
    @printf("  FP64: drift after %d sweeps (%d chains): max %.2e MW\n", NSW, C, maximum(dr))
end
@testset "Float32 build: the same sweeps drift (arrays)" begin
    (s32, dr32, _, _), (r32, _), _, _ = swept(D.F32, cfg)
    @test eltype(dr32) == Float32 && all(isfinite, dr32) && s32.r == reshape(r32, size(s32.r, 1), :)
    @printf("  Float32: drift after %d sweeps: max %.2e MW (FP64 %.2e)\n", NSW, maximum(dr32), maximum(dr))
end

# ---------------------------------------------------------------- B. compiled
if COMPILE
    Reactant.set_default_backend("cpu")
    @testset "rebuild compiled = arrays" begin
        R = Reactant.to_rarray
        S = deepcopy(X.S); D.unflat!(S, st)
        f, tc = D.rebuild_program(X.M, D.rebuild_dims(X.K, X.M, X.ix), R, (R(st), R(S.kg), R(S.ks), R(S.dnet), R(S.cren), R(S.capren));
                                  compile=true)
        o = f(R(st), R(S.kg), R(S.ks), R(S.dnet), R(S.cren), R(S.capren))
        d = max(maximum(abs.(Array(o[1].r) .- st2.r)), maximum(abs.(Array(o[2]) .- dr)), maximum(abs.(Array(o[3]) .- op) ./ op))
        @printf("  compile %.0f s; max |compiled − arrays| %.2e\n", tc, d)
        @test d < 1e-6
    end
    @testset "Metropolis driver compiled: drift recorded, continued across a stop and resume" begin
        cfg = rcfg("mh")
        cfg.checkpoint = joinpath(dirname(cfg.out), "snapshot.h5"); cfg.stop_file = joinpath(dirname(cfg.out), "STOP")
        rm(cfg.checkpoint; force=true)
        touch(cfg.stop_file)                                  # stops at the first host checkpoint
        r1 = D.run_device(cfg)
        @test r1.exitcode == ML.EXIT_CHECKPOINTED
        sn = D.read_device_snapshot(cfg.checkpoint, cfg, r1.P)
        @test size(sn.extra["residual_drift"]) == (cfg.nchains, 1)
        @test size(sn.extra["guard_root"], 2) == cfg.nchains && size(sn.extra["guard_root"], 3) == 1 && size(sn.extra["guard_cap"]) == size(sn.extra["guard_root"])
        r2 = D.run_device(cfg)
        @test r2.exitcode == 0 && r2.restarts == 1
        ML.HDF5.h5open(cfg.out) do f
            dr = read(f["device/residual_drift"])
            @printf("  drift per chain and checkpoint: %s MW\n", string(round.(dr; sigdigits=2)))
            @test size(dr) == (cfg.nchains, cfg.mlda_warmup + cfg.nsamples) && dr[:, 1] == sn.extra["residual_drift"][:, 1]
            @test all(isfinite, dr) && maximum(dr) < 1e-6
            @test haskey(f, "device_profile/rebuild") && haskey(f, "device_profile/compile_rebuild")
            gf, gr, gc = read(f["device/guard_families"]), read(f["device/guard_root"]), read(f["device/guard_cap"])
            nck = cfg.mlda_warmup + cfg.nsamples
            @test size(gr) == (length(gf), cfg.nchains, nck) && size(gc) == size(gr)
            @test gr[:, :, 1] == sn.extra["guard_root"][:, :, 1] && gc[:, :, 1] == sn.extra["guard_cap"][:, :, 1]
            @test read(f["device/guard_row_sweeps"]) == nck * cfg.nchains * cfg.device_sweeps
            println("  guards (root, cap) per family: ", join(("$x $(Int(sum(gr[i, :, :])))/$(Int(sum(gc[i, :, :])))" for (i, x) in enumerate(gf)), ", "))
        end
    end
else
    println("ESM_DEVICE_COMPILE is not 1: the compiled half is skipped")
end
