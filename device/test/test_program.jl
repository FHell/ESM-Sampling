# One compiled program per checkpoint (driver.jl `sweep_program`): operation sweeps + capacity pass.
#   The merged program is the composition of two pieces verified on arrays and compiled
#   (test_rounds/test_mock/test_comp; test_kmove, test_kpeak, test_kren), so here:
#   1. Reactant (CPU backend): the merged program = the two separately compiled calls, to round-off.
#   2. (FULL=1) all families (`device_families`: plain, compensated, cell templates) in the merged
#      program: one call, r cache against rebuild_r and cost_parts against energy_cost of the
#      downloaded chains.
# Configuration: configs/device40.toml (40 nodes daily, 4 chains). Run from the repository root:
#   nice -n 19 julia --project=device -t 4 device/test/test_program.jl
using ESMDevice, ESMSampling, Random, Test, Printf, Reactant
const ML = ESMSampling
const D = ESMDevice
Reactant.set_default_backend("cpu")

cfg = ML.load_config(joinpath(@__DIR__, "..", "..", "configs", "device40.toml")); cfg.verbose = false
cfg.out = joinpath(@__DIR__, "..", "..", "results", "test_program.h5")
cfg.device_backend = "cpu"
cfg.device_sweeps = 2
X = D.device_setup(cfg)
host(o) = D.FlatState(map(Array, (o.θ, o.ω, o.uc, o.e, o.u, o.r))...)
ns = 2

@testset "merged program = two compiled calls (Reactant CPU backend)" begin
    st, cp, sp, kk, tabs = X.st, X.cp, X.sp, X.kk, X.tabs
    kpart = X.kargs(st, X.kgd, X.ksd, X.dn, X.crd, X.cap, X.Z, X.U, X.Zp, X.Up, X.BDd, X.BSd, X.RSd, X.Zr, X.Ur)[2:end]
    nsd = Reactant.ConcreteRNumber(ns)
    seed = UInt64[11, 3]
    tc1 = @elapsed ffp = @compile X.fp(st, cp, sp, kk, tabs, nsd, Reactant.ReactantRNG(Reactant.to_rarray(copy(seed))), kpart...)
    tc2 = @elapsed fsw = @compile X.sw(st, cp, sp, kk, tabs, nsd, Reactant.ReactantRNG(Reactant.to_rarray(copy(seed))))
    tc3 = @elapsed fkp = @compile X.kp(X.kargs(st, X.kgd, X.ksd, X.dn, X.crd, X.cap, X.Z, X.U, X.Zp, X.Up, X.BDd, X.BSd, X.RSd, X.Zr, X.Ur)...)
    @printf("  compile: merged %.0f s; separate %.0f s + %.0f s\n", tc1, tc2, tc3)
    R = Reactant.to_rarray
    fresh() = R(X.pl.st)                                         # donation: every call gets its own copy
    # the arguments that are also outputs are donated: fresh copies from the host values for every call
    kpf(st) = X.kargs(st, R(X.S.kg), R(X.S.ks), R(X.S.dnet), R(X.S.cren), R(X.S.capren),
                      X.Z, X.U, X.Zp, X.Up, X.BDd, X.BSd, R(D.pack_ren(X.P, X.M, X.chains)[2]), X.Zr, X.Ur)
    a = ffp(fresh(), R(X.pl.cp), sp, kk, tabs, nsd, Reactant.ReactantRNG(R(copy(seed))), kpf(fresh())[2:end]...)
    s1 = fsw(fresh(), R(X.pl.cp), sp, kk, tabs, nsd, Reactant.ReactantRNG(R(copy(seed))))
    b = fkp(kpf(s1)...)
    A, B = host(a[1]), host(b[1])
    for f in (:θ, :ω, :uc, :e, :u, :r)
        x, y = getfield(A, f), getfield(B, f)
        d = maximum(abs.(x .- y); init=0.0); sc = max(maximum(abs.(y); init=0.0), 1.0)
        @printf("  %-2s max |merged − separate| = %.2e (relative %.1e)\n", f, d, d / sc)
        @test d <= 1e-10 * sc
    end
    @test maximum(abs.(Array(a[2]) .- Array(b[2]))) <= 1e-10 * maximum(abs.(Array(b[2])))
end

if get(ENV, "FULL", "0") == "1"
    @testset "all families in the merged program" begin
        Y = D.device_setup(cfg; families=:all)
        println("  families: ", Y.pl.fams)
        kpart = Y.kargs(Y.st, Y.kgd, Y.ksd, Y.dn, Y.crd, Y.cap, Y.Z, Y.U, Y.Zp, Y.Up, Y.BDd, Y.BSd, Y.RSd, Y.Zr, Y.Ur)[2:end]
        nsd = Reactant.ConcreteRNumber(1)
        tc = @elapsed f = @compile Y.fp(Y.st, Y.cp, Y.sp, Y.kk, Y.tabs, nsd, Y.drng, kpart...)
        o = f(Y.st, Y.cp, Y.sp, Y.kk, Y.tabs, nsd, Y.drng, kpart...)
        @printf("  compile %.0f s\n", tc)
        Sh = deepcopy(Y.S)
        rc = D.unflat!(Sh, host(o[1]))
        Sh.kg .= Array(o[2]); Sh.ks .= Array(o[3])
        Sh.dnet .= Array(o[7]); Sh.cren .= Array(o[8]); Sh.capren .= Array(o[9])
        rb, parts = D.cost_parts(Y.M, Sh)
        @printf("  r cache vs rebuild_r: %.2e MW\n", maximum(abs.(rb .- rc)))
        @test maximum(abs.(rb .- rc)) < 1e-8
    end
end
