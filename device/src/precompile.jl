# Precompile workload (PrecompileTools): traces and compiles, on Reactant's CPU backend, the programs
# every device run uses, on a tiny synthetic model (3 nodes, 8 steps; no data file), so that the Julia
# code of the tracing is cached in the package image. XLA's own compile of a run's programs stays.
#
# Off by default (the device code changes often; every edit would pay the workload). On with the
# standard PrecompileTools preference in device/LocalPreferences.toml:
#   [ESMDevice]
#   precompile_workload = true
# (hpc/prepare_env.sh writes it for the cluster). Nothing is kept: no globals, no compiled thunks.

using Preferences: @load_preference
using PrecompileTools: @setup_workload, @compile_workload

"A tiny synthetic device model and state (C chains, T steps): 3 nodes on a triangle, two units per
node without fleet commitment, one storage unit per node, one link; no line losses and a lossless
link (the share-preserving families need them), soft limits and merged vintages."
function toy_device(; C=2, T=8)
    N, nl = 3, 3
    b0, b1 = [1, 2, 1], [2, 3, 3]
    Dl, Al = zeros(N, nl), zeros(N, nl)
    for e in 1:nl
        Dl[b0[e], e] += 1; Dl[b1[e], e] -= 1; Al[b0[e], e] += 1; Al[b1[e], e] += 1
    end
    oh(rows) = (Mx = zeros(length(rows), N); for (i, n) in enumerate(rows); Mx[i, n] = 1.0; end; Mx)
    gn, sn = [1, 2, 3, 1, 2, 3], [1, 2, 3]
    Ug, Us = length(gn), length(sn)
    ψ = hinge_from([0.0, 0.9, 1.0], [0.0, 0.0, 0.1], 10.0)
    Lq = hinge_from([0.0, 0.5, 1.0], [0.0, 0.25, 1.0], 2.0)
    f(x, n) = fill(x, n)
    M = DeviceModel(map(ftfield, (N, T, 6.0, 3000.0, 5.0, 1e4, 1.0, 0.1, Dl, Al, f(50.0, nl), f(1000.0, nl), f(0.0, nl), ψ, Lq,
        oh([1]), oh([2]), [500.0], [1.0],
        oh(gn), [20.0, 30.0, 40.0, 80.0, 90.0, 70.0], [25.0, 30.0, 35.0, 80.0, 90.0, 70.0], [100.0, 0.0, 50.0, 0.0, 0.0, 0.0],
        f(1.0, Ug), [0.3, 0.3, 0.3, 0.5, 0.5, 0.5], [0.3, 0.3, 0.3, 0.5, 0.5, 0.5], f(0.0, Ug), f(0.3, Ug), f(10.0, Ug), f(5.0, Ug),
        f(0.0, Ug), fill(true, Ug), f(0.0, Ug), f(5000.0, Ug), f(1e5, Ug),
        oh(sn), f(6.0, Us), f(0.9, Us), f(0.9, Us), f(0.0, Us), f(0.1, Us), f(0.0, Us), fill(true, Us), f(0.0, Us), f(2000.0, Us),
        f(5e4, Us), false))...)
    rng = MersenneTwister(1)
    S = DeviceState(map(ftfield, (0.01 .* randn(rng, C, T, N), rand(rng, C, T, Ug), zeros(C, T, Ug), rand(rng, C, T, Us),
                    0.2 .* randn(rng, C, T, 1), fill(800.0, C, Ug), fill(300.0, C, Us), 1000.0 .+ 200.0 .* randn(rng, C, T, N),
                    zeros(C), zeros(C)))...)
    # one partition level: one cell of the three nodes with both generator groups (units 1–3, 4–6)
    ci = CellInfo([[[1, 2, 3]]], [[[1, 2]]], Dict((g, n) => (g - 1) * 3 + n for g in 1:2 for n in 1:3), [1], [false, false])
    return M, S, ci
end

"Trace and compile (CPU backend) the operation sweeps of `fams` and the capacity pass of the toy model."
function precompile_programs(; fams=(FAMILIES_FAST..., :spair0, :tsoc, :tpot, :sblk2, :cell1, :cellpair1))
    M, S, ci = toy_device()
    C, T, N = size(S.θ, 1), M.T, M.N
    cls = zeros(C, T * N)
    pl = plan(M, S, 4e-5; families=fams, cls=cls, cells=ci)
    R(x) = Reactant.to_rarray(ftconv(x))
    st, cp, sp, k, tabs = R(pl.st), R(pl.cp), R(pl.sp), R(pl.k), R(pl.dev)
    rng = Reactant.ReactantRNG(R(UInt64[1, 2]))
    ns = Reactant.ConcreteRNumber(1)
    fv = Val(pl.fams)
    sw(st, cp, sp, k, tabs, ns, rng) = run_sweeps(st, cp, sp, k, tabs, ns, rng, fv)
    Reactant.@compile sw(st, cp, sp, k, tabs, ns, rng)
    # capacity pass (no bias, no renewable units)
    Ug, Us = size(S.kg, 2), size(S.ks, 2)
    units = kmove_units(M)
    gq = [q for (kk, q) in units if kk == 1]; sq = [q for (kk, q) in units if kk == 2]
    RP = RenParams(M.Δt, M.γk, M.c_shed, M.c_dump, zeros(1, N), zeros(1), zeros(1), zeros(1), zeros(1))
    RS = RenState(zeros(C, 1), zeros(C, T, 1))
    dims = Val((C, T, N, Ug, Us, 1, 1))
    kp(args...) = capacity_pass(args..., dims; σ=ftconv(0.1), βk=ftconv(4e-8), βv=ftconv(4e-5), physical=true)
    np_ = 1
    a = (st, R(S.kg), R(S.ks), R(S.dnet), R(S.cren), R(S.capren), R(KParams(M)), R(randn(C, length(units), np_)),
         R(rand(C, length(units), np_)), R(randn(C, length(gq), np_)), R(rand(C, length(gq), np_)), R(gq), R(sq),
         R(fill(100.0, length(gq))), R(fill(1.0 / N, C, N)), nothing, nothing, nothing, nothing, R(RP), R(RS),
         R(randn(C, 1, np_)), R(rand(C, 1, np_)), R(Int[]), nothing)
    Reactant.@compile kp(a...)
    return nothing
end

"The module's mutable globals that `plan` sets (restored after the workload: nothing leaks into the image)."
const PLAN_GLOBALS = (JUMP_FAMILIES, JUMP_CAP, UC_COLS, PROFILES, PROFILE_SEASON)
const PLAN_REFS = (CELLINFO, CELL_BUCKETED, SCHED_FAMILIES, TDIR_TOPO, TDIR_SOURCE, TDIR_CHAINS, DENSE_BUDGET)

function with_plan_globals(f)
    saved = map(deepcopy, PLAN_GLOBALS)
    savedr = map(r -> r[], PLAN_REFS)
    try
        f()
    finally
        for (g, v) in zip(PLAN_GLOBALS, saved)
            empty!(g); g isa AbstractSet ? union!(g, v) : merge!(g, v)
        end
        for (r, v) in zip(PLAN_REFS, savedr)
            r[] = v
        end
    end
end

if @load_preference("precompile_workload", false)
    @setup_workload begin
        @compile_workload begin
            Reactant.set_default_backend("cpu")
            with_plan_globals(precompile_programs)
        end
    end
end
