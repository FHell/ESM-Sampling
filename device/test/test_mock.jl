# Phase-2 verification against the CPU reference with fixed mock uniforms (no statistics).
# 40 nodes × 6 h, all model terms on (setup as test_cost.jl), C = 2 chains.
#   1. The same directions on both sides: every site of two rounds of every plain family (explicit
#      unit/node and step, CPU move built with prep_node!/finalize!/link_soc! as in local_sweep!).
#   2. The same uniforms: (u1, u2) per site and chain from a fixed Weyl sequence; on the CPU a
#      MockRNG (rand returns the next given value) passed to ESMSampling.sample_line!; on the device
#      the same values as an input array (cpumap: the CPU's u → t map of the in-segment offset).
#   3. t per draw compared (device on its own points, and the device sampler on the CPU walk's
#      points); the device t committed on both sides (CPU commit_block!, device scatter-add) round
#      after round; then node values, r and the cost compared (device cost_parts, CPU energy_cost of
#      the CPU state and of the downloaded device state via to_host!).
#   4. The same rounds traced by Reactant (CPU backend) with the same given uniforms: compared with
#      the Julia-array run.
# Run: cd device && nice -n 19 julia --project=. -t 4 test/test_mock.jl   (≈ 3–5 min, mostly the compile)
include("setup40.jl")
include("helpers.jl")
using Test, Random
using Reactant: to_rarray
# work rules: traced runs are checked on Reactant's CPU backend (no CUDA stage, no 4 GB limit);
# DEVICE_BACKEND=gpu runs them on the GPU
Reactant.set_default_backend(get(ENV, "DEVICE_BACKEND", "cpu"))

# MockRNG: helpers.jl

const NSITE = parse(Int, get(ENV, "NSITE", "1500"))       # sites per round (a prefix of the schedule round)
const DEV_RE = get(ENV, "DEVICE_REACTANT", "1") == "1"
C = 2
P, M, ix, S, ss = setup40(; C=C, uc=get(ENV, "UC", "") == "none" ? String[] : ["gas", "nuclear"])   # UC=none: no fleet commitment
E = P.E
β = P.βv
FAMSET = get(ENV, "FAMSET", "plain") == "fast" ? D.FAMILIES_FAST : D.FAMILIES     # FAMSET=fast: :gen0/:pair0 split
pl = D.plan(M, S, β; bmax=Dict(:theta => 4096), cpumap=true, families=FAMSET)
LM = ML.LocalModel(P, ss[1].ℓ)
LCs = [ML.LocalCost(P, LM, s) for s in ss]
st0 = cpu_r_state(pl, LCs)                                  # the CPU residual on the device side too
fams = pl.fams
rounds = NamedTuple{fams}(Tuple([r[1:min(end, NSITE)] for r in pl.sched[f][1:min(2, end)]] for f in fams))
tabs = NamedTuple{fams}(Tuple(D.device_table(pl, f, D.host_table(f, pl.tp, rounds[f])) for f in fams))
weyl(n, off) = [mod(0.5 + (off + k) * 0.6180339887498949, 1.0) for k in 1:n]
off = Ref(0)
U = NamedTuple{fams}(Tuple(begin
    B, R = size(tabs[f].on)
    x = reshape(weyl(C * B * 3 * R, off[]), C, B, 3, R); off[] += length(x); x
end for f in fams))
nsites = sum(length(r) for f in fams for r in rounds[f])
println("families ", fams, ": ", sum(length(rounds[f]) for f in fams), " rounds, ", nsites, " sites × ", C, " chains")

# ---------------------------------------------------------------------------
# rounds: device (Julia arrays) and CPU side by side
# ---------------------------------------------------------------------------
# Per round: (a) the device draws t with its own points (all kinks, rays to ±∞); (b) the CPU draws
# with sample_line! and the same (u1, u2); (c) the device sampler is run on the CPU walk's points
# (`sample_points`, the device potential at the CPU's points). Both sides then commit the device's
# t, so the states stay comparable round after round.
# The CPU walk inserts points that are not kinks (beyond the last breakpoint it steps by |t|,
# doubling while the slope is ≤ 0, then by 2·40 nats/slope) and stops 40 nats above the running
# minimum; the dense form has every kink and exact rays. Same distribution, but a different
# (u1, u2) → t map where the CPU split a ray: (a) can differ there, (c) must agree everywhere.
padpts(v, K) = vcat(v, zeros(K - length(v)))
st = st0
scpu = [copy(s) for s in ss]
bufs = [Float64[] for _ in 1:C]
stats = Dict(:n => 0, :own_ok => 0, :own_bad => 0, :cpupts_bad => 0, :both0 => 0, :improper_mismatch => 0, :cpu_cancel => 0, :cancel_bad => 0)
worst_own = 0.0; worst_cpu = 0.0
tcpu_time = 0.0
for f in fams, r in 1:size(tabs[f].on, 2)
    global st, worst_own, worst_cpu, tcpu_time
    tb, _ = D.build(Val(f), st, pl.cp, pl.sp, tabs[f], r, pl.k)
    st2, t = D.do_round(Val(f), st, pl.cp, pl.sp, pl.k, tabs[f], r, U[f][:, :, :, r])
    td = t[:, :, 1]
    sites = rounds[f][r]
    B = size(tabs[f].on, 1)
    tc = zeros(C, B)
    pts = [Float64[0.0] for _ in 1:C, _ in 1:B]
    walked = falses(C, B)
    tcpu_time += @elapsed for c in 1:C
        LC = LCs[c]
        for (b, site) in enumerate(sites)
            bm = cpu_move(P, LM, LC, ix, f, site)
            bm === nothing && continue                    # κ_min mask of the pair family: t = 0
            tc[c, b] = ML.sample_line!(MockRNG([U[f][c, b, 1, r], U[f][c, b, 2, r]]), bm, β, bufs[c])
            pts[c, b] = isempty(bm.lb.pts) ? Float64[0.0] : copy(bm.lb.pts)   # empty: the CPU walk failed (improper)
            walked[c, b] = !isempty(bm.lb.pts)
            ML.commit_block!(bm, scpu[c].v, td[c, b])     # commit the device's t on the CPU
        end
    end
    K = maximum(length.(pts))
    Pc = zeros(C, B, K)
    for c in 1:C, b in 1:B; Pc[c, b, :] .= sort(padpts(pts[c, b], K)); end
    tcp = D.sample_points(tb, pl.k, Pc, U[f][:, :, 1:1, r], U[f][:, :, 2:2, r])[:, :, 1]
    for c in 1:C, (b, site) in enumerate(sites)
        stats[:n] += 1
        if tc[c, b] == 0.0 && !walked[c, b]                   # CPU improper (walk failed)
            td[c, b] == 0.0 ? (stats[:both0] += 1) : (stats[:improper_mismatch] += 1)
            continue
        end
        d1 = abs(td[c, b] - tc[c, b]) / max(1.0, abs(tc[c, b]))
        d2 = abs(tcp[c, b] - tc[c, b]) / max(1.0, abs(tc[c, b]))
        if d2 > 1e-9 && maximum(abs.(pts[c, b])) > 1e12
            # CPU ray overshoot: after a segment that is flat to round-off (slope ≈ 1e-16 > 0) the CPU
            # walk steps by 2·40/slope ≈ 1e15–1e18; on that segment its log-mass −c_a + logseg
            # (c_a ≈ 1e20) and its offset a + (L − s′) lose all precision (CPU t = 0.0 or exactly the
            # breakpoint). Not comparable; the device has no such step (exact rays from the last kink,
            # masses and offsets anchored at the lower end).
            stats[:cpu_cancel] += 1
            stats[:cpu_cancel] <= 3 && @printf("  CPU ray overshoot: %s round %d site %s chain %d: CPU t %.6g, device %.6g (CPU walk point at %.1e)\n",
                                               f, r, string(site), c, tc[c, b], td[c, b], maximum(abs.(pts[c, b])))
            continue
        end
        worst_cpu = max(worst_cpu, d2)
        d1 <= 1e-9 ? (stats[:own_ok] += 1; worst_own = max(worst_own, d1)) : (stats[:own_bad] += 1)
        if d2 > 1e-9
            stats[:cpupts_bad] += 1
            stats[:cpupts_bad] <= 5 && println("  CPU-points draw differs: $f round $r site $site chain $c: device $(tcp[c, b]) CPU $(tc[c, b])")
        end
    end
    st = st2
end
stJ = st
for c in 1:C; ML.evaluate!(P, scpu[c]); end
@printf("CPU: %.1f s\n", tcpu_time)

"CPU caches as a loop state (node values and r, device layout)."
function cpu_flat(LCs)
    T = M.T
    fl(Y) = reshape(permutedims(Y), 1, :)
    θ = vcat([fl(LC.Y[E.fθ]) for LC in LCs]...)
    ω = vcat([reshape(reduce(hcat, [LC.Y[E.gens[g].vf][i, :] for (g, i) in ix.gidx]), 1, :) for LC in LCs]...)
    uc = vcat([reshape(reduce(hcat, [E.gens[g].ucf > 0 ? LC.Y[E.gens[g].ucf][i, :] : zeros(T) for (g, i) in ix.gidx]), 1, :) for LC in LCs]...)
    e = vcat([reshape(reduce(hcat, [LC.Y[E.stors[a].fe][i, :] for (a, i) in ix.sidx]), 1, :) for LC in LCs]...)
    u = vcat([fl(LC.Y[E.flink]) for LC in LCs]...)
    r = vcat([fl(LC.r) for LC in LCs]...)
    return D.FlatState(θ, ω, uc, e, u, r)
end

@testset "mock uniforms: device = CPU" begin
    # t per draw
    n = stats[:n]
    @printf("  %d draws: CPU improper (walk failed, t = 0) %d, device t = 0 there too: %d (mismatch %d)\n",
            n, stats[:both0] + stats[:improper_mismatch], stats[:both0], stats[:improper_mismatch])
    @printf("  (a) device on its own points vs CPU: %d agree to round-off (max %.1e relative), %d differ (CPU walk split a ray)\n",
            stats[:own_ok], worst_own, stats[:own_bad])
    @printf("  (c) device sampler on the CPU walk points vs CPU: max |Δt|/max(1, |t|) = %.1e, %d beyond 1e-9\n", worst_cpu, stats[:cpupts_bad])
    @printf("  CPU ray overshoot (CPU walk point beyond 1e12 and the CPU-points draw differs: CPU precision loss; excluded): %d draws\n", stats[:cpu_cancel])
    @test stats[:improper_mismatch] == 0
    @test stats[:cpupts_bad] == 0
    @test stats[:cpu_cancel] <= 1e-3 * n
    # states after all rounds
    stC = cpu_flat(LCs)
    for fld in (:θ, :ω, :uc, :e, :u, :r)
        a, b = getfield(stJ, fld), getfield(stC, fld)
        d = maximum(abs.(a .- b)); sc = max(maximum(abs.(b)), 1.0)
        @printf("  %-2s max |device − CPU| = %.2e (max |x| %.2e, relative %.1e; bitwise equal entries %.1f %%)\n",
                fld, d, sc, d / sc, 100 * count(a .== b) / length(a))
        @test d <= 1e-10 * sc
    end
    # cost: device cost_parts vs CPU energy_cost (CPU state; downloaded device state)
    S1 = deepcopy(S); rc = D.unflat!(S1, stJ)
    rb, parts = D.cost_parts(M, S1)
    @printf("  r cache vs rebuild_r: %.2e MW\n", maximum(abs.(rb .- rc)))
    @test maximum(abs.(rb .- rc)) < 1e-8
    for c in 1:C
        sd = D.to_host!(P, M, ix, S1, c, copy(ss[c]))
        r1, r2 = abs(parts.op[c] / scpu[c].parts.op - 1), abs(sd.parts.op / scpu[c].parts.op - 1)
        @printf("  chain %d: op device cost_parts %.10e, CPU energy_cost %.10e (rel %.1e); energy_cost of the downloaded device state rel %.1e; max |Δv| %.1e\n",
                c, parts.op[c], scpu[c].parts.op, r1, r2, maximum(abs.(sd.v .- scpu[c].v)))
        @test r1 < 1e-11
        @test r2 < 1e-11
    end
end

if DEV_RE
    @testset "mock uniforms: Reactant = Julia arrays" begin
        fv = Val(fams)
        runU(st, cp, sp, k, tabs, U) = D.run_rounds(st, cp, sp, k, tabs, 1, nothing, U, fv)
        args = (to_rarray(st0), to_rarray(pl.cp), to_rarray(pl.sp), to_rarray(pl.k), to_rarray(tabs), to_rarray(U))
        tc = @elapsed f = @compile runU(args...)
        o = f(args...)
        oR = D.FlatState(map(Array, (o.θ, o.ω, o.uc, o.e, o.u, o.r))...)
        @printf("  compile %.0f s\n", tc)
        for fld in (:θ, :ω, :uc, :e, :u, :r)
            a, b = getfield(oR, fld), getfield(stJ, fld)
            d = maximum(abs.(a .- b)); sc = max(maximum(abs.(b)), 1.0)
            @printf("  %-2s max |reactant − julia| = %.2e (relative %.1e; bitwise equal entries %.1f %%)\n", fld, d, d / sc, 100 * count(a .== b) / length(a))
            @test d <= 1e-10 * sc
        end
    end
end
