# α = 1 smoke test of `kop` (docs/system/moves/capacity.md): the chains40 configuration at
# α = 1 (level-0 warmup shortened to `warmup0`, bias lifted to the fine level and frozen),
# from the heuristic start: one standalone kop pass with the split of the log-target change,
# then `nsw` fine steps with and without kop from the same start, b̂ and the capacities per
# sweep. Usage:
#   nice -n 19 julia --project=. -t 4 scripts/mixing/run_kop_a1smoke.jl [nsw] [warmup0]
include(joinpath(@__DIR__, "common.jl"))
nsw = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
w0 = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300
cfg, P, sL, info = mix_setup("configs/chains40.toml"; use_bias=true, warmup0=w0, kop=true)
L = ML.nlevels(P) - 1
LM = ML.LocalModel(P, L)
E = P.E
carriers = [String(sp.name)[3:end] for sp in E.kspecs]
tot(s) = [sum(k) / 1e3 for k in ML.knodes(P, s.ℓ, s.k)]
parts(s) = (b=ML.bias_term(P, s), capex=s.parts.capex, op=s.parts.op, lm=ML.log_measure(E, P.lk[L+1], P.lv[L+1], s.k), L=s.L)
"The three parts of b̂ = ĝ − r Ē ĝ + (1 − r) ĥ of a state (from a fresh bias cache)."
function bparts(s)
    BC = ML.BiasCache(P, s)
    return (g=BC.g, rgm=P.bias.ratio * BC.gm, h=(1 - P.bias.ratio) * BC.h, firm=BC.firm)
end
showb(label, s) = (q = bparts(s); @printf("   %s: ĝ %.5g  −rĒĝ %.5g  (1−r)ĥ %.5g  firm %.4g GW\n", label, q.g, -q.rgm, q.h, q.firm / 1e3); flush(stdout))
function report(label, s0, s1)
    a, b = parts(s0), parts(s1)
    @printf("%s: ΔL %+.4g = −Δb̂ %+.4g − β_kΔK %+.4g − β_vΔC %+.4g + Δlog J %+.4g\n", label, b.L - a.L, -(b.b - a.b),
            -P.βk * (b.capex - a.capex), -P.βv * (b.op - a.op), b.lm - a.lm)
    println("   totals GW (", join(carriers, ", "), "): ", round.(tot(s0), digits=1), " → ", round.(tot(s1), digits=1))
    flush(stdout)
end
s0 = ML.evaluate!(P, copy(sL))
s1, nm = ML.kop_sweep!(MersenneTwister(11), P, LM, copy(s0))
report("one kop pass from the heuristic start ($nm moves)", s0, s1)
showb("start", s0); showb("after the kop pass", s1)
for kop in (true, false)
    cfg.kop = kop
    S = ML.new_sampler(P, s0, cfg)
    rng = MersenneTwister(12)
    x = copy(s0)
    println("fine steps with kop = $kop:")
    for it in 1:nsw
        a = time()
        y = ML.fine_local_step(rng, P, x, S)
        el = time() - a
        p0, p1 = parts(x), parts(y)
        @printf("  sweep %2d  %.1fs  b̂ %.5g (Δ %+.3g)  L %.6g  β_kK %.4g  β_vC %.4g  totals %s\n", it, el, p1.b, p1.b - p0.b, p1.L,
                P.βk * p1.capex, P.βv * p1.op, join((@sprintf("%.1f", t) for t in tot(y)), " "))
        flush(stdout)
        x = y
    end
    showb("after $nsw fine steps (kop = $kop)", x)
end
println("done")
