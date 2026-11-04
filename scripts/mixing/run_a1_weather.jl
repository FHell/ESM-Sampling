# Weather and capacity diagnostics at α = 1 (fixed level-0 bias, utilization measure) from the
# end states of scripts/mixing/run_a1.jl: per weather move family the acceptance, |t|, CPU time
# per move and the squared change of the annual means relative to their prior variance; regime
# moves; and the force decomposition of the capacities including the bias. Usage:
#   nice -n 19 julia --project=. -t 4 scripts/mixing/run_a1_weather.jl
include(joinpath(@__DIR__, "driver.jl"))
include(joinpath(@__DIR__, "kop.jl"))
include(joinpath(@__DIR__, "wblock.jl"))
include(joinpath(@__DIR__, "forces.jl"))
include(joinpath(@__DIR__, "prior_spectral.jl"))
using Serialization

P = deserialize("results/mixing/a1u_problem.jls")
chains = deserialize(length(ARGS) >= 1 ? ARGS[1] : "results/mixing/a1w_chains.jls")
W = P.W
L = ML.nlevels(P) - 1
LM = ML.LocalModel(P, L)
Vd = deserialize("results/mixing/z0dirs.jls")
n0_, T0_ = ML.n0(W), W.lays[1].T
fr = [z0row(W, f, 1) for f in eachindex(W.specs)]
# prior variance of the annual means (average over prior regime draws)
Aavg = spzeros(n0_ * T0_, n0_)
for o in 1:n0_, τ in 1:T0_
    Aavg[(τ-1)*n0_+o, o] = 1 / T0_
end
Cacc = zeros(n0_, n0_)
for sd in 1:3
    mq, _, _ = ML.sample_weather(MersenneTwister(100 + sd), W, 0)
    Fq = cholesky(Symmetric(var_precision(W, mq)))
    Cacc .+= Matrix(Aavg' * (Fq \ Matrix(Aavg))) ./ 3
end
sdann = sqrt.(diag(Cacc))

function famstats(rng, P, LM, s, fam, nmv)
    a = time()
    s1, ts = if fam[1] == :single
        z0_block_sweep!(rng, P, LM, s; B=1, nmoves=nmv, rows=[(f, j) for f in eachindex(W.specs) for j in 1:W.lays[1].m[f]])
    elseif fam[1] == :row
        z0_block_sweep!(rng, P, LM, s; B=fam[2], nmoves=nmv)
    else
        z0_block_sweep!(rng, P, LM, s; B=fam[2], nmoves=nmv, V=Vd)
    end
    el = time() - a
    dzm = z0_rowmeans(P, s1.z0) .- z0_rowmeans(P, s.z0)
    nz = ts[ts.!=0.0]
    rel = (dzm[fr] ./ sdann[fr]) .^ 2 ./ nmv
    @printf("  %-14s acc %.2f  median|t| %.2e  %.1f ms/move  Δ(annual mean)²/var per move (load,on,off,solar): %s\n", string(fam[1:min(end, 2)]),
            length(nz) / nmv, isempty(nz) ? 0.0 : median(abs.(nz)), 1000el / nmv, join([@sprintf("%.1e", x) for x in rel], " "))
    flush(stdout)
end

for c in (1, 3)
    s = chains[c]
    ML.evaluate!(P, s)
    println("chain $c: z0 annual means (load, onwind, offwind, solar) = ", round.(z0_rowmeans(P, s.z0)[fr], digits=3),
            "  in prior sd: ", round.(z0_rowmeans(P, s.z0)[fr] ./ sdann[fr], digits=2))
    rng = MersenneTwister(50 + c)
    for (fam, nmv) in (((:single,), 60), ((:row, 16), 20), ((:vblock, 16), 20), ((:vblock, 64), 10), ((:vblock, 365), 4))
        famstats(rng, P, LM, s, fam, nmv)
    end
    for B in (1, 4)
        a = time()
        _, st = regime_block_sweep!(rng, P, LM, s; B=B, nmoves=100)
        @printf("  regime block B=%d: proposals changing m %d/100, accepted %d, median |βΔC| %.1f nats  [%.2fs]\n", B, st[1], st[2], st[3], time() - a)
    end
    # capacity forces with the bias, averaged over operation states
    K = ML.new_sampler(P, s, ML.Config()).loc
    states = [deepcopy(s)]
    x = deepcopy(s)
    for _ in 1:6
        for _ in 1:2
            x = ML.local_sweep!(rng, P, LM, x; wsets=K.wsets, extra=K.extra)[1]
        end
        push!(states, deepcopy(x))
    end
    Fs = [unit_forces(P, LM, y) for y in states]
    for cr in ["onwind", "solar", "offwind", "gas", "battery", "H2"]
        ks = [sum(f.total_ks for f in F if f.carrier == cr) for F in Fs]
        kp = [sum(f.total_kop for f in F if f.carrier == cr) for F in Fs]
        db = [sum(f.dbias for f in F if f.carrier == cr) for F in Fs]
        @printf("  %-8s carrier force: kscale %9.1f ± %7.1f | kop %9.1f ± %7.1f | bias part %9.1f\n", cr, mean(ks), std(ks) / sqrt(length(ks)),
                mean(kp), std(kp) / sqrt(length(kp)), mean(db))
    end
    flush(stdout)
end
