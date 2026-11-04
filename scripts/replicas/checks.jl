# Record of the weather-process checks (ledger E22) and the dispatch-lag scan (E23), run
# interactively in one session with these calls. From the repository root:
#   nice -n 19 julia --project=. -t 4 scripts/replicas/checks.jl
include(joinpath(@__DIR__, "..", "langevin", "setup40.jl"))
include(joinpath(@__DIR__, "replicas.jl"))
mkpath("results/replicas")
W = P.W; F = VarFactors(W)
T0, n = W.lays[1].T, ML.n0(W)

# ---- E22 deterministic checks ----
rngA, rngB = MersenneTwister(5), MersenneTwister(5)
m, z0, ws = ML.sample_weather(rngA, W, 1)
mB = zeros(Int, T0); mB[1] = ML.sample_log(rngB, log.(W.π0))
for τ in 2:T0; mB[τ] = ML.sample_log(rngB, log.(W.Ptr[mB[τ-1], :])); end
Eb = zeros(n, T0); for τ in 1:T0; Eb[:, τ] .= randn(rngB, n); end
println("generator: m equal ", mB == m, ", max|Δz0| = ", maximum(abs.(z0 .- z0_from_innov(W, F, mB, Eb))))
e = z0_innov(W, F, m, z0)
println("round trip: ", maximum(abs.(z0_from_innov(W, F, m, e) .- z0)))
vals = [begin eq = randn(MersenneTwister(100 + q), n, T0) .* (1 + q)
            ML.logp_z0(W, m, z0_from_innov(W, F, m, eq))[1] + 0.5 * sum(abs2, eq) end for q in 1:5]
println("log p + |e|²/2 spread: ", maximum(vals) - minimum(vals))
let θ = 0.3, c = cos(θ), sn = sin(θ), e1 = randn(MersenneTwister(1), 50), e2 = randn(MersenneTwister(2), 50)
    lq(a, b) = -0.5 * sum(abs2, a) - 0.5 * sum(abs2, b .- c .* a) / sn^2
    println("pCN balance: ", lq(e1, e2) - lq(e2, e1))
end
let a = 100, b = 102, N = 40000, rw = MersenneTwister(3), cnt = Dict{Vector{Int},Int}()
    for _ in 1:N
        key = regime_window(rw, W, m, a, b)[a:b]; cnt[key] = get(cnt, key, 0) + 1
    end
    ex = Dict([i, j, l] => W.Ptr[m[a-1], i] * W.Ptr[i, j] * W.Ptr[j, l] * W.Ptr[l, m[b+1]] for i in 1:W.K, j in 1:W.K, l in 1:W.K)
    Z = sum(values(ex)); chi = 0.0; df = 0
    for (kk, p) in ex
        p /= Z; p * N > 5 || continue
        chi += (get(cnt, kk, 0) - N * p)^2 / (N * p); df += 1
    end
    println("regime window χ² = $chi on $(df - 1) df")
end

# ---- E22 long run against exact draws ----
kref = ML.knodes(P, sL.ℓ, sL.k)
regfeat(m) = vcat([mean(m .== k) for k in 1:W.K], count(diff(m) .!= 0))
allfeat(s) = vcat(featvec(wfeatures(P, s.xs[end], kref)), regfeat(s.m))
fnames = vcat(featnames(P), ["regime $k" for k in 1:W.K], "switches")
rx = MersenneTwister(2024)
exact = reduce(hcat, [begin m_, z_, w_ = ML.sample_weather(rx, W, 1); allfeat(ML.make_state(P, 1, m_, z_, w_, sL.k, sL.v)) end for _ in 1:2000])
function wchain(seed, θ, nstep; B=7)
    r = MersenneTwister(seed)
    m_, z_, w_ = ML.sample_weather(r, W, 1)
    s = ML.make_state(P, 1, m_, z_, w_, sL.k, sL.v)
    out = zeros(length(fnames), nstep)
    for i in 1:nstep
        s = weather_step(r, P, F, s, θ; regime=true, regmode=:window, B)
        out[:, i] = allfeat(s)
    end
    out
end
function bm_se(x; nb=40)
    b = length(x) ÷ nb
    mb = [mean(x[(i-1)*b+1:i*b]) for i in 1:nb]
    return std(mb) / sqrt(nb), b * var(mb) / var(x)
end
function compare(exact, ch; burn=200)
    for j in axes(exact, 1)
        x = ch[j, burn+1:end]; y = exact[j, :]
        se, τ = bm_se(x)
        z = (mean(x) - mean(y)) / sqrt(se^2 + var(y) / length(y))
        @printf("%-18s exact %9.4f ± %8.4f   chain %9.4f ± %8.4f   z %+5.2f  τ %6.1f\n", fnames[j], mean(y), std(y), mean(x), std(x), z, τ)
    end
end
ch03 = wchain(11, 0.3, 4000; B=7); compare(exact, ch03)              # fails on the regime frequencies
ch03b = wchain(12, 0.3, 6000; B=60); compare(exact, ch03b; burn=300)  # passes
serialize("results/replicas/prior_test.jls", (; exact, ch03, ch03b, fnames))

# ---- E23 dispatch lag against weather speed (k fixed, one replica) ----
κs = [500.0, 1000.0, 200.0, 300.0, 100.0, 100.0]
Cl = deserialize("results/langevin/lgv_low_C.jls")                    # E19 C low end state
s_eq = copy(Cl.state)
for _ in 1:20; global s_eq = lgv_op!(MersenneTwister(3), P, S.loc, LM, s_eq); end
scan = Dict{Any,Any}()
for (θ, nreg, comp) in ((0.0, 0, true), (0.03, 0, true), (0.1, 0, true), (0.3, 0, true), (0.1, 0, false), (0.1, 10, true),
                        (0.01, 0, true), (0.02, 0, true))
    ctl = RepCtl(; ηw=zeros(6), κs, θ, nreg, comp, regB=60, probe_every=40, adapt=false)
    tr = NamedTuple[]; pb = NamedTuple[]
    rep_run!(MersenneTwister(1), [MersenneTwister(500)], MersenneTwister(77), P, S, [copy(s_eq)], 120, ctl;
             trace=tr, probes=pb, log_every=40, label="θ=$θ nreg=$nreg comp=$comp")
    scan[(θ, nreg, comp)] = (; tr, pb)
end
let ctl = RepCtl(; ηw=zeros(6), κs, θ=0.3, nreg=0, comp=true, probe_every=0, adapt=false), tr = NamedTuple[],
    rr = MersenneTwister(1), reps = [copy(s_eq)]
    rep_run!(rr, [MersenneTwister(501)], MersenneTwister(77), P, S, reps, 1, ctl; trace=tr, log_every=1000)
    ctl.θ = 0.0
    rep_run!(rr, [MersenneTwister(501)], MersenneTwister(77), P, S, reps, 119, ctl; trace=tr, it0=1, log_every=1000)
    scan[(:burst, 0.3)] = (; tr)
end
serialize("results/replicas/lagscan.jls", scan)
