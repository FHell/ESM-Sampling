# Shared part of opmix.jl and opmix_hpc.jl (ledger E17, E21, E27): the set-up from a chains file
# and the recorded quantities. Expects `cfg`, `chainsfile` and `ci` to be defined.
import ESMSampling as ML
using Random, Statistics, LinearAlgebra, Printf, Serialization

P, sL, info = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
chains = deserialize(chainsfile)
s = ML.evaluate!(P, copy(chains[ci]))
L = ML.nlevels(P) - 1
@assert s.ℓ == L
S = ML.new_sampler(P, s, cfg)
K = S.loc
LM = ML.local_model(K, P, L)
LM0 = ML.LocalModel(P, 0)
E = P.E
lk, lv = P.lk[L+1], P.lv[L+1]
T = LM.T
region = [LM0.cell[n] for n in 1:E.sys.N]            # 40 nodes → the 10 level-0 regions
nreg = maximum(region)
season = [min(4, 1 + (t - 1) * 4 ÷ T) for t in 1:T]  # quarters of the year
names = ML.quantity_names(P)
iop = findfirst(==("opex"), names); ish = findfirst(==("shed_MWh"), names)

"Per-carrier aggregate capacity score: Σ over units of s_k (nats per MW added at every unit)."
function carrier_scores(P, s)
    Xn = ML.weather_nodes(P.W, s.ℓ, s.xs[end])
    _, (gkc, gko, _, _) = ML.energy_cost(E, lk, lv, s.k, s.v, Xn; grad=true)
    sk = -P.βv .* gko .+ ML.log_measure_grad(E, lk, lv, s.k)
    out = Float64[]
    for f in eachindex(E.kspecs)
        U = ML.cols(E.kspecs[f].B, lk.m[f])
        push!(out, sum(U * vec(ML.fieldview(sk, lk, E.kspecs, f))))
    end
    return out
end

"Stored energy (MWh) per storage carrier, region and quarter (mean over the quarter's instants)."
function soc_rq(P, s)
    LC = ML.LocalCost(P, LM, s)
    R = zeros(length(E.stors), nreg, 4)
    cnt = zeros(4)
    for t in 1:T; cnt[season[t]] += 1; end
    for (si, st) in enumerate(E.stors)
        Y = LC.Y[st.fe]
        for i in axes(Y, 1)
            k = LC.kst[si][i]
            k > 1.0 || continue
            r = region[st.bus[i]]
            for t in 1:T
                R[si, r, season[t]] += clamp(Y[i, t], 0.0, 1.0) * k * st.hours[i]
            end
        end
    end
    for q in 1:4; R[:, :, q] ./= cnt[q]; end
    return R
end

"Stored energy (MWh) per storage unit and quarter: rows (carrier, unit) in E.stors order."
function unit_q(P, s)
    LC = ML.LocalCost(P, LM, s)
    Q = zeros(sum(length(st.bus) for st in E.stors), 4)
    cnt = zeros(4)
    for t in 1:T; cnt[season[t]] += 1; end
    r = 0
    for (si, st) in enumerate(E.stors)
        Y = LC.Y[st.fe]
        for i in axes(Y, 1)
            r += 1
            k = LC.kst[si][i]
            for t in 1:T
                Q[r, season[t]] += clamp(Y[i, t], 0.0, 1.0) * k * st.hours[i] / cnt[season[t]]
            end
        end
    end
    return Q
end


"One operation sweep of the kernel: local sweep (with sxpair at rate sx), storage multiscale sweep
with probability 1/2, and for \"slong\"/\"sx\" the long-range storage sweep (nx sxlong per unit)."
function opmix_sweep!(rng, s, kernel, nlong, nx, sx)
    s, _ = ML.local_sweep!(rng, P, LM, s; wsets=K.wsets, extra=K.extra, sxpair=sx)
    if rand(rng) < 0.5 && LM.complete[E.fθ]
        isempty(K.pools) && append!(K.pools, [ML.BlockPool(P, 140) for _ in 1:Threads.maxthreadid()])
        s, _ = ML.storage_ms_sweep!(rng, P, LM, s; pools=K.pools)
    end
    lst = nothing
    if kernel in ("slong", "sx")
        isempty(K.pools) && append!(K.pools, [ML.BlockPool(P, 140) for _ in 1:Threads.maxthreadid()])
        s, lst = ML.storage_long_sweep!(rng, P, LM, s; pools=K.pools, nmoves=nlong, nxmoves=nx)
    end
    return s, lst
end

new_record(kernel, nlong, nx, sx) =
    Dict{String,Any}("op" => Float64[], "shed" => Float64[], "score" => Vector{Float64}[], "soc" => Array{Float64,3}[],
                     "time" => Float64[], "knames" => [string(f.name) for f in E.kspecs],
                     "snames" => [st.carrier for st in E.stors], "kind" => ci, "kernel" => kernel, "nlong" => nlong,
                     "unitq" => Matrix{Float64}[], "nx" => nx, "sx" => sx, "lstats" => Matrix{Float64}[],
                     "ustore" => [(st.carrier, E.sys.nodes[st.bus[i]], st.hours[i]) for st in E.stors for i in eachindex(st.bus)])

"Record the quantities of state s after a sweep of `dt` seconds."
function record!(rec, s, dt, lst)
    push!(rec["time"], dt)
    lst === nothing || push!(rec["lstats"], lst)
    q = ML.quantities(P, s)
    push!(rec["op"], q[iop]); push!(rec["shed"], q[ish])
    push!(rec["score"], carrier_scores(P, s))
    push!(rec["soc"], soc_rq(P, s))
    push!(rec["unitq"], unit_q(P, s))
    return q
end
