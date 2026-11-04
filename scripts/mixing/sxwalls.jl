# Ledger E26: where the storage moves stop. At a state of a chains40 chain, line draws (no commit)
# of `slong` (per carrier), `sxlong` (per pair of carriers, same node or across nodes) and
# `sxpair`; per family the energy moved per draw and the walls on both sides of each walk
# (scripts/mixing/walls.jl): line limits, storage limits (box, power, the charge/discharge switch,
# which for reservoir hydro is the zero-discharge wall), gas or other generators at a clamp (at the
# move's own nodes or at leak nodes), compensation kinks, cells turning into shed.
#
# Usage: nice -n 19 julia --project=. -t 4 sxwalls.jl <chains.jls> <chain index> [draws per family] [out.txt]
import ESMSampling as ML
using Random, Statistics, Printf, Serialization
include(joinpath(@__DIR__, "walls.jl"))

function wall_report(io, P, LM, s; n=300, seed=1)
    E = P.E
    T = LM.T
    LC = ML.LocalCost(P, LM, s)
    pool = ML.BlockPool(P, 3)
    jac = NTuple{5,Float64}[]
    rng = MersenneTwister(seed)
    msl = ML.SLong(pool.wss[1], pool.wss[2], pool.wss[3], P, LM, LC)
    msx = ML.SXLong(pool.wss[1], pool.wss[2], pool.wss[3], P, LM, LC)
    wa, ws = ML.MoveWS(P), ML.MoveWS(P)
    res = Dict{String,Tuple{Vector{Float64},Dict{String,Int}}}()
    function record!(lab, m, own, EA, t)
        en, cnt = get!(() -> (Float64[], Dict{String,Int}()), res, lab)
        push!(en, abs(t) * EA)
        for (_, _, ks) in walls(m, own), k in ks
            cnt[k] = get(cnt, k, 0) + 1
        end
    end
    carrier(si) = E.stors[si].carrier
    for fam in (:slong, :sxlong, :sxpair)
        for _ in 1:n
            si, i = rand(rng, LM.sxunits)
            EA = LC.kst[si][i] * E.stors[si].hours[i]
            if fam == :slong
                ML.prep_slong!(msl, si, i, rand(rng, 1:T), rand(rng, 1:T-1), jac) || continue
                t = ML.sample_line!(rng, msl, P.βv, pool.wss[1].secprev)
                record!("slong " * carrier(si), msl, (E.stors[si].bus[i],), EA, t)
            elseif fam == :sxlong
                j, sj, jj = ML.sx_partner(rng, P, LM, LC, si, i)
                j == 0 && continue
                ML.prep_sxlong!(msx, si, i, sj, jj, j, rand(rng, 1:T), rand(rng, 1:T-1), jac) || continue
                t = ML.sample_line!(rng, msx, P.βv, pool.wss[1].secprev)
                record!("sxlong " * carrier(si) * "-" * carrier(sj) * (j == 1 ? " same node" : " cross"), msx,
                        (E.stors[si].bus[i], E.stors[sj].bus[jj]), EA, t)
            else
                τ = rand(rng, 1:T); τm = mod(τ - 2, T) + 1
                ML.clear!(ws); ML.clear!(wa)
                r = Xoshiro(rand(rng, UInt))
                ML.prep_sxpair!(copy(r), wa, ws, P, LM, LC, τm, τ, jac) || continue
                a = rand(r, LM.sxunits)                                   # the same draws: A, then the partner
                j, sj, jj = ML.sx_partner(r, P, LM, LC, a...; κmin=1.0)
                ML.finalize!(ws, P, LM, LC, τ); ML.finalize!(wa, P, LM, LC, τm); ML.link_soc!(wa, ws, LM)
                bm = ML.BlockMove([wa, ws], 2, [τm, τ], P, LM, LC, Float64[], Float64[])
                append!(bm.bps, wa.bps); append!(bm.bps, ws.bps)
                t = ML.sample_line!(rng, bm, P.βv, ws.secprev)
                EA = LC.kst[a[1]][a[2]] * E.stors[a[1]].hours[a[2]]
                record!("sxpair " * carrier(a[1]) * "-" * carrier(sj) * (j == 1 ? " same node" : " cross"), bm,
                        (E.stors[a[1]].bus[a[2]], E.stors[sj].bus[jj]), EA, t)
            end
        end
    end
    for (lab, (en, cnt)) in sort(collect(res))
        nside = 2 * length(en)
        @printf(io, "%-28s draws %4d  |t|E per draw median %8.3g mean %8.3g MWh\n", lab, length(en), median(en), mean(en))
        for (k, v) in sort(collect(cnt); by=last, rev=true)
            @printf(io, "      %-40s %5.1f %% of the walk sides\n", k, 100 * v / nside)
        end
    end
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    chainsfile, ci = ARGS[1], parse(Int, ARGS[2])
    n = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 300
    cfg = ML.load_config("configs/chains40.toml"); cfg.verbose = false
    P, _, _ = ML.setup(cfg; rng=MersenneTwister(cfg.seed))
    s = ML.evaluate!(P, copy(deserialize(chainsfile)[ci]))
    LM = ML.local_model(ML.new_sampler(P, s, cfg).loc, P, s.ℓ)
    io = length(ARGS) >= 4 ? open(ARGS[4], "w") : stdout
    wall_report(io, P, LM, s; n=n)
    io === stdout || close(io)
end
