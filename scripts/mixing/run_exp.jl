# General experiment runner for the mixing diagnosis. Usage:
#   nice -n 19 julia --project=. -t 4 scripts/mixing/run_exp.jl <name> <nsweeps> [extras] [key=value ...]
# extras: comma-separated list of moves added to every fine step:
#   none | kop (kop unit + carrier moves) | kopunit | kopcarrier |
#   vblock16 | vblock64 | vblock365 (joint z0 block moves, 10 per step) | regime1 | regime4 |
#   sms (storage multiscale sweep of src/flowmoves.jl)
# key=value: overrides of the configs/chains40.toml keys (fine_weather=0.0, measure=utilization, ...)
include(joinpath(@__DIR__, "driver.jl"))
include(joinpath(@__DIR__, "kop.jl"))
include(joinpath(@__DIR__, "wblock.jl"))
using Serialization

name = ARGS[1]
nsw = parse(Int, ARGS[2])
exlist = length(ARGS) >= 3 ? split(ARGS[3], ",") : ["none"]
ov = Dict{Symbol,Any}()
for a in ARGS[4:end]
    k, v = split(a, "=")
    T = fieldtype(ML.Config, Symbol(k))
    ov[Symbol(k)] = T <: AbstractString ? String(v) : parse(T, v)
end
out = "results/mixing/$name.h5"
cfg, P, sL, info = mix_setup("configs/chains40.toml"; ov...)
println("overrides: ", ov, "  extras: ", exlist)

# joint z0 directions (vblock): results/mixing/z0dirs.jls, the leading eigenvectors of the prior
# covariance of the annual means (scripts/mixing/prior_spectral.jl)
extras = Tuple{String,Function}[]
Vdir = nothing
for e in exlist
    e == "none" && continue
    if e == "kop"
        kopcalls = Ref(0)                 # the first two passes run with the consistency checks
        push!(extras, ("kop", (rng, P, LM, s) -> kop_sweep!(rng, P, LM, s; mode=:both, check=(kopcalls[] += 1) <= 2)[1]))
    elseif e == "kopunit"
        push!(extras, ("kop", (rng, P, LM, s) -> kop_sweep!(rng, P, LM, s; mode=:unit)[1]))
    elseif e == "kopcarrier"
        push!(extras, ("kop", (rng, P, LM, s) -> kop_sweep!(rng, P, LM, s; mode=:carrier)[1]))
    elseif startswith(e, "vblock")
        B = parse(Int, e[7:end])
        global Vdir
        if Vdir === nothing
            Vdir = deserialize("results/mixing/z0dirs.jls")
        end
        push!(extras, (e, (rng, P, LM, s) -> z0_block_sweep!(rng, P, LM, s; B=B, nmoves=10, V=Vdir)[1]))
    elseif e == "sms"
        push!(extras, ("sms", (rng, P, LM, s) -> ML.storage_ms_sweep!(rng, P, LM, s)[1]))
    elseif startswith(e, "regime")
        B = parse(Int, e[7:end])
        push!(extras, (e, (rng, P, LM, s) -> regime_block_sweep!(rng, P, LM, s; B=B, nmoves=100)[1]))
    else
        error("unknown extra $e")
    end
end
rec = run_mix(cfg, P, sL; nsweeps=nsw, out=out, label=name, every=25, extras=extras)
save_rec(out, rec, nsw)
serialize("results/mixing/$(name)_chains.jls", rec["chains"])
println("done, wall ", rec["wall"])
