# Weather move efficiency at the fine level (docs/mixing_diagnosis.md): from given states, each
# weather move family alternates with operation sweeps; recorded per sweep: the z0 row means and
# seasonal means, the regimes, the family's moves and acceptance, wall time. The efficiency is the
# expected squared jump of the annual and seasonal means per sweep and per CPU second, relative to
# their prior variance.

include(joinpath(@__DIR__, "wblock.jl"))
include(joinpath(@__DIR__, "driver.jl"))

"""
    weather_family_run(rng, P, LM, K, s, fam; nsweeps, opsweeps) -> NamedTuple

fam: (:fine, frac0, frac) the standard fine weather sweep; (:vblock, B, V, nmoves);
(:rowblock, B, nmoves); (:regime, B, nmoves).
"""
function weather_family_run(rng, P, LM, K, s, fam; nsweeps::Int=30, opsweeps::Int=1)
    n0 = ML.n0(P.W)
    zm = zeros(n0, nsweeps + 1); zs = zeros(4n0, nsweeps + 1); rg = zeros(Int, nsweeps + 1)
    zm[:, 1] .= z0_rowmeans(P, s.z0); zs[:, 1] .= z0_seasonal(P, s.z0); rg[1] = count(s.m .!= 0)
    m0 = copy(s.m)
    tw, to = 0.0, 0.0
    nmv, nacc = 0, 0
    tabs = Float64[]
    regch = 0
    for it in 1:nsweeps
        a = time()
        for _ in 1:opsweeps
            s = ML.local_sweep!(rng, P, LM, s; wsets=K.wsets, extra=K.extra)[1]
        end
        b = time(); to += b - a
        mold = copy(s.m)
        if fam[1] == :fine
            s, st = ML.fine_weather_sweep!(rng, P, LM, s; frac=fam[3], frac0=fam[2], pools=K.wpools)
            nmv += Int(st[1, 1]); nacc += Int(st[1, 2])
        elseif fam[1] == :vblock
            s, ts = z0_block_sweep!(rng, P, LM, s; B=fam[2], nmoves=fam[4], V=fam[3])
            nmv += length(ts); nacc += count(!=(0.0), ts); append!(tabs, abs.(ts))
        elseif fam[1] == :rowblock
            s, ts = z0_block_sweep!(rng, P, LM, s; B=fam[2], nmoves=fam[3])
            nmv += length(ts); nacc += count(!=(0.0), ts); append!(tabs, abs.(ts))
        elseif fam[1] == :regime
            s, st = regime_block_sweep!(rng, P, LM, s; B=fam[2], nmoves=fam[3])
            nmv += st[1]; nacc += st[2]
        end
        tw += time() - b
        regch += count(s.m .!= mold)
        zm[:, it+1] .= z0_rowmeans(P, s.z0); zs[:, it+1] .= z0_seasonal(P, s.z0)
    end
    return (; s, zm, zs, tw, to, nmv, nacc, tabs, regch, nsweeps)
end

"Mean squared jump per sweep of the rows `rows` of a trace matrix."
msj(Z, rows) = [mean(abs2, diff(Z[r, :])) for r in rows]
