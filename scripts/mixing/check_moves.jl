# Deterministic consistency checks of the prototype moves (docs/mixing_diagnosis.md), in the
# style of test/test_local.jl and test/test_flow.jl:
# * kop: incremental log-target change against the global evaluation, caches against a rebuild,
#   and the curve check: from the moved state the move with -u returns exactly to the start and
#   its line log-density is the negative of the forward one (same curve, same density from both
#   ends, Jacobians inverse); the pro-rata coordinates (σ, e) of the compensators are unchanged.
# * z0 block (joint direction) and regime block moves: incremental against global cost, caches,
#   anomalies against a fresh evaluation (inside the moves with check = true), and the reverse
#   z0 move returning to the start.
include(joinpath(@__DIR__, "kop.jl"))
include(joinpath(@__DIR__, "wblock.jl"))

"Apply a kop move of `group` with a given u deterministically; returns (s1, lf(u))."
function kop_apply(P, LM, s, group, u)
    s1 = copy(s); s1.k = copy(s.k); s1.v = copy(s.v)
    LC = ML.LocalCost(P, LM, s1)
    Kraw = ML.knodes(P, LM.ℓ, s1.k)
    terms = [kop_term(P, LM, LC, Kraw, kind, gi, i) for (kind, gi, i) in group
             if Kraw[kind == 1 ? P.E.gens[gi].kf : P.E.stors[gi].kf][i] >= KOP_KMIN]
    lo = maximum(log(KOP_KMIN / kt.k0) for kt in terms)
    u < lo && return nothing, NaN, LC             # outside the truncated line (the move never goes there)
    lf = sum(kop_eval(P, LM, LC, kt, kt.k0 * exp(u))[1] for kt in terms) + length(terms) * u
    for kt in terms
        k1 = kt.k0 * exp(u)
        kop_eval(P, LM, LC, kt, k1; commit=true, s=s1)
        kop_setk!(P, LM, LC, s1, kt, k1)
    end
    return ML.evaluate!(P, s1), lf, LC
end

"Pro-rata coordinates (σ, e) of the compensators of a node at a step."
function prorata_coords(P, LM, LC, n, τ, self)
    E = P.E
    ps, Qs = Float64[], 0.0
    for (f, ii) in LM.genunits[n]
        (f, ii) == self && continue
        E.gens[LM.role[f][2]].wf > 0 && continue
        κ = ML.κunit(P, LM, LC, f, ii, τ)
        abs(κ) >= KOP_KMIN || continue
        push!(ps, κ * LC.Y[f][ii, τ]); Qs += κ * clamp(LC.Y[f][ii, τ], 0.0, 1.0)
    end
    isempty(ps) && return (Float64[], NaN)
    Q = LC.r[n, τ] + Qs
    return ps ./ sum(ps), sum(ps) - max(Q, 0.0)
end

function check_kop(P, LM, s; rng=MersenneTwister(1), us=(0.3, -0.5, 1.2))
    ok = true
    for grp in kop_units(P, LM; kinds=(:renewable, :storage)), u in us
        for group in ([grp[rand(rng, 1:length(grp))]], grp)
            s1, lf1, LC1 = kop_apply(P, LM, s, group, u)
            s1 === nothing && continue
            s2, lf2, LC2 = kop_apply(P, LM, s1, group, -u)
            ek = maximum(abs.(s2.k .- s.k)); ev = maximum(abs.(s2.v .- s.v))
            ed = abs(lf1 + lf2)
            # pro-rata coordinates of the first unit's node, at every step
            kind, gi, i = group[1]
            n = kind == 1 ? P.E.gens[gi].bus[i] : P.E.stors[gi].bus[i]
            LC0 = ML.LocalCost(P, LM, s)
            eσ = 0.0
            for τ in 1:LM.T
                σ0, e0 = prorata_coords(P, LM, LC0, n, τ, (0, 0))
                σ1, e1 = prorata_coords(P, LM, LC1, n, τ, (0, 0))
                isempty(σ0) && continue
                eσ = max(eσ, maximum(abs.(σ1 .- σ0)), abs(e1 - e0) / max(1.0, abs(e0)))
            end
            good = ek < 1e-6 && ev < 1e-9 && ed < 1e-6 * max(1, abs(lf1)) && eσ < 1e-8
            ok &= good
            good || @warn "kop curve check failed" group[1] length(group) u ek ev ed eσ
        end
    end
    # incremental vs global and caches inside a sweep
    s3, us3 = kop_sweep!(rng, P, LM, s; mode=:both, check=true)
    return ok
end

function check_z0(P, LM, s, V; rng=MersenneTwister(2))
    LC = ML.LocalCost(P, LM, s)
    pool = ML.BlockPool(P, 70)
    bm = ML.BlockMove(pool.wss, 0, Int[], P, LM, LC, Float64[], Float64[])
    jac, sp = NTuple{5,Float64}[], Float64[]
    s1 = copy(s); s1.z0 = copy(s.z0); s1.v = copy(s.v); s1.xs = [copy(x) for x in s.xs]
    Dz = V[:, 1] * ones(16)'
    t = 0.0
    for _ in 1:20
        t = z0_dir_move!(rng, P, LM, LC, s1, bm, jac, sp, 100, 115, Dz; check=true)
        t != 0.0 && break
    end
    # reverse: apply -t along the same direction deterministically (commit without sampling)
    LC1 = ML.LocalCost(P, LM, s1)
    bm1 = ML.BlockMove(pool.wss, 0, Int[], P, LM, LC1, Float64[], Float64[])
    Dm = Matrix{Float64}[]; row = 0
    for ff in eachindex(P.W.specs)
        push!(Dm, Matrix(Dz[row+1:row+P.W.lays[1].m[ff], :])); row += P.W.lays[1].m[ff]
    end
    ch = ML.wdelta_chain(P.W, s1.m, ML.WDelta(0, 100, 115, Dm), s1.ℓ)
    ML.prep_weather_move!(bm1, P, LM, LC1, ch, jac)
    ML.mv_eval!(bm1, -t; mode=1)
    ML.commit_block!(bm1, s1.v, -t)
    dz = zeros(length(s1.z0))
    for ff in eachindex(P.W.specs)
        ML.fieldview(dz, P.W.lays[1], P.W.specs, ff)[:, 100:115] .= Dm[ff]
    end
    s1.z0 .-= t .* dz; s1.xs[1] === s1.z0 || (s1.xs[1] .-= t .* dz)
    ML.apply_wdelta!(s1, P.W, ch[2:end], -t)
    ev = maximum(abs.(s1.v .- s.v)); ez = maximum(abs.(s1.z0 .- s.z0)); ex = maximum(abs.(s1.xs[end] .- s.xs[end]))
    good = ev < 1e-8 && ez < 1e-12 && ex < 1e-10
    good || @warn "z0 reverse check failed" t ev ez ex
    return good, t
end
