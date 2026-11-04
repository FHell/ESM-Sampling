# Capacity transport pair ("kxpair", prototype for docs/mixing_diagnosis.md).
#
# Moves renewable capacity of one carrier between the two ends (n, m) of an AC line, with the
# power flow following: k_n → k_n + t, k_m → k_m − ρ t (ρ = Σ_τ A_n,τ / Σ_τ A_m,τ, annual
# energy neutral), and at every step a re-routing of g_τ t from n to m through the whole AC
# network, Δθ_τ = g_τ t L⁺(e_n − e_m) (projected to the potential space), with
# g_τ = (A_n,τ + ρ A_m,τ)/2, so that each node's balance changes only by half the profile
# mismatch ±(A_n,τ − ρ A_m,τ) t/2. Everything is affine in t with coefficients fixed by
# (x, network, k ≥ κmin selection), so no Jacobian enters; t is drawn from the exact line
# conditional by slice sampling with a fixed width, truncated to k ≥ κmin at both ends.

include(joinpath(@__DIR__, "kop.jl"))

"AC line endpoint pairs (n, m) where both nodes have a unit of renewable group gi."
function kx_pairs(P, gi)
    sys = P.E.sys
    g = P.E.gens[gi]
    pos = Dict(b => i for (i, b) in enumerate(g.bus))
    out = NTuple{4,Int}[]
    for e in eachindex(sys.line_s)
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        (haskey(pos, a) && haskey(pos, b)) && push!(out, (a, b, pos[a], pos[b]))
    end
    return out
end

"Potential change per unit re-routing n → m, projected to mean zero on the component."
function kx_dtheta(P, LM, n, m)
    GS = LM.GS
    x = zeros(P.E.sys.N)
    ML.reroute!(x, GS, n, m)
    nd = GS.nodes[GS.comp[n]]
    μ = mean(x[nd])
    x[nd] .-= μ
    return x
end

"""
Log line density of a kxpair move at t (capex, renewable costs, all line terms, the two
balance terms); with commit = true also writes the caches and the coefficients.
"""
function kx_eval(P, LM, LC, s, gi, n, m, in_, im, ρ, gτ, dF, dθ, kn0, km0, t; commit::Bool=false)
    E = P.E
    sys = E.sys
    Δt, cv, κ0, T = LM.Δt, E.c_viol, E.κ0, LM.T
    g = E.gens[gi]
    sp = E.kspecs[g.kf]
    kn1, km1 = kn0 + t, km0 - ρ * t
    kcn0, kcn1 = clamp(kn0, sp.lo[in_], sp.hi[in_]), clamp(kn1, sp.lo[in_], sp.hi[in_])
    kcm0, kcm1 = clamp(km0, sp.lo[im], sp.hi[im]), clamp(km1, sp.lo[im], sp.hi[im])
    cap = ML.f_capex(E, g.kf)
    Δcap = cap[in_] * (kn1 - kn0) + E.γk * cap[in_] * (abs(kn1 - kcn1) - abs(kn0 - kcn0)) +
           cap[im] * (km1 - km0) + E.γk * cap[im] * (abs(km1 - kcm1) - abs(km0 - kcm0))
    Δop = 0.0
    cn, cm = LM.cell[n], LM.cell[m]
    lines = findall(x -> abs(x) > 1e-12, dF)
    lv = P.lv[LM.ℓ+1]
    for τ in 1:T
        An, Am = LC.A[gi][in_, τ], LC.A[gi][im, τ]
        Δop += ML.gen_cost(g, in_, An, kcn1, kcn1 + κ0, Δt, cv, 1.0)[1] - ML.gen_cost(g, in_, An, kcn0, kcn0 + κ0, Δt, cv, 1.0)[1]
        Δop += ML.gen_cost(g, im, Am, kcm1, kcm1 + κ0, Δt, cv, 1.0)[1] - ML.gen_cost(g, im, Am, kcm0, kcm0 + κ0, Δt, cv, 1.0)[1]
        x = gτ[τ] * t                       # re-routed MW n → m
        dSn = x - An * (kcn1 - kcn0)
        dSm = -x - Am * (kcm1 - kcm0)
        Sn0, Sm0 = LC.S[cn, τ], LC.S[cm, τ]
        ncn, ncm = LM.ncell[cn], LM.ncell[cm]
        Δop += Δt * ncn * (ML.φbal(E, (Sn0 + dSn) / ncn) - ML.φbal(E, Sn0 / ncn))
        Δop += Δt * ncm * (ML.φbal(E, (Sm0 + dSm) / ncm) - ML.φbal(E, Sm0 / ncm))
        for e in lines
            f0 = LC.F[e, τ]
            f1 = f0 + x * dF[e]
            Δop += cv * Δt * (ML.line_eval(E.lines, e, sys.line_s[e], f1)[1] - ML.line_eval(E.lines, e, sys.line_s[e], f0)[1])
            commit && (LC.F[e, τ] = f1)
        end
        if commit
            LC.r[n, τ] += dSn; LC.r[m, τ] += dSm
            LC.S[cn, τ] += dSn; LC.S[cm, τ] += dSm
            fθ = E.fθ
            mθ = lv.m[fθ]; o = lv.off[fθ] + (τ - 1) * mθ
            Ut = E.vspecs[fθ].B.Ut
            for i in eachindex(dθ)
                d = x * dθ[i]
                d == 0.0 && continue
                LC.Y[fθ][i, τ] += d
                for q in nzrange(Ut, i)
                    j = rowvals(Ut)[q]
                    j <= mθ && (s.v[o+j] += d * nonzeros(Ut)[q])
                end
            end
        end
    end
    if commit
        lk = P.lk[LM.ℓ+1]
        Ut = sp.B.Ut
        for (ii, Δ) in ((in_, kn1 - kn0), (im, km1 - km0))
            for q in nzrange(Ut, ii)
                s.k[lk.off[g.kf]+rowvals(Ut)[q]] += Δ * nonzeros(Ut)[q]
            end
        end
        LC.kgen[gi][in_] = kcn1; LC.kgen[gi][im] = kcm1
    end
    return -P.βk * Δcap - P.βv * Δop
end

"""
    kxpair_sweep!(rng, P, LM, s; nmoves, w, check) -> (s, ts)

`nmoves` capacity transport pairs: a renewable group and a line with the group at both ends,
uniformly; slice width `w` MW (fixed).
"""
function kxpair_sweep!(rng, P, LM, s; nmoves::Int=40, w::Float64=200.0, check::Bool=false, κmin::Float64=KOP_KMIN)
    E = P.E
    s = copy(s); s.k = copy(s.k); s.v = copy(s.v)
    LC = ML.LocalCost(P, LM, s)
    groups = [gi for (gi, g) in enumerate(E.gens) if g.kf > 0 && g.vf == 0 && !isempty(kx_pairs(P, gi))]
    pairs = Dict(gi => kx_pairs(P, gi) for gi in groups)
    θcache = Dict{Tuple{Int,Int},Tuple{Vector{Float64},Vector{Float64}}}()
    sys = E.sys
    ts = zeros(nmoves)
    for q in 1:nmoves
        gi = rand(rng, groups)
        n, m, in_, im = rand(rng, pairs[gi])
        rand(rng, Bool) && ((n, m, in_, im) = (m, n, im, in_))
        Kraw = ML.knodes(P, LM.ℓ, s.k)
        kf = E.gens[gi].kf
        kn0, km0 = Kraw[kf][in_], Kraw[kf][im]
        (kn0 >= κmin && km0 >= κmin) || continue
        dθ, dF = get!(θcache, (n, m)) do
            dθ = kx_dtheta(P, LM, n, m)
            dF = [sys.line_b[e] * (dθ[sys.line_bus0[e]] - dθ[sys.line_bus1[e]]) for e in eachindex(sys.line_s)]
            (dθ, dF)
        end
        An, Am = LC.A[gi][in_, :], LC.A[gi][im, :]
        ρ = sum(An) / max(sum(Am), 1e-9)
        gτ = (An .+ ρ .* Am) ./ 2
        lo, hi = κmin - kn0, (km0 - κmin) / ρ
        lf(t) = (t < lo || t > hi) ? -Inf : kx_eval(P, LM, LC, s, gi, n, m, in_, im, ρ, gτ, dF, dθ, kn0, km0, t)
        t = ML.slice_scalar(rng, lf, 0.0, w, lo)
        t > hi && (t = 0.0)
        t == 0.0 && continue
        L0 = s.L
        pred = check ? lf(t) : 0.0
        kx_eval(P, LM, LC, s, gi, n, m, in_, im, ρ, gτ, dF, dθ, kn0, km0, t; commit=true)
        ts[q] = t
        if check
            ML.evaluate!(P, s)
            err = (s.L - L0) - pred
            abs(err) > 1e-6 * max(1.0, abs(pred)) && @warn "kxpair: incremental and global log target differ" err pred
        end
    end
    ML.evaluate!(P, s)
    if check
        LC2 = ML.LocalCost(P, LM, s)
        errY = maximum(maximum(abs.(LC.Y[f] .- LC2.Y[f]); init=0.0) for f in eachindex(LC.Y))
        errF = maximum(abs.(LC.F .- LC2.F)); errS = maximum(abs.(LC.S .- LC2.S))
        @info "kxpair cache check" errY errF errS
    end
    return s, ts
end
