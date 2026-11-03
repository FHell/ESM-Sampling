# Time hierarchy: transport-first (T2) prolongation from a coarse time level to a
# finer one, and diagnostics of intra-block structure (docs/flow_hierarchy.md §12).

"""
    prolong_t2(P, sc, sw) -> sf

Prolong the state `sc` of level ℓ−1 to level ℓ (same spatial resolution, finer
time) with the fine weather of the reference state `sw`, keeping the coarse
transport (angles, links, block boundary states of charge with linear
interpolation) and applying the T2 map inside every block. Per node and block,
the coarse shares σ_i = p_i/P and excess e = P − Q₊ are kept, and the fine
dispatch is p_i(τ) = σ_i (Q_τ₊ + e) with Q_τ the fine requirement. Block-constant
dispatch (zero details) is the alternative `prolong_constant`.
"""
function prolong_t2(P::Problem, sc::State, sw::State; clampbox::Bool=false)
    sf = prolong_constant(P, sc, sw)
    E = P.E
    ℓ = sf.ℓ
    lv, lvc = P.lv[ℓ+1], P.lv[ℓ]
    @assert lv.M == lvc.M "prolong_t2 needs the same spatial resolution at both levels"
    r = lvc.T == 0 ? 1 : lv.T ÷ lvc.T
    LMc, LMf = LocalModel(P, ℓ - 1), LocalModel(P, ℓ)
    LCc, LCf = LocalCost(P, LMc, sc), LocalCost(P, LMf, sf)
    N = E.sys.N
    newY = [copy(Y) for Y in LCf.Y]
    for n in 1:N
        units = [(f, i) for (f, i) in LMf.units[n] if LMf.role[f][1] == ROLE_GEN && LMf.complete[f]]
        isempty(units) && continue
        for b in 1:lvc.T
            κc = [κunit(P, LMc, LCc, f, i, b) for (f, i) in units]
            ps = κc .* [LCc.Y[f][i, b] for (f, i) in units]
            Pb = sum(ps)
            abs(Pb) < 1e-6 && continue
            Qb = LCc.r[n, b] + sum(κc .* clamp.([LCc.Y[f][i, b] for (f, i) in units], 0.0, 1.0))
            eb = Pb - max(Qb, 0.0)
            σ = ps ./ Pb
            for o in 1:r
                τ = (b - 1) * r + o
                κf = [κunit(P, LMf, LCf, f, i, τ) for (f, i) in units]
                Qτ = LCf.r[n, τ] + sum(κf .* clamp.([LCf.Y[f][i, τ] for (f, i) in units], 0.0, 1.0))
                Pτ = max(Qτ, 0.0) + eb
                for (q, (f, i)) in enumerate(units)
                    abs(κf[q]) > 1e-9 || continue
                    ω = σ[q] * Pτ / κf[q]
                    newY[f][i, τ] = clampbox ? clamp(ω, 0.0, 1.0) : ω      # clamped: the residual absorbs the rest
                end
            end
        end
    end
    # node values → coefficients for the generator fields (complete level: Uᵀ)
    for (f, s) in enumerate(E.vspecs)
        LMf.role[f][1] == ROLE_GEN || continue
        Z = fieldview(sf.v, lv, E.vspecs, f)
        Z .= cols(s.B, lv.m[f])' * newY[f]
    end
    return evaluate!(P, sf)
end

"T2 prolongation with the dispatch clamped to the unit boxes (the residual absorbs the rest)."
prolong_t2c(P::Problem, sc::State, sw::State) = prolong_t2(P, sc, sw; clampbox=true)

"Block-constant prolongation of the operation (zero details), fine weather of `sw`."
function prolong_constant(P::Problem, sc::State, sw::State)
    ℓ = sc.ℓ + 1
    k = prolong_to(sc.k, P.lk, P.trk, P.E.kspecs, ℓ - 1, ℓ)
    v = prolong_to(sc.v, P.lv, P.trv, P.E.vspecs, ℓ - 1, ℓ)
    s = State(ℓ, copy(sw.m), copy(sw.z0), sw.ws[1:ℓ], sw.xs[1:ℓ+1], k, v, -Inf, sc.parts)
    return evaluate!(P, s)
end

"""
Intra-block detail profiles of the node values of field f at a level with blocks
of r steps: a matrix (r × (support · blocks)) of deviations from the block means
(STEP fields) or from linear interpolation between block boundaries (INSTANT).
"""
function block_details(P::Problem, s::State, f::Int, r::Int)
    lv = P.lv[s.ℓ+1]
    Y = Matrix(nodevals(s.v, lv, P.E.vspecs, f))
    S, T = size(Y)
    nb = T ÷ r
    out = zeros(r, S * nb)
    inst = P.E.vspecs[f].kind == INSTANT
    q = 0
    for b in 1:nb, i in 1:S
        q += 1
        seg = Y[i, (b-1)*r+1:b*r]
        if inst
            a, c = Y[i, (b-1)*r+1], Y[i, mod(b * r, T)+1]
            out[:, q] .= seg .- [(1 - (o - 1) / r) * a + ((o - 1) / r) * c for o in 1:r]
        else
            out[:, q] .= seg .- mean(seg)
        end
    end
    return out
end

"Fraction of the second moment explained by the leading 1..kmax principal components of the columns of X."
function explained(X::AbstractMatrix; kmax::Int=4)
    C = X * X' ./ max(size(X, 2), 1)
    λ = sort(max.(eigvals(Symmetric(C)), 0.0); rev=true)
    tot = sum(λ)
    tot == 0 && return zeros(min(kmax, length(λ)))
    return cumsum(λ)[1:min(kmax, length(λ))] ./ tot
end
