# Fields, level layouts and the level transfer (splitting) operators.
#
# A level is a pair (M_ℓ global Laplace modes, step length h_ℓ hours). The
# state of a field at level ℓ is a matrix of mode coefficients × time steps,
# stored contiguously in a flat vector. Going from level ℓ to ℓ+1 adds
#   * time details of the existing modes (if h_ℓ₊₁ < h_ℓ), and
#   * all time steps of the newly added modes (if M_ℓ₊₁ > M_ℓ).
# Step fields are block averaged in time (details are orthonormal Helmert
# contrasts within each block); instant fields are decimated (details are the
# deviations of the interior instants from linear interpolation).

@enum FieldKind KFIELD STEP INSTANT

struct FieldSpec
    name::Symbol
    kind::FieldKind
    B::ModeBasis
    lo::Vector{Float64}   # node-level box on the support (may be ±Inf)
    hi::Vector{Float64}
end

struct Layout
    M::Int
    h::Int
    T::Int
    m::Vector{Int}
    off::Vector{Int}
    n::Int
end

ntime(T::Int, s::FieldSpec) = s.kind == KFIELD ? 1 : T
ntime(lay::Layout, s::FieldSpec) = ntime(lay.T, s)

function Layout(specs::Vector{FieldSpec}, M::Int, h::Int)
    @assert 8760 % h == 0
    T = 8760 ÷ h
    m = [modes_at(s.B, M) for s in specs]
    off = zeros(Int, length(specs))
    n = 0
    for (f, s) in enumerate(specs)
        off[f] = n
        n += m[f] * ntime(T, s)
    end
    return Layout(M, h, T, m, off, n)
end

fieldview(z::AbstractVector, lay::Layout, specs, f::Int) =
    reshape(view(z, lay.off[f]+1:lay.off[f]+lay.m[f]*ntime(lay, specs[f])), lay.m[f], ntime(lay, specs[f]))

iscomplete(lay::Layout, specs, f::Int) = lay.m[f] == nmodes(specs[f].B)

"Node values of field `f`: U[:, 1:m] * Z."
function nodevals(z::AbstractVector, lay::Layout, specs, f::Int)
    return cols(specs[f].B, lay.m[f]) * fieldview(z, lay, specs, f)
end

"Orthonormal within-block contrasts (r × r-1), orthogonal to the constant."
function helmert(r::Int)
    H = zeros(r, r - 1)
    for k in 1:r-1
        H[1:k, k] .= 1.0
        H[k+1, k] = -k
        H[:, k] ./= sqrt(k * (k + 1))
    end
    return H
end

"""
Transfer between a coarse and a fine layout of the same fields.

Detail coordinates of field f are stored as `[ξT (m_c, r-1, T_c); ξS (m_f-m_c, r, T_c)]`
for time-dependent fields and `ξS (m_f-m_c)` for k fields. `dblock[i]` is the
coarse time step of detail coordinate i (0 for k fields).
"""
struct Transfer
    r::Int
    H::Matrix{Float64}
    doff::Vector{Int}
    dT::Vector{Int}       # length of the ξT part per field
    n::Int
    dblock::Vector{Int32}
end

function Transfer(specs, lc::Layout, lf::Layout)
    @assert lc.h % lf.h == 0 && all(lf.m .>= lc.m)
    r = lc.h ÷ lf.h
    doff = zeros(Int, length(specs))
    dT = zeros(Int, length(specs))
    dblock = Int32[]
    n = 0
    for (f, s) in enumerate(specs)
        doff[f] = n
        mc, mf = lc.m[f], lf.m[f]
        if s.kind == KFIELD
            append!(dblock, zeros(Int32, mf - mc))
            n += mf - mc
        else
            dT[f] = mc * (r - 1) * lc.T
            for j in 1:lc.T, o in 1:r-1, i in 1:mc
                push!(dblock, j)
            end
            for j in 1:lc.T, o in 1:r, i in 1:mf-mc
                push!(dblock, j)
            end
            n += dT[f] + (mf - mc) * r * lc.T
        end
    end
    return Transfer(r, helmert(r), doff, dT, n, dblock)
end

function detail_views(ξ::AbstractVector, tr::Transfer, lc::Layout, lf::Layout, specs, f::Int)
    mc, mf, r = lc.m[f], lf.m[f], tr.r
    o = tr.doff[f]
    if specs[f].kind == KFIELD
        return nothing, view(ξ, o+1:o+mf-mc)
    end
    ξT = reshape(view(ξ, o+1:o+tr.dT[f]), mc, r - 1, lc.T)
    ξS = reshape(view(ξ, o+tr.dT[f]+1:o+tr.dT[f]+(mf-mc)*r*lc.T), mf - mc, r, lc.T)
    return ξT, ξS
end

"Split a fine state into its coarse state and detail coordinates."
function split_state(tr::Transfer, lc::Layout, lf::Layout, specs, zf::AbstractVector)
    zc = zeros(lc.n)
    ξ = zeros(tr.n)
    r, H = tr.r, tr.H
    for (f, s) in enumerate(specs)
        mc, mf = lc.m[f], lf.m[f]
        Zf = fieldview(zf, lf, specs, f)
        Zc = fieldview(zc, lc, specs, f)
        ξT, ξS = detail_views(ξ, tr, lc, lf, specs, f)
        if s.kind == KFIELD
            Zc[:, 1] .= Zf[1:mc, 1]
            ξS .= Zf[mc+1:mf, 1]
            continue
        end
        Tf = lf.T
        for j in 1:lc.T
            t0 = (j - 1) * r
            for i in 1:mc
                if s.kind == STEP
                    acc = 0.0
                    for o in 1:r
                        acc += Zf[i, t0+o]
                    end
                    Zc[i, j] = acc / r
                    for k in 1:r-1
                        acc = 0.0
                        for o in 1:r
                            acc += H[o, k] * Zf[i, t0+o]
                        end
                        ξT[i, k, j] = acc
                    end
                else
                    a, b = Zf[i, t0+1], Zf[i, mod(t0 + r, Tf)+1]
                    Zc[i, j] = a
                    for o in 1:r-1
                        ξT[i, o, j] = Zf[i, t0+1+o] - ((1 - o / r) * a + (o / r) * b)
                    end
                end
            end
            for i in mc+1:mf, o in 1:r
                ξS[i-mc, o, j] = Zf[i, t0+o]
            end
        end
    end
    return zc, ξ
end

"Inverse of `split_state`."
function merge_state(tr::Transfer, lc::Layout, lf::Layout, specs, zc::AbstractVector, ξ::AbstractVector)
    zf = zeros(lf.n)
    r, H = tr.r, tr.H
    for (f, s) in enumerate(specs)
        mc, mf = lc.m[f], lf.m[f]
        Zf = fieldview(zf, lf, specs, f)
        Zc = fieldview(zc, lc, specs, f)
        ξT, ξS = detail_views(ξ, tr, lc, lf, specs, f)
        if s.kind == KFIELD
            Zf[1:mc, 1] .= Zc[:, 1]
            Zf[mc+1:mf, 1] .= ξS
            continue
        end
        Tc = lc.T
        for j in 1:Tc
            t0 = (j - 1) * r
            for i in 1:mc
                if s.kind == STEP
                    for o in 1:r
                        acc = Zc[i, j]
                        for k in 1:r-1
                            acc += H[o, k] * ξT[i, k, j]
                        end
                        Zf[i, t0+o] = acc
                    end
                else
                    a, b = Zc[i, j], Zc[i, mod(j, Tc)+1]
                    Zf[i, t0+1] = a
                    for o in 1:r-1
                        Zf[i, t0+1+o] = (1 - o / r) * a + (o / r) * b + ξT[i, o, j]
                    end
                end
            end
            for i in mc+1:mf, o in 1:r
                Zf[i, t0+o] = ξS[i-mc, o, j]
            end
        end
    end
    return zf
end

"Restrict a state from level `from` down to level `to` (to ≤ from)."
function restrict_to(z::AbstractVector, lays::Vector{Layout}, trs, specs, from::Int, to::Int)
    for ℓ in from:-1:to+1
        z, _ = split_state(trs[ℓ], lays[ℓ], lays[ℓ+1], specs, z)  # layouts are 1-based: level ℓ ↦ lays[ℓ+1]
    end
    return z
end

"Prolongate (zero details) from level `from` up to level `to`."
function prolong_to(z::AbstractVector, lays::Vector{Layout}, trs, specs, from::Int, to::Int)
    for ℓ in from+1:to
        z = merge_state(trs[ℓ], lays[ℓ], lays[ℓ+1], specs, z, zeros(trs[ℓ].n))
    end
    return z
end
