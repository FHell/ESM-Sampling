# Hard constraints Ω_ℓ(k) × K_ℓ of a level, as coordinate boxes plus sparse
# linear rows a_c·q ≤ b_c on the concatenated coordinates q = [k; v; extra].

struct Constraints
    n::Int
    lo::Vector{Float64}
    hi::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int}    # nc × n
    At::SparseMatrixCSC{Float64,Int}   # n × nc (column c is row c of A)
    b::Vector{Float64}
end

nrows(c::Constraints) = length(c.b)

mutable struct RowBuilder
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
    b::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
end

RowBuilder(n) = RowBuilder(Int[], Int[], Float64[], Float64[], fill(-Inf, n), fill(Inf, n))

"Add a ≤ row Σ vals[i] q[idx[i]] ≤ rhs; single-entry rows become coordinate boxes."
function add_row!(rb::RowBuilder, idx, vals, rhs; drop=1e-12)
    keep = findall(v -> abs(v) > drop, vals)
    isempty(keep) && (@assert rhs >= -1e-9 "infeasible constant row"; return)
    if length(keep) == 1
        j, a = idx[keep[1]], vals[keep[1]]
        a > 0 ? (rb.hi[j] = min(rb.hi[j], rhs / a)) : (rb.lo[j] = max(rb.lo[j], rhs / a))
        return
    end
    c = length(rb.b) + 1
    for i in keep
        push!(rb.I, c)
        push!(rb.J, idx[i])
        push!(rb.V, vals[i])
    end
    push!(rb.b, rhs)
end

function finish(rb::RowBuilder, n)
    A = sparse(rb.I, rb.J, rb.V, length(rb.b), n)
    return Constraints(n, rb.lo, rb.hi, A, sparse(A'), rb.b)
end

"Outer box of the mode coefficients of a field whose node values lie in [lo, hi]."
function coefficient_box(U::AbstractMatrix, lo, hi)
    m = size(U, 2)
    clo, chi = zeros(m), zeros(m)
    for j in 1:m, i in axes(U, 1)
        a, b = U[i, j] * lo[i], U[i, j] * hi[i]
        clo[j] += min(a, b)
        chi[j] += max(a, b)
    end
    return clo, chi
end

"""
    build_constraints(E, lk, lv; nextra=0)

Hard constraints of a level on q = [k; v; extra]. Complete fields get their
exact node-level constraints, incomplete fields an outer coefficient box.
"""
function build_constraints(E::EnergyModel, lk::Layout, lv::Layout; nextra::Int=0)
    n = lk.n + lv.n + nextra
    rb = RowBuilder(n)
    E.regularized && return finish(rb, n)   # all constraints are priced
    sys = E.sys

    function field_boxes!(lay, specs, f, base)
        s = specs[f]
        all(isinf, s.lo) && all(isinf, s.hi) && return
        m = lay.m[f]
        U = view(s.B.U, :, 1:m)
        nt = ntime(lay, s)
        if iscomplete(lay, specs, f)
            for t in 1:nt
                idx = base .+ lay.off[f] .+ (t - 1) * m .+ (1:m)
                for i in axes(U, 1)
                    isfinite(s.hi[i]) && add_row!(rb, idx, U[i, :], s.hi[i])
                    isfinite(s.lo[i]) && add_row!(rb, idx, -U[i, :], -s.lo[i])
                end
            end
        else
            clo, chi = coefficient_box(U, s.lo, s.hi)
            for t in 1:nt, j in 1:m
                q = base + lay.off[f] + (t - 1) * m + j
                rb.lo[q] = max(rb.lo[q], clo[j])
                rb.hi[q] = min(rb.hi[q], chi[j])
            end
        end
    end

    for f in eachindex(E.kspecs)
        field_boxes!(lk, E.kspecs, f, 0)
    end
    vb = lk.n
    for f in eachindex(E.vspecs)
        field_boxes!(lv, E.vspecs, f, vb)
    end

    # line limits (θ)
    f = E.fθ
    m = lv.m[f]
    U = view(E.vspecs[f].B.U, :, 1:m)
    if iscomplete(lv, E.vspecs, f)
        for t in 1:lv.T
            idx = vb .+ lv.off[f] .+ (t - 1) * m .+ (1:m)
            for e in eachindex(sys.line_s)
                a = sys.line_b[e] .* (U[sys.line_bus0[e], :] .- U[sys.line_bus1[e], :])
                add_row!(rb, idx, a, sys.line_s[e])
                add_row!(rb, idx, -a, sys.line_s[e])
            end
        end
    else
        for t in 1:lv.T, j in 1:m
            q = vb + lv.off[f] + (t - 1) * m + j
            rb.lo[q], rb.hi[q] = E.θlo[j], E.θhi[j]
        end
    end

    # storage: energy may be lost but never created
    Δt = Float64(lv.h)
    for s in E.stors
        iscomplete(lv, E.vspecs, s.fp) || continue
        m = lv.m[s.fp]
        U = E.vspecs[s.fp].B.U
        T = lv.T
        for t in 1:T, i in eachindex(s.bus)
            t1 = mod(t, T) + 1
            ie1 = vb + lv.off[s.fe] + (t1 - 1) * m
            ie0 = vb + lv.off[s.fe] + (t - 1) * m
            ip = vb + lv.off[s.fp] + (t - 1) * m
            im = vb + lv.off[s.fm] + (t - 1) * m
            u = U[i, :]
            c = s.hours[i] / Δt
            idx = vcat(ie1 .+ (1:m), ie0 .+ (1:m), ip .+ (1:m), im .+ (1:m))
            vals = vcat(c .* u, -c .* u, -s.eta_in[i] .* u, u ./ s.eta_out[i])
            add_row!(rb, idx, vals, s.qt[i])
        end
    end
    return finish(rb, n)
end

"Feasibility of q under the constraints, with a relative tolerance."
function isfeasible(c::Constraints, q::AbstractVector; tol=1e-9)
    @inbounds for i in 1:c.n
        x = q[i]
        (x < c.lo[i] - tol * (1 + abs(c.lo[i])) || x > c.hi[i] + tol * (1 + abs(c.hi[i]))) && return false
    end
    nrows(c) == 0 && return true
    Aq = c.A * view(q, 1:c.n)
    @inbounds for j in eachindex(c.b)
        Aq[j] > c.b[j] + tol * (1 + abs(c.b[j])) && return false
    end
    return true
end
