# Reflective HMC on a polytope (§5.1).
#
# The position drift of each leapfrog step is the exact billiard flow inside
# {lo ≤ q ≤ hi, A q ≤ b}: free motion with specular reflections (in the metric
# of the mass matrix) at the walls. The flow is volume preserving and
# reversible, so the usual Metropolis correction keeps the target invariant.
# Wall hits are processed as events in a priority queue; positions are synced
# lazily, so a reflection costs O(nnz of the wall · log n) and the whole drift
# stays local.

mutable struct EventHeap
    t::Vector{Float64}
    id::Vector{Int}
    ver::Vector{Int}
end
EventHeap() = EventHeap(Float64[], Int[], Int[])
Base.isempty(h::EventHeap) = isempty(h.t)
function Base.empty!(h::EventHeap)
    empty!(h.t); empty!(h.id); empty!(h.ver)
    h
end

function heap_push!(h::EventHeap, t, id, ver)
    push!(h.t, t); push!(h.id, id); push!(h.ver, ver)
    i = length(h.t)
    while i > 1
        p = i ÷ 2
        h.t[p] <= h.t[i] && break
        heap_swap!(h, i, p)
        i = p
    end
end

function heap_swap!(h, i, j)
    h.t[i], h.t[j] = h.t[j], h.t[i]
    h.id[i], h.id[j] = h.id[j], h.id[i]
    h.ver[i], h.ver[j] = h.ver[j], h.ver[i]
end

function heap_pop!(h::EventHeap)
    t, id, ver = h.t[1], h.id[1], h.ver[1]
    last = length(h.t)
    heap_swap!(h, 1, last)
    pop!(h.t); pop!(h.id); pop!(h.ver)
    n = length(h.t)
    i = 1
    while true
        l, r = 2i, 2i + 1
        s = i
        l <= n && h.t[l] < h.t[s] && (s = l)
        r <= n && h.t[r] < h.t[s] && (s = r)
        s == i && break
        heap_swap!(h, i, s)
        i = s
    end
    return t, id, ver
end

struct BilliardWS
    v::Vector{Float64}
    tlast::Vector{Float64}
    ver::Vector{Int}
    mark::Vector{Int}
    heap::EventHeap
end
BilliardWS(C::Constraints) = BilliardWS(zeros(C.n), zeros(C.n), zeros(Int, C.n + nrows(C)), zeros(Int, nrows(C)), EventHeap())

@inline qat(q, ws, i, t) = q[i] + ws.v[i] * (t - ws.tlast[i])

@inline function sync!(q, ws, i, t)
    q[i] += ws.v[i] * (t - ws.tlast[i])
    ws.tlast[i] = t
end

function schedule_box!(q, ws, C, i, t)
    ws.ver[i] += 1
    vi = ws.v[i]
    if vi > 0 && isfinite(C.hi[i])
        heap_push!(ws.heap, t + max(C.hi[i] - qat(q, ws, i, t), 0.0) / vi, i, ws.ver[i])
    elseif vi < 0 && isfinite(C.lo[i])
        heap_push!(ws.heap, t + max(qat(q, ws, i, t) - C.lo[i], 0.0) / (-vi), i, ws.ver[i])
    end
end

function schedule_row!(q, ws, C, c, t)
    id = C.n + c
    ws.ver[id] += 1
    At = C.At
    s = C.b[c]
    rate = 0.0
    mag = 0.0
    @inbounds for k in At.colptr[c]:At.colptr[c+1]-1
        j = At.rowval[k]
        a = At.nzval[k]
        s -= a * qat(q, ws, j, t)
        rate += a * ws.v[j]
        mag += abs(a * ws.v[j])
    end
    if rate > 1e-12 * mag && rate > 0
        heap_push!(ws.heap, t + max(s, 0.0) / rate, id, ws.ver[id])
    end
end

"Billiard drift of duration τ. Returns (ok, number of reflections)."
function drift!(q, p, minv, τ, C::Constraints, ws::BilliardWS; max_events=50 * C.n + 1000)
    n, nc = C.n, nrows(C)
    @. ws.v = minv * p
    fill!(ws.tlast, 0.0)
    empty!(ws.heap)
    for i in 1:n
        schedule_box!(q, ws, C, i, 0.0)
    end
    for c in 1:nc
        schedule_row!(q, ws, C, c, 0.0)
    end
    A, At = C.A, C.At
    nev = 0
    stamp = 0
    while !isempty(ws.heap)
        t, id, ver = heap_pop!(ws.heap)
        t > τ && break
        ver == ws.ver[id] || continue
        nev += 1
        nev > max_events && return false, nev
        if id <= n
            i = id
            sync!(q, ws, i, t)
            q[i] = ws.v[i] > 0 ? C.hi[i] : C.lo[i]
            p[i] = -p[i]
            ws.v[i] = -ws.v[i]
            schedule_box!(q, ws, C, i, t)
            for k in A.colptr[i]:A.colptr[i+1]-1
                schedule_row!(q, ws, C, A.rowval[k], t)
            end
        else
            c = id - n
            av, ama = 0.0, 0.0
            for k in At.colptr[c]:At.colptr[c+1]-1
                j = At.rowval[k]
                sync!(q, ws, j, t)
                a = At.nzval[k]
                av += a * ws.v[j]
                ama += a * a * minv[j]
            end
            coef = 2av / ama
            stamp += 1
            for k in At.colptr[c]:At.colptr[c+1]-1
                j = At.rowval[k]
                p[j] -= coef * At.nzval[k]
                ws.v[j] = minv[j] * p[j]
            end
            for k in At.colptr[c]:At.colptr[c+1]-1
                j = At.rowval[k]
                schedule_box!(q, ws, C, j, t)
                for kk in A.colptr[j]:A.colptr[j+1]-1
                    c2 = A.rowval[kk]
                    if ws.mark[c2] != stamp
                        ws.mark[c2] = stamp
                        schedule_row!(q, ws, C, c2, t)
                    end
                end
            end
        end
    end
    for i in 1:n
        sync!(q, ws, i, τ)
    end
    return true, nev
end

mutable struct HMCState
    ε::Float64
    nleap::Int
    minv::Vector{Float64}
    # dual averaging
    δ::Float64
    μ::Float64
    Hbar::Float64
    logεbar::Float64
    it::Int
    # Welford accumulators for the mass matrix
    wn::Int
    wmean::Vector{Float64}
    wM2::Vector{Float64}
    # statistics
    nacc::Int
    ntot::Int
    nref::Int
end

HMCState(ε, nleap, minv; δ=0.65) =
    HMCState(ε, nleap, copy(minv), δ, log(10ε), 0.0, log(ε), 0, 0, zeros(length(minv)), zeros(length(minv)), 0, 0, 0)

"""
    hmc_step!(rng, q, potential, C, st, ws)

One HMC transition; `potential(q)` returns `(U, ∇U)` (U = Inf outside the
support). Returns the acceptance probability.
"""
function hmc_step!(rng, q::Vector{Float64}, potential, C::Constraints, st::HMCState, ws::BilliardWS)
    minv = st.minv
    p = randn(rng, length(q)) ./ sqrt.(minv)
    U0, g = potential(q)
    H0 = U0 + 0.5 * sum(minv .* p .^ 2)
    qn, pn = copy(q), copy(p)
    L = rand(rng, max(1, st.nleap ÷ 2):st.nleap)
    U1 = Inf
    ok = true
    for _ in 1:L
        @. pn -= 0.5 * st.ε * g
        ok, nev = drift!(qn, pn, minv, st.ε, C, ws)
        st.nref += nev
        ok || break
        U1, g = potential(qn)
        isfinite(U1) || (ok = false; break)
        @. pn -= 0.5 * st.ε * g
    end
    st.ntot += 1
    (ok && isfeasible(C, qn; tol=1e-7)) || return 0.0
    H1 = U1 + 0.5 * sum(minv .* pn .^ 2)
    acc = isfinite(H1) ? min(1.0, exp(H0 - H1)) : 0.0
    if rand(rng) < acc
        q .= qn
        st.nacc += 1
    end
    return acc
end

"Dual-averaging step size adaptation (Hoffman & Gelman 2014)."
function adapt_stepsize!(st::HMCState, acc)
    st.it += 1
    γ, t0, κ = 0.05, 10.0, 0.75
    η = 1 / (st.it + t0)
    st.Hbar = (1 - η) * st.Hbar + η * (st.δ - acc)
    logε = st.μ - sqrt(st.it) / γ * st.Hbar
    w = st.it^(-κ)
    st.logεbar = w * logε + (1 - w) * st.logεbar
    st.ε = exp(logε)
end

finish_stepsize!(st::HMCState) = (st.ε = exp(st.logεbar))

function accumulate_mass!(st::HMCState, q)
    st.wn += 1
    d = q .- st.wmean
    st.wmean .+= d ./ st.wn
    st.wM2 .+= d .* (q .- st.wmean)
end

"Update the diagonal inverse mass from the window, regularized towards the old one."
function update_mass!(st::HMCState)
    n = st.wn
    n < 10 && return
    var = st.wM2 ./ (n - 1)
    st.minv .= (n / (n + 5)) .* max.(var, 1e-12 .* st.minv) .+ (5 / (n + 5)) .* 1e-3 .* st.minv
    st.wn = 0
    fill!(st.wmean, 0.0)
    fill!(st.wM2, 0.0)
    # restart dual averaging around the current step size
    st.μ = log(10st.ε)
    st.Hbar = 0.0
    st.it = 0
end

"""
    descend!(q, potential, C, ws; iters, lr)

Reflective descent towards the mode, used to pre-equilibrate a chain: the
position moves along a normalized (Adam-like) negative gradient for unit time
with billiard reflections, so every iterate stays inside the polytope.
Coordinates with `mask = 0` are held fixed.
"""
function descend!(q, potential, C::Constraints, ws::BilliardWS; iters=200, lr=1e-2, β1=0.9, β2=0.999,
                  mask=ones(length(q)))
    U, g = potential(q)
    m1, m2 = zeros(length(q)), zeros(length(q))
    scale = max.(abs.(q), 1e-3)
    ones_ = ones(length(q))
    for it in 1:iters
        @. m1 = β1 * m1 + (1 - β1) * g
        @. m2 = β2 * m2 + (1 - β2) * g^2
        p = @. -lr * mask * scale * (m1 / (1 - β1^it)) / (sqrt(m2 / (1 - β2^it)) + 1e-12)
        qn = copy(q)
        ok, _ = drift!(qn, p, ones_, 1.0, C, ws)
        ok || (lr /= 2; continue)
        Un, gn = potential(qn)
        if isfinite(Un) && Un <= U
            q .= qn
            U, g = Un, gn
            lr *= 1.05
        else
            lr /= 2
        end
    end
    return U
end
