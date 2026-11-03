# Balance-preserving pair moves for the fast block (the energy-model version of
# the toy's slack-node pairs; docs/toy_results.md).
#
# At a time step τ of a spatially complete level, pick unit 1 (generator, storage
# charge or discharge) at node n and unit 2 at node m. Shift t MW of injection from
# unit 2 to unit 1:  Δω₁ = t/κ₁, Δω₂ = −t/κ₂ with κ the injection per unit of
# utilization (āk for generators, −k for charging, +k for discharging). If n ≠ m
# the flow is re-routed through the AC network, Δθ = t·L⁺(e_n − e_m), so every
# nodal residual stays unchanged. Each move touches only the coefficients of step
# τ, and only the term of step τ changes, so all steps can be updated in parallel.
# t is drawn by slice sampling along the line (exact for the conditional), with
# state-independent pair selection, so every move is π-reversible.

struct PairSet
    units::Vector{Vector{NTuple{3,Int}}}     # per node: (kind, group, index); kind 1 = gen, 2 = charge, 3 = discharge
    Lplus::Matrix{Float64}                   # pseudo-inverse of the susceptance Laplacian (per AC component)
    comp::Vector{Int}
end

function PairSet(E::EnergyModel)
    sys = E.sys
    N = sys.N
    units = [NTuple{3,Int}[] for _ in 1:N]
    for (gi, g) in enumerate(E.gens), (i, n) in enumerate(g.bus)
        g.vf > 0 || continue
        push!(units[n], (1, gi, i))
    end
    for (si, s) in enumerate(E.stors), (i, n) in enumerate(s.bus)
        s.fp > 0 || continue
        push!(units[n], (2, si, i)); push!(units[n], (3, si, i))
    end
    L = zeros(N, N)
    for (a, b, w) in zip(sys.line_bus0, sys.line_bus1, sys.line_b)
        L[a, a] += w; L[b, b] += w; L[a, b] -= w; L[b, a] -= w
    end
    Zc = component_indicators(N, sys.line_bus0, sys.line_bus1)
    comp = [argmax(abs.(Zc[i, :])) for i in 1:N]
    Lplus = pinv(L + Zc * Zc') - Zc * Zc'          # L⁺ on the complement of the component constants
    return PairSet(units, Lplus, comp)
end

"Injection per unit utilization κ and the (field, support index) of a unit at step τ."
function unit_info(E::EnergyModel, ctx::SweepCtx, ws::StepWS, u::NTuple{3,Int})
    kind, gi, i = u
    if kind == 1
        g = E.gens[gi]
        A = g.wf > 0 ? clamp(ws.x[g.wf][i], 0.0, 1.0) : g.avail[i]
        return A * ctx.kgen[gi][i], g.vf, i
    end
    s = E.stors[gi]
    k = ctx.kst[gi][i]
    return kind == 2 ? (-k, s.fp, i) : (k, s.fm, i)
end

"Slice sampling of t for the move v[idx] .+= t .* coef."
function slice_vec!(rng, v, idx::Vector{Int}, coef::Vector{Float64}, lf, w; maxsteps=16)
    base = v[idx]
    setto!(t) = (@inbounds for q in eachindex(idx); v[idx[q]] = base[q] + t * coef[q]; end)
    logy = lf() - randexp(rng)
    L = -w * rand(rng)
    R = L + w
    J = floor(Int, maxsteps * rand(rng))
    K = maxsteps - 1 - J
    setto!(L)
    while J > 0 && lf() > logy
        L -= w; setto!(L); J -= 1
    end
    setto!(R)
    while K > 0 && lf() > logy
        R += w; setto!(R); K -= 1
    end
    for _ in 1:200
        t = L + rand(rng) * (R - L)
        setto!(t)
        lf() > logy && return abs(t)
        t < 0 ? (L = t) : (R = t)
    end
    setto!(0.0)
    return 0.0
end

"""
    pair_sweep!(rng, P, G, s, PS, widths, stats; nmoves, p_same, κmin)

`nmoves` pair moves per time step of the level of `s` (weather and capacities
fixed), in parallel over time steps. With probability `p_same` the partner is at
the same node, otherwise at a uniformly chosen node of the same AC component
(with flow re-routing). `widths[1:2]` are slice widths in MW for same-node and
cross-node pairs; `stats[fam, 1:2]` accumulate (count, Σ|t|).
"""
function pair_sweep!(rng, P::Problem, G::GibbsState, s::State, PS::PairSet, widths, stats;
                     nmoves::Int=P.E.sys.N, p_same=0.5, κmin=1.0, iweight::Float64=0.0, ε::Float64=0.05,
                     p_single::Float64=0.0, p_theta::Float64=0.0)
    ℓ = s.ℓ
    E = P.E
    lv = P.lv[ℓ+1]
    @assert all(iscomplete(lv, E.vspecs, f) for f in eachindex(E.vspecs)) "pair moves need a spatially complete level"
    T = lv.T
    N = E.sys.N
    ctx = SweepCtx(P, s.k; ℓ=ℓ)
    x = s.xs[end]
    Uθ = E.vspecs[E.fθ].B.U
    mθ = lv.m[E.fθ]
    seeds = rand(rng, UInt64, T)
    nfam = size(stats, 1)
    loc = [zeros(size(stats)) for _ in 1:T]
    Threads.@threads :static for τ in 1:T
        r = Xoshiro(seeds[τ])
        ws = G.wss[Threads.threadid()]
        term() = step_term(P, ℓ, ctx, ws, s.v, x, s.z0, s.m, τ)
        for _ in 1:nmoves
            n = rand(r, 1:N)
            uu = rand(r)
            if uu < p_single + p_theta
                # node-level moves: one unit at node n (changes only r_n), or the potential of node n
                if uu < p_single
                    isempty(PS.units[n]) && continue
                    term()
                    κ1, f1, i1 = unit_info(E, ctx, ws, rand(r, PS.units[n]))
                    abs(κ1) < κmin && continue
                    m1 = lv.m[f1]; o1 = lv.off[f1] + (τ - 1) * m1
                    idx = collect(o1+1:o1+m1); coef = E.vspecs[f1].B.U[i1, 1:m1] ./ κ1
                    fam = 3
                else
                    nrm = sqrt(sum(abs2, view(Uθ, n, 1:mθ)))
                    nrm > 1e-8 || continue                # node without own potential (single-node AC component)
                    oθ = lv.off[E.fθ] + (τ - 1) * mθ
                    idx = collect(oθ+1:oθ+mθ); coef = Uθ[n, 1:mθ] ./ nrm
                    fam = 4
                end
                dt = slice_vec!(r, s.v, idx, coef, () -> -term(), widths[fam])
                loc[τ][fam, 1] += 1; loc[τ][fam, 2] += dt
                continue
            end
            isempty(PS.units[n]) && continue
            same = rand(r) < p_same
            m = same ? n : rand(r, 1:N)
            (m == n && !same) && continue
            PS.comp[m] == PS.comp[n] || continue
            isempty(PS.units[m]) && continue
            term()                                   # fills ws.x (weather) and ws.y (node values) at τ
            u1 = rand(r, PS.units[n])
            if iweight > 0                           # partner weighted towards interior (marginal) units
                wts = [unit_weight(E, ws, u, ε) for u in PS.units[m]]
                u1 in PS.units[m] && (wts[findfirst(==(u1), PS.units[m])] = 0.0)
                sum(wts) > 0 || continue
                u2 = PS.units[m][sample_index(r, wts)]
                qf = pair_q(E, ws, PS, n, m, u1, u2, ε)
            else
                u2 = rand(r, PS.units[m])
            end
            u1 == u2 && continue
            κ1, f1, i1 = unit_info(E, ctx, ws, u1)
            κ2, f2, i2 = unit_info(E, ctx, ws, u2)
            (abs(κ1) < κmin || abs(κ2) < κmin) && continue
            idx = Int[]; coef = Float64[]
            m1, m2 = lv.m[f1], lv.m[f2]
            o1 = lv.off[f1] + (τ - 1) * m1
            o2 = lv.off[f2] + (τ - 1) * m2
            U1, U2 = E.vspecs[f1].B.U, E.vspecs[f2].B.U
            for j in 1:m1
                push!(idx, o1 + j); push!(coef, U1[i1, j] / κ1)
            end
            for j in 1:m2
                push!(idx, o2 + j); push!(coef, -U2[i2, j] / κ2)
            end
            if n != m
                dθ = PS.Lplus[:, n] .- PS.Lplus[:, m]
                oθ = lv.off[E.fθ] + (τ - 1) * mθ
                cθ = view(Uθ, :, 1:mθ)' * dθ
                for j in 1:mθ
                    push!(idx, oθ + j); push!(coef, cθ[j])
                end
            end
            # merge duplicate indices (same field for both units)
            if f1 == f2
                d = Dict{Int,Float64}()
                for q in eachindex(idx); d[idx[q]] = get(d, idx[q], 0.0) + coef[q]; end
                idx = collect(keys(d)); coef = [d[k] for k in idx]
            end
            fam = same ? 1 : 2
            base = s.v[idx]
            dt = slice_vec!(r, s.v, idx, coef, () -> -term(), widths[fam])
            if iweight > 0
                term()                               # node values at the new point
                qr = pair_q(E, ws, PS, n, m, u1, u2, ε)
                if rand(r) >= min(1.0, qr / qf)      # selection probabilities differ: MH correction
                    s.v[idx] .= base
                    dt = 0.0
                end
            end
            loc[τ][fam, 1] += 1; loc[τ][fam, 2] += dt
        end
    end
    stats .+= sum(loc)
    return evaluate!(P, s)
end

"Selection weight of a unit: ε + 1 if its utilization at τ is interior (a marginal unit)."
function unit_weight(E::EnergyModel, ws::StepWS, u::NTuple{3,Int}, ε)
    kind, gi, i = u
    ω = kind == 1 ? ws.y[E.gens[gi].vf][i] : ws.y[kind == 2 ? E.stors[gi].fp : E.stors[gi].fm][i]
    return ε + Float64(0.02 < ω < 0.98)
end

function sample_index(rng, w)
    u = rand(rng) * sum(w)
    acc = 0.0
    for i in eachindex(w)
        acc += w[i]
        u <= acc && return i
    end
    return length(w)
end

"""
Probability (up to the node-choice factors, which are symmetric) of selecting the
unordered pair {u1 at n, u2 at m}: either u1 first (uniform at n) and u2 as the
weighted partner at m, or u2 first (uniform at m) and u1 as the weighted partner at n.
"""
function pair_q(E::EnergyModel, ws::StepWS, PS::PairSet, n, m, u1, u2, ε)
    function part(first, nf, partner, np)
        wts = [unit_weight(E, ws, u, ε) for u in PS.units[np]]
        first in PS.units[np] && (wts[findfirst(==(first), PS.units[np])] = 0.0)
        tot = sum(wts)
        tot > 0 || return 0.0
        return (1 / length(PS.units[nf])) * unit_weight(E, ws, partner, ε) / tot
    end
    return part(u1, n, u2, m) + part(u2, m, u1, n)
end

"""
    storage_shift_sweep!(rng, P, G, s, PS, width, stats; wmax, nmoves)

Balance-preserving intertemporal storage moves at a spatially complete level:
storage unit u at node n charges δ MW more at step τ₁, carries the energy
η⁺δΔt through the instants τ₁+1 … τ₂ and discharges η⁺η⁻δ MW more at τ₂; partner
units at node n inject δ more at τ₁ and η⁺η⁻δ less at τ₂. The storage dynamics
and all nodal residuals are unchanged; only steps τ₁ … τ₂ are affected. Windows of
length ≤ `wmax` are drawn inside blocks of length 2·wmax, and blocks of one parity
class are updated in parallel. Selection is state-independent.
"""
function storage_shift_sweep!(rng, P::Problem, G::GibbsState, s::State, PS::PairSet, width::Ref{Float64}, stats;
                              wmax::Int=14, nmoves::Int=8, κmin=1.0)
    ℓ = s.ℓ
    E = P.E
    lv = P.lv[ℓ+1]
    T = lv.T
    Δt = Float64(lv.h)
    ctx = SweepCtx(P, s.k; ℓ=ℓ)
    x = s.xs[end]
    B = 2wmax
    nb = cld(T, B)
    stores = [(n, u) for n in eachindex(PS.units) for u in PS.units[n] if u[1] == 2]   # (node, charge unit)
    for parity in shuffle(rng, [0, 1])
        blocks = [b for b in 0:nb-1 if b % 2 == parity && !(isodd(nb) && b == nb - 1 && parity == 0)]
        parity == 1 && isodd(nb) && push!(blocks, nb - 1)
        seeds = rand(rng, UInt64, length(blocks))
        loc = [zeros(2) for _ in blocks]
        Threads.@threads :static for q in eachindex(blocks)
            r = Xoshiro(seeds[q])
            ws = G.wss[Threads.threadid()]
            a0 = blocks[q] * B + 1
            b0 = min(a0 + B - 1, T)
            b0 - a0 < 1 && continue
            for _ in 1:nmoves
                n, (_, si, i) = rand(r, stores)
                st = E.stors[si]
                k = ctx.kst[si][i]
                (k < κmin || st.hours[i] <= 0) && continue
                τ1 = rand(r, a0:b0-1)
                τ2 = rand(r, τ1+1:min(τ1 + wmax, b0))
                p1 = rand(r, PS.units[n]); p2 = rand(r, PS.units[n])
                (p1[2] == si && p1[1] != 1) && continue        # partners must not be this storage unit
                (p2[2] == si && p2[1] != 1) && continue
                step_term(P, ℓ, ctx, ws, s.v, x, s.z0, s.m, τ1)
                κp1, fp1, ip1 = unit_info(E, ctx, ws, p1)
                step_term(P, ℓ, ctx, ws, s.v, x, s.z0, s.m, τ2)
                κp2, fp2, ip2 = unit_info(E, ctx, ws, p2)
                (abs(κp1) < κmin || abs(κp2) < κmin) && continue
                ηi, ηo = st.eta_in[i], st.eta_out[i]
                ηi <= 0 && continue
                d = Dict{Int,Float64}()
                addf!(f, τ, ii, c) = (m = lv.m[f]; o = lv.off[f] + (τ - 1) * m; U = E.vspecs[f].B.U;
                                      for j in 1:m; d[o+j] = get(d, o + j, 0.0) + c * U[ii, j]; end)
                addf!(st.fp, τ1, i, 1 / k)
                for τ in τ1+1:τ2
                    addf!(st.fe, τ, i, ηi * Δt / (st.hours[i] * k))
                end
                addf!(st.fm, τ2, i, ηi * ηo / k)
                addf!(fp1, τ1, ip1, 1 / κp1)
                addf!(fp2, τ2, ip2, -ηi * ηo / κp2)
                idx = collect(keys(d)); coef = [d[j] for j in idx]
                lf = () -> -sum(step_term(P, ℓ, ctx, ws, s.v, x, s.z0, s.m, τ) for τ in τ1:τ2)
                dt = slice_vec!(r, s.v, idx, coef, lf, width[])
                loc[q][1] += 1; loc[q][2] += dt
            end
        end
        stats .+= sum(loc)
    end
    return evaluate!(P, s)
end
