# Device model and exact cost (docs/reactant_design.md, phase 1).
#
# Array-first code over a chain dimension: the same functions run on Julia Arrays
# (reference) and traced by Reactant (XLA on CPU or GPU). Scope as in the design §1:
# the complete level, storage = soc, renewables = load, regularized constraints. All
# model-completion terms are included (line losses and soft limits as hinge sums,
# lossy links, merged vintages, fleet unit commitment, CO₂ price through mc).
#
# Layout: Julia column-major [c, τ, i] (chain fastest). Gathers and scatters are
# dense contractions with static incidence matrices (exact, traceable; to be replaced
# by gathers/scatters for full Europe, design §4.5).
module ESMDevice

using ESMSampling
using LinearAlgebra
using Printf
using Random
using Reactant
using Reactant
using Reactant: @trace
const ML = ESMSampling

export DeviceModel, DeviceIndex, DeviceState, pack, to_device, to_host!, rebuild_r, cost_parts

"Piecewise-linear function of u ≥ 0 as a hinge sum: g(u) = v0 + s0 u + Σ_j d_j (u − U_j)₊."
struct Hinge{V}
    v0::Float64
    s0::Float64
    U::V
    d::V
end

function hinge_from(U::Vector{Float64}, V::Vector{Float64}, send::Float64)
    n = length(U)
    sl = [(V[j+1] - V[j]) / (U[j+1] - U[j]) for j in 1:n-1]
    push!(sl, send)
    return Hinge(V[1] - sl[1] * U[1], sl[1], U[2:n], diff(sl))
end

@inline function hinge(h::Hinge, u)
    g = h.v0 .+ h.s0 .* u
    for j in 1:length(h.d)                  # a few knots, unrolled at trace time
        g = g .+ h.d[j:j] .* max.(u .- h.U[j:j], 0.0)
    end
    return g
end

"Static model shared by all chains (complete level ℓ of `P`)."
struct DeviceModel{V,Mx,Bv,H}
    N::Int
    T::Int
    Δt::Float64
    c_shed::Float64
    c_dump::Float64
    c_viol::Float64
    κ0::Float64
    γk::Float64
    # lines: incidence (N × L), susceptance, rating, loss coefficient, line-term hinges (u = |f|/s)
    Dl::Mx                 # +1 at bus0, −1 at bus1
    Al::Mx                 # +1 at both ends (losses)
    lb::V
    ls::V
    lα::V
    ψ::H                            # overload in units of s (cost c_viol · s · ψ)
    Lq::H                           # loss interpolant of u²
    # links
    K0::Mx                 # (U_l × N) bus0 one-hot
    K1::Mx
    kcap::V
    η::V
    # dispatchable units (generator groups with a utilization field)
    G::Mx                  # (U_g × N) node one-hot
    gmc::V; gmc0::V; gk0::V; gav::V
    gef::V; gef0::V
    guc::V                # 1 if committable
    ggmin::V; gsu::V; gnl::V
    gkfix::V              # fixed capacity (non-investable)
    ginv::Bv
    glo::V; ghi::V; gcap::V
    # storage (SoC form)
    S::Mx                  # (U_s × N)
    sh::V; sηin::V; sηout::V; sqt::V; smc::V
    skfix::V; sinv::Bv
    slo::V; shi::V; scap::V
    brownfield::Bool
end

"Host-side index of the device items: (group, index) of every dispatchable and storage unit."
struct DeviceIndex
    gidx::Vector{Tuple{Int,Int}}
    sidx::Vector{Tuple{Int,Int}}
end

"Per-chain state: node values (C, T, items), raw capacities (C, items), and the host-side load term."
mutable struct DeviceState{A3,A2,A1}
    θ::A3        # (C, T, N)
    ω::A3        # (C, T, U_g)
    uc::A3       # (C, T, U_g) commitment (ignored where not committable)
    e::A3        # (C, T, U_s)
    u::A3        # (C, T, U_l)
    kg::A2       # (C, U_g) raw capacities
    ks::A2       # (C, U_s)
    dnet::A3     # (C, T, N): load − renewable output A·k (host, from x and k_ren)
    cren::A1     # (C,) renewable running cost (host)
    capren::A1   # (C,) capex of the renewable capacities (host)
end

onehot(rows, N) = (M = zeros(length(rows), N); for (i, n) in enumerate(rows); M[i, n] = 1.0; end; M)

"Pack the static device model of the complete level ℓ; returns (model, host index)."
function pack(P::ML.Problem, ℓ::Int)
    E = P.E
    sys = E.sys
    lv = P.lv[ℓ+1]
    @assert all(ML.iscomplete(lv, E.vspecs, f) for f in eachindex(E.vspecs)) "device model: complete level only"
    @assert ML.issoc(E) && E.renewables == :load && E.regularized
    N, nl = sys.N, length(sys.line_s)
    Dl, Al = zeros(N, nl), zeros(N, nl)
    for e in 1:nl
        Dl[sys.line_bus0[e], e] += 1; Dl[sys.line_bus1[e], e] -= 1
        Al[sys.line_bus0[e], e] += 1; Al[sys.line_bus1[e], e] += 1
    end
    LT = E.lines
    ψ = hinge_from(LT.U, LT.Ψ, LT.sΨ)
    Lq = hinge_from(LT.U, LT.L, LT.sL)
    gi = [(gi, i) for (gi, g) in enumerate(E.gens) if g.vf > 0 for i in eachindex(g.bus)]
    gp(f) = [f(E.gens[a], b) for (a, b) in gi]
    si = [(si, i) for (si, s) in enumerate(E.stors) for i in eachindex(s.bus)]
    sp(f) = [f(E.stors[a], b) for (a, b) in si]
    kspec(kf, i) = kf > 0 ? E.kspecs[kf] : nothing
    return DeviceModel(N, lv.T, Float64(lv.h), E.c_shed, E.c_dump, E.c_viol, E.κ0, E.γk,
        Dl, Al, copy(sys.line_b), copy(sys.line_s), copy(LT.α), ψ, Lq,
        onehot(sys.link_bus0, N), onehot(sys.link_bus1, N), copy(sys.link_cap), copy(E.link_eta),
        onehot(gp((g, i) -> g.bus[i]), N), gp((g, i) -> g.mc[i]), gp((g, i) -> g.mc0[i]), gp((g, i) -> g.k0[i]),
        gp((g, i) -> g.avail[i]), gp((g, i) -> g.ef[i]), gp((g, i) -> g.ef0[i]),
        gp((g, i) -> Float64(g.ucf > 0)), gp((g, i) -> g.uc[1]), gp((g, i) -> g.uc[2]), gp((g, i) -> g.uc[3]),
        gp((g, i) -> g.kfix[i]), gp((g, i) -> g.kf > 0),
        gp((g, i) -> g.kf > 0 ? E.kspecs[g.kf].lo[i] : 0.0), gp((g, i) -> g.kf > 0 ? E.kspecs[g.kf].hi[i] : Inf),
        gp((g, i) -> g.kf > 0 ? g.capex[i] : 0.0),
        onehot(sp((s, i) -> s.bus[i]), N), sp((s, i) -> s.hours[i]), sp((s, i) -> s.eta_in[i]), sp((s, i) -> s.eta_out[i]),
        sp((s, i) -> s.qt[i]), sp((s, i) -> s.mc[i]), sp((s, i) -> s.kfix[i]), sp((s, i) -> s.kf > 0),
        sp((s, i) -> s.kf > 0 ? E.kspecs[s.kf].lo[i] : 0.0), sp((s, i) -> s.kf > 0 ? E.kspecs[s.kf].hi[i] : Inf),
        sp((s, i) -> s.kf > 0 ? s.capex[i] : 0.0), E.brownfield), DeviceIndex(gi, si)
end

"Device state of C copies of level-ℓ states `ss` (vector of States, one per chain)."
function to_device(P::ML.Problem, M::DeviceModel, ix::DeviceIndex, ss::Vector{ML.State})
    E = P.E
    C = length(ss)
    ℓ = ss[1].ℓ
    lv, lk = P.lv[ℓ+1], P.lk[ℓ+1]
    T, N = M.T, M.N
    Ug, Us, Ul = length(ix.gidx), length(ix.sidx), length(M.kcap)
    θ, ω, uc = zeros(C, T, N), zeros(C, T, Ug), zeros(C, T, Ug)
    e, u = zeros(C, T, Us), zeros(C, T, Ul)
    kg, ks = zeros(C, Ug), zeros(C, Us)
    dnet = zeros(C, T, N)
    cren, capren = zeros(C), zeros(C)
    for (c, s) in enumerate(ss)
        Y(f) = ML.nodevals(s.v, lv, E.vspecs, f)
        K(f) = vec(ML.nodevals(s.k, lk, E.kspecs, f))
        θ[c, :, :] .= Y(E.fθ)'
        E.flink > 0 && (u[c, :, :] .= Y(E.flink)')
        for (q, (gi, i)) in enumerate(ix.gidx)
            g = E.gens[gi]
            ω[c, :, q] .= Y(g.vf)[i, :]
            g.ucf > 0 && (uc[c, :, q] .= Y(g.ucf)[i, :])
            kg[c, q] = g.kf > 0 ? K(g.kf)[i] : g.kfix[i]
        end
        for (q, (si, i)) in enumerate(ix.sidx)
            st = E.stors[si]
            e[c, :, q] .= Y(st.fe)[i, :]
            ks[c, q] = st.kf > 0 ? K(st.kf)[i] : st.kfix[i]
        end
        # host part: load, renewable output A·k (renewables as load) and their costs, capex of renewables
        Xn = ML.weather_nodes(P.W, ℓ, s.xs[end])
        dn = max.(Xn[1], 0.0)                     # N × T
        cr = 0.0
        for g in E.gens
            g.vf == 0 || continue
            if g.kf > 0
                sp = E.kspecs[g.kf]; kr = K(g.kf); kcl = clamp.(kr, sp.lo, sp.hi)
                capren[c] += sum(g.capex .* kr) + E.γk * sum(g.capex .* abs.(kr .- kcl)) - (E.brownfield ? sum(g.capex .* sp.lo) : 0.0)
            end
            kv = g.kf > 0 ? clamp.(K(g.kf), E.kspecs[g.kf].lo, E.kspecs[g.kf].hi) : g.kfix
            A = g.wf > 0 ? clamp.(Xn[g.wf], 0.0, 1.0) : repeat(g.avail, 1, T)
            for t in 1:T, i in eachindex(g.bus)
                p = A[i, t] * kv[i]
                dn[g.bus[i], t] -= p
                cr += M.Δt * g.mc[i] * p
            end
        end
        dnet[c, :, :] .= dn'
        cren[c] = cr
    end
    return DeviceState(θ, ω, uc, e, u, kg, ks, dnet, cren, capren)
end

"""
Write chain c of the device state back into the level-ℓ state `s` (mode coefficients
from node values: at the complete level every basis is orthonormal, so Uᵀ Y is exact;
θ is projected off the component constants, which removes the free gauge).
Returns the evaluated state.
"""
function to_host!(P::ML.Problem, M::DeviceModel, ix::DeviceIndex, S::DeviceState, c::Int, s::ML.State)
    E = P.E
    lv, lk = P.lv[s.ℓ+1], P.lk[s.ℓ+1]
    A3(x) = Array(x)
    θ, ω, uc, e, u = A3(S.θ), A3(S.ω), A3(S.uc), A3(S.e), A3(S.u)
    kg, ks = Array(S.kg), Array(S.ks)
    setv!(f, Y) = (ML.fieldview(s.v, lv, E.vspecs, f) .= ML.cols(E.vspecs[f].B, lv.m[f])' * Y)
    setv!(E.fθ, permutedims(θ[c, :, :]))
    E.flink > 0 && setv!(E.flink, permutedims(u[c, :, :]))
    for (gi, g) in enumerate(E.gens)
        g.vf > 0 || continue
        qs = [q for (q, (a, i)) in enumerate(ix.gidx) if a == gi]
        setv!(g.vf, permutedims(ω[c, :, qs]))
        g.ucf > 0 && setv!(g.ucf, permutedims(uc[c, :, qs]))
        if g.kf > 0
            ML.fieldview(s.k, lk, E.kspecs, g.kf) .= ML.cols(E.kspecs[g.kf].B, lk.m[g.kf])' * reshape(kg[c, qs], :, 1)
        end
    end
    for (si, st) in enumerate(E.stors)
        qs = [q for (q, (a, i)) in enumerate(ix.sidx) if a == si]
        setv!(st.fe, permutedims(e[c, :, qs]))
        if st.kf > 0
            ML.fieldview(s.k, lk, E.kspecs, st.kf) .= ML.cols(E.kspecs[st.kf].B, lk.m[st.kf])' * reshape(ks[c, qs], :, 1)
        end
    end
    return ML.evaluate!(P, s)
end

# ---------------------------------------------------------------------------
# cost (array code)
# ---------------------------------------------------------------------------

"(C, T, X) × (X, Y) → (C, T, Y)."
@inline tmul(A, B) = reshape(reshape(A, size(A, 1) * size(A, 2), size(A, 3)) * B, size(A, 1), size(A, 2), size(B, 2))
"(C, T, Y) × (X, Y)ᵀ → (C, T, X)."
@inline tmulT(A, B) = reshape(reshape(A, size(A, 1) * size(A, 2), size(A, 3)) * transpose(B), size(A, 1), size(A, 2), size(B, 1))
"Next instant along τ (cyclic)."
@inline nextτ(A) = cat(A[:, 2:end, :], A[:, 1:1, :]; dims=2)
rowv(v) = reshape(v, 1, 1, :)
rowk(v) = reshape(v, size(v, 1), 1, size(v, 2))

"Clamped capacities and the capex part (per chain)."
function capacities(M::DeviceModel, kraw, lo, hi, kfix, inv, cap)
    kc = ifelse.(reshape(inv, 1, :), clamp.(kraw, reshape(lo, 1, :), reshape(hi, 1, :)), reshape(kfix, 1, :))
    ex = abs.(kraw .- kc)
    capex = vec(sum((kraw .* reshape(inv, 1, :)) .* reshape(cap, 1, :) .+ M.γk .* reshape(cap, 1, :) .* ex; dims=2))
    M.brownfield && (capex = capex .- sum(cap .* lo .* inv))
    return kc, capex
end

"""
Residual r (C, T, N) and the cost terms of every step: returns (r, parts) with
parts a NamedTuple of per-chain vectors (gen, storage, penalty, shed, dump, co2, capex).
"""
function cost_parts(M::DeviceModel, S::DeviceState)
    Δt, cv, κ0 = M.Δt, M.c_viol, M.κ0
    # lines: flows, overload and losses
    f = tmul(S.θ, M.Dl) .* rowv(M.lb)                        # (C, T, L): f = b (θ_a − θ_b)
    sl = rowv(M.ls)
    uabs = max.(f, -f) ./ sl                                   # not abs.(a .* b): miscompiled by Reactant 0.2.304
    ov = sl .* hinge(M.ψ, uabs)
    loss = rowv(M.lα) .* sl .^ 2 .* hinge(M.Lq, uabs)
    r = S.dnet .+ tmulT(f, M.Dl) .+ tmulT(0.5 .* loss, M.Al)
    pen = vec(sum(cv * Δt .* ov; dims=(2, 3)))
    # links: signed, lossy at the receiving end
    if length(M.kcap) > 0
        yc = clamp.(S.u, -1.0, 1.0)
        uu = yc .* rowv(M.kcap)
        η = rowv(M.η)
        j0 = uu .+ (1 .- η) .* max.(-uu, 0.0)
        j1 = -uu .+ (1 .- η) .* max.(uu, 0.0)
        r = r .+ tmul(j0, M.K0) .+ tmul(j1, M.K1)
        pen = pen .+ vec(sum(cv * Δt .* rowv(M.kcap) .* abs.(S.u .- yc); dims=(2, 3)))
    end
    # dispatchable units: merit order of the vintages, walls, commitment
    kg, capg = capacities(M, S.kg, M.glo, M.ghi, M.gkfix, M.ginv, M.gcap)
    kgr = rowk(kg)
    A = rowv(M.gav)
    yc = clamp.(S.ω, 0.0, 1.0)
    p = A .* kgr .* yc
    q = min.(rowv(M.gk0), kgr)
    oldfirst = rowv(M.gmc0) .<= rowv(M.gmc)
    plo = A .* ifelse.(oldfirst, q, kgr .- q)
    clo = ifelse.(oldfirst, rowv(M.gmc0), rowv(M.gmc)); chi = ifelse.(oldfirst, rowv(M.gmc), rowv(M.gmc0))
    elo = ifelse.(oldfirst, rowv(M.gef0), rowv(M.gef)); ehi = ifelse.(oldfirst, rowv(M.gef), rowv(M.gef0))
    hasv = (rowv(M.gk0) .> 0) .& (rowv(M.gmc0) .!= rowv(M.gmc))
    cg = ifelse.(hasv, clo .* min.(p, plo) .+ chi .* max.(p .- plo, 0.0), rowv(M.gmc) .* p)
    eg = ifelse.(hasv, elo .* min.(p, plo) .+ ehi .* max.(p .- plo, 0.0), rowv(M.gef) .* p)
    w = kgr .+ κ0
    gen = vec(sum(Δt .* cg; dims=(2, 3))) .+ S.cren
    co2 = vec(sum(Δt .* eg; dims=(2, 3)))
    pen = pen .+ vec(sum(cv * Δt .* w .* abs.(S.ω .- yc); dims=(2, 3)))
    r = r .- tmul(p, M.G)
    # fleet unit commitment (committable units only)
    ucm = rowv(M.guc)
    ucc, uc1 = clamp.(S.uc, 0.0, 1.0), clamp.(nextτ(S.uc), 0.0, 1.0)
    κ = A .* kgr
    cnl = ucm .* (Δt .* rowv(M.gnl) .* κ .* ucc .+ rowv(M.gsu) .* κ .* max.(uc1 .- ucc, 0.0))
    cwall = ucm .* (cv * Δt .* w .* (max.(yc .- ucc, 0.0) .+ max.(rowv(M.ggmin) .* ucc .- yc, 0.0) .+ abs.(S.uc .- ucc)))
    gen = gen .+ vec(sum(cnl; dims=(2, 3)))
    pen = pen .+ vec(sum(cwall; dims=(2, 3)))
    # storage, SoC form
    ks, caps = capacities(M, S.ks, M.slo, M.shi, M.skfix, M.sinv, M.scap)
    ksr = rowk(ks)
    ye, ye1 = S.e, nextτ(S.e)
    ec, ec1 = clamp.(ye, 0.0, 1.0), clamp.(ye1, 0.0, 1.0)
    h, ηin, ηout, qt = rowv(M.sh), rowv(M.sηin), rowv(M.sηout), rowv(M.sqt)
    d = h .* (ec1 .- ec) ./ Δt .- qt
    pos = d .>= 0
    pc = ifelse.(pos, ifelse.(ηin .> 0, d ./ ifelse.(ηin .> 0, ηin, 1.0), 0.0), 0.0)
    mc_ = ifelse.(pos, 0.0, min.(-d .* ηout, 1.0))
    ws_ = ksr .+ κ0
    cst = Δt .* rowv(M.smc) .* ksr .* (pc .+ mc_)
    spen = cv .* ws_ .* h .* abs.(ye .- ec) .+ cv * Δt .* ws_ .* max.(pc .- 1.0, 0.0) .+
           ifelse.(.!pos .& (qt .<= 0), cv * Δt .* ws_ .* max.(-d .* ηout .- 1.0, 0.0), 0.0) .+
           ifelse.(pos .& (ηin .<= 0), cv * Δt .* ws_ .* d, 0.0)
    storage = vec(sum(cst; dims=(2, 3)))
    pen = pen .+ vec(sum(spen; dims=(2, 3)))
    r = r .+ tmul(ksr .* (pc .- mc_), M.S)
    # balance (complete level: the cells are the nodes)
    shed = vec(sum(Δt .* max.(r, 0.0); dims=(2, 3)))
    dump = vec(sum(Δt .* max.(-r, 0.0); dims=(2, 3)))
    op = gen .+ storage .+ M.c_shed .* shed .+ M.c_dump .* dump .+ pen
    capex = capg .+ caps .+ S.capren
    return r, (; op, gen, storage, shed, dump, penalty=pen, co2, capex)
end

"Residual only (the cache of the device kernel, rebuilt every sweep)."
rebuild_r(M::DeviceModel, S::DeviceState) = cost_parts(M, S)[1]

# phase 2: term batch, exact dense line sampler, plain families, schedule and device loop
include("terms.jl")
include("jump.jl")          # slope-jump form of the line sampler
include("rounds.jl")
include("comp.jl")         # phase 3: compensated moves (spair, tsocL/tsoc, tpotL/tpot)
include("cells.jl")        # phase 3: cell templates (cell, cellpair)
include("blocks.jl")       # phase 3: storage block moves (plateau, shapes)
export TermBatch, sample_dense, eval_terms, FlatState, plan, run_rounds, check_conflicts, unflat!

include("kmove.jl")
include("bias_dev.jl")
include("kren.jl")
include("driver.jl")
include("score.jl")          # capacity score (replica–Langevin sampler)
include("langevin_dev.jl")   # Langevin capacity step, dispatch following, replica batch, weather upload
include("snapshot_dev.jl")   # state snapshots and resume of the device drivers
include("replica_run.jl")    # host loop of the replica–Langevin sampler
export run_device, run_replicas_device, cap_score, lgv_pass, weather_update, kscale_mh!, kmove_units, kscale_mh_loop, kpeak_mh_loop, KParams, pack_bias, delta_bias, commit_bias!, ren_units, pack_ren, kren_mh_loop, peak_sets_dev

end
