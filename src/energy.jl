# Energy system model (docs/system/model.md) in utilization coordinates on the
# graph-Haar hierarchy (optionally with transport modes, basis = :flow).
#
# Fields
#   k:  one KFIELD per investable carrier (generator power, storage power with
#       fixed energy-to-power ratio), node box [0, k̄].
#   v:  θ (potentials, gauge modes removed), ω_G per dispatchable generator
#       carrier, ω_C for DC links (identity basis), the state of charge ω_e
#       (INSTANT) per storage carrier, commitment fields (INSTANT) per committable
#       carrier. Renewables are load: output A·k, curtailment is nodal dump.
#   x:  load and availability per weather carrier (STEP).
#
# All constraints are priced: node-level violations of a field are priced at
# c_viol (energy units), and the cost is evaluated on the clamped values. The
# balance is priced on the projection of the nodal residual onto the level's
# load modes, φ(U_m U_mᵀ r), the analogue of aggregating the balance over a
# cluster: only imbalances resolved at the level are priced.

struct GenGroup
    carrier::String
    bus::Vector{Int}
    vf::Int
    kf::Int          # k field (0 for fixed fleet)
    wf::Int          # weather field (0 for constant availability)
    kfix::Vector{Float64}
    kbar::Vector{Float64}
    capex::Vector{Float64}
    mc::Vector{Float64}      # effective marginal cost of the (new) vintage, incl. CO₂ (€/MWh)
    avail::Vector{Float64}
    mcf::Vector{Float64}     # marginal cost without CO₂
    ef::Vector{Float64}      # emission factor (t CO₂ / MWh_el)
    k0::Vector{Float64}      # existing capacity (old vintage; brownfield floor of k), 0 without brownfield
    mc0::Vector{Float64}     # effective marginal cost of the old vintage
    mc0f::Vector{Float64}
    ef0::Vector{Float64}
    ucf::Int                 # commitment field (fleet unit commitment), 0 if not committable
    uc::NTuple{3,Float64}    # (minimum stable fraction, start-up cost €/MW, no-load cost €/MWh committed)
end
GenGroup(carrier, bus, vf, kf, wf, kfix, kbar, capex, mc, avail) =
    GenGroup(carrier, bus, vf, kf, wf, kfix, kbar, capex, mc, avail, copy(mc), zeros(length(mc)), zeros(length(mc)),
             copy(mc), copy(mc), zeros(length(mc)), 0, (0.0, 0.0, 0.0))

"Whether unit i of g has two vintages with different marginal costs (a merit-order kink)."
@inline hasvintage(g::GenGroup, i) = g.k0[i] > 0 && g.mc0[i] != g.mc[i]

"""
Merged co-located vintages (docs/sampling_formulations.md §3.2): the unit holds
the existing capacity k0 (old vintage) and the new capacity k − k0; output is
dispatched in merit order, so the cost of output p = A k y is the convex
piecewise-linear C(p) = c_lo min(p, A q_lo) + c_hi (p − A q_lo)₊, with q_lo the
capacity of the cheaper vintage. Returns (cost rate €/h, dC/dy, dC/dk at fixed y,
emissions t/h) at the clamped utilization yc. Without vintages C = mc·p.
"""
@inline function gen_vint(g::GenGroup, i, A, k, yc)
    p = A * k * yc
    if !hasvintage(g, i)
        return g.mc[i] * p, g.mc[i] * A * k, g.mc[i] * A * yc, g.ef[i] * p
    end
    q = min(g.k0[i], k)                       # old capacity (k ≥ k0 inside the box)
    oldfirst = g.mc0[i] <= g.mc[i]
    qlo = oldfirst ? q : k - q
    clo, chi = oldfirst ? (g.mc0[i], g.mc[i]) : (g.mc[i], g.mc0[i])
    elo, ehi = oldfirst ? (g.ef0[i], g.ef[i]) : (g.ef[i], g.ef0[i])
    plo = A * qlo
    if p <= plo
        c, em, mcp = clo * p, elo * p, clo
    else
        c, em, mcp = clo * plo + chi * (p - plo), elo * plo + ehi * (p - plo), chi
    end
    # dC/dk at fixed y: p moves with k (A y), and the kink moves with k when the new vintage is cheaper
    dk = mcp * A * yc
    if !oldfirst && g.k0[i] < k                # qlo = k − k0: dC/dqlo = A (clo − chi) when p > plo
        p > plo && (dk += A * (clo - chi))
    end
    return c, mcp * A * k, dk, em
end

"Utilization of the vintage kink of unit i at capacity k (NaN without vintages)."
@inline function vint_kink(g::GenGroup, i, k)
    hasvintage(g, i) || return NaN
    k > 0 || return NaN
    q = min(g.k0[i], k)
    return (g.mc0[i] <= g.mc[i] ? q : k - q) / k
end

"""
Fleet unit commitment (PyPSA's linearized UC): commitment c = κ/k ∈ [0, 1]
(an instant field), output y ∈ [g_min c, c] (walls priced at c_viol), no-load
cost a per MWh committed, start-up cost s per MW started, (c_{τ+1} − c_τ)₊.
Cost of step τ for unit i (A k the available capacity): returns the cost and,
with `grad`, (∂/∂yc, ∂/∂u, ∂/∂u1, ∂/∂k).
"""
@inline function uc_cost(g::GenGroup, i, A, k, w, Δt, cv, yc, u, u1)
    gmin, su, nl = g.uc
    uc, uc1 = clamp(u, 0.0, 1.0), clamp(u1, 0.0, 1.0)
    κ = A * k
    return Δt * nl * κ * uc + su * κ * max(uc1 - uc, 0.0) +
           cv * Δt * w * (max(yc - uc, 0.0) + max(gmin * uc - yc, 0.0) + abs(u - uc))
end

function uc_grad(g::GenGroup, i, A, k, w, Δt, cv, yc, u, u1)
    gmin, su, nl = g.uc
    uc, uc1 = clamp(u, 0.0, 1.0), clamp(u1, 0.0, 1.0)
    κ = A * k
    dy = du = du1 = 0.0
    dk = A * (Δt * nl * uc + su * max(uc1 - uc, 0.0))
    inu, inu1 = 0.0 < u < 1.0, 0.0 < u1 < 1.0
    if uc1 > uc
        inu && (du -= su * κ); inu1 && (du1 += su * κ)
    end
    inu && (du += Δt * nl * κ)
    if yc > uc
        dy += cv * Δt * w; inu && (du -= cv * Δt * w); dk += cv * Δt * (yc - uc)
    end
    if gmin * uc > yc
        dy -= cv * Δt * w; inu && (du += cv * Δt * w * gmin); dk += cv * Δt * (gmin * uc - yc)
    end
    if !inu
        du += cv * Δt * w * sign(u - uc); dk += cv * Δt * abs(u - uc)
    end
    return dy, du, du1, dk
end

"""
Line terms in relative flow u = |f|/s on a shared grid `U` (docs/model_completion.md §3):
overload cost rate c_viol·s·ψ(u) €/h (hard: ψ = (u − 1)₊; soft: the marginal price
rises linearly from 0 at s to c_viol at (1+δ)s, then stays c_viol) and losses
α s² L(u) MW, L the piecewise-linear interpolant of u² (exact at the grid points,
linear beyond the last one). Both are convex and piecewise linear in f, so the
exact line sampler stays exact; their kinks (f = 0, ±U_j s) are primary breakpoints.
"""
struct LineTerms
    U::Vector{Float64}
    L::Vector{Float64}
    Ψ::Vector{Float64}
    sL::Float64            # slope of L beyond the last grid point
    sΨ::Float64            # slope of ψ beyond the last grid point
    α::Vector{Float64}     # loss coefficient per line (0 without losses)
    losses::Bool
    soft::Bool
end

function LineTerms(alpha::Vector{Float64}; losses::Bool=false, segments::Int=4, umax::Float64=1.5,
                   thermal::Symbol=:hard, band::Float64=0.1, band_segments::Int=4)
    U = [0.0, 1.0]
    losses && append!(U, collect(0:ceil(Int, umax * segments)) ./ segments)
    soft = thermal == :soft
    soft && append!(U, 1.0 .+ band .* (1:band_segments) ./ band_segments)
    U = sort(unique(round.(U; digits=12)))
    L = losses ? U .^ 2 : zeros(length(U))
    ψ(u) = soft ? (u <= 1 ? 0.0 : u <= 1 + band ? (u - 1)^2 / (2band) : band / 2 + (u - 1 - band)) : max(u - 1, 0.0)
    Ψ = ψ.(U)
    n = length(U)
    sL = losses ? (L[n] - L[n-1]) / (U[n] - U[n-1]) : 0.0
    sΨ = 1.0
    return LineTerms(U, L, Ψ, sL, sΨ, losses ? copy(alpha) : zeros(length(alpha)), losses, soft)
end

"(overload ψ, loss L, dψ/du, dL/du) at relative flow u ≥ 0."
@inline function lineterm_u(LT::LineTerms, u::Float64)
    U = LT.U
    n = length(U)
    if u >= U[n]
        return LT.Ψ[n] + LT.sΨ * (u - U[n]), LT.L[n] + LT.sL * (u - U[n]), LT.sΨ, LT.sL
    end
    j = searchsortedlast(U, u)
    h = U[j+1] - U[j]
    a = (u - U[j]) / h
    dΨ = (LT.Ψ[j+1] - LT.Ψ[j]) / h
    dL = (LT.L[j+1] - LT.L[j]) / h
    return LT.Ψ[j] + a * (LT.Ψ[j+1] - LT.Ψ[j]), LT.L[j] + a * (LT.L[j+1] - LT.L[j]), dΨ, dL
end

"(s·ψ — the overload cost rate divided by c_viol, in MW — and the loss in MW) of line e at flow f."
@inline function line_eval(LT::LineTerms, e::Int, s_e::Float64, f::Float64)
    ψ, L, _, _ = lineterm_u(LT, abs(f) / s_e)
    return s_e * ψ, LT.α[e] * s_e^2 * L
end

"d/df of (s·ψ, loss) of line e at flow f."
@inline function line_deval(LT::LineTerms, e::Int, s_e::Float64, f::Float64)
    _, _, dψ, dL = lineterm_u(LT, abs(f) / s_e)
    sg = f > 0 ? 1.0 : (f < 0 ? -1.0 : 0.0)
    return sg * dψ, sg * LT.α[e] * s_e * dL
end

"Kink flows of line e (both signs, and 0 with losses) for the primary breakpoints."
function line_kinks!(out::Vector{Float64}, LT::LineTerms, s_e::Float64)
    LT.losses && push!(out, 0.0)
    for u in LT.U
        u > 0 || continue
        push!(out, u * s_e, -u * s_e)
    end
    return out
end

"Residual changes (bus0, bus1) of a signed lossy link with flow u (losses at the receiving end)."
@inline link_inj(η, u) = (u + (1 - η) * max(-u, 0.0), -u + (1 - η) * max(u, 0.0))
"d/du of `link_inj`."
@inline link_dinj(η, u) = u > 0 ? (1.0, -η) : (u < 0 ? (η, -1.0) : (1.0, -1.0))

struct StorGroup
    carrier::String
    bus::Vector{Int}
    fp::Int                   # 0: no separate charge field in the state-of-charge form
    fm::Int                   # 0: no separate discharge field
    fe::Int
    kf::Int
    kfix::Vector{Float64}
    kbar::Vector{Float64}
    hours::Vector{Float64}
    eta_in::Vector{Float64}
    eta_out::Vector{Float64}
    qt::Vector{Float64}       # inflow per unit power
    capex::Vector{Float64}
    mc::Vector{Float64}
end

struct EnergyModel
    sys::System
    Ug::AbstractMatrix{Float64}
    λg::Vector{Float64}
    kspecs::Vector{FieldSpec}
    vspecs::Vector{FieldSpec}
    wspecs::Vector{FieldSpec}
    gens::Vector{GenGroup}
    stors::Vector{StorGroup}
    fθ::Int
    flink::Int
    c_shed::Float64
    c_dump::Float64
    c_viol::Float64
    γk::Float64
    θlo::Vector{Float64}   # outer bounds of the θ mode coefficients
    θhi::Vector{Float64}
    regularized::Bool      # all constraints priced (always true)
    κ0::Float64            # capacity floor [MW] of the penalty weights (keeps the target proper)
    parts::Vector{Vector{Int}}   # graph-Haar partitions (cell label per node, coarse → fine)
    measure::Symbol              # reference measure of dispatchable units: :utilization (dω) or :physical (dp = k dω)
    storage::Symbol              # :soc (state-of-charge trajectory; the only storage form)
    renewables::Symbol           # :load (output A·k, curtailment = nodal dump; the only renewable form)
    flow::Any                    # FlowBasis (basis = :flow) or nothing
    lines::LineTerms             # line losses and thermal limits
    link_eta::Vector{Float64}    # link efficiencies (ones: lossless)
    co2::Base.RefValue{Float64}  # canonical CO₂ price μ (€/t)
    brownfield::Bool
end

"Set the canonical CO₂ price μ: mc = mc_fuel + μ ef for every vintage."
function set_co2_price!(E::EnergyModel, μ::Float64)
    E.co2[] = μ
    for g in E.gens
        g.mc .= g.mcf .+ μ .* g.ef
        g.mc0 .= g.mc0f .+ μ .* g.ef0
    end
    return E
end

"""
Default fleet unit-commitment parameters per carrier (assumptions, not in the
PyPSA-Eur data): (minimum stable fraction, start-up cost €/MW, no-load cost €/MWh
committed).
"""
const UC_DEFAULTS = Dict("gas" => (0.3, 40.0, 4.0), "coal" => (0.4, 80.0, 3.0), "lignite" => (0.5, 100.0, 3.0),
                         "nuclear" => (0.5, 150.0, 1.0), "biomass" => (0.3, 40.0, 2.0), "oil" => (0.2, 20.0, 5.0),
                         "waste" => (0.4, 40.0, 2.0))

"Whether storage is in the state-of-charge form (always true; kept for the device checks)."
issoc(E::EnergyModel) = E.storage == :soc

"Weights of the capacity bases: the node potentials (finite positive), else 1."
function kweights(kbar)
    ok = isfinite.(kbar) .& (kbar .> 0)
    any(ok) || return ones(length(kbar))
    fill = median(kbar[ok])
    return [ok[i] ? kbar[i] : fill for i in eachindex(kbar)]
end

function build_energy_model(sys::System; c_shed=3000.0, c_dump=5.0, c_viol=c_shed, γk=10.0,
                            κ0=1.0, basis=:haar, measure=:utilization, flow_kw=NamedTuple(), flow_theta=:nested,
                            co2_price=nothing, brownfield::Bool=false, losses::Bool=false, loss_segments::Int=4,
                            thermal::Symbol=:hard, thermal_band::Float64=0.1, thermal_segments::Int=4,
                            link_losses::Bool=false, uc=String[], uc_params=Dict{String,Any}())
    N = sys.N
    Zc = component_indicators(N, sys.line_bus0, sys.line_bus1)
    edges = transfer_edges(sys)
    FB = nothing
    @assert basis in (:haar, :flow) "unknown basis $basis (haar or flow)"
    # nested partition: whole network → countries → nodes (graph Haar, ε = 0),
    # sparse, with balanced binary contrasts inside every parent cell
    cidx = Dict(c => i for (i, c) in enumerate(unique(sys.country)))
    parts = [ones(Int, N), [cidx[c] for c in sys.country], collect(1:N)]
    Ug, gcounts, _ = haar_basis(parts; edges=edges)
    λg = Float64.(1:N)
    comp = [argmax(abs.(Zc[i, :])) for i in 1:N]
    mkbasis = (sup; weights=nothing) -> haar_field_basis(parts, gcounts, sup; edges=edges, weights=weights)
    θbasis = haar_field_basis([comp, sys.country, collect(1:N)], gcounts, collect(1:N); edges=edges, drop_top=true)
    if basis == :flow       # transport modes from the proxy ensemble first (model.md §2)
        FB = flow_basis(sys; flow_kw...)
        # :nested — the modes are the coarse angle coordinates (dense complement);
        # :haar — sparse Haar angles, the modes only as transport moves (large networks)
        flow_theta == :nested && (θbasis = flow_theta_basis(FB, θbasis, N))
    end
    kspecs, vspecs, wspecs = FieldSpec[], FieldSpec[], FieldSpec[]

    # weather fields
    wnames = [:load; [s for s in (:onwind, :offwind, :solar) if haskey(sys.weather, s)]]
    for s in wnames
        nodes, _ = sys.weather[s]
        S = length(nodes)
        push!(wspecs, FieldSpec(s, STEP, mkbasis(nodes), fill(-Inf, S), fill(Inf, S)))
    end

    # potentials: gauge modes (one constant per AC component) removed
    push!(vspecs, FieldSpec(:theta, STEP, θbasis, fill(-Inf, N), fill(Inf, N)))
    fθ = 1

    gens = GenGroup[]
    G = sys.gen
    if brownfield && !isempty(sys.bfgen)       # fixed brownfield fleets as further carriers
        G = Dict{Symbol,Any}(k => vcat(v, sys.bfgen[k]) for (k, v) in sys.gen if haskey(sys.bfgen, k))
    end
    μdata = sys.co2_data
    ucset = Set(String.(uc))
    for c in unique(G[:carrier])
        u = findall(==(c), G[:carrier])
        bus = G[:bus][u]
        @assert allunique(bus)
        S = length(u)
        k0 = brownfield ? copy(G[:k0][u]) : zeros(S)
        isren = haskey(sys.weather, Symbol(c))
        if isren                         # output A·k exactly; curtailment is the negative nodal residual (dump)
            vf = 0
        else
            push!(vspecs, FieldSpec(Symbol("g_", c), STEP, mkbasis(bus), zeros(S), ones(S)))
            vf = length(vspecs)
        end
        invest = all(G[:invest][u] .> 0.5)
        kf = 0
        kbar = copy(G[:kbar][u])
        if invest
            brownfield && (kbar .= max.(kbar, 1.1 .* k0))        # room above the existing capacity
            # coarse capacities spread in proportion to the node potentials (weighted Haar);
            # brownfield: the existing capacity is the floor (no retirement), capex on k − k0
            push!(kspecs, FieldSpec(Symbol("k_", c), KFIELD, mkbasis(bus; weights=kweights(kbar)), copy(k0), kbar))
            kf = length(kspecs)
        else
            k0 .= 0.0                                          # fixed fleet: one vintage
        end
        wf = something(findfirst(s -> String(s.name) == c, wspecs), 0)
        if wf > 0
            @assert wspecs[wf].B.support == bus "weather support of $c does not match generators"
        end
        ef, ef0 = copy(G[:ef][u]), copy(G[:ef0][u])
        mcf, mc0f = G[:mc][u] .- μdata .* ef, G[:mc0][u] .- μdata .* ef0
        ucf = 0
        ucp = (0.0, 0.0, 0.0)
        if c in ucset && vf > 0
            ucp = Tuple(Float64.(get(uc_params, c, get(UC_DEFAULTS, c, (0.3, 40.0, 3.0)))))
            push!(vspecs, FieldSpec(Symbol("uc_", c), INSTANT, mkbasis(bus), zeros(S), ones(S)))
            ucf = length(vspecs)
        end
        push!(gens, GenGroup(c, bus, vf, kf, wf, G[:k][u], kbar, G[:capex][u], copy(G[:mc][u]), G[:avail][u],
                             mcf, ef, k0, copy(G[:mc0][u]), mc0f, ef0, ucf, ucp))
    end

    flink = 0
    nl = length(sys.link_cap)
    if nl > 0
        push!(vspecs, FieldSpec(:links, STEP, identity_basis(nl, N), fill(-1.0, nl), fill(1.0, nl)))
        flink = length(vspecs)
    end

    stors = StorGroup[]
    St = sys.stor
    for c in unique(St[:carrier])
        u = findall(==(c), St[:carrier])
        bus = St[:bus][u]
        @assert allunique(bus)
        S = length(u)
        B = mkbasis(bus)
        push!(vspecs, FieldSpec(Symbol("se_", c), INSTANT, B, zeros(S), ones(S)))   # state of charge
        fe = length(vspecs)
        fp_, fm_ = 0, 0
        invest = all(St[:invest][u] .> 0.5)
        kf = 0
        if invest
            push!(kspecs, FieldSpec(Symbol("k_", c), KFIELD, mkbasis(bus; weights=kweights(St[:kbar][u])), zeros(S), St[:kbar][u]))
            kf = length(kspecs)
        end
        push!(stors, StorGroup(c, bus, fp_, fm_, fe, kf, St[:k][u], St[:kbar][u], St[:hours][u],
                               St[:eta_in][u], St[:eta_out][u], St[:inflow][u], St[:capex][u], St[:mc][u]))
    end

    θlo, θhi = theta_mode_bounds(sys, vspecs[fθ].B; scale=thermal == :soft ? 1 + 2thermal_band : 1.0)
    LT = LineTerms(sys.line_alpha; losses=losses, segments=loss_segments, thermal=thermal,
                   band=thermal_band, band_segments=thermal_segments)
    E = EnergyModel(sys, Ug, λg, kspecs, vspecs, wspecs, gens, stors, fθ, flink,
                    c_shed, c_dump, c_viol, γk, θlo, θhi, true, κ0, parts, measure, :soc, :load, FB,
                    LT, link_losses ? copy(sys.link_eta) : ones(length(sys.link_cap)), Ref(μdata), brownfield)
    co2_price === nothing || set_co2_price!(E, Float64(co2_price))
    return E
end

"Outer bounds of the θ mode coefficients over the line-limit polytope (LPs)."
function theta_mode_bounds(sys::System, B::ModeBasis; scale::Float64=1.0)
    r = nmodes(B)
    lo, hi = zeros(r), zeros(r)
    nl = length(sys.line_s)
    D = zeros(nl, r)
    for e in 1:nl
        D[e, :] .= sys.line_b[e] .* (B.Ut[:, sys.line_bus0[e]] .- B.Ut[:, sys.line_bus1[e]])
    end
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, y[1:r])
    @constraint(model, -scale .* sys.line_s .<= D * y .<= scale .* sys.line_s)
    for j in 1:r, sense in (MOI.MIN_SENSE, MOI.MAX_SENSE)
        @objective(model, sense, y[j])
        optimize!(model)
        @assert termination_status(model) == MOI.OPTIMAL "θ mode bound LP failed"
        sense == MOI.MIN_SENSE ? (lo[j] = objective_value(model)) : (hi[j] = objective_value(model))
    end
    return lo, hi
end

# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------

"""
State-of-charge storage form: charge/discharge utilizations of unit i at a step
from the clamped state of charge at its two ends. d = h(e₁ − e)/Δt − q;
charging d/η⁺ if d ≥ 0, else discharge min(−d η⁻, 1) (energy beyond the power
limit is spilled).
"""
@inline function soc_power(s::StorGroup, i, Δt, ec, ec1)
    d = s.hours[i] * (ec1 - ec) / Δt - s.qt[i]
    d >= 0 && return (s.eta_in[i] > 0 ? d / s.eta_in[i] : 0.0, 0.0)
    return (0.0, min(-d * s.eta_out[i], 1.0))
end

"""
Cost of one storage unit at one step in the SoC form: running cost, box
penalty on the state of charge, penalty for charging beyond the power limit.
Returns (cost, d, residual contribution k(p⁺ − p⁻)).
"""
@inline function soc_cost(s::StorGroup, i, k, w, Δt, cv, ye, ye1)
    ec, ec1 = clamp(ye, 0.0, 1.0), clamp(ye1, 0.0, 1.0)
    d = s.hours[i] * (ec1 - ec) / Δt - s.qt[i]
    η = s.eta_in[i]
    if d >= 0
        pc, mc = η > 0 ? d / η : 0.0, 0.0
    else
        pc, mc = 0.0, min(-d * s.eta_out[i], 1.0)
    end
    c = Δt * s.mc[i] * k * (pc + mc) + cv * w * s.hours[i] * abs(ye - ec)
    pc > 1 && (c += cv * Δt * w * (pc - 1))
    # beyond the power limit, energy can only be spilled by units with inflow; otherwise a violation
    (d < 0 && s.qt[i] <= 0) && (ex = -d * s.eta_out[i] - 1.0; ex > 0 && (c += cv * Δt * w * ex))
    (d >= 0 && η <= 0) && (c += cv * Δt * w * d)    # no charging (reservoir hydro): gain beyond inflow is a violation
    return c, d, k * (pc - mc)
end

"Gradient of the SoC-form storage terms w.r.t. the state-of-charge node values (and k into dk)."
function soc_grad(s::StorGroup, Ye, Ec, kv, dk, Gr, Δt, cv, κ0)
    S, T = size(Ye)
    dEc = zeros(S, T)
    dYe = zeros(S, T)
    @inbounds for t in 1:T, i in 1:S
        t1 = mod(t, T) + 1
        h = s.hours[i]
        d = h * (Ec[i, t1] - Ec[i, t]) / Δt - s.qt[i]
        k, w = kv[i], kv[i] + κ0
        g = Gr[s.bus[i], t]
        if d >= 0 && s.eta_in[i] <= 0
            dCdd = cv * Δt * w
            dk[i] += cv * Δt * d
        elseif d >= 0
            pc = d / s.eta_in[i]
            dCdd = (Δt * s.mc[i] * k + k * g + (pc > 1 ? cv * Δt * w : 0.0)) / s.eta_in[i]
            dk[i] += Δt * s.mc[i] * pc + g * pc + (pc > 1 ? cv * Δt * (pc - 1) : 0.0)
        else
            mc = -d * s.eta_out[i]
            dCdd = mc < 1 ? -s.eta_out[i] * (Δt * s.mc[i] * k - k * g) : (s.qt[i] <= 0 ? -s.eta_out[i] * cv * Δt * w : 0.0)
            mcc = min(mc, 1.0)
            dk[i] += Δt * s.mc[i] * mcc - g * mcc + ((mc > 1 && s.qt[i] <= 0) ? cv * Δt * (mc - 1) : 0.0)
        end
        dEc[i, t1] += dCdd * h / Δt
        dEc[i, t] -= dCdd * h / Δt
    end
    @inbounds for t in 1:T, i in 1:S
        y = Ye[i, t]
        if 0.0 < y < 1.0
            dYe[i, t] = dEc[i, t]
        else
            dYe[i, t] = cv * (kv[i] + κ0) * s.hours[i] * sign(y - Ec[i, t])
            dk[i] += cv * s.hours[i] * abs(y - Ec[i, t])
        end
    end
    return dYe
end

"Column-wise projection of a (support × T) matrix onto the first m basis vectors."
function project_cols(B::ModeBasis, m::Int, R::AbstractMatrix)
    out = similar(R, Float64)
    tmp = zeros(max(m, length(B.support)))
    for t in axes(R, 2)
        project!(view(out, :, t), B, m, view(R, :, t), tmp)
    end
    return out
end

"Weather data map at node level: d̄ = (d)₊, ā = clip(a, 0, 1)."
datamap(s::FieldSpec, x) = s.name == :load ? max(x, 0.0) : clamp(x, 0.0, 1.0)
datamap_deriv(s::FieldSpec, x) = s.name == :load ? Float64(x > 0) : Float64(0 < x < 1)

"""
Node-level values of a v or k field: returns `(Y, Yc, free)` with `Yc` the
clamped values used in the cost (`Yc === Y` and `free` for unbounded fields).
"""
function field_nodes(E::EnergyModel, z, lay::Layout, specs, f::Int)
    Y = nodevals(z, lay, specs, f)
    s = specs[f]
    if all(isinf, s.lo) && all(isinf, s.hi)
        return Y, Y, true
    end
    return Y, clamp.(Y, s.lo, s.hi), false
end

struct CostParts
    capex::Float64
    op::Float64
    gen::Float64
    storage::Float64
    shed::Float64       # MWh
    dump::Float64       # MWh
    penalty::Float64    # € (coarse-level relaxations)
    co2::Float64        # t CO₂ (per year)
end
CostParts(a, b, c, d, e, f, g) = CostParts(a, b, c, d, e, f, g, NaN)

"""
    energy_cost(E, lk, lv, zk, zv, Xn; grad=false)

Annualized capex ⟨c_k,k⟩ and operational cost C_x(k,v) at the level with
layouts `lk`, `lv`, for node-level weather `Xn[f]` (raw, before the data map).
With `grad=true` also returns gradients `(gk_capex, gk_op, gv, gX)`; `gX[f]` is
the gradient of C with respect to the raw node-level weather.
"""
function energy_cost(E::EnergyModel, lk::Layout, lv::Layout, zk, zv, Xn; grad=false)
    sys = E.sys
    N, T, Δt = sys.N, lv.T, Float64(lv.h)
    ks, vs = E.kspecs, E.vspecs

    # ---- capacities
    kn = Vector{Vector{Float64}}(undef, length(ks))
    kc = similar(kn)
    capex = 0.0
    gkn_capex = grad ? [zeros(nsupport(s.B)) for s in ks] : nothing
    for (f, s) in enumerate(ks)
        Y = vec(nodevals(zk, lk, ks, f))
        kn[f] = Y
        cap = f_capex(E, f)
        E.brownfield && (capex -= dot(cap, s.lo))           # existing capacity is sunk: capex on k − k0
        kc[f] = clamp.(Y, s.lo, s.hi)
        ex = Y .- kc[f]
        capex += dot(cap, Y) + E.γk * dot(cap, abs.(ex))
        grad && (gkn_capex[f] .= cap .+ E.γk .* cap .* sign.(ex))
    end
    dkc = grad ? [zeros(length(kc[f])) for f in eachindex(kc)] : nothing

    r = zeros(N, T)
    W = [map(x -> datamap(E.wspecs[f], x), Xn[f]) for f in eachindex(Xn)]
    for t in 1:T, n in 1:N
        r[n, t] = W[1][n, t]   # load field is the first weather field and covers all nodes
    end
    @assert E.wspecs[1].B.support == collect(1:N)

    pen = 0.0
    cgen = 0.0
    cst = 0.0
    c_viol = E.c_viol

    # ---- potentials and line flows
    Θ = nodevals(zv, lv, vs, E.fθ)
    nl = length(sys.line_s)
    F = zeros(nl, T)
    LT = E.lines
    for t in 1:T, e in 1:nl
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        fe = sys.line_b[e] * (Θ[a, t] - Θ[b, t])
        F[e, t] = fe
        ov, loss = line_eval(LT, e, sys.line_s[e], fe)
        r[a, t] += fe + 0.5 * loss                # losses are drawn half at each end
        r[b, t] += -fe + 0.5 * loss
        pen += c_viol * Δt * ov
    end

    # ---- DC links
    if E.flink > 0
        Yl, Lc, lfree = field_nodes(E, zv, lv, vs, E.flink)
        for t in 1:T, l in eachindex(sys.link_cap)
            u = sys.link_cap[l] * Lc[l, t]
            j0, j1 = link_inj(E.link_eta[l], u)
            r[sys.link_bus0[l], t] += j0
            r[sys.link_bus1[l], t] += j1
            lfree || (pen += c_viol * Δt * sys.link_cap[l] * abs(Yl[l, t] - Lc[l, t]))
        end
    end

    # ---- generators
    gen_cache = Vector{Any}(undef, length(E.gens))
    co2 = 0.0
    for (gi, g) in enumerate(E.gens)
        Y, Yc, comp = g.vf > 0 ? field_nodes(E, zv, lv, vs, g.vf) : (ones(length(g.bus), T), ones(length(g.bus), T), true)
        kv = g.kf > 0 ? kc[g.kf] : g.kfix
        A = g.wf > 0 ? W[g.wf] : repeat(g.avail, 1, T)
        for t in 1:T, i in eachindex(g.bus)
            p = A[i, t] * kv[i] * Yc[i, t]
            c, _, _, em = gen_vint(g, i, A[i, t], kv[i], Yc[i, t])
            cgen += Δt * c
            co2 += Δt * em
            r[g.bus[i], t] -= p
        end
        if g.ucf > 0                                  # fleet unit commitment
            Yu = nodevals(zv, lv, vs, g.ucf)
            for t in 1:T, i in eachindex(g.bus)
                cu = uc_cost(g, i, A[i, t], kv[i], kv[i] + E.κ0, Δt, c_viol, Yc[i, t], Yu[i, t], Yu[i, mod(t, T)+1])
                cnl = Δt * g.uc[3] * A[i, t] * kv[i] * clamp(Yu[i, t], 0.0, 1.0) +
                      g.uc[2] * A[i, t] * kv[i] * max(clamp(Yu[i, mod(t, T)+1], 0.0, 1.0) - clamp(Yu[i, t], 0.0, 1.0), 0.0)
                cgen += cnl
                pen += cu - cnl
            end
        end
        if !comp
            for t in 1:T, i in eachindex(g.bus)
                ex = abs(Y[i, t] - Yc[i, t])
                ex > 0 && (pen += c_viol * Δt * (kv[i] + E.κ0) * ex)
            end
        end
        gen_cache[gi] = (Y, Yc, comp, kv, A)
    end

    # ---- storage
    st_cache = Vector{Any}(undef, length(E.stors))
    for (si, s) in enumerate(E.stors)                 # state-of-charge form
        Ye, Ec, _ = field_nodes(E, zv, lv, vs, s.fe)
        kv = s.kf > 0 ? kc[s.kf] : s.kfix
        for t in 1:T, i in eachindex(s.bus)
            c, _, j = soc_cost(s, i, kv[i], kv[i] + E.κ0, Δt, c_viol, Ye[i, t], Ye[i, mod(t, T)+1])
            pc, mc = soc_power(s, i, Δt, Ec[i, t], Ec[i, mod(t, T)+1])
            cst += Δt * s.mc[i] * kv[i] * (pc + mc)
            pen += c - Δt * s.mc[i] * kv[i] * (pc + mc)
            r[s.bus[i], t] += j
        end
        st_cache[si] = (Ye, Ec, kv)
    end

    # ---- priced balance, on the level-ℓ projection of the nodal residual
    #      (the analogue of aggregating the balance over a cluster)
    Bl = E.wspecs[1].B
    mr = modes_at(Bl, lv.M)
    rproj = mr < N
    re = rproj ? project_cols(Bl, mr, r) : r
    shed, dump, cbal = 0.0, 0.0, 0.0
    for t in 1:T, n in 1:N
        x = re[n, t]
        if x > 0
            shed += Δt * x
        else
            dump -= Δt * x
        end
    end
    cbal = E.c_shed * shed + E.c_dump * dump
    op = cgen + cst + cbal + pen
    parts = CostParts(capex, op, cgen, cst, shed, dump, pen, co2)
    grad || return parts

    # ===================== gradients =====================
    Gr = similar(r)
    for t in 1:T, n in 1:N
        x = re[n, t]
        Gr[n, t] = Δt * (x > 0 ? E.c_shed : (x < 0 ? -E.c_dump : 0.0))
    end
    rproj && (Gr = project_cols(Bl, mr, Gr))
    gv = zeros(lv.n)
    gX = [zeros(size(x)) for x in Xn]
    for t in 1:T, n in 1:N
        gX[1][n, t] = Gr[n, t] * datamap_deriv(E.wspecs[1], Xn[1][n, t])
    end

    # θ
    dΘ = zeros(N, T)
    for t in 1:T, e in 1:nl
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        dψ, dL = line_deval(LT, e, sys.line_s[e], F[e, t])
        dF = Gr[a, t] * (1 + 0.5 * dL) + Gr[b, t] * (-1 + 0.5 * dL)
        dF += c_viol * Δt * dψ
        dΘ[a, t] += sys.line_b[e] * dF
        dΘ[b, t] -= sys.line_b[e] * dF
    end
    add_field_grad!(gv, lv, vs, E.fθ, dΘ)

    # links
    if E.flink > 0
        Y, Lc, comp = field_nodes(E, zv, lv, vs, E.flink)
        dY = zeros(size(Y))
        for t in 1:T, l in eachindex(sys.link_cap)
            if comp || -1.0 < Y[l, t] < 1.0
                d0, d1 = link_dinj(E.link_eta[l], sys.link_cap[l] * Lc[l, t])
                dY[l, t] = sys.link_cap[l] * (Gr[sys.link_bus0[l], t] * d0 + Gr[sys.link_bus1[l], t] * d1)
            else
                dY[l, t] = c_viol * Δt * sys.link_cap[l] * sign(Y[l, t] - Lc[l, t])
            end
        end
        add_field_grad!(gv, lv, vs, E.flink, dY)
    end

    # generators and storage (function barriers: the caches are heterogeneous)
    for (gi, g) in enumerate(E.gens)
        Y, Yc, comp, kv, A = gen_cache[gi]
        dk = g.kf > 0 ? dkc[g.kf] : zeros(length(kv))
        gXw = g.wf > 0 ? gX[g.wf] : nothing
        Xw = g.wf > 0 ? Xn[g.wf] : nothing
        dY = gen_grad(g, Y, Yc, comp, kv, A, dk, gXw, Xw, Gr, Δt, c_viol, E.κ0)
        if g.ucf > 0
            Yu = nodevals(zv, lv, vs, g.ucf)
            dU = zeros(size(Yu))
            for t in 1:T, i in eachindex(g.bus)
                t1 = mod(t, T) + 1
                dy, du, du1, dkk = uc_grad(g, i, A[i, t], kv[i], kv[i] + E.κ0, Δt, c_viol, Yc[i, t], Yu[i, t], Yu[i, t1])
                (comp || 0.0 < Y[i, t] < 1.0) && (dY[i, t] += dy)
                dU[i, t] += du; dU[i, t1] += du1
                dk[i] += dkk
                # availability enters the committed capacity A k (weather-dependent groups have no UC)
            end
            add_field_grad!(gv, lv, vs, g.ucf, dU)
        end
        g.vf > 0 && add_field_grad!(gv, lv, vs, g.vf, dY)
    end
    for (si, s) in enumerate(E.stors)
        Ye, Ec, kv = st_cache[si]
        dk = s.kf > 0 ? dkc[s.kf] : zeros(length(kv))
        add_field_grad!(gv, lv, vs, s.fe, soc_grad(s, Ye, Ec, kv, dk, Gr, Δt, c_viol, E.κ0))
    end

    # capacities: chain through the clamp and the modes
    gk_capex = zeros(lk.n)
    gk_op = zeros(lk.n)
    for (f, s) in enumerate(ks)
        mask = Float64.((s.lo .< kn[f]) .& (kn[f] .< s.hi))
        add_field_grad!(gk_capex, lk, ks, f, reshape(gkn_capex[f], :, 1))
        add_field_grad!(gk_op, lk, ks, f, reshape(mask .* dkc[f], :, 1))
    end
    return parts, (gk_capex, gk_op, gv, gX)
end

function f_capex(E::EnergyModel, f::Int)
    for g in E.gens
        g.kf == f && return g.capex
    end
    for s in E.stors
        s.kf == f && return s.capex
    end
    error("no capex for k field $f")
end

"Accumulate the mode-space gradient Uᵀ dY of field f into g."
function add_field_grad!(g::AbstractVector, lay::Layout, specs, f::Int, dY::AbstractMatrix)
    U = cols(specs[f].B, lay.m[f])
    G = fieldview(g, lay, specs, f)
    mul!(G, U', dY, 1.0, 1.0)
    return g
end

"""
Log density of the reference measure relative to Lebesgue measure on the
utilization coordinates. With `measure = :physical` the dispatch of dispatchable
units (generators without weather-dependent availability, and the state of charge
of storage) is measured in MW (MWh), dp = k dω, which adds
n_u log(k_u + κ0) per investable unit u whose fields are complete at the level
(n_u = number of its utilization coordinates). In utilization coordinates the
cost-confined dispatch of a unit occupies a volume ∝ k^(−n_conf), an entropic
force against capacity; the physical measure removes it (then integrating out
idle capacity gives the capex of the peak dispatch, see docs/physical_formulation.md).
Renewables keep the utilization (curtailment share) measure: their physical
volume vanishes wherever the availability does.
"""
function log_measure(E::EnergyModel, lk::Layout, lv::Layout, zk)
    E.measure == :physical || return 0.0
    val = 0.0
    kc = nothing
    for g in E.gens
        (g.kf > 0 && g.wf == 0 && g.vf > 0 && iscomplete(lv, E.vspecs, g.vf) && iscomplete(lk, E.kspecs, g.kf)) || continue
        kc === nothing && (kc = [clamp.(vec(nodevals(zk, lk, E.kspecs, f)), s.lo, s.hi) for (f, s) in enumerate(E.kspecs)])
        val += (g.ucf > 0 && iscomplete(lv, E.vspecs, g.ucf) ? 2 : 1) * lv.T * sum(log.(kc[g.kf] .+ E.κ0))
    end
    for st in E.stors
        (st.kf > 0 && iscomplete(lv, E.vspecs, st.fe) && iscomplete(lk, E.kspecs, st.kf)) || continue
        kc === nothing && (kc = [clamp.(vec(nodevals(zk, lk, E.kspecs, f)), s.lo, s.hi) for (f, s) in enumerate(E.kspecs)])
        val += lv.T * sum(log.(kc[st.kf] .+ E.κ0))
    end
    return val
end

"""
Gradient of `log_measure` w.r.t. the capacity coefficients: per unit with physical
measure, n_u/(k_c + κ0) at its node inside the box (0 where the capacity is clamped).
"""
function log_measure_grad(E::EnergyModel, lk::Layout, lv::Layout, zk)
    g = zeros(lk.n)
    E.measure == :physical || return g
    function add!(f, n)
        sp = E.kspecs[f]
        y = vec(nodevals(zk, lk, E.kspecs, f))
        dk = [(sp.lo[i] < y[i] < sp.hi[i]) ? n / (y[i] + E.κ0) : 0.0 for i in eachindex(y)]
        add_field_grad!(g, lk, E.kspecs, f, reshape(dk, :, 1))
    end
    for gg in E.gens
        (gg.kf > 0 && gg.wf == 0 && gg.vf > 0 && iscomplete(lv, E.vspecs, gg.vf) && iscomplete(lk, E.kspecs, gg.kf)) || continue
        add!(gg.kf, (gg.ucf > 0 && iscomplete(lv, E.vspecs, gg.ucf) ? 2 : 1) * lv.T)
    end
    for st in E.stors
        (st.kf > 0 && iscomplete(lv, E.vspecs, st.fe) && iscomplete(lk, E.kspecs, st.kf)) || continue
        add!(st.kf, lv.T)
    end
    return g
end

"Total capacity per investable carrier (MW), from the node values."
capacity_totals(E::EnergyModel, lk::Layout, zk) =
    [sum(nodevals(zk, lk, E.kspecs, f)) for f in eachindex(E.kspecs)]

function gen_grad(g::GenGroup, Y, Yc, comp, kv, A, dk, gXw, Xw, Gr, Δt, c_viol, κ0)
    T = size(Y, 2)
    dY = zeros(size(Y))
    @inbounds for t in 1:T, i in eachindex(g.bus)
        _, dcy, dck, _ = gen_vint(g, i, A[i, t], kv[i], Yc[i, t])
        Gn = Gr[g.bus[i], t]
        if comp || (0.0 < Y[i, t] < 1.0)
            dY[i, t] = Δt * dcy - A[i, t] * kv[i] * Gn
        else
            dY[i, t] = c_viol * Δt * (kv[i] + κ0) * sign(Y[i, t] - Yc[i, t])
            dk[i] += c_viol * Δt * abs(Y[i, t] - Yc[i, t])
        end
        dk[i] += Δt * dck - A[i, t] * Yc[i, t] * Gn
        if gXw !== nothing
            x = Xw[i, t]
            gXw[i, t] += kv[i] * Yc[i, t] * (Δt * g.mc[i] - Gn) * Float64(0 < x < 1)
        end
    end
    return dY
end

