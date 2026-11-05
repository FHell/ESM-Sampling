# LP baselines for the validation (docs/system/validation.md §2, campaign H2): the
# capacity-expansion LP on the historical weather years of the data file, single years and
# stacked over several years (shared capacities, operation per year, equal weights), and
# MGA envelopes (minimum and maximum total capacity per carrier at a cost slack ε).

"""
    historical_weather(E, lv, year) -> Vector{Matrix{Float64}}

Node-level weather of a historical year of the data file at the time step of `lv` (raw,
before the data map): the hourly data `sys.weather[field]` averaged over the steps of the
layout (`aggregate`, as in the weather-model fit). `year` is a calendar year of
`sys.years`. At a spatially complete level this equals the weather model's training data
`weather_nodes(W, ℓ, W.data[ℓ+1][y])` (to round-off).
"""
function historical_weather(E::EnergyModel, lv::Layout, year::Int)
    sys = E.sys
    y = findfirst(==(year), sys.years)
    y === nothing && error("year $year is not in the data file (years $(first(sys.years))–$(last(sys.years)))")
    lv.M >= sys.N || error("historical weather: the level ($(lv.M), $(lv.h)) is not spatially complete ($(sys.N) nodes)")
    out = Matrix{Float64}[]
    for s in E.wspecs
        nodes, X = sys.weather[s.name]
        nodes == s.B.support || error("weather field $(s.name): data nodes differ from the field's support")
        push!(out, aggregate(X[:, :, y:y], lv.h)[:, :, 1])
    end
    return out
end

"The weather after the data map (load ≥ 0, availabilities in [0, 1]): the input of the LP."
datamapped(E::EnergyModel, X) = [map(x -> datamap(E.wspecs[f], x), X[f]) for f in eachindex(X)]

"HiGHS options of the second attempt when a solve does not end optimal: IPX without presolve."
const LP_FALLBACK = ("solver" => "ipx", "presolve" => "off")

"""
Result of the multi-year LP: shared capacities, the operational cost and the summary
statistics of each year (`lp_year_stats`), and per year the `LPSolution` of its block
(objective = capex + that year's opex; duals per unit weight).
"""
struct MultiYearLP
    years::Vector{Int}
    weights::Vector{Float64}
    objective::Float64                      # capex + Σ_y w_y opex_y
    capex::Float64
    k::Vector{Vector{Float64}}
    opex::Vector{Float64}
    stats::Vector{Dict{String,Float64}}
    sols::Vector{LPSolution}
    status::String
end

"""
    solve_lp_years(E, lv, years; weights, keep_model, solver options) -> (MultiYearLP, lpm)

The capacity-expansion LP on the historical `years` (calendar years of the data file) at
the time step of `lv` (spatially complete): shared capacities, one operation block per year
with its storage and commitment cyclic over that year, objective capex + Σ w_y opex_y with
equal weights by default. One year is the single-year LP of `solve_lp`. `lpm` is the built
model (`lp_build`), for the MGA sweep (`mga_sweep!`). A solve that does not end optimal is
repeated once with the options `fallback` (`LP_FALLBACK`).
"""
function solve_lp_years(E::EnergyModel, lv::Layout, years::Vector{Int}; weights=fill(1.0 / length(years), length(years)),
                        fallback=LP_FALLBACK, kw...)
    Ws = [datamapped(E, historical_weather(E, lv, y)) for y in years]
    lpm = lp_build(E, lv, Ws; weights=weights, kw...)
    optimize!(lpm.model)
    st = termination_status(lpm.model)
    defaults = nothing
    if !(st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL))      # HiPO's tolerance check after postsolve: retry with IPX
        @warn "multi-year LP over $years: $st; repeating with IPX without presolve"
        defaults = [k => get_attribute(lpm.model, k) for (k, _) in fallback]
        foreach(((k, v),) -> set_attribute(lpm.model, k, v), fallback)
        optimize!(lpm.model)
        st = termination_status(lpm.model)
    end
    st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || error("multi-year LP over $years: $st")
    kval = [value.(lpm.kv[f]) for f in eachindex(E.kspecs)]
    capex = value(lpm.capex)
    opex = [value(op) for op in lpm.ops]
    sols = [lp_extract(E, lv, kval, lpm.blocks[y], Ws[y], capex + opex[y], capex; weight=weights[y]) for y in eachindex(years)]
    stats = [lp_year_stats(E, lv, lpm.blocks[y], kval, Ws[y]) for y in eachindex(years)]
    res = MultiYearLP(years, collect(Float64, weights), objective_value(lpm.model), capex, kval, opex, stats, sols, string(st))
    defaults === nothing || foreach(((k, v),) -> set_attribute(lpm.model, k, v), defaults)   # (discards the solution)
    return res, lpm
end

"""
Summary of one operation block of a solved LP: shed, dump and renewable curtailment (MWh/a),
CO₂ emissions (t/a), the energy per generator and storage carrier (MWh/a; storage:
discharge), and `curtail_cost_eur`: the cost of renewable output below A·k that the sampler
would charge and the LP would not. Renewables produce A·k exactly in the LP, as in the
sampler (their excess is nodal dump at `c_dump`), so it is zero up to the solver tolerance. One
loophole of the LP remains: simultaneous charge and discharge disposes of energy in the round-trip
losses, which the sampler's SoC form cannot (`storage_cycling_MWh`, `cycling_loss_MWh`; binaries
would be needed to exclude it). The LP objective is therefore a lower bound of the sampler's
minimum, and the sampler's cost at the LP solution is about the objective + `cycling_cost_eur`
(c_dump × the energy lost), an upper bound (test/test_lp_baseline.jl).
"""
function lp_year_stats(E::EnergyModel, lv::Layout, blk, kval, W)
    Δt = Float64(lv.h)
    d = Dict{String,Float64}()
    d["shed_MWh"] = Δt * sum(value.(blk.shed))
    d["dump_MWh"] = Δt * sum(value.(blk.dump))
    curt, ccost, co2 = 0.0, 0.0, 0.0
    for (gi, g) in enumerate(E.gens)
        p, A, _, _, plos = blk.pg[gi]
        pv = value.(p)
        kk = g.kf > 0 ? kval[g.kf] : g.kfix
        d["energy_MWh_" * g.carrier] = get(d, "energy_MWh_" * g.carrier, 0.0) + Δt * sum(pv)
        if g.wf > 0
            cu = A .* kk .- pv
            curt += Δt * sum(max.(cu, 0.0))
            ccost += Δt * sum((E.c_dump .+ g.mc) .* cu)
        end
        for i in eachindex(g.bus)
            if haskey(plos, i)                 # two vintages in merit order
                oldfirst = g.mc0[i] <= g.mc[i]
                elo, ehi = oldfirst ? (g.ef0[i], g.ef[i]) : (g.ef[i], g.ef0[i])
                plo = value.(plos[i])
                co2 += Δt * sum(elo .* plo .+ ehi .* (pv[i, :] .- plo))
            else
                co2 += Δt * g.ef[i] * sum(pv[i, :])
            end
        end
    end
    cyc, closs = 0.0, 0.0
    for (si, s) in enumerate(E.stors)
        pp, pm = value.(blk.ps[si][1]), value.(blk.ps[si][2])
        d["energy_MWh_" * s.carrier] = get(d, "energy_MWh_" * s.carrier, 0.0) + Δt * sum(pm)
        # simultaneous charge and discharge: energy the LP disposes of in the round-trip losses,
        # which the sampler's SoC form cannot (it must dump it at c_dump)
        for i in eachindex(s.bus)
            y = Δt * sum(min.(pp[i, :], pm[i, :]))
            cyc += y
            closs += y * (1 / s.eta_out[i] - s.eta_in[i])
        end
    end
    d["storage_cycling_MWh"] = cyc
    d["cycling_loss_MWh"] = closs
    d["cycling_cost_eur"] = E.c_dump * closs
    d["curtail_MWh"] = curt
    d["curtail_cost_eur"] = ccost
    d["co2_t"] = co2
    return d
end

"Total capacity (MW) per k field and per country: `Dict(field => Dict(country => MW))`."
function capacity_by_country(E::EnergyModel, kval)
    countries = sort(unique(E.sys.country))
    out = Dict{String,Dict{String,Float64}}()
    for (f, sp) in enumerate(E.kspecs)
        bus = sp.B.support
        out[String(sp.name)] = Dict(c => sum((kval[f][i] for i in eachindex(bus) if E.sys.country[bus[i]] == c); init=0.0)
                                    for c in countries)
    end
    return out
end

"""
    mga_sweep!(lpm, E, C, eps, fields; done, on_result, fallback) -> lpm

MGA envelopes on a built and solved LP `lpm` (`lp_build`, optimum `C`): adds the cost bound
`obj ≤ (1 + ε) C` (scaled to a right-hand side of about 1e5) and, for every ε in `eps` and k field `f` in `fields` (indices into
`E.kspecs`), minimises and maximises the total capacity `Σ_i k_f[i]`. The model is reused:
only the bound's right-hand side and the objective change between the solves (HiGHS's
interior-point solver starts each solve afresh).

HiGHS's interior-point solver (HiPO) often reports a primal-feasible point of these
capacity objectives as `OTHER_ERROR` (dual residuals above tolerance after postsolve). A
solve that does not end `OPTIMAL` is repeated with the options `fallback` (default: IPX
without presolve, which converges on them); if that is not optimal either, a primal-feasible
point within the cost bound (relative 1e-6) is kept with the status `feasible:<status>`,
anything else is recorded as failed (NaN).

Calls `on_result(ε, f, sense, r)` after every solve, with
`r = (; status, attempts, cost, total, k, seconds)` (`k`: all fields' node values). Triples
`(ε, f, sense)` in `done` are skipped (a resumed sweep). Afterwards the bound is removed and
the cost objective restored.
"""
function mga_sweep!(lpm, E::EnergyModel, C::Float64, eps, fields; done=Set{Tuple{Float64,Int,Symbol}}(),
                    on_result=(a...) -> nothing, fallback=LP_FALLBACK)
    model = lpm.model
    c0 = constant(lpm.obj)
    # the bound in units of S = 10^(⌊log₁₀ C⌋ − 4) €/a, so that its right-hand side is 1e4–1e5
    # like the other rows': unscaled (1e11) HiPO and IPX stall on the maximisations
    S = 10.0^(floor(log10(abs(C))) - 4)
    cb = @constraint(model, (1 / S) * lpm.obj <= (1 + first(eps)) * C / S)
    defaults = [k => get_attribute(model, k) for (k, _) in fallback]
    optimal(m) = termination_status(m) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
    for ε in eps
        set_normalized_rhs(cb, ((1 + ε) * C - c0) / S)
        for f in fields, sense in (:min, :max)
            (Float64(ε), f, sense) in done && continue
            @objective(model, sense == :min ? MIN_SENSE : MAX_SENSE, sum(lpm.kv[f]))
            t = @elapsed optimize!(model)
            attempts = 1
            if !optimal(model)
                foreach(((k, v),) -> set_attribute(model, k, v), fallback)
                t += @elapsed optimize!(model)
                attempts = 2
            end
            st = string(termination_status(model))
            feas = primal_status(model) in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT) &&
                   value(lpm.obj) <= (1 + ε) * C * (1 + 1e-6)
            ok = optimal(model) || feas
            optimal(model) || (st = feas ? "feasible:" * st : st)
            k = ok ? [value.(lpm.kv[g]) for g in eachindex(E.kspecs)] : [fill(NaN, length(lpm.kv[g])) for g in eachindex(E.kspecs)]
            r = (; status=st, attempts, cost=ok ? value(lpm.obj) : NaN, total=ok ? sum(k[f]) : NaN, k, seconds=t)
            # (after reading the solution: changing an option discards it)
            attempts == 2 && foreach(((k, v),) -> set_attribute(model, k, v), defaults)
            on_result(Float64(ε), f, sense, r)
        end
    end
    delete(model, cb)
    @objective(model, Min, lpm.obj)
    return lpm
end
