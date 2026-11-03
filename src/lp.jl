# Deterministic capacity expansion LP (perfect foresight on one weather
# year) at a spatially complete level. Used to initialize the chains and to
# calibrate the temperatures (§1.5).
#
# The interior-point solution without crossover lies in the relative interior
# of the optimal face. Coordinates pinned there are the cost-confining
# directions (each carries 1/β of expected excess); the others span the flat
# optimal face and carry none.

struct LPSolution
    objective::Float64
    capex::Float64
    k::Vector{Vector{Float64}}           # node values per k field
    v::Vector{Matrix{Float64}}           # node values per v field (utilization coordinates)
    n_v_free::Int                        # operational coordinates strictly inside their bounds
    n_v_total::Int
    n_k_free::Int
    n_k_total::Int
    # zero-temperature scores (sensitivities of the optimal operational cost)
    dF_dX::Vector{Matrix{Float64}}       # w.r.t. node weather per weather field (load, availability)
    dF_dk::Vector{Vector{Float64}}       # w.r.t. node capacities per k field (operational part)
end

"""
    solve_lp(E, lk, lv, W)

`W[f]` are node-level weather values after the data map at the level of
`lv` (which must be complete in space).
"""
function solve_lp(E::EnergyModel, lk::Layout, lv::Layout, W; silent=true, time_limit=Inf,
                  kfixed=nothing, crossover=false, return_model=false, solver="ipm")
    sys = E.sys
    N, T, Δt = sys.N, lv.T, Float64(lv.h)
    model = Model(HiGHS.Optimizer)
    silent && set_silent(model)
    set_attribute(model, "solver", solver)
    set_attribute(model, "run_crossover", crossover ? "on" : "off")
    isfinite(time_limit) && set_attribute(model, "time_limit", Float64(time_limit))

    kv = Dict{Int,Any}()
    for (f, s) in enumerate(E.kspecs)
        lo = kfixed === nothing ? (E.brownfield ? copy(s.lo) : zeros(nsupport(s.B))) : kfixed[f]
        hi = kfixed === nothing ? s.hi : kfixed[f]
        kv[f] = @variable(model, [i in 1:nsupport(s.B)], lower_bound = lo[i], upper_bound = hi[i])
    end
    θ = @variable(model, [1:N, 1:T])
    bal = [AffExpr(W[1][n, t]) for n in 1:N, t in 1:T]
    obj = AffExpr(0.0)
    capex = AffExpr(0.0)
    for g in E.gens, f in (g.kf,)
        f > 0 && add_to_expression!(capex, sum(g.capex .* kv[f]))
        (f > 0 && E.brownfield) && add_to_expression!(capex, -sum(g.capex .* E.kspecs[f].lo))   # sunk existing capacity
    end
    for s in E.stors
        s.kf > 0 && add_to_expression!(capex, sum(s.capex .* kv[s.kf]))
    end
    add_to_expression!(obj, capex)

    for e in eachindex(sys.line_s)
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        fl = @expression(model, [t in 1:T], sys.line_b[e] * (θ[a, t] - θ[b, t]))
        @constraint(model, -sys.line_s[e] .<= fl .<= sys.line_s[e])
        for t in 1:T
            add_to_expression!(bal[a, t], fl[t])
            add_to_expression!(bal[b, t], -1.0, fl[t])
        end
    end
    u = nothing
    if E.flink > 0
        nl = length(sys.link_cap)
        # signed lossy link: u = u⁺ − u⁻, losses (1 − η)|u| at the receiving end
        up = @variable(model, [l in 1:nl, t in 1:T], lower_bound = 0.0, upper_bound = sys.link_cap[l])
        um = @variable(model, [l in 1:nl, t in 1:T], lower_bound = 0.0, upper_bound = sys.link_cap[l])
        u = (up, um)
        for l in 1:nl, t in 1:T
            η = E.link_eta[l]
            add_to_expression!(bal[sys.link_bus0[l], t], up[l, t]); add_to_expression!(bal[sys.link_bus0[l], t], -η, um[l, t])
            add_to_expression!(bal[sys.link_bus1[l], t], um[l, t]); add_to_expression!(bal[sys.link_bus1[l], t], -η, up[l, t])
        end
    end
    pg = Dict{Int,Any}()
    for (gi, g) in enumerate(E.gens)
        S = length(g.bus)
        A = g.wf > 0 ? W[g.wf] : repeat(g.avail, 1, T)
        p = @variable(model, [1:S, 1:T], lower_bound = 0.0)
        pc = Matrix{ConstraintRef}(undef, S, T)
        κ = g.ucf > 0 ? @variable(model, [1:S, 1:T], lower_bound = 0.0) : nothing   # committed capacity
        pg[gi] = (p, A, pc, κ)
        gmin, su, nl_ = g.uc
        for i in 1:S
            vint = hasvintage(g, i)
            if vint                                   # merit order of the two vintages (convex)
                plo = @variable(model, [1:T], lower_bound = 0.0)
                oldfirst = g.mc0[i] <= g.mc[i]
                clo, chi = oldfirst ? (g.mc0[i], g.mc[i]) : (g.mc[i], g.mc0[i])
            end
            for t in 1:T
                cap = g.kf > 0 ? A[i, t] * kv[g.kf][i] : A[i, t] * g.kfix[i]
                if κ === nothing
                    pc[i, t] = @constraint(model, p[i, t] <= cap)
                else
                    pc[i, t] = @constraint(model, κ[i, t] <= cap)
                    @constraint(model, p[i, t] <= κ[i, t])
                    @constraint(model, p[i, t] >= gmin * κ[i, t])
                    sgm = @variable(model, lower_bound = 0.0)
                    @constraint(model, sgm >= κ[i, mod(t, T)+1] - κ[i, t])
                    add_to_expression!(obj, Δt * nl_, κ[i, t])
                    add_to_expression!(obj, su, sgm)
                end
                add_to_expression!(bal[g.bus[i], t], -1.0, p[i, t])
                if vint
                    qlo = oldfirst ? A[i, t] * g.k0[i] : (g.kf > 0 ? A[i, t] * (kv[g.kf][i] - g.k0[i]) : 0.0)
                    @constraint(model, plo[t] <= qlo)
                    @constraint(model, plo[t] <= p[i, t])
                    add_to_expression!(obj, Δt * clo, plo[t])
                    add_to_expression!(obj, Δt * chi, p[i, t] - plo[t])
                else
                    add_to_expression!(obj, Δt * g.mc[i], p[i, t])
                end
            end
        end
    end
    ps = Dict{Int,Any}()
    for (si, s) in enumerate(E.stors)
        S = length(s.bus)
        pp = @variable(model, [1:S, 1:T], lower_bound = 0.0)
        pm = @variable(model, [1:S, 1:T], lower_bound = 0.0)
        e = @variable(model, [1:S, 1:T], lower_bound = 0.0)
        crefs = ConstraintRef[]
        ps[si] = (pp, pm, e, crefs)
        for i in 1:S
            K = s.kf > 0 ? kv[s.kf][i] : s.kfix[i]
            for t in 1:T
                push!(crefs, @constraint(model, pp[i, t] <= K))
                push!(crefs, @constraint(model, pm[i, t] <= K))
                push!(crefs, @constraint(model, e[i, t] <= s.hours[i] * K))
                t1 = mod(t, T) + 1
                @constraint(model, e[i, t1] - e[i, t] <= Δt * (s.eta_in[i] * pp[i, t] - pm[i, t] / s.eta_out[i] + s.qt[i] * K))
                add_to_expression!(bal[s.bus[i], t], pp[i, t])
                add_to_expression!(bal[s.bus[i], t], -1.0, pm[i, t])
                add_to_expression!(obj, Δt * s.mc[i], pp[i, t])
                add_to_expression!(obj, Δt * s.mc[i], pm[i, t])
            end
        end
    end
    shed = @variable(model, [1:N, 1:T], lower_bound = 0.0)
    dump = @variable(model, [1:N, 1:T], lower_bound = 0.0)
    balc = @constraint(model, bal .== shed .- dump)
    add_to_expression!(obj, Δt * E.c_shed * sum(shed) + Δt * E.c_dump * sum(dump))
    @objective(model, Min, obj)
    optimize!(model)
    @assert termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) "LP: $(termination_status(model))"

    # ---- convert to utilization coordinates and count unpinned coordinates
    rel(x, lo, hi) = (hi - lo) > 1e-9 && (x - lo) > 1e-6 * (hi - lo) && (hi - x) > 1e-6 * (hi - lo)
    kval = [value.(kv[f]) for f in eachindex(E.kspecs)]
    n_k_free = sum(count(i -> rel(kval[f][i], 0.0, E.kspecs[f].hi[i]), eachindex(kval[f])) for f in eachindex(kval); init=0)
    n_k_total = sum(length, kval)
    ntot = 0
    vvals = [zeros(nsupport(s.B), ntime(lv, s)) for s in E.vspecs]
    nfree = 0
    θv = value.(θ)
    vvals[E.fθ] .= θv .- mean(θv; dims=1)
    nfree += (N - 1) * T
    ntot += (N - 1) * T
    for t in 1:T, e in eachindex(sys.line_s)
        fl = sys.line_b[e] * (θv[sys.line_bus0[e], t] - θv[sys.line_bus1[e], t])
        abs(fl) > sys.line_s[e] * (1 - 1e-6) && (nfree -= 1)
    end
    if E.flink > 0
        uv = value.(u[1]) .- value.(u[2])
        vvals[E.flink] .= uv ./ sys.link_cap
        nfree += count(l -> abs(uv[l]) < sys.link_cap[l[1]] * (1 - 1e-6), CartesianIndices(uv))
        ntot += length(uv)
    end
    for (gi, g) in enumerate(E.gens)
        g.vf > 0 || continue                     # renewables as negative load: no dispatch coordinate
        p, A, _, κ = pg[gi]
        pv = value.(p)
        κv = κ === nothing ? nothing : value.(κ)
        for i in eachindex(g.bus), t in 1:T
            K = (g.kf > 0 ? kval[g.kf][i] : g.kfix[i]) * A[i, t]
            vvals[g.vf][i, t] = K > 1e-6 ? clamp(pv[i, t] / K, 0.0, 1.0) : 0.5
            K > 1e-6 && (ntot += 1; rel(pv[i, t], 0.0, K) && (nfree += 1))
            κv === nothing || (vvals[g.ucf][i, t] = K > 1e-6 ? clamp(κv[i, t] / K, 0.0, 1.0) : 0.5)
        end
    end
    for (si, s) in enumerate(E.stors)
        pp, pm, e, _ = ps[si]
        ppv, pmv, ev = value.(pp), value.(pm), value.(e)
        for i in eachindex(s.bus), t in 1:T
            K = s.kf > 0 ? kval[s.kf][i] : s.kfix[i]
            # one coordinate per instant (state of charge); confined if the state of
            # charge is at a bound or the unit idles/saturates on either side
            ok = K > 1e-6 && s.hours[i] > 1e-6
            vvals[s.fe][i, t] = ok ? clamp(ev[i, t] / (s.hours[i] * K), 0.0, 1.0) : 0.5
            if ok
                tm = mod(t - 2, T) + 1
                idle(τ) = ppv[i, τ] < 1e-6 * K && pmv[i, τ] < 1e-6 * K
                sat(τ) = ppv[i, τ] > K * (1 - 1e-6) || pmv[i, τ] > K * (1 - 1e-6)
                free = rel(ev[i, t], 0, s.hours[i] * K) && !idle(t) && !idle(tm) && !sat(t) && !sat(tm)
                nfree += free; ntot += 1
            end
        end
    end
    # sensitivities: ∂obj/∂rhs = dual (minimization). Load sits on the rhs of the
    # balance with a minus sign; a capacity constraint p ≤ A·k has rhs A·k.
    dF_dX = [zeros(size(w)) for w in W]
    dF_dX[1] .= -dual.(balc)
    dF_dk = [zeros(length(x)) for x in kval]
    for (gi, g) in enumerate(E.gens)
        _, A, pc, _ = pg[gi]
        μ = dual.(pc)                    # ≤ 0
        kk = g.kf > 0 ? kval[g.kf] : g.kfix
        g.wf > 0 && (dF_dX[g.wf] .+= μ .* kk)
        g.kf > 0 && (dF_dk[g.kf] .+= vec(sum(μ .* A; dims=2)))
    end
    for (si, s) in enumerate(E.stors)
        s.kf > 0 || continue
        _, _, _, crefs = ps[si]
        μ = reshape(dual.(crefs), 3, lv.T, length(s.bus))
        for i in eachindex(s.bus)
            dF_dk[s.kf][i] = sum(μ[1, :, i]) + sum(μ[2, :, i]) + s.hours[i] * sum(μ[3, :, i])
        end
    end
    sol = LPSolution(objective_value(model), value(capex), kval, vvals, nfree, ntot, n_k_free, n_k_total, dF_dX, dF_dk)
    return return_model ? (sol, model, balc) : sol
end

"A strictly interior reference point of the operation (node values per v field)."
function interior_center(E::EnergyModel, lv::Layout)
    out = Matrix{Float64}[]
    for (f, s) in enumerate(E.vspecs)
        S, nt = nsupport(s.B), ntime(lv, s)
        val = f == E.fθ || f == E.flink ? 0.0 : 0.5
        push!(out, fill(val, S, nt))
    end
    return out
end

"Mode coordinates of node-level field values on a layout."
function to_modes(lay::Layout, specs, vals)
    z = zeros(lay.n)
    for f in eachindex(specs)
        U = cols(specs[f].B, lay.m[f])
        fieldview(z, lay, specs, f) .= U' * reshape(vals[f], size(U, 1), :)
    end
    return z
end

"Solve the LP or load it from an HDF5 cache keyed by `key`."
function solve_lp_cached(E::EnergyModel, lk::Layout, lv::Layout, W, path::String)
    if isfile(path)
        return h5open(path, "r") do f
            nk, nv = read(f["nk"]), read(f["nv"])
            LPSolution(read(f["objective"]), read(f["capex"]), [read(f["k$i"]) for i in 1:nk],
                       [read(f["v$i"]) for i in 1:nv], read(f["n_v_free"]), read(f["n_v_total"]),
                       read(f["n_k_free"]), read(f["n_k_total"]),
                       [read(f["dX$i"]) for i in 1:read(f["nw"])], [read(f["dk$i"]) for i in 1:nk])
        end
    end
    lp = solve_lp(E, lk, lv, W)
    mkpath(dirname(path))
    h5open(path, "w") do f
        f["nk"], f["nv"] = length(lp.k), length(lp.v)
        f["objective"], f["capex"] = lp.objective, lp.capex
        f["n_v_free"], f["n_v_total"], f["n_k_free"], f["n_k_total"] = lp.n_v_free, lp.n_v_total, lp.n_k_free, lp.n_k_total
        foreach(i -> (f["k$i"] = lp.k[i]), eachindex(lp.k))
        foreach(i -> (f["v$i"] = lp.v[i]), eachindex(lp.v))
        f["nw"] = length(lp.dF_dX)
        foreach(i -> (f["dX$i"] = lp.dF_dX[i]), eachindex(lp.dF_dX))
        foreach(i -> (f["dk$i"] = lp.dF_dk[i]), eachindex(lp.dF_dk))
    end
    return lp
end
