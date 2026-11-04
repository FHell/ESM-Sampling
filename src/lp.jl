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
`lv` (which must be complete in space). The environment variable `ESM_LP_THREADS`, if
set, is HiGHS's `threads` option (otherwise HiGHS's default). With `return_model = true`
also returns the JuMP model and the balance constraints, `(sol, model, balc)`; the
multi-year LP and the MGA sweep (lp_baseline.jl) use `lp_build` directly.
"""
function solve_lp(E::EnergyModel, lk::Layout, lv::Layout, W; silent=true, time_limit=Inf,
                  kfixed=nothing, crossover=false, return_model=false, solver="ipm")
    lpm = lp_build(E, lv, [W]; silent=silent, time_limit=time_limit, kfixed=kfixed, crossover=crossover, solver=solver)
    model = lpm.model
    optimize!(model)
    @assert termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) "LP: $(termination_status(model))"
    kval = [value.(lpm.kv[f]) for f in eachindex(E.kspecs)]
    sol = lp_extract(E, lv, kval, lpm.blocks[1], W, objective_value(model), value(lpm.capex))
    return return_model ? (sol, model, lpm.blocks[1].balc) : sol
end

"A HiGHS model with the options of the LPs (`ESM_LP_THREADS`: HiGHS threads)."
function lp_model(; silent=true, time_limit=Inf, crossover=false, solver="ipm")
    model = Model(HiGHS.Optimizer)
    silent && set_silent(model)
    set_attribute(model, "solver", solver)
    set_attribute(model, "run_crossover", crossover ? "on" : "off")
    isfinite(time_limit) && set_attribute(model, "time_limit", Float64(time_limit))
    lpth = get(ENV, "ESM_LP_THREADS", "")          # shared nodes: HiGHS must not take every core (docs/system/hpc.md §3)
    if !isempty(lpth)
        set_attribute(model, "threads", parse(Int, lpth))
        # HiGHS keeps one global thread scheduler, created with its default size (half the
        # hardware threads) before any option is read; a different `threads` then makes the
        # solve fail ("global scheduler has already been initialized"). Resetting it makes the
        # next solve create it with this option. (One LP at a time per process.)
        HiGHS.Highs_resetGlobalScheduler(1)
    end
    return model
end

"""
    lp_build(E, lv, Ws; weights, kfixed, solver options) -> (; model, kv, capex, obj, ops, blocks)

The capacity-expansion LP over the weather years `Ws` (node-level weather after the data
map, one entry per year, each at the time step of `lv`): shared capacities `kv`, one
operation block per year with its own cyclic storage and commitment (`lp_operation!`),
objective `capex + Σ_y weights[y] · ops[y]` (`obj`). One year with weight 1 is the
single-year LP of `solve_lp`. Not solved.
"""
function lp_build(E::EnergyModel, lv::Layout, Ws; weights=fill(1.0 / length(Ws), length(Ws)), kfixed=nothing, kw...)
    @assert length(weights) == length(Ws)
    model = lp_model(; kw...)
    kv = Dict{Int,Any}()
    for (f, s) in enumerate(E.kspecs)
        lo = kfixed === nothing ? (E.brownfield ? copy(s.lo) : zeros(nsupport(s.B))) : kfixed[f]
        hi = kfixed === nothing ? s.hi : kfixed[f]
        kv[f] = @variable(model, [i in 1:nsupport(s.B)], lower_bound = lo[i], upper_bound = hi[i])
    end
    capex = AffExpr(0.0)
    for g in E.gens, f in (g.kf,)
        f > 0 && add_to_expression!(capex, sum(g.capex .* kv[f]))
        (f > 0 && E.brownfield) && add_to_expression!(capex, -sum(g.capex .* E.kspecs[f].lo))   # sunk existing capacity
    end
    for s in E.stors
        s.kf > 0 && add_to_expression!(capex, sum(s.capex .* kv[s.kf]))
    end
    obj = AffExpr(0.0)
    add_to_expression!(obj, capex)
    ops, blocks = AffExpr[], Any[]
    for (y, W) in enumerate(Ws)
        op, blk = lp_operation!(model, E, lv, kv, W)
        add_to_expression!(obj, weights[y], op)
        push!(ops, op); push!(blocks, blk)
    end
    @objective(model, Min, obj)
    return (; model, kv, capex, obj, ops, blocks, weights)
end

"""
The operation of one weather year `W` at the capacities `kv`: potentials and line flows,
DC links, dispatch (vintages, commitment), storage (cyclic over the year), shed and dump,
the nodal balance. Returns `(op, block)`: the operational cost and the variables.
"""
function lp_operation!(model, E::EnergyModel, lv::Layout, kv, W)
    sys = E.sys
    N, T, Δt = sys.N, lv.T, Float64(lv.h)
    θ = @variable(model, [1:N, 1:T])
    bal = [AffExpr(W[1][n, t]) for n in 1:N, t in 1:T]
    obj = AffExpr(0.0)
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
        plos = Dict{Int,Any}()
        pg[gi] = (p, A, pc, κ, plos)
        gmin, su, nl_ = g.uc
        for i in 1:S
            vint = hasvintage(g, i)
            if vint                                   # merit order of the two vintages (convex)
                plo = @variable(model, [1:T], lower_bound = 0.0)
                plos[i] = plo
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
    return obj, (; θ, u, pg, ps, shed, dump, balc)
end

"""
The `LPSolution` of one operation block of a solved LP, at the capacities `kval`: the
operation in utilization coordinates, the free-coordinate counts and the zero-temperature
scores. The duals are divided by `weight` (the block's weight in the objective), so that
they are those of the block's own year.
"""
function lp_extract(E::EnergyModel, lv::Layout, kval, blk, W, objective, capex; weight=1.0)
    sys = E.sys
    N, T = sys.N, lv.T
    (; θ, u, pg, ps, balc) = blk
    # ---- convert to utilization coordinates and count unpinned coordinates
    rel(x, lo, hi) = (hi - lo) > 1e-9 && (x - lo) > 1e-6 * (hi - lo) && (hi - x) > 1e-6 * (hi - lo)
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
        p, A, _, κ, _ = pg[gi]
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
    dF_dX[1] .= -dual.(balc) ./ weight
    dF_dk = [zeros(length(x)) for x in kval]
    for (gi, g) in enumerate(E.gens)
        _, A, pc, _, _ = pg[gi]
        μ = dual.(pc) ./ weight           # ≤ 0
        kk = g.kf > 0 ? kval[g.kf] : g.kfix
        g.wf > 0 && (dF_dX[g.wf] .+= μ .* kk)
        g.kf > 0 && (dF_dk[g.kf] .+= vec(sum(μ .* A; dims=2)))
    end
    for (si, s) in enumerate(E.stors)
        s.kf > 0 || continue
        _, _, _, crefs = ps[si]
        μ = reshape(dual.(crefs), 3, lv.T, length(s.bus)) ./ weight
        for i in eachindex(s.bus)
            dF_dk[s.kf][i] = sum(μ[1, :, i]) + sum(μ[2, :, i]) + s.hours[i] * sum(μ[3, :, i])
        end
    end
    return LPSolution(objective, capex, kval, vvals, nfree, ntot, n_k_free, n_k_total, dF_dX, dF_dk)
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

"""
Solve the LP or load it from the HDF5 cache entry `path` (or from the read-only `mirror`
directory, copied in). The entry is written atomically (`atomic_write`).
"""
function solve_lp_cached(E::EnergyModel, lk::Layout, lv::Layout, W, path::String; mirror::String="")
    rd(p) = h5open(p, "r") do f
        nk, nv = read(f["nk"]), read(f["nv"])
        LPSolution(read(f["objective"]), read(f["capex"]), [read(f["k$i"]) for i in 1:nk],
                   [read(f["v$i"]) for i in 1:nv], read(f["n_v_free"]), read(f["n_v_total"]),
                   read(f["n_k_free"]), read(f["n_k_total"]),
                   [read(f["dX$i"]) for i in 1:read(f["nw"])], [read(f["dk$i"]) for i in 1:nk])
    end
    lp0 = cache_read(rd, path, mirror)
    lp0 === nothing || return lp0
    lp = solve_lp(E, lk, lv, W)
    atomic_write(path) do tmp
        write_lp_entry(tmp, lp)
    end
    return lp
end

"Write an LP solution as an HDF5 cache entry."
function write_lp_entry(path::String, lp::LPSolution)
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
    return path
end
