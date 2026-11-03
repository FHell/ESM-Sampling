# Recursive decomposed LP start (docs/flow_hierarchy.md §14): a near-optimal
# initial state for large networks from many small LPs, no full-size LP.
#
#   depth 1     countries (cells), joint investment and operation at the coarse
#               time step, transport model between cells on the summed cut
#               capacities of the base lines and links;
#   depth d     inside every parent cell: its children (regions of ≤ cap nodes,
#               or nodes at the last depth), capacity totals per technology
#               group fixed from the parent, flows of base lines already assigned
#               at coarser depths entered as fixed border injections, transfers
#               between the children free (summed cut capacities; real DC lines
#               with KVL once the children are nodes); the transfers are
#               distributed to the base lines by capacity share;
#   operation   per deepest parent region and block of `block_days`, at the fine
#               time step: node capacities fixed, border injections fixed (coarse
#               step constant), states of charge fixed at the block ends.
# The node angles are then recomputed globally from the injections (sparse grid
# solver), so the balance holds exactly; line limits may be exceeded (the sampler
# prices that). All LPs are small; the work is linear in nodes × steps and every
# subproblem of one depth is independent.

"Labels per depth: countries, then (if needed) regions of ≤ cap nodes by recursive Fiedler bisection, then nodes."
function decomposition_partitions(E::EnergyModel; cap::Int=50)
    sys = E.sys
    N = sys.N
    country = E.parts[2]
    W = zeros(N, N)
    for (a, b, s) in zip(sys.line_bus0, sys.line_bus1, sys.line_s)
        W[a, b] += s; W[b, a] += s
    end
    for (a, b, s) in zip(sys.link_bus0, sys.link_bus1, sys.link_cap)
        W[a, b] += s; W[b, a] += s
    end
    regions = zeros(Int, N)
    nreg = 0
    function bisect!(nodes)
        if length(nodes) <= cap
            nreg += 1
            regions[nodes] .= nreg
            return
        end
        Ws = W[nodes, nodes]
        L = Diagonal(vec(sum(Ws; dims=2))) - Ws
        F = eigen(Symmetric(L + 1e-9I))
        v = F.vectors[:, 2]
        m = median(v)
        A, B = nodes[v .<= m], nodes[v .> m]
        if isempty(A) || isempty(B)
            A, B = nodes[1:end÷2], nodes[end÷2+1:end]
        end
        bisect!(A); bisect!(B)
    end
    for c in unique(country)
        bisect!(findall(==(c), country))
    end
    parts = [country]
    nreg > length(unique(country)) && push!(parts, regions)
    push!(parts, collect(1:N))
    return parts
end

"An aggregated unit of a cell LP: a technology group restricted to a cell."
struct DUnit
    kind::Int            # 1 generator, 2 storage
    group::Int
    cell::Int
    members::Vector{Int} # support indices within the group
    kfix::Float64
    kbar::Float64
    invest::Bool
    capex::Float64
    mc::Float64
    A::Vector{Float64}   # availability profile (generators)
    hours::Float64
    eta_in::Float64
    eta_out::Float64
    qt::Float64
end

"""
Aggregated units of the cells `cellids` (labels `lab` over the nodes) from the
energy model's groups, with availability profiles from the node weather `Xn` at
the LP resolution (capacity-weighted means).
"""
function aggregate_units(E::EnergyModel, Xn, lab::Vector{Int}, cellids)
    units = DUnit[]
    cidx = Dict(c => j for (j, c) in enumerate(cellids))
    T = size(Xn[1], 2)
    for (gi, g) in enumerate(E.gens), c in cellids
        mem = [i for (i, n) in enumerate(g.bus) if lab[n] == c]
        isempty(mem) && continue
        inv = g.kf > 0
        wts = inv ? [isfinite(g.kbar[i]) ? g.kbar[i] : 1.0 for i in mem] : g.kfix[mem]
        sw = sum(wts); sw > 0 || (wts = ones(length(mem)); sw = length(mem))
        A = g.wf > 0 ? vec(sum(clamp.(Xn[g.wf][mem, :], 0.0, 1.0) .* wts; dims=1)) ./ sw :
                       fill(sum(g.avail[mem] .* wts) / sw, T)
        push!(units, DUnit(1, gi, cidx[c], mem, sum(g.kfix[mem]), inv ? sum(min.(g.kbar[mem], 1e7)) : sum(g.kfix[mem]),
                           inv, sum(g.capex[mem] .* wts) / sw, sum(g.mc[mem] .* wts) / sw, A, 0.0, 1.0, 1.0, 0.0))
    end
    for (si, s) in enumerate(E.stors), c in cellids
        mem = [i for (i, n) in enumerate(s.bus) if lab[n] == c]
        isempty(mem) && continue
        inv = s.kf > 0
        kb = inv ? min.(s.kbar[mem], 1e7) : s.kfix[mem]
        w = sum(kb); w > 0 || (w = 1.0)
        push!(units, DUnit(2, si, cidx[c], mem, sum(s.kfix[mem]), sum(kb), inv,
                           sum(s.capex[mem] .* kb) / w, sum(s.mc[mem] .* kb) / w, Float64[],
                           sum(s.hours[mem] .* kb) / w, sum(s.eta_in[mem] .* kb) / w, sum(s.eta_out[mem] .* kb) / w,
                           sum(s.qt[mem] .* kb) / w))
    end
    return units
end

"""
    cell_lp(E, nC, D, units, edges; Δt, kmode, ktotal, kfixed, soc_ends, capex_scale)

LP on `nC` cells over the columns of `D` (cell × step demand including fixed net
exports). `edges`: (a, b, cap, b_susc) — b_susc > 0 gives DC flow with angles
(KVL), 0 a transport edge. `kmode`: :free (bounds), :total (Σ over the units of
a group key = `ktotal[key]`, key = (kind, group)), or :fixed (`kfixed[u]`).
`soc_ends`: nothing (cyclic) or Vector of (e_start, e_end) per storage unit (MWh).
Returns a NamedTuple with k, p, pp, pm, e (MWh, steps+1 instants), f, shed, dump.
"""
function cell_lp(E::EnergyModel, nC::Int, D::Matrix{Float64}, units::Vector{DUnit}, edges;
                 Δt::Float64, kmode::Symbol=:free, ktotal=nothing, kfixed=nothing, soc_ends=nothing, capex_scale=1.0)
    T = size(D, 2)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "solver", "ipm"); set_attribute(model, "run_crossover", "off")
    nu = length(units)
    k = Vector{Any}(undef, nu)
    for (u, U) in enumerate(units)
        if kmode == :fixed
            k[u] = kfixed[u]
        elseif U.invest
            k[u] = @variable(model, lower_bound = 0.0, upper_bound = max(U.kbar, 0.0))
        else
            k[u] = U.kfix
        end
    end
    if kmode == :total
        keys_ = unique((U.kind, U.group) for U in units if U.invest)
        for key in keys_
            us = [u for (u, U) in enumerate(units) if U.invest && (U.kind, U.group) == key]
            @constraint(model, sum(k[u] for u in us) == ktotal[key])
        end
    end
    bal = [AffExpr(-D[c, t]) for c in 1:nC, t in 1:T]
    obj = AffExpr(0.0)
    p = Dict{Int,Any}(); pp = Dict{Int,Any}(); pm = Dict{Int,Any}(); e = Dict{Int,Any}()
    for (u, U) in enumerate(units)
        U.invest && kmode != :fixed && add_to_expression!(obj, capex_scale * U.capex, k[u])
        if U.kind == 1
            p[u] = @variable(model, [1:T], lower_bound = 0.0)
            @constraint(model, [t in 1:T], p[u][t] <= U.A[t] * k[u])
            for t in 1:T
                add_to_expression!(bal[U.cell, t], p[u][t])
                add_to_expression!(obj, Δt * U.mc, p[u][t])
            end
        else
            pp[u] = @variable(model, [1:T], lower_bound = 0.0)
            pm[u] = @variable(model, [1:T], lower_bound = 0.0)
            # energy in MW-equivalent units ẽ = e/h (e in MWh): the dynamics row is divided by h,
            # which keeps the LP well scaled for reservoirs with h ≫ Δt
            h = max(U.hours, 1e-6)
            e[u] = @variable(model, [1:T+1], lower_bound = 0.0)
            @constraint(model, [t in 1:T], pp[u][t] <= k[u])
            @constraint(model, [t in 1:T], pm[u][t] <= k[u])
            @constraint(model, [t in 1:T+1], e[u][t] <= k[u])
            @constraint(model, [t in 1:T], e[u][t+1] - e[u][t] <= (Δt / h) * (U.eta_in * pp[u][t] - pm[u][t] / U.eta_out + U.qt * k[u]))
            if soc_ends === nothing
                @constraint(model, e[u][T+1] == e[u][1])
            else
                j = count(x -> x.kind == 2, units[1:u])
                @constraint(model, e[u][1] == soc_ends[j][1] / h)
                @constraint(model, e[u][T+1] >= soc_ends[j][2] / h - 1e-6)
            end
            for t in 1:T
                add_to_expression!(bal[U.cell, t], pm[u][t] - pp[u][t])
                add_to_expression!(obj, Δt * U.mc, pp[u][t] + pm[u][t])
            end
        end
    end
    ne = length(edges)
    f = @variable(model, [1:ne, 1:T])
    dc = any(ed -> ed[4] > 0, edges)
    θ = dc ? @variable(model, [1:nC, 1:T]) : nothing
    for (q, (a, b, cap, bs)) in enumerate(edges)
        @constraint(model, [t in 1:T], -cap <= f[q, t] <= cap)
        bs > 0 && @constraint(model, [t in 1:T], f[q, t] == bs * (θ[a, t] - θ[b, t]))
        for t in 1:T
            add_to_expression!(bal[a, t], -1.0, f[q, t]); add_to_expression!(bal[b, t], f[q, t])
        end
    end
    dc && @constraint(model, [t in 1:T], θ[1, t] == 0.0)
    shed = @variable(model, [1:nC, 1:T], lower_bound = 0.0)
    dump = @variable(model, [1:nC, 1:T], lower_bound = 0.0)
    @constraint(model, [c in 1:nC, t in 1:T], bal[c, t] + shed[c, t] - dump[c, t] == 0.0)
    add_to_expression!(obj, Δt * E.c_shed * sum(shed) + Δt * E.c_dump * sum(dump))
    @objective(model, Min, obj)
    t0 = time()
    optimize!(model)
    st = termination_status(model)
    get(ENV, "DECOMP_DEBUG", "") == "1" && (println(@sprintf("    cell LP %d cells × %d steps (%s): %s in %.1f s", nC, T, kmode, st, time() - t0)); flush(stdout))
    if !has_values(model)                     # IPM without a usable point: retry with dual simplex
        set_attribute(model, "solver", "simplex")
        optimize!(model)
        st = termination_status(model)
    end
    has_values(model) || error("cell LP ($nC cells, $T steps, kmode $kmode): no solution, status $st")
    val(x) = x isa Number ? Float64(x) : value(x)
    return (k=[val(k[u]) for u in 1:nu],
            p=Dict(u => value.(p[u]) for u in keys(p)), pp=Dict(u => value.(pp[u]) for u in keys(pp)),
            pm=Dict(u => value.(pm[u]) for u in keys(pm)), e=Dict(u => max(units[u].hours, 1e-6) .* value.(e[u]) for u in keys(e)),
            f=ne > 0 ? value.(f) : zeros(0, T), shed=value.(shed), dump=value.(dump), status=st)
end

"Time average over blocks of r columns."
blockmean(X::AbstractMatrix, r::Int) = r == 1 ? Matrix(X) : hcat([mean(X[:, (b-1)*r+1:b*r]; dims=2) for b in 1:size(X, 2)÷r]...)

"""
    decomposed_start(E, Xn; h_fine, h_coarse, cap, block_days, verbose) -> (kvals, vvals, info)

Node capacities per k field and node-level operation per v field at the fine
step `h_fine` for the node weather `Xn` (per weather field, support × T_fine,
raw), by the recursive decomposition. `vvals` are in the sampler's coordinates
(angles, link and generator utilizations, states of charge, and charge and
discharge for the LP storage form).
"""
function decomposed_start(E::EnergyModel, Xn; h_fine::Int, h_coarse::Int=24, cap::Int=50, block_days::Int=7, verbose=false)
    sys = E.sys
    N = sys.N
    r = h_coarse ÷ h_fine
    Tf = size(Xn[1], 2)
    Tc = Tf ÷ r
    Xc = [blockmean(x, r) for x in Xn]
    loadf = max.(Xn[1], 0.0)            # support = all nodes
    loadc = blockmean(loadf, r)
    parts = decomposition_partitions(E; cap=cap)
    depth = length(parts)
    # base transfer edges (lines and links) with node ends
    base = [(sys.line_bus0[e], sys.line_bus1[e], sys.line_s[e], sys.line_b[e]) for e in eachindex(sys.line_s)]
    nl = length(base)
    append!(base, [(sys.link_bus0[l], sys.link_bus1[l], sys.link_cap[l], 0.0) for l in eachindex(sys.link_cap)])
    Fe = fill(NaN, length(base), Tc)    # assigned base flows (coarse step), bus0 → bus1
    # capacity totals per (cell at depth d, kind, group)
    Ktot = Dict{Tuple{Int,Int,Int},Float64}()          # totals of the previous depth (labels repeat across depths)
    knode = Dict{Tuple{Int,Int},Vector{Float64}}()     # (kind, group) → node-level capacities over the group support
    socnode = Dict{Tuple{Int,Int},Matrix{Float64}}()   # (stor group, support index) → MWh at coarse instants
    tl = 0.0
    gen_p = Dict(gi => zeros(length(g.bus), Tf) for (gi, g) in enumerate(E.gens))
    st_pp = Dict(si => zeros(length(s.bus), Tf) for (si, s) in enumerate(E.stors))
    st_pm = Dict(si => zeros(length(s.bus), Tf) for (si, s) in enumerate(E.stors))
    st_e = Dict(si => zeros(length(s.bus), Tf) for (si, s) in enumerate(E.stors))
    linkflow = zeros(length(sys.link_cap), Tf)
    for d in 1:depth
        Knew = Dict{Tuple{Int,Int,Int},Float64}()
        lab = parts[d]
        parentlab = d == 1 ? ones(Int, N) : parts[d-1]
        for pc in unique(parentlab)
            nodes = findall(==(pc), parentlab)
            cells = unique(lab[nodes])
            nC = length(cells)
            cid = Dict(c => j for (j, c) in enumerate(cells))
            units = aggregate_units(E, Xc, lab, cells)
            D = zeros(nC, Tc)
            for n in nodes
                D[cid[lab[n]], :] .+= loadc[n, :]
            end
            # internal edges between the children, and fixed border exports from assigned base flows
            edges = Tuple{Int,Int,Float64,Float64}[]
            emap = Dict{Tuple{Int,Int},Int}()
            members = Vector{Vector{Int}}()                 # base edges of each internal (aggregated) edge
            last = d == depth
            for (q, (a, b, cap_, bs)) in enumerate(base)
                ina, inb = parentlab[a] == pc, parentlab[b] == pc
                if ina && inb
                    ca, cb = cid[lab[a]], cid[lab[b]]
                    ca == cb && continue
                    if last
                        push!(edges, (ca, cb, cap_, bs)); push!(members, [q])
                    else
                        key = (min(ca, cb), max(ca, cb))
                        if !haskey(emap, key)
                            push!(edges, (key[1], key[2], 0.0, 0.0)); push!(members, Int[]); emap[key] = length(edges)
                        end
                        j = emap[key]
                        edges[j] = (edges[j][1], edges[j][2], edges[j][3] + cap_, 0.0)
                        push!(members[j], q)
                    end
                elseif ina ⊻ inb
                    any(isnan, view(Fe, q, :)) && continue      # not assigned (cannot happen below depth 1)
                    inside = ina ? a : b
                    sgn = ina ? 1.0 : -1.0                      # export from the inside end
                    D[cid[lab[inside]], :] .+= sgn .* Fe[q, :]
                end
            end
            kmode = d == 1 ? :free : :total
            ktot = d == 1 ? nothing : Dict((U.kind, U.group) => get(Ktot, (pc, U.kind, U.group), 0.0) for U in units if U.invest)
            tl += @elapsed sol = cell_lp(E, nC, D, units, edges; Δt=Float64(h_coarse), kmode=kmode, ktotal=ktot)
            # record totals for the next depth, base flows by capacity share (or directly), node capacities and SoC
            for (u, U) in enumerate(units)
                Knew[(cells[U.cell], U.kind, U.group)] = sol.k[u]
            end
            for (j, mem) in enumerate(members)
                (a1, b1, _, _) = edges[j]
                tot = sum(base[q][3] for q in mem)
                for q in mem
                    a, b = base[q][1], base[q][2]
                    dir = cid[lab[a]] == a1 ? 1.0 : -1.0          # orientation of the base edge vs the internal edge
                    Fe[q, :] .= dir .* sol.f[j, :] .* (base[q][3] / tot)
                end
            end
            if last
                for (u, U) in enumerate(units)
                    grp = (U.kind, U.group)
                    v = get!(() -> zeros(U.kind == 1 ? length(E.gens[U.group].bus) : length(E.stors[U.group].bus)), knode, grp)
                    @assert length(U.members) == 1
                    v[U.members[1]] = U.invest ? sol.k[u] : U.kfix
                    U.kind == 2 && (socnode[(U.group, U.members[1])] = reshape(sol.e[u], 1, :))
                    if r == 1                                 # fine step = coarse step: this is the operation
                        i = U.members[1]
                        if U.kind == 1
                            gen_p[U.group][i, :] .= sol.p[u]
                        else
                            st_pp[U.group][i, :] .= sol.pp[u]; st_pm[U.group][i, :] .= sol.pm[u]; st_e[U.group][i, :] .= sol.e[u][1:end-1]
                        end
                    end
                end
                if r == 1
                    for (j, mem) in enumerate(members)
                        q = mem[1]
                        q > nl && (linkflow[q-nl, :] .= (cid[lab[base[q][1]]] == edges[j][1] ? 1.0 : -1.0) .* sol.f[j, :])
                    end
                end
            end
        end
        Ktot = Knew
        verbose && println(@sprintf("  decomposition depth %d/%d: %d cells, LP time so far %.1f s", d, depth, length(unique(lab)), tl))
    end
    # operation LPs at the fine step, one per block over all regions at once (block diagonal:
    # edges only inside regions, cross-region base flows fixed); not needed if fine = coarse step
    regionlab = depth >= 2 ? parts[depth-1] : ones(Int, N)
    bl = max(1, block_days * 24 ÷ h_coarse)            # coarse steps per block
    to = 0.0
    nblocks = 0
    if r == 1
        for (q, (a, b, _, _)) in enumerate(base)       # links across the deepest regions: assigned flows
            q > nl && regionlab[a] != regionlab[b] && (linkflow[q-nl, :] .= Fe[q, :])
        end
    end
    for b0 in (r > 1 ? (1:bl:Tc) : (1:0))
        nblocks += 1
        b1 = min(b0 + bl - 1, Tc)
        tf = (b0-1)*r+1:b1*r
        nodes = collect(1:N)
        cid = Dict(n => n for n in nodes)
        Xw = [x[:, tf] for x in Xn]
        units = aggregate_units(E, Xw, collect(1:N), nodes)
        D = loadf[:, tf]
        edges = Tuple{Int,Int,Float64,Float64}[]
        eidx = Int[]
        for (q, (a, b, cap_, bs)) in enumerate(base)
            if regionlab[a] == regionlab[b]
                push!(edges, (a, b, cap_, bs)); push!(eidx, q)
            else
                fl = repeat(Fe[q, b0:b1], inner=r)
                D[a, :] .+= fl; D[b, :] .-= fl
                q > nl && (linkflow[q-nl, tf] .= fl)
            end
        end
        kf = Float64[]
        ends = Tuple{Float64,Float64}[]
        for U in units
            grp = (U.kind, U.group)
            push!(kf, knode[grp][U.members[1]])
            if U.kind == 2
                E_ = socnode[(U.group, U.members[1])]
                push!(ends, (E_[1, b0], E_[1, b1+1]))
            end
        end
        to += @elapsed sol = cell_lp(E, N, D, units, edges; Δt=Float64(h_fine), kmode=:fixed, kfixed=kf, soc_ends=ends)
        for (u, U) in enumerate(units)
            i = U.members[1]
            if U.kind == 1
                gen_p[U.group][i, tf] .= sol.p[u]
            else
                st_pp[U.group][i, tf] .= sol.pp[u]; st_pm[U.group][i, tf] .= sol.pm[u]; st_e[U.group][i, tf] .= sol.e[u][1:end-1]
            end
        end
        for (j, q) in enumerate(eidx)
            q > nl && (linkflow[q-nl, tf] .= sol.f[j, :])
        end
    end
    verbose && println(@sprintf("  operation LPs: %d blocks (all regions per block), %.1f s", nblocks, to))
    # node injections → global angles (balance exact); conversion to the sampler's coordinates
    inj = -copy(loadf)
    for (gi, g) in enumerate(E.gens), (i, n) in enumerate(g.bus)
        inj[n, :] .+= gen_p[gi][i, :]
    end
    for (si, s) in enumerate(E.stors), (i, n) in enumerate(s.bus)
        inj[n, :] .+= st_pm[si][i, :] .- st_pp[si][i, :]
    end
    for (l, (a, b)) in enumerate(zip(sys.link_bus0, sys.link_bus1))
        inj[a, :] .-= linkflow[l, :]; inj[b, :] .+= linkflow[l, :]
    end
    GS = GridSolver(sys)
    θ = zeros(N, Tf)
    for c in eachindex(GS.nodes)
        nd = GS.nodes[c]
        length(nd) < 2 && continue
        k_ = length(nd) - 1
        for t in 1:Tf
            bvec = zeros(k_)
            pc = inj[nd, t] .- mean(inj[nd, t])          # component imbalance goes into the residual
            for (j, n) in enumerate(nd)
                GS.loc[n] > 0 && (bvec[GS.loc[n]] = pc[j])
            end
            y = bvec[GS.perm[c]]
            ldiv!(LowerTriangular(GS.Lf[c]), y); ldiv!(UpperTriangular(transpose(GS.Lf[c])), y)
            z = similar(y); z[GS.perm[c]] = y
            for n in nd
                θ[n, t] = GS.loc[n] > 0 ? z[GS.loc[n]] : 0.0
            end
            θ[nd, t] .-= mean(θ[nd, t])
        end
    end
    kvals = Vector{Vector{Float64}}()
    for (f, sp) in enumerate(E.kspecs)
        v = zeros(nsupport(sp.B))
        for (gi, g) in enumerate(E.gens)
            g.kf == f && (v .= knode[(1, gi)])
        end
        for (si, s) in enumerate(E.stors)
            s.kf == f && (v .= knode[(2, si)])
        end
        push!(kvals, v)
    end
    vvals = [zeros(nsupport(s.B), Tf) for s in E.vspecs]
    vvals[E.fθ] .= θ
    E.flink > 0 && (vvals[E.flink] .= linkflow ./ max.(sys.link_cap, 1e-9))
    for (gi, g) in enumerate(E.gens)
        g.vf > 0 || continue
        kv = g.kf > 0 ? knode[(1, gi)] : g.kfix
        A = g.wf > 0 ? clamp.(Xn[g.wf], 0.0, 1.0) : repeat(g.avail, 1, Tf)
        vvals[g.vf] .= [A[i, t] * kv[i] > 1e-6 ? gen_p[gi][i, t] / (A[i, t] * kv[i]) : 0.0 for i in eachindex(g.bus), t in 1:Tf]
    end
    for (si, s) in enumerate(E.stors)
        kv = s.kf > 0 ? knode[(2, si)] : s.kfix
        den = s.hours .* kv
        vvals[s.fe] .= [den[i] > 1e-6 ? st_e[si][i, t] / den[i] : 0.5 for i in eachindex(s.bus), t in 1:Tf]
        if s.fp > 0
            vvals[s.fp] .= [kv[i] > 1e-6 ? st_pp[si][i, t] / kv[i] : 0.0 for i in eachindex(s.bus), t in 1:Tf]
            vvals[s.fm] .= [kv[i] > 1e-6 ? st_pm[si][i, t] / kv[i] : 0.0 for i in eachindex(s.bus), t in 1:Tf]
        end
    end
    return kvals, vvals, (; parts, lp_time=tl + to, alloc_time=tl, op_time=to)
end
