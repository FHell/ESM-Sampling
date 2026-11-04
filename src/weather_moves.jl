# Weather moves with built-in shear at level 0 (v2 §5.1(4); the weather move of
# the transport-first formulation, docs/archive/sampling_formulations.md §8.4).
#
# A coordinate update of the weather at fixed dispatch changes the nodal residual
# and is blocked by the balance. Here a weather coordinate z_j at step τ moves
# together with a pro-rata re-dispatch: the requirement change of every balance
# cell c of the level, g_c (load up, or renewable output down, per unit t), is
# covered by the dispatchable carriers of the cell in proportion to their
# current output, σ_f = P_{f,c}/P_c, as a cell-constant change of their
# utilization (in the span of the level). The coefficients use only quantities
# that do not change along the line (σ, capacities, and the renewable
# utilizations, which the move leaves alone), so the direction is invariant and
# the move reversible; the conditional along t carries Π_c (P_c(t)/P_c)^{m_c−1}.
# The target of the line is the full level-0 term: β_v cost + bias + weather prior
# of steps τ and τ+1. Sampled by slice sampling (the prior and bias are smooth).

"""
    weather_shear_sweep!(rng, P, G, s; κmin) -> (s, nmoved, nmoves)

One pass over all level-0 weather coordinates with shear, colour classes mod 5 in
random order and steps of a class in parallel.
"""
function weather_shear_sweep!(rng, P::Problem, G::GibbsState, s::State; κmin::Float64=1.0)
    E, W = P.E, P.W
    @assert s.ℓ == 0
    lv, lw = P.lv[1], W.lays[1]
    T = lv.T
    N = E.sys.N
    ctx = SweepCtx(P, s.k)
    Bl = E.wspecs[1].B
    mr = modes_at(Bl, lv.M)
    cell = mr == N ? collect(1:N) : Bl.cells[mr]
    ncell = maximum(cell)
    comp = [gi for (gi, g) in enumerate(E.gens) if g.wf == 0 && g.vf > 0]   # dispatchable carriers
    # capacities per (carrier, cell) and the support positions of each cell
    Kc = [zeros(ncell) for _ in comp]
    members = [[Int[] for _ in 1:ncell] for _ in comp]
    for (q, gi) in enumerate(comp), (i, n) in enumerate(E.gens[gi].bus)
        Kc[q][cell[n]] += E.gens[gi].avail[i] * ctx.kgen[gi][i]
        push!(members[q][cell[n]], i)
    end
    nmoved = Threads.Atomic{Int}(0)
    ntot = Threads.Atomic{Int}(0)
    for c in shuffle(rng, collect(1:NCOLOURS))
        steps = collect(c:NCOLOURS:T)
        seeds = rand(rng, UInt64, length(steps))
        Threads.@threads :static for q in eachindex(steps)
            r = Xoshiro(seeds[q])
            ws = G.wss[Threads.threadid()]
            τ = steps[q]
            term(t) = step_term(P, 0, ctx, ws, s.v, s.z0, s.z0, s.m, t)
            coords = [(f, j) for f in eachindex(W.specs) for j in 1:lw.m[f]]
            shuffle!(r, coords)
            for (f, j) in coords
                term(τ)                           # fills ws.y (node values of v) and ws.x at τ
                # requirement change per cell per unit t
                gcell = zeros(ncell)
                U = W.specs[f].B.U
                sup = W.specs[f].B.support
                for qq in nzrange(U, j)
                    pos, a = rowvals(U)[qq], nonzeros(U)[qq] * W.scale[f]
                    n = sup[pos]
                    if f == 1
                        gcell[cell[n]] += a                        # load
                    else
                        for (gi, g) in enumerate(E.gens)           # renewable output A·k·ω at this node
                            g.wf == f || continue
                            ω = g.vf > 0 ? ws.y[g.vf][pos] : 1.0
                            gcell[cell[n]] -= a * ctx.kgen[gi][pos] * ω
                        end
                    end
                end
                # pro-rata compensation (coefficient changes of the v fields at τ)
                vidx, vcoef = Int[], Float64[]
                jac = NTuple{3,Float64}[]
                for cl in 1:ncell
                    gc = gcell[cl]
                    abs(gc) < 1e-9 && continue
                    Pf = zeros(length(comp))
                    for (qc, gi) in enumerate(comp)
                        g = E.gens[gi]
                        Kc[qc][cl] >= κmin || continue
                        for i in members[qc][cl]
                            Pf[qc] += g.avail[i] * ctx.kgen[gi][i] * ws.y[g.vf][i]
                        end
                    end
                    Pc = sum(Pf)
                    abs(Pc) < 1e-6 && continue
                    nm = count(!iszero, Pf)
                    for (qc, gi) in enumerate(comp)
                        Pf[qc] == 0.0 && continue
                        g = E.gens[gi]
                        Δω = (Pf[qc] / Pc) * gc / Kc[qc][cl]
                        Ut = E.vspecs[g.vf].B.Ut
                        m = lv.m[g.vf]
                        o = lv.off[g.vf] + (τ - 1) * m
                        for i in members[qc][cl], qq in nzrange(Ut, i)
                            jj = rowvals(Ut)[qq]
                            jj <= m || continue
                            push!(vidx, o + jj); push!(vcoef, Δω * nonzeros(Ut)[qq])
                        end
                    end
                    push!(jac, (Pc, gc, Float64(nm - 1)))
                end
                iz = lw.off[f] + (τ - 1) * lw.m[f] + j
                z0b = s.z0[iz]
                vb = s.v[vidx]
                function setto!(t)
                    s.z0[iz] = z0b + t
                    @inbounds for k in eachindex(vidx)
                        s.v[vidx[k]] = vb[k]
                    end
                    @inbounds for k in eachindex(vidx)
                        s.v[vidx[k]] += t * vcoef[k]
                    end
                end
                lf(t) = (setto!(t); lj = logjac_lin(jac, t); isfinite(lj) ? -(term(τ) + (τ < T ? term(τ + 1) : 0.0)) + lj : -Inf)
                t = slice_scalar(r, lf, 0.0, G.wz[f], -Inf)
                setto!(t)
                Threads.atomic_add!(ntot, 1)
                t != 0.0 && Threads.atomic_add!(nmoved, 1)
            end
        end
    end
    s.xs = [s.z0]
    return evaluate!(P, s), nmoved[], ntot[]
end
