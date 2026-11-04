# Renewable units on the device (renewables as load): their index, static parameters, the
# per-chain state (raw capacities and clamped availability) and the write-back to host States.
# Used by the capacity score, the Langevin step and the weather upload (replica sampler) and by
# the Metropolis renewable moves (kren.jl).

"Investable renewable units (renewables as load) in device order: (group, index)."
ren_units(P::ML.Problem) = [(gi, i) for (gi, g) in enumerate(P.E.gens) if g.vf == 0 && g.kf > 0 for i in eachindex(g.bus)]

"Static parameters of the renewable units (type-parametric scalars)."
struct RenParams{F,V,Mx}
    Δt::F; γk::F; c_shed::F; c_dump::F
    H::Mx                 # (U_r, N) node one-hot
    mc::V; lo::V; hi::V; cap::V
end

"Per-chain renewable state: raw capacities (C, U_r) and clamped availability (C, T, U_r)."
mutable struct RenState{A2,A3}
    kr::A2
    A::A3
end

"Renewable parameters and state of the chains `ss`."
function pack_ren(P::ML.Problem, M::DeviceModel, ss::Vector{ML.State})
    E = P.E
    L = ss[1].ℓ
    units = ren_units(P)
    N, T, C = M.N, M.T, length(ss)
    H = zeros(length(units), N)
    mc, lo, hi, cap = zeros(length(units)), zeros(length(units)), zeros(length(units)), zeros(length(units))
    for (q, (gi, i)) in enumerate(units)
        g = E.gens[gi]
        H[q, g.bus[i]] = 1.0
        mc[q], lo[q], hi[q], cap[q] = g.mc[i], E.kspecs[g.kf].lo[i], E.kspecs[g.kf].hi[i], g.capex[i]
    end
    kr, A = zeros(C, length(units)), zeros(C, T, length(units))
    lk = P.lk[L+1]
    for (c, s) in enumerate(ss)
        Xn = ML.weather_nodes(P.W, L, s.xs[end])
        for (q, (gi, i)) in enumerate(units)
            g = E.gens[gi]
            kr[c, q] = vec(ML.nodevals(s.k, lk, E.kspecs, g.kf))[i]
            A[c, :, q] .= clamp.(Xn[g.wf][i, :], 0.0, 1.0)
        end
    end
    return RenParams(M.Δt, M.γk, M.c_shed, M.c_dump, H, mc, lo, hi, cap), RenState(kr, A)
end

"Write the renewable capacities of chain c (device order `ren_units`) into the State `s` (coefficients)."
function ren_to_host!(P::ML.Problem, kr::AbstractMatrix, c::Int, s::ML.State)
    E = P.E
    lk = P.lk[s.ℓ+1]
    units = ren_units(P)
    for (gi, g) in enumerate(E.gens)
        (g.vf == 0 && g.kf > 0) || continue
        qs = [q for (q, (a, i)) in enumerate(units) if a == gi]
        ML.fieldview(s.k, lk, E.kspecs, g.kf) .= ML.cols(E.kspecs[g.kf].B, lk.m[g.kf])' * reshape(kr[c, qs], :, 1)
    end
    return s
end
