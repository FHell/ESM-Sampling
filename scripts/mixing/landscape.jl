# The capacity landscape along a carrier direction (docs/mixing_diagnosis.md): the log
# marginal density ℓ(u) of a common log-scale u of all units of one carrier, with the operation
# integrated out, by thermodynamic integration of the mean force dℓ/du = E_{v|k(u)}[lf'(0)].
# At every grid point the capacities are set by the kop map (utilizations fixed, pro-rata
# compensation), the operation is relaxed by local sweeps at fixed (k, x), and the force is
# averaged over further sweeps, in both parametrizations (kop and kscale); they agree when
# the operation is in equilibrium given k.

include(joinpath(@__DIR__, "forces.jl"))

"Set the capacities of the units of `carrier` to e^u times their current values by the kop map (deterministic)."
function kop_scale_carrier(P, LM, s, carrier::String, u::Float64)
    s = copy(s)
    LC = ML.LocalCost(P, LM, s)
    Kraw = ML.knodes(P, LM.ℓ, s.k)
    grp = only(g for g in kop_units(P, LM; kinds=(:renewable, :storage, :dispatch)) if
               (g[1][1] == 1 ? P.E.gens[g[1][2]].carrier : P.E.stors[g[1][2]].carrier) == carrier)
    for (kind, gi, i) in grp
        kf = kind == 1 ? P.E.gens[gi].kf : P.E.stors[gi].kf
        Kraw[kf][i] >= KOP_KMIN || continue
        kt = kop_term(P, LM, LC, Kraw, kind, gi, i)
        k1 = kt.k0 * exp(u)
        kop_eval(P, LM, LC, kt, k1; commit=true, s=s)
        kop_setk!(P, LM, LC, s, kt, k1)
    end
    return ML.evaluate!(P, s)
end

"Carrier force (sum over its units) of a force list."
carrier_force(F, c, key) = sum(getfield(f, key) for f in F if f.carrier == c)

"""
    carrier_profile(rng, P, LM, K, s, carrier; us, nrelax, nmeas) -> NamedTuple

For each u in `us`: kop-scale the carrier, relax the operation (`nrelax` local sweeps), then
average the carrier force over `nmeas` sweeps. Returns the forces and the integrated profile.
"""
function carrier_profile(rng, P, LM, K, s, carrier::String; us=range(-0.4, 0.4, length=9), nrelax::Int=20, nmeas::Int=10)
    fk, fs, parts = Float64[], Float64[], NamedTuple[]
    for u in us
        x = kop_scale_carrier(P, LM, s, carrier, u)
        for _ in 1:nrelax
            x = ML.local_sweep!(rng, P, LM, x; wsets=K.wsets, extra=K.extra)[1]
        end
        acc = zeros(6)
        for _ in 1:nmeas
            x = ML.local_sweep!(rng, P, LM, x; wsets=K.wsets, extra=K.extra)[1]
            F = unit_forces(P, LM, x)
            acc .+= [carrier_force(F, carrier, k) for k in (:total_kop, :total_ks, :capex, :meas, :cost_kop, :cost_ks)]
        end
        acc ./= nmeas
        push!(fk, acc[1]); push!(fs, acc[2])
        push!(parts, (; u, total_kop=acc[1], total_ks=acc[2], capex=acc[3], meas=acc[4], cost_kop=acc[5], cost_ks=acc[6],
                      L=x.L))
        @printf("%-8s u=%+.2f  dℓ/du kop %10.1f  kscale %10.1f  (capex %9.1f meas %8.1f cost_kop %10.1f cost_ks %9.1f)\n",
                carrier, u, acc[1], acc[2], acc[3], acc[4], acc[5], acc[6])
        flush(stdout)
    end
    # trapezoidal integration of the kop-parametrization force
    ℓ = zeros(length(us))
    for j in 2:length(us)
        ℓ[j] = ℓ[j-1] + 0.5 * (fk[j] + fk[j-1]) * (us[j] - us[j-1])
    end
    return (; us=collect(us), fk, fs, ℓ, parts)
end
