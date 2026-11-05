# Residual cache rebuild at the host checkpoints (docs/system/device.md §6.3). The device loop
# carries the nodal residual r as a cache that every commit updates by a scatter-add of its
# injection change, and the capacity and weather programs shift it. In FP64 the cache drifts by
# about 1e-10 MW per 100 sweeps; in Float32 a scatter-add at |r| ≈ 1e5 MW rounds at ≈ 1e-2 MW and
# the drift grows with the number of moves, which moves the balance potential and the shed/dump
# roots of the line draws. Both drivers therefore rebuild r from the node arrays (`cost_parts`) at
# every host checkpoint and record the drift max |r_cache − r_rebuilt| per chain (replica).

"""
    rebuild_flat(st, kg, ks, dnet, cren, capren, M, Val(dims)) -> (st′, drift, op, shed)

The loop state `st` with its residual cache rebuilt from the node arrays (`cost_parts` of the
device state), the drift max |r_cache − r_rebuilt| per chain (C,), and the operation cost and the
shed (MWh) per chain of the state. dims = (C, T, N, U_g, U_s, U_l, …). Traceable.
"""
function rebuild_flat(st::FlatState, kg, ks, dnet, cren, capren, M::DeviceModel, ::Val{dims}) where {dims}
    C, T, N, Ug, Us, Ul = dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]
    m3(x, n) = reshape(x, C, T, n) .+ 0.0          # materialized (a lazy reshape of a traced array has another type)
    S = DeviceState(m3(st.θ, N), m3(st.ω, Ug), m3(st.uc, Ug), m3(st.e, Us), m3(st.u, max(Ul, 1)), kg, ks, dnet, cren, capren)
    r, parts = cost_parts(M, S)
    rf = reshape(r, C, :) .+ 0.0
    d = rf .- st.r
    return FlatState(st.θ, st.ω, st.uc, st.e, st.u, rf), vec(maximum(max.(d, -d); dims=2)), parts.op, parts.shed
end

"""
    rebuild_program(M, dims, R; compile) -> (f, compile seconds)

`rebuild_flat` with the device model uploaded once (`R`, the driver's upload): f(st, kg, ks, dnet,
cren, capren) -> (st′, drift, op, shed), compiled by Reactant (`ftcompile`) or the plain function
on Arrays. The program changes none of its inputs, so a caller may also use only its costs.
"""
function rebuild_program(M::DeviceModel, dims::Val, R, args; compile::Bool)
    Md = R(M)
    f(st, kg, ks, dn, cr, cap, Mx) = rebuild_flat(st, kg, ks, dn, cr, cap, Mx, dims)
    compile || return ((a...) -> f(a..., Md)), 0.0
    tc = @elapsed fc = ftcompile(() -> @compile f(args..., Md))
    return ((a...) -> fc(a..., Md)), tc
end

"The dims of `rebuild_flat` for C rows of model M: (C, T, N, U_g, U_s, U_l)."
rebuild_dims(C::Int, M::DeviceModel, ix::DeviceIndex) = Val((C, M.T, M.N, length(ix.gidx), length(ix.sidx), length(M.kcap)))

"Drift records (one vector per host checkpoint) as a matrix chain × checkpoint (0 × 0 if empty)."
drift_matrix(v) = isempty(v) ? zeros(0, 0) : reduce(hcat, v)

"The drift records of a resumed Metropolis run (snapshot entry `device/residual_drift`), or none."
function resume_drift(snap)
    snap === nothing && return Vector{Vector{Float64}}()
    D = get(snap.extra, "residual_drift", zeros(0, 0))
    return [Float64.(D[:, j]) for j in 1:size(D, 2)]
end
