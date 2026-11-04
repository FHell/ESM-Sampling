# CPU references of the device families (phase-2 tests): the same move built with the CPU
# `prep_node!`/`finalize!`/`link_soc!` and evaluated as in `local_sweep!` (BlockMove over τ−1, τ).

"CPU move of family `fam` at `site` (device indices) on LocalCost LC; returns the BlockMove (or nothing if masked)."
function cpu_move(P, LM, LC, ix, fam, site; κmin=1.0, jac=NTuple{5,Float64}[], learn=nothing, chain=1)
    E = P.E
    T = LM.T
    ws, wa = ML.MoveWS(P), ML.MoveWS(P); wa.follow = false
    fam == :spair0 && (fam = :spair)                          # the same CPU move (spair)
    τ = fam == :spair ? site[2] : (fam in D.TDIR_FAMILIES || haskey(D.SPARSE_FAMILIES, fam)) ? site[1] : site[end]; τm = mod1(τ - 1, T)   # spair: (s, τ, q1, q2); tdir: (τ, d, g)
    gi(q) = ix.gidx[q]
    sf = string(fam)
    if startswith(sf, "dshp")                                  # dispatch shape of a unit pair (`prep_adshape!`)
        L = parse(Int, replace(sf, "dshp" => ""))
        q1, q2, b0, p = site
        bm = ML.BlockMove([ML.MoveWS(P) for _ in 1:L+2], 0, Int[], P, LM, LC, Float64[], Float64[])
        (g1, i1), (g2, i2) = gi(q1), gi(q2)
        ML.prep_adshape!(bm, P, LM, LC, (E.gens[g1].vf, i1), (E.gens[g2].vf, i2), D.dshapes(L)[:, p], b0, L; κmin=κmin) || return nothing
        return bm
    end
    if startswith(sf, "sblk")                                  # storage block (plateau or shape profile)
        L = parse(Int, replace(sf, "sblkL" => "", "sblk" => ""))
        s, b0, p = site
        si, i = ix.sidx[s]
        κs = LC.kst[si][i] * E.stors[si].hours[i] / LM.Δt
        wss = [ML.MoveWS(P) for _ in 1:L+2]
        bm = ML.BlockMove(wss, 0, Int[], P, LM, LC, Float64[], Float64[])
        shape = p == 1 ? nothing : D.profiles(L)[:, p]
        ML.prep_storage_ms!(bm, P, LM, LC, [(si, i)], b0, L, shape, jac; κmin=κmin, learn=learn) || return nothing
        return bm
    end
    if haskey(D.SPARSE_FAMILIES, fam)                         # sparse angle direction (τ, (nodes, d, g, pair))
        τd, (nodes, d, g, pq) = site
        rule, pair, _ = D.SPARSE_FAMILIES[fam]
        if pair                                                # the pair units (as `prep_xloc!`, `prep_random!(:xpair)`)
            (g1, i1), (g2, i2) = gi(pq[1]), gi(pq[2])
            κ1 = LC.A[g1][i1, τd] * LC.kgen[g1][i1]; κ2 = LC.A[g2][i2, τd] * LC.kgen[g2][i2]
            (abs(κ1) < κmin || abs(κ2) < κmin) && return nothing
            ML.prep_node!(ws, P, LM, E.gens[g1].vf, i1, 1 / κ1, τd)
            ML.prep_node!(ws, P, LM, E.gens[g2].vf, i2, -1 / κ2, τd)
        end
        for (n, a) in zip(nodes, d)
            a != 0.0 && ML.prep_node!(ws, P, LM, E.fθ, n, a, τd)
        end
        if rule != :none
            for (n, gg) in zip(nodes, g)
                gg != 0.0 && ML.prorata!(ws, jac, P, LM, LC, n, gg, τd; κmin=κmin, learn=learn)
            end
        end
        ML.finalize!(ws, P, LM, LC, τd)
        bm = ML.BlockMove([wa, ws], 2, [mod1(τd - 1, T), τd], P, LM, LC, Float64[], Float64[])
        ML.finalize!(wa, P, LM, LC, mod1(τd - 1, T)); ML.link_soc!(wa, ws, LM)
        append!(bm.bps, wa.bps); append!(bm.bps, ws.bps)
        return bm
    end
    if fam in D.TDIR_FAMILIES                                  # dense angle direction (τ, d, g) of chain `chain`
        τd, dd, gg = site
        d, g = dd[chain, :], gg[chain, :]
        for i in eachindex(d)
            d[i] != 0.0 && ML.prep_node!(ws, P, LM, E.fθ, i, d[i], τd)
        end
        if fam != :tdir0
            for n in eachindex(g)
                g[n] != 0.0 && ML.prorata!(ws, jac, P, LM, LC, n, g[n], τd; κmin=κmin, learn=learn)
            end
        end
        ML.finalize!(ws, P, LM, LC, τd)
        bm = ML.BlockMove([wa, ws], 2, [mod1(τd - 1, T), τd], P, LM, LC, Float64[], Float64[])
        ML.finalize!(wa, P, LM, LC, mod1(τd - 1, T)); ML.link_soc!(wa, ws, LM)
        append!(bm.bps, wa.bps); append!(bm.bps, ws.bps)
        return bm
    end
    fam = startswith(sf, "cellpair") ? :cellpair : startswith(sf, "cell") ? :cell : fam     # level buckets
    fam = fam == :gen0 ? :gen : fam == :pair0 ? :pair : D.istheta(fam) ? :theta : fam     # structure-specialised variants
    if fam == :gen
        g, i = gi(site[1]); ML.prep_node!(ws, P, LM, E.gens[g].vf, i, 1.0, τ)
    elseif fam == :uc
        g, i = gi(site[1]); ML.prep_node!(ws, P, LM, E.gens[g].ucf, i, 1.0, τ)
    elseif fam == :pair
        (g1, i1), (g2, i2) = gi(site[1]), gi(site[2])
        κ1 = LC.A[g1][i1, τ] * LC.kgen[g1][i1]; κ2 = LC.A[g2][i2, τ] * LC.kgen[g2][i2]
        (abs(κ1) < κmin || abs(κ2) < κmin) && return nothing
        ML.prep_node!(ws, P, LM, E.gens[g1].vf, i1, 1 / κ1, τ); ML.prep_node!(ws, P, LM, E.gens[g2].vf, i2, -1 / κ2, τ)
    elseif fam == :soc
        si, i = ix.sidx[site[1]]; ML.prep_node!(ws, P, LM, E.stors[si].fe, i, 1.0, τ)
    elseif fam == :link
        ML.prep_node!(ws, P, LM, E.flink, site[1], 1.0, τ)
    elseif fam == :theta
        ML.prep_node!(ws, P, LM, E.fθ, site[1], 1.0, τ)
    elseif fam == :spair
        s, _, q1, q2 = site
        si, i = ix.sidx[s]; st = E.stors[si]
        (g1, i1), (g2, i2) = gi(q1), gi(q2)
        κs = LC.kst[si][i] * st.hours[i] / LM.Δt
        κ1 = LC.A[g1][i1, τm] * LC.kgen[g1][i1]; κ2 = LC.A[g2][i2, τ] * LC.kgen[g2][i2]
        (κs < κmin || abs(κ1) < κmin || abs(κ2) < κmin) && return nothing
        ML.prep_node!(wa, P, LM, E.gens[g1].vf, i1, κs / κ1, τm)
        ML.prep_node!(ws, P, LM, st.fe, i, 1.0, τ)
        ML.prep_node!(ws, P, LM, E.gens[g2].vf, i2, -κs / κ2, τ)
    elseif fam == :tsoc || fam == :tsocL
        si, i = ix.sidx[site[1]]; st = E.stors[si]
        κs = LC.kst[si][i] * st.hours[i] / LM.Δt
        κs < κmin && return nothing
        n = st.bus[i]
        ML.prep_node!(ws, P, LM, st.fe, i, 1.0, τ)
        u = ((si, i),)                                         # nominal storage requirement (CPU `soc_qadj`, as `prep_tsoc!`)
        ML.prorata!(wa, jac, P, LM, LC, n, κs, τm; learn=learn, qadj=ML.soc_qadj(P, LM, LC, u, n, τm))
        ML.prorata!(ws, jac, P, LM, LC, n, -κs, τ; learn=learn, qadj=ML.soc_qadj(P, LM, LC, u, n, τ))
    elseif fam == :tpot || fam == :tpotL
        ML.prep_tpot!(nothing, ws, P, LM, LC, τ, jac; n=site[1], learn=learn) || return nothing
    elseif fam == :cell
        q, k, g, _ = site
        ML.cell_entries!(ws, P, LM, LM.cellunits[q][k], E.gens[g].vf, 1.0, τ)
    elseif fam == :cellpair
        q, k, g1, g2, _ = site
        cu = LM.cellunits[q][k]
        f1, f2 = E.gens[g1].vf, E.gens[g2].vf
        K1 = sum(ML.κunit(P, LM, LC, f, i, τ) for (f, i) in ML.cell_units(cu, f1))
        K2 = sum(ML.κunit(P, LM, LC, f, i, τ) for (f, i) in ML.cell_units(cu, f2))
        (abs(K1) < κmin || abs(K2) < κmin) && return nothing
        ML.cell_entries!(ws, P, LM, cu, f1, 1 / K1, τ)
        ML.cell_entries!(ws, P, LM, cu, f2, -1 / K2, τ)
    end
    ML.finalize!(ws, P, LM, LC, τ); ML.finalize!(wa, P, LM, LC, τm); ML.link_soc!(wa, ws, LM)
    bm = ML.BlockMove([wa, ws], 2, [τm, τ], P, LM, LC, Float64[], Float64[])
    append!(bm.bps, wa.bps); append!(bm.bps, ws.bps)
    return bm
end

"Device term batch of family `fam` for the given sites (one round), on the loop state st."
function dev_tb(pl, fam, sites, st=pl.st)
    ht = D.host_table(fam, pl.tp, [sites]); dt = D.device_table(fam, ht)
    tb, aux = D.build(Val(fam), st, pl.cp, pl.sp, dt, 1, pl.k)
    return tb, dt, aux
end

"Loop state with the CPU residual (for comparisons at round-off level)."
function cpu_r_state(pl, LCs)
    r = zeros(size(pl.st.r))
    for c in eachindex(LCs); r[c, :] .= vec(permutedims(LCs[c].r)); end
    return D.FlatState(pl.st.θ, pl.st.ω, pl.st.uc, pl.st.e, pl.st.u, r)
end

"Round-off scale of a site's ΔC: Δt c_shed · max |r0| · 1e-12 (the ulp level of the balance sums)."
balscale(pl, tb) = isempty(tb.r0) ? zeros(size(pl.st.θ, 1)) :
    pl.k.Δt * pl.k.cshed * 1e-12 .* vec(maximum(abs.(cat(tb.r0...; dims=3)); dims=3)) .+ 1e-12

"""
Mock-uniform comparison of device rounds with the CPU (see test_mock.jl): per round the device draws
with its own points (a), the CPU with `sample_line!` and a MockRNG, plus its Jacobian step with u3
(b), and the device sampler runs on the CPU walk's points (c); both commit the device's t.
`cpumove(c, LC, fam, site)` returns (BlockMove or nothing, jac). Returns (device state, stats, CPU states).
"""
function mock_compare(pl, st0, LCs, ss, fams, rounds, tabs, U, cpumove; P, β)
    C = size(st0.θ, 1)
    st = st0
    scpu = [copy(s) for s in ss]
    stats = Dict(:n => 0, :guard => 0, :own_ok => 0, :own_bad => 0, :cpupts_bad => 0, :improper_mismatch => 0, :overshoot => 0, :truncated => 0,
                 :jac_rej => 0, :jac_mismatch => 0, :worst_own => 0.0, :worst_cpu => 0.0)
    for f in fams, r in 1:size(tabs[f].on, 2)
        tb, _ = D.build(Val(f), st, pl.cp, pl.sp, tabs[f], r, pl.k)
        u = U[f][:, :, :, r]
        st2, tfin = D.do_round(Val(f), st, pl.cp, pl.sp, pl.k, tabs[f], r, u)
        tpre, guard = D.line_draw(Val(f), tb, pl.k, u[:, :, 1:1], u[:, :, 2:2])     # the family's form (dense or slope-jump)
        sites = rounds[f][r]
        B = size(tabs[f].on, 1)
        tc = zeros(C, B); tcpre = zeros(C, B)
        pts = [Float64[0.0] for _ in 1:C, _ in 1:B]
        walked = falses(C, B)
        for c in 1:C
            for (b, site) in enumerate(sites)
                bm, jac = cpumove(c, LCs[c], f, site)
                bm === nothing && continue
                t = ML.sample_line!(MockRNG([u[c, b, 1], u[c, b, 2]]), bm, β, Float64[])
                walked[c, b] = !isempty(bm.lb.pts)
                walked[c, b] && (pts[c, b] = copy(bm.lb.pts))
                tcpre[c, b] = t
                if t != 0.0 && !isempty(jac) && u[c, b, 3] >= exp(min(0.0, ML.logjac(jac, t, bm)))
                    t = 0.0
                end
                tc[c, b] = t
                ML.commit_block!(bm, scpu[c].v, tfin[c, b, 1])
            end
        end
        # the CPU's points inside the CPU walk's range, the device's own kinks outside it (the CPU walk
        # stops 40 nats above the running minimum; the potential may fall again further out)
        own = D.line_points(Val(f), tb, pl.k)
        merged = [begin
            lo, hi = extrema(pts[c, b]); o = own[c, b, :]
            vcat(pts[c, b], unique(o[(o .< lo) .| (o .> hi)]))
        end for c in 1:C, b in 1:B]
        K = maximum(length.(merged))
        Pc = zeros(C, B, K)
        for c in 1:C, b in 1:B; Pc[c, b, :] .= sort(vcat(merged[c, b], zeros(K - length(merged[c, b])))); end
        tcp = D.sample_points(tb, pl.k, Pc, u[:, :, 1:1], u[:, :, 2:2])
        cPc = D.eval_terms(tb, pl.k, Pc; sec=false)[1]          # potential at the merged points
        tcpf = D.jac_accept(tb, tcp, u[:, :, 3:3])[:, :, 1]
        tcp = tcp[:, :, 1]
        for c in 1:C, (b, site) in enumerate(sites)
            stats[:n] += 1
            td, tdp = tfin[c, b, 1], tpre[c, b, 1]
            if guard[c, b, 1]                       # more sign changes than kept roots: the device stays (exact guard)
                stats[:guard] += 1
                continue
            end
            if tcpre[c, b] == 0.0 && !walked[c, b]
                tdp == 0.0 || (stats[:improper_mismatch] += 1)
                continue
            end
            d2 = abs(tcp[c, b] - tcpre[c, b]) / max(1.0, abs(tcpre[c, b]))
            if d2 > 1e-9 && maximum(abs.(pts[c, b])) > 1e12
                stats[:overshoot] += 1
                continue
            end
            # the CPU walk stops 40 nats above its running minimum; where the potential falls again
            # beyond its range to within 40 nats of that minimum, the CPU draw is truncated (the device's
            # points cover the whole line), so the draws differ by construction
            if d2 > 1e-9
                lo, hi = extrema(pts[c, b])
                pin_ = [x for (x, p) in zip(cPc[c, b, :], Pc[c, b, :]) if lo <= p <= hi]
                pout = [x for (x, p) in zip(cPc[c, b, :], Pc[c, b, :]) if !(lo <= p <= hi)]
                if !isempty(pout) && !isempty(pin_) && minimum(pout) < minimum(pin_) + 40
                    stats[:truncated] += 1
                    continue
                end
            end
            stats[:worst_cpu] = max(stats[:worst_cpu], d2)
            if d2 > 1e-9
                stats[:cpupts_bad] += 1
                stats[:cpupts_bad] <= 5 && println("  CPU-points draw differs: $f round $r site $site chain $c: device $(tcp[c, b]) CPU $(tcpre[c, b])")
            end
            (tcpf[c, b] == 0.0) != (tc[c, b] == 0.0) && (stats[:jac_mismatch] += 1)    # Jacobian decision on the same t
            (tcpre[c, b] != 0.0 && tc[c, b] == 0.0) && (stats[:jac_rej] += 1)
            d1 = abs(tdp - tcpre[c, b]) / max(1.0, abs(tcpre[c, b]))
            d1 <= 1e-9 ? (stats[:own_ok] += 1; stats[:worst_own] = max(stats[:worst_own], d1)) : (stats[:own_bad] += 1)
        end
        st = st2
    end
    for c in 1:C; ML.evaluate!(P, scpu[c]); end
    return st, stats, scpu
end

"RNG returning the given values in order (`rand(rng)`); errors if more are drawn than given."
mutable struct MockRNG <: Random.AbstractRNG
    u::Vector{Float64}
    i::Int
end
MockRNG(u) = MockRNG(u, 0)
Random.rand(r::MockRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float64}}) = (r.i += 1; r.u[r.i])

"CPU caches as a loop state (node values and r, device layout)."
function cpu_flat(P, ix, LCs)
    E = P.E
    T = size(LCs[1].r, 2)
    fl(Y) = reshape(permutedims(Y), 1, :)
    θ = vcat([fl(LC.Y[E.fθ]) for LC in LCs]...)
    ω = vcat([reshape(reduce(hcat, [LC.Y[E.gens[g].vf][i, :] for (g, i) in ix.gidx]), 1, :) for LC in LCs]...)
    uc = vcat([reshape(reduce(hcat, [E.gens[g].ucf > 0 ? LC.Y[E.gens[g].ucf][i, :] : zeros(T) for (g, i) in ix.gidx]), 1, :) for LC in LCs]...)
    e = vcat([reshape(reduce(hcat, [LC.Y[E.stors[a].fe][i, :] for (a, i) in ix.sidx]), 1, :) for LC in LCs]...)
    u = E.flink > 0 ? vcat([fl(LC.Y[E.flink]) for LC in LCs]...) : zeros(length(LCs), T)
    r = vcat([fl(LC.r) for LC in LCs]...)
    return D.FlatState(θ, ω, uc, e, u, r)
end
