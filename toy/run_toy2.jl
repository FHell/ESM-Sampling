# Part D–F of the toy tests (docs/toy_results.md):
#   D  actual MLDA chains (full-space coarse chain) vs single-level kernels, uncongested, fixed c
#   E  congested toy (zonal interface W→E capped), fixed c: which kernels mix
#   F  joint (c, x) chain, congested regularized toy, λ = 0: lag of the fast block behind the weather
# Usage: julia --project=. toy/run_toy2.jl
include(joinpath(@__DIR__, "Toy.jl"))
using .Toy, Random, Statistics, Printf, LinearAlgebra

const NSIDE = 16

include(joinpath(@__DIR__, "kernels.jl"))

# ---------------------------------------------------------------- D
function part_D(; secs=15.0)
    P = grid_problem(NSIDE; βrel=200.0, seed=1)
    rng = MersenneTwister(41)
    S = Float64(P.N)
    W = westset(P)
    observe(s) = [obj(P, s), sum(s[W .& (1:P.N .!= P.istar)])]
    ex = reduce(hcat, [observe(exact_sample(rng, P, S)) for _ in 1:20000])
    ref = vec(mean(ex; dims=2)); sd = vec(std(ex; dims=2))
    println(@sprintf("\n== D: uncongested hard toy, fixed c — MLDA chains vs single-level (%.0f s each) ==", secs))
    println(@sprintf("   exact: E[β(C−C*)] − const shift ⇒ objective mean %.1f ± %.1f, west slack %.3f ± %.3f", ref[1], sd[1], ref[2], sd[2]))
    merit = merit_cells(P, 16)
    geo = block_cells(P, 4)
    variants = Any[("local", make_kernel(:local, P), nothing),
                ("pairs_slack", make_kernel(:pairs_slack, P), nothing)]
    for (lbl, cells, mode, fine) in (("MLDA transport, merit, + local", merit, :transport, :local),
                                     ("MLDA redraw, merit, + local", merit, :redraw, :local),
                                     ("MLDA redraw, geo, + local", geo, :redraw, :local),
                                     ("MLDA redraw, merit, no fine", merit, :redraw, nothing))
        st = Ref((0, 0))
        push!(variants, (lbl, make_mlda(P, Coarse(P, cells); mode=mode, fine=fine, stats=st), st))
    end
    println("   kernel                              relax time (far start)   ESS/s objective, west slack   it/s    MLDA acc   mean obj (exact $(round(ref[1], digits=1)))")
    for (lbl, k!, st) in variants
        sfar = fill(S / P.N, P.N)
        _, tsr, Or = run_for(k!, sfar, secs, rng; observe=observe)
        i = findfirst(j -> abs(Or[1, j] - ref[1]) < 3 * sd[1], eachindex(tsr))
        st === nothing || (st[] = (0, 0))
        _, ts, O = run_for(k!, exact_sample(rng, P, S), secs, rng; observe=observe)
        acc = st === nothing ? NaN : st[][2] / max(st[][1], 1)
        println(@sprintf("   %-34s %10s                  %8.1f  %8.1f          %7.0f   %6.3f     %.1f",
                         lbl, i === nothing ? ">$(secs)s" : @sprintf("%.2fs", tsr[i]), ess(O[1, :]) / ts[end], ess(O[2, :]) / ts[end],
                         length(ts) / ts[end], acc, mean(O[1, :])))
    end
end

# ---------------------------------------------------------------- E
function part_E(; secs=15.0, Ftot=64.0)
    P = grid_problem(NSIDE; βrel=200.0, seed=1)
    rng = MersenneTwister(51)
    c = ones(P.N); S = sum(c)
    W = westset(P)
    iW = findall(W)[argmax(P.a[W])]
    mkzone(s) = Zone(W, Ref(sum(c[W]) - Ftot), Ref(sum(s[W])))
    observe(s) = [obj(P, s), sum(s[W])]
    # reference: long mixture run
    s = fill(S / P.N, P.N); z = mkzone(s)
    kref = make_kernel(:mixture, P; zone=z, iW=iW)
    for _ in 1:2000; kref(rng, s); end
    R = reduce(hcat, [(kref(rng, s); observe(s)) for _ in 1:8000])
    ref = vec(mean(R; dims=2)); sd = vec(std(R; dims=2))
    println(@sprintf("\n== E: congested hard toy (west→east interface ≤ %.0f, west export would be %.0f), fixed c ==", Ftot, sum(c[W])))
    println(@sprintf("   reference (mixture, 8000 sweeps): objective %.1f ± %.1f, west slack %.3f ± %.3f (bound %.0f; ESS of west slack in reference %.0f)",
                     ref[1], sd[1], ref[2], sd[2], sum(c[W]) - Ftot, ess(R[2, :])))
    merit = merit_cells(P, 16)
    println("   kernel                              relax time (far start)   ESS/s objective, west slack   it/s    MLDA acc   mean obj, west slack")
    for name in (:local, :pairs_slack, :pairs_global, :zonal, :mixture, :mlda)
        runk(s0) = begin
            zz = mkzone(s0)
            st = Ref((0, 0))
            k! = name == :mlda ? make_mlda(P, Coarse(P, merit); mode=:redraw, fine=:local, zone=zz, stats=st) :
                                 make_kernel(name, P; zone=zz, iW=iW)
            sres, ts, O = run_for(k!, s0, secs, rng; observe=observe)
            ts, O, st
        end
        tsr, Or, _ = runk(fill(S / P.N, P.N))
        i = findfirst(j -> abs(Or[1, j] - ref[1]) < 3 * sd[1] && abs(Or[2, j] - ref[2]) < 3 * sd[2], eachindex(tsr))
        s0 = copy(s)                       # an equilibrated reference state
        ts, O, st = runk(s0)
        acc = name == :mlda ? st[][2] / max(st[][1], 1) : NaN
        println(@sprintf("   %-34s %10s                  %8.1f  %8.1f          %7.0f   %6.3f     %.1f, %.3f",
                         name, i === nothing ? ">$(secs)s" : @sprintf("%.2fs", tsr[i]), ess(O[1, :]) / ts[end], ess(O[2, :]) / ts[end],
                         length(ts) / ts[end], acc, mean(O[1, :]), mean(O[2, :])))
    end
end

# ---------------------------------------------------------------- F
function part_F(; T=400, Ftot=128.0)
    P = grid_problem(NSIDE; βrel=200.0, μrel=30.0, seed=1)
    rng = MersenneTwister(61)
    N, n = P.N, P.n
    W = westset(P)
    iW = findall(W)[argmax(P.a[W])]
    xy = [((i - 1) % n, (i - 1) ÷ n) for i in 1:N]
    σc, ℓ = 0.2, 3.0
    Σc = [σc^2 * exp(-hypot(xy[i][1] - xy[j][1], xy[i][2] - xy[j][2]) / ℓ) for i in 1:N, j in 1:N]
    Lc = cholesky(Symmetric(Σc + 1e-10I)).L
    ρ = 0.3
    cs = Vector{Vector{Float64}}(undef, T)
    c = ones(N)
    for t in 1:T
        c = 1.0 .+ sqrt(1 - ρ^2) .* (c .- 1.0) .+ ρ .* (Lc * randn(rng, N))
        cs[t] = c
    end
    margin(s, c) = sum(s[W]) - (sum(c[W]) - Ftot)
    observe(s, c) = [obj(P, s) + P.β * dot(P.a, c), margin(s, c)]   # β a·(c − s) = β C, and the interface margin
    println(@sprintf("\n== F: joint (c, x), regularized congested toy (interface ≤ %.0f), λ = 0, %d weather steps (pCN ρ = %.1f) ==", Ftot, T, ρ))
    # reference conditional moments every 4th step
    every = 4
    zr = Zone(W, Ref(0.0), Ref(0.0))
    kref = make_kernel(:mixture, P; zone=zr, iW=iW)
    s = exact_sample(rng, P, sum(cs[1]))
    cprev = cs[1]
    refm = Dict{Int,Vector{Float64}}(); refs = Dict{Int,Vector{Float64}}()
    for t in 1:T
        s .+= cs[t] .- cprev; cprev = cs[t]
        t % every == 0 || continue
        zr.bnd[] = sum(cs[t][W]) - Ftot; zr.σW[] = sum(s[W])
        for _ in 1:60; kref(rng, s); end
        O = reduce(hcat, [(kref(rng, s); observe(s, cs[t])) for _ in 1:40])
        refm[t] = vec(mean(O; dims=2)); refs[t] = vec(std(O; dims=2))
    end
    congested = mean(refm[t][2] < 3 * refs[t][2] for t in keys(refm))
    println(@sprintf("   interface binding (reference margin < 3 sd) in %.0f %% of the checked steps", 100congested))
    println("   kernel          sweeps/step   RMS lag (cost)   RMS lag (interface margin)   [≈1 = equilibrated, ≫1 = lagging]")
    for name in (:local, :pairs_slack, :zonal, :mixture), nf in (1, 4, 16)
        z = Zone(W, Ref(0.0), Ref(0.0))
        k! = make_kernel(name, P; zone=z, iW=iW)
        s = exact_sample(rng, P, sum(cs[1]))
        cprev = cs[1]
        e1, e2 = Float64[], Float64[]
        for t in 1:T
            s .+= cs[t] .- cprev; cprev = cs[t]
            z.bnd[] = sum(cs[t][W]) - Ftot; z.σW[] = sum(s[W])
            for _ in 1:nf; k!(rng, s); end
            if haskey(refm, t) && t > T ÷ 5
                o = observe(s, cs[t])
                push!(e1, (o[1] - refm[t][1]) / refs[t][1]); push!(e2, (o[2] - refm[t][2]) / refs[t][2])
            end
        end
        println(@sprintf("   %-14s %6d        %8.2f          %8.2f", name, nf, sqrt(mean(e1 .^ 2)), sqrt(mean(e2 .^ 2))))
    end
end

function main()
    # sanity: the hinge sampler reproduces the exact pair law (regularized and hard)
    rng = MersenneTwister(1)
    for μrel in (Inf, 4.0)
        P = grid_problem(4; βrel=5.0, μrel=μrel, seed=2)
        s = exact_sample(rng, P, 16.0)
        j = findfirst(!=(P.istar), 1:P.N)
        xs = [(line_move!(rng, P, s, [j, P.istar], [1.0, -1.0]); s[j]) for _ in 1:200000]
        ys = [exact_sample(rng, P, 16.0)[j] for _ in 1:50000]
        @assert abs(mean(xs) - mean(ys)) < 0.03 * std(ys) + 1e-3 "pair move check failed ($μrel): $(mean(xs)) vs $(mean(ys))"
    end
    println("hinge sampler checks passed")
    part_D()
    part_E()
    part_F()
end

main()
