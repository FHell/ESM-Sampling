# Scaling of convergence with the finest scale on the toy: grid side n = 8 … 32
# (N = 64 … 1024). For each kernel: sweeps to relax from the far start and ESS per
# sweep in equilibrium (sweep = N line moves; MLDA step = one coarse subchain over
# all cells, counted as one sweep since it touches every node once).
# Usage: julia --project=. toy/run_toy4.jl
include(joinpath(@__DIR__, "Toy.jl"))
using .Toy
include(joinpath(@__DIR__, "kernels.jl"))

function sweeps_run(k!, s0, nsw, rng, observe)
    s = copy(s0)
    O = zeros(length(observe(s)), nsw)
    t = @elapsed for it in 1:nsw
        k!(rng, s); O[:, it] = observe(s)
    end
    return O, t
end

function scaling_uncongested(ns; nsw=600)
    println("\n== S1: uncongested hard toy, β = 200/r̄ — sweeps to relax (far start, to 3 sd of exact objective) and ESS per 1000 sweeps (objective, west slack) ==")
    println("   n     N    kernel              relax   ESS/1000 obj  ESS/1000 west   ms/sweep")
    for n in ns
        P = grid_problem(n; βrel=200.0, seed=1)
        rng = MersenneTwister(100 + n)
        S = Float64(P.N)
        W = westset(P) .& (1:P.N .!= P.istar)
        observe(s) = [obj(P, s), sum(s[W])]
        ex = reduce(hcat, [observe(exact_sample(rng, P, S)) for _ in 1:4000])
        ref = vec(mean(ex; dims=2)); sd = vec(std(ex; dims=2))
        ncell = max(4, P.N ÷ 16)
        kernels = Any[("local", make_kernel(:local, P)), ("pairs_slack", make_kernel(:pairs_slack, P)),
                      ("MLDA redraw merit", make_mlda(P, Coarse(P, merit_cells(P, ncell)); mode=:redraw, fine=nothing)),
                      ("MLDA redraw geo", make_mlda(P, Coarse(P, block_cells(P, 4)); mode=:redraw, fine=nothing))]
        for (lbl, k!) in kernels
            Of, _ = sweeps_run(k!, fill(S / P.N, P.N), nsw, rng, observe)
            i = findfirst(j -> abs(Of[1, j] - ref[1]) < 3sd[1], 1:nsw)
            O, t = sweeps_run(k!, exact_sample(rng, P, S), nsw, rng, observe)
            println(@sprintf("   %-4d %-5d %-18s %6s   %10.1f   %10.1f   %9.2f", n, P.N, lbl, i === nothing ? ">$nsw" : string(i),
                             1000 * ess(O[1, :]) / nsw, 1000 * ess(O[2, :]) / nsw, 1000t / nsw))
        end
        flush(stdout)
    end
end

function scaling_congested(ns; nsw=600)
    println("\n== S2: congested hard toy (interface = half the west export) — same metrics, reference from a long mixture run ==")
    println("   n     N    kernel              relax   ESS/1000 obj  ESS/1000 west   ms/sweep")
    for n in ns
        P = grid_problem(n; βrel=200.0, seed=1)
        rng = MersenneTwister(200 + n)
        c = ones(P.N); S = sum(c)
        Wset = westset(P)
        Ftot = 0.5 * sum(c[Wset])
        iW = findall(Wset)[argmax(P.a[Wset])]
        observe(s) = [obj(P, s), sum(s[Wset])]
        z = Zone(Wset, Ref(sum(c[Wset]) - Ftot), Ref(0.0))
        sync(k!) = (rng, s) -> (z.σW[] = sum(s[Wset]); k!(rng, s))
        kref = sync(make_kernel(:mixture, P; zone=z, iW=iW))
        s = fill(S / P.N, P.N)
        for _ in 1:1000; kref(rng, s); end
        R = reduce(hcat, [(kref(rng, s); observe(s)) for _ in 1:4000])
        ref = vec(mean(R; dims=2)); sd = vec(std(R; dims=2))
        seq = copy(s)
        for (lbl, name) in (("local", :local), ("pairs_slack", :pairs_slack), ("zonal", :zonal), ("mixture", :mixture))
            k! = sync(make_kernel(name, P; zone=z, iW=iW))
            Of, _ = sweeps_run(k!, fill(S / P.N, P.N), nsw, rng, observe)
            i = findfirst(j -> all(abs.(Of[:, j] .- ref) .< 3 .* sd), 1:nsw)
            O, t = sweeps_run(k!, copy(seq), nsw, rng, observe)
            println(@sprintf("   %-4d %-5d %-18s %6s   %10.1f   %10.1f   %9.2f", n, P.N, lbl, i === nothing ? ">$nsw" : string(i),
                             1000 * ess(O[1, :]) / nsw, 1000 * ess(O[2, :]) / nsw, 1000t / nsw))
        end
        flush(stdout)
    end
end

scaling_uncongested((8, 12, 16, 24, 32))
scaling_congested((8, 12, 16, 24, 32))
