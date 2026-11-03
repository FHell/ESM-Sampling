# NCMC as the coarse step on the toy (docs/toy_results.md, part G):
# large-scale perturbation (log-scaling of cell slack totals, shares transported,
# slack node absorbs) in K stages, relaxation of the details between stages with a
# π_f-reversible kernel, acceptance exp(−W) with the work W in the fine potential.
# Usage: julia --project=. toy/run_toy3.jl
include(joinpath(@__DIR__, "Toy.jl"))
using .Toy
include(joinpath(@__DIR__, "kernels.jl"))

const NSIDE = 16

"Full potential (−β a·s + node and interface penalties); Inf if a hard constraint is violated."
function Ufull(P, s, zone)
    U = -P.β * dot(P.a, s)
    if isinf(P.μ)
        any(<(0), s) && return Inf
        zone !== nothing && sum(s[zone.inW]) < zone.bnd[] && return Inf
    else
        U += P.β * P.μ * sum(x -> max(-x, 0.0), s)
        zone !== nothing && (U += P.β * P.μ * max(zone.bnd[] - sum(s[zone.inW]), 0.0))
    end
    return U
end

function make_ncmc(P, cells, M, τ, K, relax!; zone=nothing, stats=Ref((0, 0)))
    return (rng, s) -> begin
        moved = cells[randperm(rng, length(cells))[1:M]]
        η = τ .* randn(rng, M)
        s0 = copy(s)
        W = 0.0
        for k in 1:K
            U0 = Ufull(P, s, zone)
            logJ = 0.0
            for (q, C) in enumerate(moved)
                ρ = exp(η[q] / K)
                σ = sum(s[C])
                s[C] .*= ρ
                s[P.istar] += σ * (1 - ρ)
                logJ += length(C) * η[q] / K
            end
            zone === nothing || (zone.σW[] = sum(s[zone.inW]))
            U1 = Ufull(P, s, zone)
            W += U1 - U0 - logJ
            isfinite(W) || break
            k < K && relax!(rng, s)
        end
        n, a = stats[]
        if isfinite(W) && log(rand(rng)) < -W
            stats[] = (n + 1, a + 1)
        else
            s .= s0
            zone === nothing || (zone.σW[] = sum(s[zone.inW]))
            stats[] = (n + 1, a)
        end
    end
end

combine(k1, k2) = (rng, s) -> (k1(rng, s); k2(rng, s))

function bench(label, k!, s_far, s_eq, ref, sd, observe, secs, rng; stats=nothing)
    sres, tsr, Or = run_for(k!, s_far, secs, rng; observe=observe)
    i = findfirst(j -> all(abs.(Or[:, j] .- ref) .< 3 .* sd), eachindex(tsr))
    stats === nothing || (stats[] = (0, 0))
    _, ts, O = run_for(k!, s_eq, secs, rng; observe=observe)
    acc = stats === nothing ? NaN : stats[][2] / max(stats[][1], 1)
    println(@sprintf("   %-44s %9s   %8.1f %8.1f   %7.0f   %6.3f   %.1f, %.3f", label,
                     i === nothing ? ">$(round(Int, secs))s" : @sprintf("%.2fs", tsr[i]),
                     ess(O[1, :]) / ts[end], ess(O[2, :]) / ts[end], length(ts) / ts[end], acc, mean(O[1, :]), mean(O[2, :])))
end

const HDR = "   kernel                                         relax        ESS/s obj, west    it/s     acc      mean obj, west"

function part_G1(; secs=12.0)
    P = grid_problem(NSIDE; βrel=200.0, seed=1)
    rng = MersenneTwister(71)
    S = Float64(P.N)
    W = westset(P) .& (1:P.N .!= P.istar)
    observe(s) = [obj(P, s), sum(s[W])]
    ex = reduce(hcat, [observe(exact_sample(rng, P, S)) for _ in 1:20000])
    ref = vec(mean(ex; dims=2)); sd = vec(std(ex; dims=2))
    println(@sprintf("\n== G1: NCMC coarse steps, uncongested hard toy (exact obj %.1f ± %.1f, west %.3f ± %.3f) ==", ref[1], sd[1], ref[2], sd[2]))
    println(HDR)
    merit, geo = merit_cells(P, 16), block_cells(P, 4)
    far() = fill(S / P.N, P.N)
    eq() = exact_sample(rng, P, S)
    loc = make_kernel(:local, P)
    bench("local only", loc, far(), eq(), ref, sd, observe, secs, rng)
    for (cname, cells) in (("merit", merit), ("geo", geo)), M in (1, length(cells)), K in (1, 4, 16)
        τ = M == 1 ? 0.3 : 0.1
        st = Ref((0, 0))
        k! = combine(loc, make_ncmc(P, cells, M, τ, K, loc; stats=st))
        bench("local + NCMC($cname, M=$M, K=$K, τ=$τ)", k!, far(), eq(), ref, sd, observe, secs, rng; stats=st)
    end
end

function part_G2(; secs=12.0, Ftot=64.0)
    P = grid_problem(NSIDE; βrel=200.0, μrel=30.0, seed=1)
    rng = MersenneTwister(81)
    c = ones(P.N); S = sum(c)
    Wset = westset(P)
    iW = findall(Wset)[argmax(P.a[Wset])]
    mkzone(s) = Zone(Wset, Ref(sum(c[Wset]) - Ftot), Ref(sum(s[Wset])))
    observe(s) = [obj(P, s), sum(s[Wset])]
    s = fill(S / P.N, P.N); z = mkzone(s)
    kref = make_kernel(:mixture, P; zone=z, iW=iW)
    for _ in 1:2000; kref(rng, s); end
    R = reduce(hcat, [(kref(rng, s); observe(s)) for _ in 1:8000])
    ref = vec(mean(R; dims=2)); sd = vec(std(R; dims=2))
    seq = copy(s)
    println(@sprintf("\n== G2: NCMC coarse steps, congested regularized toy (reference obj %.1f ± %.1f, west %.3f ± %.3f) ==", ref[1], sd[1], ref[2], sd[2]))
    println(HDR)
    merit, geo = merit_cells(P, 16), block_cells(P, 4)
    for base in (:local, :pairs_slack)
        zf = mkzone(fill(S / P.N, P.N))
        kb0 = make_kernel(base, P; zone=zf, iW=iW)
        bench("$base only", (rng, s) -> (zf.σW[] = sum(s[Wset]); kb0(rng, s)), fill(S / P.N, P.N), copy(seq), ref, sd, observe, secs, rng)
        for (cname, cells) in (("merit", merit), ("geo", geo)), M in (1, length(cells)), K in (1, 8)
            τ = M == 1 ? 0.3 : 0.1
            st = Ref((0, 0))
            zz = mkzone(fill(S / P.N, P.N))
            kb = make_kernel(base, P; zone=zz, iW=iW)
            k! = combine(kb, make_ncmc(P, cells, M, τ, K, kb; zone=zz, stats=st))
            # the zone tracker must follow whichever start is used
            kk = (rng, s) -> (zz.σW[] = sum(s[Wset]); k!(rng, s))
            bench("$base + NCMC($cname, M=$M, K=$K)", kk, fill(S / P.N, P.N), copy(seq), ref, sd, observe, secs, rng; stats=st)
        end
    end
end

function main()
    part_G1()
    part_G2()
end
main()
