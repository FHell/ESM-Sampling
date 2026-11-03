# Tests of the alternatives on the toy (docs/toy_mlda.md).
# Usage: julia --project=. toy/run_toy.jl
include(joinpath(@__DIR__, "Toy.jl"))
using .Toy, Random, Statistics, Printf, LinearAlgebra

const NSIDE = 16

# --------------------------------------------------------------------------
# sanity checks of the exact line sampler
# --------------------------------------------------------------------------
function check_line_sampler()
    rng = MersenneTwister(1)
    for (lo, hi, g) in ((0.0, 1.0, 3.0), (0.0, 1.0, -3.0), (0.0, Inf, 2.0), (-Inf, 0.0, -2.0))
        x = [Toy.trexp(rng, lo, hi, g) for _ in 1:200000]
        if isfinite(lo) && isfinite(hi)
            m = g == 0 ? 0.5 : (1 / g - 1 / expm1(g)) # mean of ∝ e^{-gt} on [0,1]
            @assert abs(mean(x) - m) < 0.01 "trexp mean $(mean(x)) vs $m"
        end
        @assert all(lo .<= x .<= hi)
    end
    # line_move! leaves the exact law of a 2-node pair invariant (regularized)
    P = grid_problem(4; βrel=5.0, μrel=4.0, seed=2)
    s = exact_sample(rng, P, 16.0)
    j = findfirst(!=(P.istar), 1:P.N)
    xs = Float64[]
    for _ in 1:200000
        line_move!(rng, P, s, [j, P.istar], [1.0, -1.0])
        push!(xs, s[j])
    end
    ys = [exact_sample(rng, P, 16.0)[j] for _ in 1:50000]
    @assert abs(mean(xs) - mean(ys)) < 0.03 * std(ys) + 1e-3 "pair move: $(mean(xs)) vs $(mean(ys))"
    println("line sampler checks passed")
end

# --------------------------------------------------------------------------
# A: operation at fixed weather — direction families
# --------------------------------------------------------------------------
function sampler_sweep(name, P, cells_q, edges)
    N = P.N
    if name == :local
        return (rng, s) -> (for _ in 1:N; line_move!(rng, P, s, local_dir(P, rand(rng, 1:N))...); end)
    elseif name == :local_cells
        dirs = [cell_dir(P, C) for C in cells_q]
        return (rng, s) -> begin
            for _ in 1:N; line_move!(rng, P, s, local_dir(P, rand(rng, 1:N))...); end
            for d in shuffle(rng, dirs); line_move!(rng, P, s, d...); end
        end
    elseif name == :pairs_nbr
        return (rng, s) -> (for _ in 1:N; (i, j) = edges[rand(rng, 1:length(edges))]; line_move!(rng, P, s, [i, j], [1.0, -1.0]); end)
    elseif name == :pairs_global
        return (rng, s) -> (for _ in 1:N; i, j = rand(rng, 1:N), rand(rng, 1:N); i != j && line_move!(rng, P, s, [i, j], [1.0, -1.0]); end)
    elseif name == :pairs_slack
        return (rng, s) -> (for _ in 1:N; j = rand(rng, 1:N); j != P.istar && line_move!(rng, P, s, [j, P.istar], [1.0, -1.0]); end)
    end
end

function part_A(P, label; nsweeps=1500)
    rng = MersenneTwister(11)
    S = Float64(P.N)
    edges = [(i, j) for i in 1:P.N for j in P.nbrs[i] if i < j]
    cells_q = quadtree_cells(P)
    west = [i for i in 1:P.N if (i - 1) % P.n < P.n ÷ 2 && i != P.istar]
    obs(s) = (P.β * dot(P.r, s), sum(s[west]))
    ex = [obs(exact_sample(rng, P, S)) for _ in 1:20000]
    exE, exW = mean(first.(ex)), mean(last.(ex))
    println(@sprintf("\n== A [%s]: exact E[β r·s] = %.1f (N−1 = %d), E[west-half slack] = %.4f ==", label, exE, P.N - 1, exW))
    println("  sampler        sweeps to |excess−exact|<10% (far start)   equilibrium ESS/1000 sweeps: excess, west slack   μs/sweep")
    for name in (:local, :local_cells, :pairs_nbr, :pairs_global, :pairs_slack)
        sweep! = sampler_sweep(name, P, cells_q, edges)
        # relaxation from the box centre of the simplex
        s = fill(S / P.N, P.N)
        relax = -1
        for it in 1:nsweeps
            sweep!(rng, s)
            if abs(obs(s)[1] - exE) < 0.1 * exE
                relax = it; break
            end
        end
        # equilibrium mixing from an exact draw
        s = exact_sample(rng, P, S)
        E, Wt = Float64[], Float64[]
        t = @elapsed for it in 1:nsweeps
            sweep!(rng, s)
            o = obs(s); push!(E, o[1]); push!(Wt, o[2])
        end
        println(@sprintf("  %-13s %10s                                    %7.1f   %7.1f        %8.0f   (mean excess %.1f)",
                         name, relax < 0 ? ">$(nsweeps)" : string(relax), 1000 * ess(E) / nsweeps, 1000 * ess(Wt) / nsweeps,
                         1e6 * t / nsweeps, mean(E)))
    end
end

# --------------------------------------------------------------------------
# B: MLDA variants on exact equilibrium states
# --------------------------------------------------------------------------
logπ0_galerkin(P, σ, rbar) = hard(P) ? (σ < 0 ? -Inf : -P.β * rbar * σ) : -P.β * (rbar * σ + P.μ * max(-σ, 0.0))
logπ0_gamma(P, σ, rbar, n) = σ <= 0 ? -Inf : (n - 1) * log(σ) - P.β * rbar * σ
hard(P) = isinf(P.μ)

function draw_galerkin(rng, P, rbar)
    hard(P) && return randexp(rng) / (P.β * rbar)
    λp, λm = P.β * rbar, P.β * (P.μ - rbar)
    return rand(rng) < (1 / λp) / (1 / λp + 1 / λm) ? randexp(rng) / λp : -randexp(rng) / λm
end
draw_gamma(rng, P, rbar, n) = sum(randexp(rng) for _ in 1:n) / (P.β * rbar)

"""
One proposal of the given variant from an exact equilibrium state; returns log α.
variants: :galerkin, :gamma (fixed details), :transport (shares, hard only), :ncmc (K stages).
"""
function proposal_logα(rng, P, cells, M, variant; K=1)
    S = Float64(P.N)
    s = exact_sample(rng, P, S)
    moved = randperm(rng, length(cells))[1:M]
    U0 = potential(P, s)
    sn = copy(s)
    logq = 0.0          # log π0(σ) − log π0(σ')
    work = 0.0
    if variant == :ncmc
        Δs = Float64[]
        for c in moved
            C = cells[c]; n = length(C); rbar = mean(P.r[C]); σ = sum(s[C])
            σn = draw_gamma(rng, P, rbar, n)
            logq += logπ0_gamma(P, σ, rbar, n) - logπ0_gamma(P, σn, rbar, n)
            push!(Δs, σn - σ)
        end
        for k in 1:K
            Ub = potential(P, sn)
            for (q, c) in enumerate(moved)
                C = cells[c]
                sn[C] .+= Δs[q] / (K * length(C))
                sn[P.istar] -= Δs[q] / K
            end
            Ua = potential(P, sn)
            isinf(Ua) && return -Inf
            work += Ua - Ub
            for c in moved            # relaxation at fixed cell totals: within-cell pairs
                C = cells[c]
                for _ in 1:length(C)
                    i, j = rand(rng, C), rand(rng, C)
                    i != j && line_move!(rng, P, sn, [i, j], [1.0, -1.0])
                end
            end
        end
        return -work + logq
    end
    corr = 0.0
    for c in moved
        C = cells[c]; n = length(C); rbar = mean(P.r[C]); σ = sum(s[C])
        if variant == :galerkin
            σn = draw_galerkin(rng, P, rbar)
            logq += logπ0_galerkin(P, σ, rbar) - logπ0_galerkin(P, σn, rbar)
            sn[C] .+= (σn - σ) / n
        elseif variant == :gamma
            σn = draw_gamma(rng, P, rbar, n)
            logq += logπ0_gamma(P, σ, rbar, n) - logπ0_gamma(P, σn, rbar, n)
            sn[C] .+= (σn - σ) / n
        elseif variant == :transport
            σn = draw_gamma(rng, P, rbar, n)
            sn[C] .*= σn / σ
            corr += P.β * rbar * (σn - σ)      # log π_c(s') − log π_c(s) = −β r̄ (σ' − σ)
        end
        sn[P.istar] -= σn - σ
    end
    U1 = potential(P, sn)
    isinf(U1) && return -Inf
    variant == :transport && return -(U1 - U0) + corr
    return -(U1 - U0) + logq
end

function part_B(P, label; nprop=300)
    rng = MersenneTwister(21)
    geo = Toy.block_cells(P, 4)
    mer = Toy.merit_cells(P, length(geo))
    spread(cells) = mean(std(P.r[C]) / mean(P.r[C]) for C in cells)
    println(@sprintf("\n== B [%s]: %d cells of ≈%d nodes; Δr/r̄ per cell: geographic %.2f, merit-order %.2f ==",
                     label, length(geo), round(Int, mean(length.(geo))), spread(geo), spread(mer)))
    summarize(la) = @sprintf("acc %.3f, median log α %9.2e", mean(min.(1.0, exp.(la))), median(la))
    Ms = (1, 2, 4, 8, length(geo))
    for (variant, cellsets) in ((:galerkin, (("geo", geo),)), (:gamma, (("geo", geo), ("merit", mer))),
                                (:transport, (("geo", geo), ("merit", mer))))
        variant == :transport && !hard(P) && continue
        for (cname, cells) in cellsets
            line = join([@sprintf("M=%-2d %s", M, summarize([proposal_logα(rng, P, cells, M, variant) for _ in 1:nprop])) for M in Ms], " | ")
            println(@sprintf("  %-9s %-5s %s", variant, cname, line))
        end
    end
    for M in (1, length(geo)), K in (1, 4, 16, 64)
        la = [proposal_logα(rng, P, geo, M, :ncmc; K=K) for _ in 1:(K >= 64 ? 60 : 150)]
        println(@sprintf("  ncmc      geo   M=%-2d K=%-3d %s", M, K, summarize(la)))
    end
end

# --------------------------------------------------------------------------
# C: weather tilt by mean force (regularized toy)
# --------------------------------------------------------------------------
function part_C(P; T=3000, nrelax=3, navg=3)
    rng = MersenneTwister(31)
    N, n = P.N, P.n
    xy = [((i - 1) % n, (i - 1) ÷ n) for i in 1:N]
    σc, ℓ = 0.2, 3.0
    Σc = [σc^2 * exp(-hypot(xy[i][1] - xy[j][1], xy[i][2] - xy[j][2]) / ℓ) for i in 1:N, j in 1:N]
    Lc = cholesky(Symmetric(Σc + 1e-10I)).L
    μc = ones(N)
    shift_per_λ = -Σc * P.r                       # ∇F ≈ −r in the near-vertex regime
    λ = 0.5σc / mean(abs.(shift_per_λ))
    pairs_slack!(s) = (for _ in 1:N; j = rand(rng, 1:N); j != P.istar && line_move!(rng, P, s, [j, P.istar], [1.0, -1.0]); end)
    ρ = 0.3
    a1 = sqrt(1 - ρ^2)
    println(@sprintf("\n== C [regularized, μ/max r = %.0f]: tilt e^{λF}, λ = %.3g (α = 0 joint chain would tilt by β = %.3g, %.0f× stronger) ==",
                     P.μ / maximum(P.r), λ, P.β, P.β / λ))
    for λrun in (0.0, λ)
        c = copy(μc)
        x_s = exact_sample(rng, P, sum(c))     # s = c − Lx
        csum = zeros(N); gsum = zeros(N); cnt = 0
        for t in 1:T
            # fast block: relax, then estimate the force ∇F = E[−μ 1{s<0}]
            for _ in 1:nrelax; pairs_slack!(x_s); end
            g = zeros(N)
            for _ in 1:navg
                pairs_slack!(x_s)
                g .+= -P.μ .* (x_s .< 0)
            end
            g ./= navg
            # slow block: pCN step for c with the force λ g (x fixed ⇒ s shifts with c)
            cn = μc .+ a1 .* (c .- μc) .+ (1 - a1) .* (Σc * (λrun .* g)) .+ ρ .* (Lc * randn(rng, N))
            x_s .+= cn .- c
            c = cn
            if t > T ÷ 5
                csum .+= c; gsum .+= g; cnt += 1
            end
        end
        cm = csum ./ cnt .- μc
        target = λrun .* shift_per_λ
        rel = λrun == 0 ? norm(cm) / (σc * sqrt(N)) : norm(cm .- target) / norm(target)
        println(@sprintf("  λ = %-8.3g mean shift of c: |Δc|/node %.4f (target %.4f); %s %.3f; force vs −r: corr %.3f, rel. error %.3f",
                         λrun, mean(abs.(cm)), mean(abs.(target)), λrun == 0 ? "‖Δc‖/(σ_c√N)" : "rel. error of shift",
                         rel, cor(gsum ./ cnt, -P.r), norm(gsum ./ cnt .+ P.r) / norm(P.r)))
    end
end

function main()
    check_line_sampler()
    Ph = grid_problem(NSIDE; βrel=200.0, seed=1)
    Pr = grid_problem(NSIDE; βrel=200.0, μrel=30.0, seed=1)
    part_A(Ph, "hard")
    part_A(Pr, "regularized μ = 30 max r")
    part_B(Ph, "hard")
    part_B(Pr, "regularized μ = 30 max r")
    Pr3 = grid_problem(NSIDE; βrel=200.0, μrel=3.0, seed=1)
    part_B(Pr3, "regularized μ = 3 max r")
    part_C(Pr)
end

main()
