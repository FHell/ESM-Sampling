# Kernels shared by the toy test scripts (expects `using .Toy`).
using Random, Statistics, Printf, LinearAlgebra

# ---------------------------------------------------------------- kernels
struct Zone
    inW::BitVector
    bnd::Base.RefValue{Float64}
    σW::Base.RefValue{Float64}
end
zt(z::Zone) = (z.inW, z.bnd[], z.σW)
zt(::Nothing) = nothing

function make_kernel(name, P; zone=nothing, iW=0)
    N = P.N
    mv!(rng, s, idx, coef) = line_move!(rng, P, s, idx, coef; zone=zt(zone))
    partner(j) = (zone !== nothing && zone.inW[j]) ? iW : P.istar
    if name == :local
        return (rng, s) -> (for _ in 1:N; mv!(rng, s, local_dir(P, rand(rng, 1:N))...); end)
    elseif name == :pairs_slack
        return (rng, s) -> (for _ in 1:N; j = rand(rng, 1:N); j != P.istar && mv!(rng, s, [j, P.istar], [1.0, -1.0]); end)
    elseif name == :pairs_global
        return (rng, s) -> (for _ in 1:N; i, j = rand(rng, 1:N), rand(rng, 1:N); i != j && mv!(rng, s, [i, j], [1.0, -1.0]); end)
    elseif name == :zonal
        return (rng, s) -> (for _ in 1:N; j = rand(rng, 1:N); p = partner(j); j != p && mv!(rng, s, [j, p], [1.0, -1.0]); end)
    elseif name == :mixture
        return (rng, s) -> (for _ in 1:N
            j = rand(rng, 1:N); p = rand(rng) < 0.5 ? partner(j) : P.istar
            j != p && mv!(rng, s, [j, p], [1.0, -1.0])
        end)
    end
    error("unknown kernel $name")
end

# ------------------------------------------------- MLDA with a full-space coarse chain
# π_c: the toy with reduced costs averaged within cells (and no zonal interface).
# Its cell totals are Gamma(n, β r̄) (truncated by the slack node) and its shares are
# uniform, so the coarse subchain is exact Gibbs. w = π_f/π_c = exp(β Σ (a_i − ā_C) s_i)
# (× the zonal indicator under hard constraints).
struct Coarse
    cells::Vector{Vector{Int}}
    rbar::Vector{Float64}
    abar_of::Vector{Float64}
end
Coarse(P, cells) = (ab = zeros(P.N); for C in cells; ab[C] .= mean(P.a[C]); end;
                    Coarse(cells, [mean(P.r[C]) for C in cells], ab))

function logw(P, cs::Coarse, s, zone)
    lw = 0.0
    for C in cs.cells, i in C
        lw += P.β * (P.a[i] - cs.abar_of[i]) * s[i]
    end
    if zone !== nothing
        sum(s[zone.inW]) < zone.bnd[] && return -Inf
    end
    return lw
end

function draw_trunc_gamma(rng, n, λ, ub)
    for _ in 1:1000
        x = sum(randexp(rng) for _ in 1:n) / λ
        x <= ub && return x
    end
    return ub * rand(rng)
end

function coarse_subchain!(rng, P, cs::Coarse, s, mode, J)
    for _ in 1:J
        q = rand(rng, eachindex(cs.cells))
        C = cs.cells[q]
        σ = sum(s[C])
        σn = draw_trunc_gamma(rng, length(C), P.β * cs.rbar[q], σ + s[P.istar])
        if mode == :transport && σ > 0
            s[C] .*= σn / σ
        else
            u = randexp(rng, length(C)); s[C] .= σn .* u ./ sum(u)
        end
        s[P.istar] += σ - σn
    end
end

function make_mlda(P, cs; mode=:redraw, fine=:local, zone=nothing, stats=Ref((0, 0)))
    finek = fine === nothing ? nothing : make_kernel(fine, P; zone=zone)
    return (rng, s) -> begin
        finek === nothing || finek(rng, s)
        ψ = copy(s)
        coarse_subchain!(rng, P, cs, ψ, mode, length(cs.cells))
        la = logw(P, cs, ψ, zone) - logw(P, cs, s, zone)
        n, a = stats[]
        if log(rand(rng)) < la
            s .= ψ
            zone === nothing || (zone.σW[] = sum(s[zone.inW]))
            stats[] = (n + 1, a + 1)
        else
            stats[] = (n + 1, a)
        end
    end
end

# ------------------------------------------------------------- helpers
westset(P) = BitVector([(i - 1) % P.n < P.n ÷ 2 for i in 1:P.N])
obj(P, s) = -P.β * dot(P.a, s)                       # = β(C − const)

function run_for(kernel!, s0, seconds, rng; observe)
    s = copy(s0)
    ts, O = Float64[], Vector{Vector{Float64}}()
    t0 = time()
    while time() - t0 < seconds
        kernel!(rng, s)
        push!(ts, time() - t0); push!(O, observe(s))
    end
    return s, ts, reduce(hcat, O)
end

