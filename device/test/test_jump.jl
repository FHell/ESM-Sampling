# The two forms of the line sampler (docs/system/device.md §3), deterministic, Julia arrays:
#   1. the potential is linear between consecutive points of either form (midpoints evaluated
#      directly): every kink of βΔC is one of the points (this fails if a kink is missed, e.g. a
#      balance-slot root interpolated across a SoC injection kink);
#   2. the slope-jump potential at its points (jumps integrated from 0) equals the direct
#      evaluation;
#   3. with the same uniforms both forms draw the same t (up to round-off), apart from guarded sites.
# Model A: 40 nodes × 6 h, all terms (tracking: LINEAR families); model B: no losses (T2 families).
# Every round of every family is checked on a prefix of NSITE sites.
# Run: cd device && nice -n 19 julia --project=. -t 4 --heap-size-hint=3G test/test_jump.jl   (CASE=A|B|C, default ABC)
include("setup40.jl")
using Test, Random

const NSITE = parse(Int, get(ENV, "NSITE", "100"))

"Rows `rows` of a device table (fields whose first dimension is the site axis)."
function subtab(tab, rows)
    B = size(tab.on, 1)
    NamedTuple{keys(tab)}(Tuple(x isa AbstractArray && size(x, 1) == B ? copy(selectdim(x, 1, rows)) : x for x in values(tab)))
end

"Largest deviation from linearity of βΔC between consecutive distinct points q (C, B, N), in nats beyond round-off."
function nonlinearity(tb, k, q)
    N = size(q, 3)
    qs = sort(q; dims=3)
    a, b = qs[:, :, 1:N-1], qs[:, :, 2:N]
    ca, cb = D.eval_terms(tb, k, a; sec=false)[1], D.eval_terms(tb, k, b; sec=false)[1]
    cm = D.eval_terms(tb, k, 0.5 .* (a .+ b); sec=false)[1]
    dev = abs.(cm .- 0.5 .* (ca .+ cb)) .- 1e-9 .* (abs.(ca) .+ abs.(cb))
    return dropdims(maximum(ifelse.(b .> a, dev, 0.0); dims=3); dims=3)
end

function jump_case(name; tracking_model::Bool, fams, cells=false, C=2)
    P, M, ix, S, ss = setup40(; C=C, losses=tracking_model, link_losses=tracking_model)
    LM = ML.LocalModel(P, ss[1].ℓ)
    ci = cells ? D.cell_info(P, LM, ix) : nothing
    fams == :cells && (fams = D.cell_families(ci))
    cls = zeros(C, M.T * M.N)
    for c in 1:C; cls[c, :] .= vec(permutedims(Float64.(S.dnet[c, :, :] .< 0))); end
    pl = D.plan(M, S, P.βv; families=fams, cls=cls, cells=ci, jump=:all)
    @testset "$name" begin
        for f in pl.fams, r in 1:size(pl.dev[f].on, 2)
            tab = pl.dev[f]
            rows = findall(tab.on[:, r] .> 0)[1:min(NSITE, end)]
            isempty(rows) && continue
            st = subtab(tab, rows)
            tb, _ = D.build(Val(f), pl.st, pl.cp, pl.sp, st, r, pl.k)
            K = D.nroots(Val(f))
            B = length(rows)
            rng = MersenneTwister(r)
            u1, u2 = rand(rng, C, B, 1), rand(rng, C, B, 1)
            # dense form
            dp, overd = D.dense_points(tb, pl.k, zeros(C, B, 1); K=K)
            td = ifelse.(overd, 0.0, D.sample_points(tb, pl.k, dp, u1, u2))
            # slope-jump form (with the plan's cap)
            tj, overj = D.sample_jump(tb, pl.k, u1, u2; K=K, cap=D.jumpcap(Val(f)))
            pts, jmp, s0, _, nz0, s0abs = D.jump_parts(tb, pl.k; K=K)
            q, J, _, J0abs = D.jump_points(pts, jmp, nz0, 0)
            pe, c = D.jump_values(q, J, s0, pl.k.β, s0abs, J0abs)
            cdir = D.eval_terms(tb, pl.k, pe; sec=false)[1]
            ok = .!overd[:, :, 1] .& .!overj[:, :, 1]       # guarded sites stay at t = 0 (their lists may miss points)
            dcs = dropdims(maximum(abs.(c .- cdir) ./ (1.0 .+ abs.(cdir)); dims=3); dims=3)
            dc = maximum(dcs[ok]; init=0.0)
            nd, nj = maximum(nonlinearity(tb, pl.k, dp)[ok]; init=0.0), maximum(nonlinearity(tb, pl.k, q)[ok]; init=0.0)
            dt = abs.(tj .- td)[:, :, 1] ./ max.(1.0, abs.(td[:, :, 1]))
            @printf("  %-10s r %2d: %4d sites  nonlinearity dense %.1e, slope-jump %.1e nats; potential %.1e; t %.1e (%d > 1e-9); guard dense %d, slope-jump %d\n",
                    f, r, B, nd, nj, dc, maximum(dt[ok]; init=0.0), count(dt[ok] .> 1e-9), count(overd), count(overj))
            @test nd <= 1e-6
            @test nj <= 1e-6
            @test dc <= 1e-9
            @test count(dt[ok] .> 1e-9) <= 0.002 * length(dt)
            @test count(overj) <= 0.02 * length(overj)
            @test count(overj .& .!overd) <= 0.01 * length(overj)    # the slope-jump cap
        end
    end
end

const CASES = get(ENV, "CASE", "ABC")      # CASE=B: only case B
'A' in CASES && jump_case("A (all terms; LINEAR)"; tracking_model=true, fams=(:spair, :tsocL, :tpotL, :sblkL2, :sblkL4, :sblkL8))
'B' in CASES && jump_case("B (no losses; T2)"; tracking_model=false, fams=(:spair, :tsoc, :tpot, :sblk2, :sblk4, :sblk16))
'C' in CASES && jump_case("C (all terms; cell templates)"; tracking_model=true, fams=:cells, cells=true)
