# The host redraw of the random site choices between device calls (sources.jl, device.md §3):
# deterministic checks on arrays. For every redrawn family: the redrawn tables have the plan's
# shapes (an upload, no recompile), their rounds are conflict-free (so every round stays a
# π-invariant kernel: the choices do not depend on the state), two redraws differ, the same
# stream gives the same tables, few sites are dropped, and one round runs on them.
# Run: cd device && nice -n 19 julia --project=. -t 4 test/test_redraw.jl
include("setup40.jl")
using Test, Random

P, M, ix, S, ss = setup40(; C=2, losses=false, link_losses=false)
LM = ML.LocalModel(P, ss[1].ℓ)
ci = D.cell_info(P, LM, ix)
D.set_sources!(P, LM, ix)
cls = zeros(2, M.T * M.N)
fams = (:spair, :sblk2, :sblkg2, :slong, :xloc, :xpair, :sxpair, :sxpairn, :sxlong, :cell1, :cellpairu2)
pl = D.plan(M, S, P.βv; families=fams, cls=cls, cells=ci, rng=MersenneTwister(11))

@testset "site redraws between device calls" begin
    @test all(D.redrawn, pl.fams)
    rd1, drop1, h1 = D.redraw_tables(pl, M, MersenneTwister(1))
    rd2, _, h2 = D.redraw_tables(pl, M, MersenneTwister(2))
    rd1b, _, _ = D.redraw_tables(pl, M, MersenneTwister(1))
    nsite = sum(Int(sum(pl.host[f].on)) for f in pl.fams)
    println("redrawn: ", keys(rd1), "; dropped $drop1 of $nsite sites")
    @test Set(keys(rd1)) == Set(pl.fams)
    @test drop1 <= 0.01 * nsite
    for f in keys(rd1)
        @test keys(rd1[f]) == keys(pl.dev[f])
        @test all(size(getfield(rd1[f], x)) == size(getfield(pl.dev[f], x)) for x in keys(rd1[f]))   # plan shapes
        @test D.check_conflicts(f, h1[f])[1] && D.check_conflicts(f, h2[f])[1]                        # conflict-free
        @test any(getfield(rd1[f], x) != getfield(rd2[f], x) for x in keys(rd1[f]))                   # a new draw
        @test all(getfield(rd1[f], x) == getfield(rd1b[f], x) for x in keys(rd1[f]))                  # reproducible
        tab = rd1[f]
        u = rand(MersenneTwister(3), size(pl.st.θ, 1), size(tab.on, 1), 3)
        st2, t = D.do_round(Val(f), pl.st, pl.cp, pl.sp, pl.k, tab, 1, u)
        @test all(isfinite, t) && all(isfinite, st2.r) && all(isfinite, st2.e)
        @printf("  %-10s %6d sites (plan %6d), %5d of %5d moved in round 1\n", f, Int(sum(tab.on)), Int(sum(pl.host[f].on)),
                count(!=(0.0), t), length(t))
    end
end
