# Reactant (CPU backend) against the Julia-array run, family by family (device testing order, step 2):
# the same rounds, the same uniforms, one compiled program per family; prints the compile time and the
# largest relative difference of the loop state. 40 nodes × 6 h, all model terms (CASE=A, LINEAR
# families) or without line losses and lossy links (CASE=B, share-preserving), 2 chains.
#   FAMS=tsoc,tpot (default: the case's families), NSITE (default 60) sites per round, NROUND (2).
# Run: cd device && nice -n 19 julia --project=. -t 4 test/test_reactant_fams.jl
include("setup40.jl")
using Test, Random
using Reactant: to_rarray
Reactant.set_default_backend("cpu")

const NSITE = parse(Int, get(ENV, "NSITE", "60"))
const NROUND = parse(Int, get(ENV, "NROUND", "2"))
const CASE = get(ENV, "CASE", "B")
weyl(n, off) = [mod(0.5 + (off + k) * 0.6180339887498949, 1.0) for k in 1:n]

P, M, ix, S, ss = setup40(; C=2, losses=CASE == "A", link_losses=CASE == "A")
LM = ML.LocalModel(P, ss[1].ℓ)
ci = D.cell_info(P, LM, ix)
C = 2
cls = zeros(C, M.T * M.N)
for c in 1:C; cls[c, :] .= vec(permutedims(Float64.(S.dnet[c, :, :] .< 0))); end
D.TDIR_CHAINS[] = C
D.TDIR_SOURCE[] = (fam, τ, rng, c) -> begin
    sys = P.E.sys
    rand(rng) < 0.3 && return D.tline_dir(sys, LM, rand(rng, eachindex(sys.line_s)))
    d = randn(rng, sys.N); d .-= sum(d) / sys.N; d ./= sqrt(sum(abs2, d))
    D.tdir_g(sys, d)
end
deffams = CASE == "A" ? (:spair, :tsocL, :tpotL, :sblkL2, :sblkL8, :tdirL, :tdir0) : (:tsoc, :tpot, :sblk2, :sblk8, :tdir, :tdir0)
fams = haskey(ENV, "FAMS") ? Tuple(Symbol.(split(ENV["FAMS"], ","))) : deffams

@testset "Reactant (CPU backend) = Julia arrays, case $CASE" begin
    for f in fams
        pl = D.plan(M, S, P.βv; families=(f,), cls=cls, bmax=Dict(f => NSITE), cells=ci)
        f in pl.fams || (println("  $f: no sites"); continue)
        rounds = NamedTuple{(f,)}((pl.sched[f][1:min(NROUND, end)],))
        tabs = NamedTuple{(f,)}((D.device_table(f, D.host_table(f, pl.tp, rounds[f])),))
        B, R = size(tabs[f].on)
        U = NamedTuple{(f,)}((reshape(weyl(C * B * 3 * R, 0), C, B, 3, R),))
        fv = Val((f,))
        stJ = D.run_rounds(pl.st, pl.cp, pl.sp, pl.k, tabs, 1, nothing, U, fv)
        runU(st, cp, sp, k, tabs, U) = D.run_rounds(st, cp, sp, k, tabs, 1, nothing, U, fv)
        args = (to_rarray(pl.st), to_rarray(pl.cp), to_rarray(pl.sp), to_rarray(pl.k), to_rarray(tabs), to_rarray(U))
        tc = @elapsed fc = Reactant.@compile runU(args...)
        o = fc(args...)
        worst = 0.0
        for fld in (:θ, :ω, :uc, :e, :u, :r)
            a, b = Array(getfield(o, fld)), getfield(stJ, fld)
            worst = max(worst, maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1.0))
        end
        moved = count(stJ.ω .!= pl.st.ω) + count(stJ.θ .!= pl.st.θ) + count(stJ.e .!= pl.st.e)
        @printf("  %-10s %d rounds × %d sites: compile %5.0f s, max relative |reactant − julia| %.1e (%d entries moved), %s form\n",
                f, R, B, tc, worst, moved, D.jumpform(Val(f)) ? "slope-jump" : "dense")
        flush(stdout)
        @test worst <= 1e-10
        GC.gc()
    end
end
