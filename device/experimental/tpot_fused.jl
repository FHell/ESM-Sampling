# The fused line-draw kernel on one round of `:tpot` (docs/fused_kernel_exploration.md).
#   1. Julia arrays: `sample_jump` (the package's slope-jump draw) against `sample_jump_fused` (the
#      kernel's algorithm, `fused_draw_ref`) on the same term batch and uniforms;
#   2. GPU: the round compiled with the XLA draw (`do_round`) and with the kernel (`do_round_fused`),
#      the same inputs, t and the loop state compared; the kernel's t against the Julia reference;
#   3. timings of the two compiled rounds and of the per-column part alone (`jump_parts`).
# 40 nodes × H hours (default 6), no line losses and lossy links (the share-preserving `:tpot`),
# C chains (default 4). Environment: C, H, NSITE (sites per round; default: the plan's round),
# NT (threads per block, 256), NREP (timing repetitions, 5), FAM (default tpot: any slope-jump family).
# Run from the repository root (a few minutes; the setup dominates):
#   timeout 600 nice -n 19 julia --project=device/experimental -t 4 device/experimental/tpot_fused.jl
using ESMDevice, ESMSampling, Reactant, CUDA, KernelAbstractions, Random, Printf, Statistics
include(joinpath(@__DIR__, "fused_jump.jl"))
const ML = ESMSampling
include(joinpath(@__DIR__, "..", "test", "setup40.jl"))

const C = parse(Int, get(ENV, "C", "4"))
const H = parse(Int, get(ENV, "H", "6"))
const NT = parse(Int, get(ENV, "NT", "256"))
const NREP = parse(Int, get(ENV, "NREP", "5"))
const FAM = Symbol(get(ENV, "FAM", "tpot"))
nsite = parse(Int, get(ENV, "NSITE", "0"))
weyl(n, off) = [mod(0.5 + (off + k) * 0.6180339887498949, 1.0) for k in 1:n]
# the model file: ESM_DATA, else the repository's, else the main checkout's (a worktree has no data/)
const DATA = get(ENV, "ESM_DATA", let cands = [joinpath(@__DIR__, "..", "..", "data", "processed", "model.h5"),
                                               joinpath(@__DIR__, "..", "..", "..", "..", "..", "data", "processed", "model.h5")]
    i = findfirst(isfile, cands)
    i === nothing ? cands[1] : cands[i]
end)

tsetup = @elapsed begin
    P, M, ix, S, ss = setup40(; C=C, losses=false, link_losses=false, data=DATA, levels=H == 6 ? nothing : [(10, H), (40, H)])
    LM = ML.LocalModel(P, ss[1].ℓ)
    ci = D.cell_info(P, LM, ix)
    cls = zeros(C, M.T * M.N)
    for c in 1:C; cls[c, :] .= vec(permutedims(Float64.(S.dnet[c, :, :] .< 0))); end
    D.BLOCK_GROUPS[] = D.block_groups(P, ix)
    for L in D.BLOCK_LENGTHS
        D.set_profiles!(L, hcat([vcat(0.0, rand(MersenneTwister(L + j), L - 1) .- 0.3, 0.0) for j in 1:2]...))
    end
    bmax = nsite > 0 ? Dict(FAM => nsite) : Dict{Symbol,Int}()
    pl = D.plan(M, S, P.βv; families=(FAM,), cls=cls, cells=ci, jump=:all, backend="gpu", bmax=bmax)
end
tab = pl.dev[FAM]
B, R = size(tab.on)
@printf("setup %.0f s: %s, T = %d, C = %d, %d rounds of %d sites, cap %d, K = %d\n", tsetup, FAM, M.T, C, R, B, D.jumpcap(Val(FAM)), D.nroots(Val(FAM)) isa Val{4} ? 4 : 2)
fv = Val(FAM)
U = reshape(weyl(C * B * 3, 0), C, B, 3)
u1, u2 = U[:, :, 1:1], U[:, :, 2:2]

# 1. Julia arrays: the package's draw against the kernel's algorithm
tb, _ = D.build(fv, pl.st, pl.cp, pl.sp, tab, 1, pl.k)
qa, Ja, s0, over, nz0, s0abs = fused_entries(tb, pl.k; K=D.nroots(fv))
N = size(qa, 1)
@printf("entries per site N = %d (NP = %d), cap M = %d, shared memory %d bytes per block\n", N, nextpow(2, N + 1),
        D.jumpcap(fv) > 0 ? D.jumpcap(fv) : N + 1, fused_shmem(N, D.jumpcap(fv) > 0 ? D.jumpcap(fv) : N + 1, NT))
ta = @elapsed (tj, oj) = D.sample_jump(tb, pl.k, u1, u2; K=D.nroots(fv), cap=D.jumpcap(fv))
tr = @elapsed (tf, of) = sample_jump_fused(tb, pl.k, u1, u2; K=D.nroots(fv), cap=D.jumpcap(fv), nt=NT)
on = tab.on[:, 1] .> 0
rel(a, b) = maximum(abs.(a .- b) ./ max.(1.0, abs.(b)); init=0.0)
dt = abs.(tj .- tf)[:, :, 1] ./ max.(1.0, abs.(tj[:, :, 1]))
@printf("arrays: sample_jump %.1f s, fused reference %.1f s; max relative |t_fused − t_jump| %.2e (%d of %d sites > 1e-12, %d > 1e-9); guards equal: %s (%d guarded); %d sites moved\n",
        ta, tr, maximum(dt), count(dt .> 1e-12), length(dt), count(dt .> 1e-9), oj == of, count(oj), count(tj .!= 0.0))

# 2. GPU: the compiled round with the XLA draw and with the kernel
Reactant.set_default_backend("gpu")
R_ = Reactant.to_rarray
args = (R_(pl.st), R_(pl.cp), R_(pl.sp), R_(pl.k), R_(tab), R_(U))
fx(st, cp, sp, k, tab, u) = D.do_round(fv, st, cp, sp, k, tab, 1, u)
ff(st, cp, sp, k, tab, u) = do_round_fused(fv, st, cp, sp, k, tab, 1, u; nt=NT)
function fparts(st, cp, sp, k, tab, u)                          # the per-column part alone
    tb, _ = D.build(fv, st, cp, sp, tab, 1, k)
    qa, Ja, s0, over, nz0, s0abs = fused_entries(tb, k; K=D.nroots(fv))
    return qa, Ja, s0, nz0, s0abs
end
tcx = @elapsed cx = @compile sync=true donated_args=:none fx(args...)
@printf("compile: XLA round %.0f s\n", tcx); flush(stdout)
tcf = @elapsed cf = @compile sync=true donated_args=:none ff(args...)
@printf("compile: fused round %.0f s\n", tcf); flush(stdout)
tcp = @elapsed cp_ = @compile sync=true donated_args=:none fparts(args...)
@printf("compile: parts only %.0f s\n", tcp); flush(stdout)
ox = cx(args...); of_ = cf(args...)
txg, tfg = Array(ox[2])[:, :, 1], Array(of_[2])[:, :, 1]
dg = abs.(txg .- tfg) ./ max.(1.0, abs.(txg))
@printf("GPU: max relative |t_fused − t_xla| %.2e (%d of %d sites > 1e-12, %d > 1e-9); XLA vs arrays %.2e; fused vs reference %.2e\n",
        maximum(dg), count(dg .> 1e-12), length(dg), count(dg .> 1e-9), rel(txg, tj[:, :, 1]), rel(tfg, tf[:, :, 1]))
for fld in (:θ, :ω, :uc, :e, :u, :r)
    a, b = Array(getfield(ox[1], fld)), Array(getfield(of_[1], fld))
    @printf("  %-2s max |fused − xla| = %.2e (relative %.1e)\n", fld, maximum(abs.(a .- b)), rel(a, b))
end

# 3. timings (sync = true: the call returns when the program has finished)
timeit(f) = (f(args...); ts = [@elapsed f(args...) for _ in 1:NREP]; (minimum(ts), median(ts)))
mx, mdx = timeit(cx); mf, mdf = timeit(cf); mp, mdp = timeit(cp_)
nsites = C * count(on)
@printf("time per round (min / median of %d): XLA %.4f / %.4f s, fused %.4f / %.4f s, parts only %.4f / %.4f s; %d real sites\n",
        NREP, mx, mdx, mf, mdf, mp, mdp, nsites)
@printf("per site: XLA %.1f µs, fused %.1f µs, parts %.1f µs; global part in XLA ≈ %.4f s (%.0f %% of the round), fused kernel ≈ %.4f s; speed-up of the round %.2fx\n",
        1e6 * mx / nsites, 1e6 * mf / nsites, 1e6 * mp / nsites, mx - mp, 100 * (mx - mp) / mx, mf - mp, mx / mf)
