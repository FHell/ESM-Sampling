# The one-thread-per-site kernel of the dense draw on the `:soc` family (docs/fused_kernel_exploration.md §6).
#   1. Julia arrays: the package's dense draw (`line_draw`) against the kernel's algorithm
#      (`soc_draw_ref`), and the sites' points against `dense_points`: bitwise;
#   2. GPU: the draws alone (kernel vs reference, vs the XLA draw; the XLA draw vs its array form),
#      then `do_round` against `do_round_soc` (t and the loop state), compile times;
#   3. the CPU backend with `raise = true` (the kernel raised to HLO): the project's normal testing
#      contract if it compiles; compared with the reference;
#   4. timings of the draws and the rounds (laptop GPU: prototype-internal only).
# 40 nodes × H hours (default 6), C chains (default 4); the plan's soc rounds. Environment: C, H, NT
# (threads per block, 128), NREP (timing repetitions, 5). Run from the repository root:
#   timeout 900 nice -n 19 julia --project=device/experimental -t 4 device/experimental/soc_fused.jl
using ESMDevice, ESMSampling, Reactant, CUDA, KernelAbstractions, Random, Printf, Statistics
include(joinpath(@__DIR__, "fused_soc.jl"))
const ML = ESMSampling
include(joinpath(@__DIR__, "..", "test", "setup40.jl"))

const C = parse(Int, get(ENV, "C", "4"))
const H = parse(Int, get(ENV, "H", "6"))
const NT = parse(Int, get(ENV, "NT", "128"))
const NREP = parse(Int, get(ENV, "NREP", "5"))
weyl(n, off) = [mod(0.5 + (off + k) * 0.6180339887498949, 1.0) for k in 1:n]
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
    pl = D.plan(M, S, P.βv; families=(:soc,), cls=cls, cells=ci, jump=:auto, backend="gpu")
end
tab = pl.dev[:soc]
B, R = size(tab.on)
fs = Val(:soc)
@printf("setup %.0f s: soc, T = %d, C = %d, %d rounds of %d sites (%d real in round 1)\n", tsetup, M.T, C, R, B, count(tab.on[:, 1] .> 0))
U = reshape(weyl(C * B * 3, 0), C, B, 3)
u1, u2 = U[:, :, 1:1], U[:, :, 2:2]
rel(a, b) = maximum(abs.(a .- b) ./ max.(1.0, abs.(b)); init=0.0)

# 1. Julia arrays: the package's dense draw against the kernel's algorithm, and the points
tb, _ = D.build(fs, pl.st, pl.cp, pl.sp, tab, 1, pl.k)
So = tb.st.soc
@printf("soc batch: %d columns, %d slots, moving %s / %s, points %s\n", length(So.slot), tb.st.nr, So.mve, So.mve1, D.dense_size(tb, pl.k; K=D.nroots(fs)))
ta = @elapsed (tj, oj) = D.line_draw(fs, tb, pl.k, u1, u2)
tr = @elapsed (tf, of) = sample_soc_fused(tb, pl.k, u1, u2; K=D.nroots(fs))
@printf("arrays: line_draw %.1f s, reference %.1f s; max relative |t_ref − t_dense| %.2e (%d of %d bitwise different); guards equal %s (%d); moved %d\n",
        ta, tr, rel(tf[:, :, 1], tj[:, :, 1]), count(tf .!= tj), length(tj), oj == of, count(oj), count(tj .!= 0.0))
pd, _ = D.dense_points(tb, pl.k, u1 .* 0.0; K=D.nroots(fs))
NP = size(pd, 3); CB = C * B
f2(X, m) = X === nothing ? nothing : reshape(X, CB * m)
flds = (f2(So.e0, 2), f2(So.ae, 2), f2(So.e10, 2), f2(So.ae1, 2), f2(So.k, 2), f2(So.w, 2), f2(So.h, 2), f2(So.ηin, 2), f2(So.ηout, 2),
        f2(So.qt, 2), f2(So.mc, 2), f2(tb.st.r0, 2), reshape(u1, CB), reshape(u2, CB))
ptsref = zeros(NP, CB)
soc_draw_ref(flds..., pl.k.Δt, pl.k.cv, pl.k.β, pl.k.cshed, pl.k.cdump, pl.k.cpumap, 2, 2, 2, NP, Tuple(So.mve), Tuple(So.mve1), Tuple(So.slot); pts=ptsref)
dp = abs.(reshape(permutedims(pd, (3, 1, 2)), NP, CB) .- ptsref)
@printf("points: %d per site; max |Δ| %.2e, %d of %d sites differ\n", NP, maximum(dp), count(any(dp .> 0.0; dims=1)), CB)

# 2. GPU: the draws alone and the rounds
Reactant.set_default_backend("gpu")
R_ = Reactant.to_rarray
args = (R_(pl.st), R_(pl.cp), R_(pl.sp), R_(pl.k), R_(tab), R_(U))
fx(st, cp, sp, k, tab, u) = D.do_round(fs, st, cp, sp, k, tab, 1, u)
ff(st, cp, sp, k, tab, u) = do_round_soc(st, cp, sp, k, tab, 1, u; nt=NT)
function fdraw(st, cp, sp, k, tab, u)
    tb, _ = D.build(fs, st, cp, sp, tab, 1, k)
    return sample_soc_fused(tb, k, u[:, :, 1:1], u[:, :, 2:2]; K=D.nroots(fs), nt=NT)
end
function fdrawx(st, cp, sp, k, tab, u)
    tb, _ = D.build(fs, st, cp, sp, tab, 1, k)
    return D.line_draw(fs, tb, k, u[:, :, 1:1], u[:, :, 2:2])
end
timed(name, f) = (t = @elapsed (c, s) = Reactant.Profiler.@timed_compile sync=true donated_args=:none f(args...); @printf("compile: %s %.1f s (trace %.1f, mlir %.1f, xla %.1f)\n", name, t, s.tracing_time_μs / 1e6, s.mlir_time_μs / 1e6, s.xla_time_μs / 1e6); flush(stdout); c)
cd_ = timed("fused draw", fdraw)
cdx = timed("XLA draw", fdrawx)
od, odx = cd_(args...), cdx(args...)
tdg, tdx = Array(od[1])[:, :, 1], Array(odx[1])[:, :, 1]
@printf("GPU draws: fused vs reference %.2e (%d bitwise different), fused vs XLA draw %.2e, XLA draw vs its array form %.2e (%d bitwise different); guards equal %s\n",
        rel(tdg, tf[:, :, 1]), count(tdg .!= tf[:, :, 1]), rel(tdg, tdx), rel(tdx, tj[:, :, 1]), count(tdx .!= tj[:, :, 1]), Array(od[2]) == Array(odx[2]))
cx = timed("XLA round", fx)
cf = timed("fused round", ff)
ox = cx(args...); of_ = cf(args...)
txg, tfg = Array(ox[2])[:, :, 1], Array(of_[2])[:, :, 1]
@printf("GPU rounds: max relative |t_fused − t_xla| %.2e (%d bitwise different of %d); %d sites moved (XLA), %d (fused)\n",
        rel(tfg, txg), count(tfg .!= txg), length(txg), count(txg .!= 0.0), count(tfg .!= 0.0))
for fld_ in (:θ, :ω, :uc, :e, :u, :r)
    a, b = Array(getfield(ox[1], fld_)), Array(getfield(of_[1], fld_))
    @printf("  %-2s max |fused − xla| = %.2e (relative %.1e)\n", fld_, maximum(abs.(a .- b)), rel(a, b))
end

# 3. the CPU backend: the kernel raised to HLO (`raise = true`), and the kernel compiled for the host
#    as it is (Reactant lowers the kernel call to LLVM-dialect code on a CPU client)
Reactant.set_default_backend("cpu")
Rc_ = Reactant.to_rarray
argsc = (Rc_(pl.st), Rc_(pl.cp), Rc_(pl.sp), Rc_(pl.k), Rc_(tab), Rc_(U))
for (name, raise) in (("raise = true", true), ("no raise", false))
    try
        tcr = @elapsed cdr = raise ? (@compile raise=true sync=true donated_args=:none fdraw(argsc...)) :
                                     (@compile sync=true donated_args=:none fdraw(argsc...))
        odr = cdr(argsc...)
        tdr = Array(odr[1])[:, :, 1]
        @printf("CPU backend, %s: compiled in %.1f s; vs reference %.2e (%d bitwise different), vs the GPU kernel %.2e; guards equal %s\n",
                name, tcr, rel(tdr, tf[:, :, 1]), count(tdr .!= tf[:, :, 1]), rel(tdr, tdg), Array(odr[2]) == Array(of))
    catch err
        println("CPU backend, ", name, ": FAILED: ", first(sprint(showerror, err), 300))
    end
end
Reactant.set_default_backend("gpu")

# 4. timings (sync = true), program size and memory
timeit(f) = (f(args...); ts = [@elapsed f(args...) for _ in 1:NREP]; (minimum(ts), median(ts)))
function fbuild(st, cp, sp, k, tab, u)                          # the gathers of `build` alone (baseline)
    tb, _ = D.build(fs, st, cp, sp, tab, 1, k)
    So = tb.st.soc
    return So.e0, So.ae, So.e10, So.ae1, So.k, So.w, So.h, So.ηin, So.ηout, So.qt, So.mc, tb.st.r0
end
cb_ = @compile sync=true donated_args=:none fbuild(args...)
mx, mdx = timeit(cx); mf, mdf = timeit(cf); mdr, mddr = timeit(cd_); mxd, mdxd = timeit(cdx); mb_, mdb_ = timeit(cb_)
nsites = C * count(tab.on[:, 1] .> 0)
@printf("time per round (min / median of %d): XLA %.4f / %.4f s, fused %.4f / %.4f s; draws alone: XLA %.4f / %.4f s, fused %.4f / %.4f s; build only %.4f / %.4f s; %d real sites\n",
        NREP, mx, mdx, mf, mdf, mxd, mdxd, mdr, mddr, mb_, mdb_, nsites)
@printf("per site: round XLA %.2f µs, fused %.2f µs; draw XLA %.2f µs, fused %.2f µs; speed-up of the round %.2fx, of the draw %.2fx\n",
        1e6 * mx / nsites, 1e6 * mf / nsites, 1e6 * mxd / nsites, 1e6 * mdr / nsites, mx / mf, mxd / mdr)
nops(m) = (s = string(m); count("stablehlo.", s) + count("enzymexla.", s) + count("chlo.", s))
println("HLO ops (optimized module): XLA round ", nops(@code_hlo optimize=true donated_args=:none fx(args...)), ", fused round ", nops(@code_hlo optimize=true donated_args=:none ff(args...)))
const API = Reactant.MLIR.API
dev = Reactant.XLA.default_device(Reactant.XLA.default_backend())
function memstats(exec)                                          # the compiled executable's memory statistics
    st = Ref(API.JLCompiledMemoryStats(ntuple(_ -> Int64(0), 11)...))
    API.PjRtLoadedExecutableGetCompiledMemoryStats(exec.exec.exec, st)
    return st[]
end
function peakrun(f, a; n=3)                                      # the device allocator's peak during n runs
    f(a...)
    API.PjRtDeviceClearMemoryStats(dev.device)
    base = Reactant.XLA.allocatorstats(dev).bytes_in_use
    for _ in 1:n; f(a...); end
    return Reactant.XLA.allocatorstats(dev).peak_bytes_in_use - base
end
MB(x) = round(x / 2^20; digits=1)
for (name, c) in (("XLA round", cx), ("fused round", cf))
    m = memstats(c)
    @printf("%-12s temporaries %.1f MB, arguments %.1f MB, outputs %.1f MB, peak %.1f MB; allocator peak above the resident buffers %.1f MB\n",
            name, MB(m.temp_size_in_bytes), MB(m.argument_size_in_bytes), MB(m.output_size_in_bytes), MB(m.peak_memory_in_bytes), MB(peakrun(c, args)))
end
