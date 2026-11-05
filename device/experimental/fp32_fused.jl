# Both fused kernels in Float32 against the package's Float32 build (docs/fused_kernel_exploration.md §7):
# the tpot global-part kernel (400 and NSITE2 = 1000 sites per chain) and the soc dense-draw kernel (the
# plan's round), each as the draw alone and as the round, against `ESMDevice.F32`'s XLA form on the same
# Float32 inputs and against the FP64 reference draws on the same sites (the Float32 round-off scale).
# No bitwise or 1e-12 contract in Float32: reported are max relative |Δt|, counts > 1e-4 and > 1e-2, guard
# agreement, and the timings, compile times and temporaries, as in tpot_fused.jl / soc_fused.jl.
# Environment: C (2), H (24), NT (256), NREP (5), NSITE2 (1000). Run from the repository root:
#   timeout 1800 nice -n 19 julia --project=device/experimental -t 4 device/experimental/fp32_fused.jl
using ESMDevice, ESMSampling, Reactant, CUDA, KernelAbstractions, Random, Printf, Statistics
include(joinpath(@__DIR__, "fused_jump.jl"))
include(joinpath(@__DIR__, "fused_soc.jl"))
const ML = ESMSampling
const F = ESMDevice.F32
include(joinpath(@__DIR__, "..", "test", "setup40.jl"))

const C = parse(Int, get(ENV, "C", "2"))
const H = parse(Int, get(ENV, "H", "24"))
const NT = parse(Int, get(ENV, "NT", "256"))
const NREP = parse(Int, get(ENV, "NREP", "5"))
const NSITE2 = parse(Int, get(ENV, "NSITE2", "1000"))
weyl(n, off) = [mod(0.5 + (off + k) * 0.6180339887498949, 1.0) for k in 1:n]
const DATA = get(ENV, "ESM_DATA", let cands = [joinpath(@__DIR__, "..", "..", "data", "processed", "model.h5"),
                                               joinpath(@__DIR__, "..", "..", "..", "..", "..", "data", "processed", "model.h5")]
    i = findfirst(isfile, cands)
    i === nothing ? cands[1] : cands[i]
end)
fv = Val(:tpot); fs = Val(:soc)

# the FP64 setup and plans (the references), and the same in the Float32 build (its own module globals)
P, M, ix, S, ss = setup40(; C=C, losses=false, link_losses=false, data=DATA, levels=H == 6 ? nothing : [(10, H), (40, H)])
LM = ML.LocalModel(P, ss[1].ℓ)
cls = zeros(C, M.T * M.N)
for c in 1:C; cls[c, :] .= vec(permutedims(Float64.(S.dnet[c, :, :] .< 0))); end
profiles(L) = hcat([vcat(0.0, rand(MersenneTwister(L + j), L - 1) .- 0.3, 0.0) for j in 1:2]...)
ci = D.cell_info(P, LM, ix); D.BLOCK_GROUPS[] = D.block_groups(P, ix); for L in D.BLOCK_LENGTHS; D.set_profiles!(L, profiles(L)); end
Mf, ixf = F.pack(P, ss[1].ℓ); Sf = F.to_device(P, Mf, ixf, ss)
cif = F.cell_info(P, LM, ixf); F.BLOCK_GROUPS[] = F.block_groups(P, ixf); for L in F.BLOCK_LENGTHS; F.set_profiles!(L, profiles(L)); end
plans(mod, Mx, Sx, cx) = (mod.plan(Mx, Sx, P.βv; families=(:tpot,), cls=cls, cells=cx, jump=:all, backend="gpu", bmax=Dict(:tpot => 200)),
                          mod.plan(Mx, Sx, P.βv; families=(:tpot,), cls=cls, cells=cx, jump=:all, backend="gpu", bmax=Dict(:tpot => NSITE2)),
                          mod.plan(Mx, Sx, P.βv; families=(:soc,), cls=cls, cells=cx, jump=:auto, backend="gpu"))
pl, pl2, pls = plans(D, M, S, ci)
plf, plf2, plfs = plans(F, Mf, Sf, cif)
@printf("tpot cap: FP64 %d, Float32 %d; soc sites per round %d\n", D.jumpcap(fv), F.jumpcap(fv), size(pls.dev[:soc].on, 1))
rel(a, b) = maximum(abs.(a .- b) ./ max.(1.0, abs.(b)); init=0.0)
d64(a) = Float64.(Array(a)[:, :, 1])
stats(a, b) = (d = abs.(a .- b) ./ max.(1.0, abs.(b)); (maximum(d), count(d .> 1e-4), count(d .> 1e-2)))
R_ = Reactant.to_rarray
Reactant.set_default_backend("gpu")
function tcomp(name, f, a)
    t = @elapsed (c, s) = F.ftcompile(() -> Reactant.Profiler.@timed_compile sync=true donated_args=:none f(a...))
    @printf("compile %s: %.1f s (trace %.1f, mlir %.1f, xla %.1f)\n", name, t, s.tracing_time_μs / 1e6, s.mlir_time_μs / 1e6, s.xla_time_μs / 1e6); flush(stdout)
    return c
end
timeit(f, a) = (f(a...); ts = [@elapsed f(a...) for _ in 1:NREP]; (minimum(ts), median(ts)))
const API = Reactant.MLIR.API
dev = Reactant.XLA.default_device(Reactant.XLA.default_backend())
memstats(exec) = (st = Ref(API.JLCompiledMemoryStats(ntuple(_ -> Int64(0), 11)...)); API.PjRtLoadedExecutableGetCompiledMemoryStats(exec.exec.exec, st); st[])
MB(x) = round(x / 2^20; digits=1)
state_line(ox, of) = join([@sprintf("%s %.1e", f, rel(Float64.(Array(getfield(of[1], f))), Float64.(Array(getfield(ox[1], f))))) for f in (:θ, :ω, :uc, :e, :u, :r)], " ")

# ---- tpot: the FP64 reference draws, then the Float32 draws and rounds
for (p64, pf, label) in ((pl, plf, "tpot 400"), (pl2, plf2, "tpot $(C * NSITE2)"))
    tab = p64.dev[:tpot]; tabf = pf.dev[:tpot]
    B = size(tab.on, 1)
    U = reshape(weyl(C * B * 3, 0), C, B, 3); Uf = Float32.(U)
    tb, _ = D.build(fv, p64.st, p64.cp, p64.sp, tab, 1, p64.k)
    tj, _ = D.sample_jump(tb, p64.k, U[:, :, 1:1], U[:, :, 2:2]; K=D.nroots(fv), cap=D.jumpcap(fv))
    tref = tj[:, :, 1]
    tbf, _ = F.build(fv, pf.st, pf.cp, pf.sp, tabf, 1, pf.k)
    tj32, _ = F.sample_jump(tbf, pf.k, Uf[:, :, 1:1], Uf[:, :, 2:2]; K=F.nroots(fv), cap=F.jumpcap(fv))
    tf32, _ = sample_jump_fused(tbf, pf.k, Uf[:, :, 1:1], Uf[:, :, 2:2]; K=F.nroots(fv), cap=F.jumpcap(fv), nt=NT)
    @printf("%s Float32 arrays: XLA form vs FP64 %.2e (>1e-4: %d, >1e-2: %d); kernel reference vs FP64 %.2e (>1e-4: %d, >1e-2: %d); kernel vs XLA form %.2e\n", label,
            stats(Float64.(tj32[:, :, 1]), tref)..., stats(Float64.(tf32[:, :, 1]), tref)..., stats(Float64.(tf32[:, :, 1]), Float64.(tj32[:, :, 1]))[1])
    args = (R_(pf.st), R_(pf.cp), R_(pf.sp), R_(pf.k), R_(tabf), R_(Uf))
    fx(st, cp, sp, k, tab, u) = F.do_round(fv, st, cp, sp, k, tab, 1, u)
    ff(st, cp, sp, k, tab, u) = do_round_fused(fv, st, cp, sp, k, tab, 1, u; nt=NT)
    function fdraw(st, cp, sp, k, tab, u; mode=0)
        tb, _ = F.build(fv, st, cp, sp, tab, 1, k)
        return sample_jump_fused(tb, k, u[:, :, 1:1], u[:, :, 2:2]; K=F.nroots(fv), cap=F.jumpcap(fv), nt=NT, mode=mode)
    end
    fdraw1(a...) = fdraw(a...; mode=1); fdraw2(a...) = fdraw(a...; mode=2)
    function fdrawx(st, cp, sp, k, tab, u)
        tb, _ = F.build(fv, st, cp, sp, tab, 1, k)
        return F.sample_jump(tb, k, u[:, :, 1:1], u[:, :, 2:2]; K=F.nroots(fv), cap=F.jumpcap(fv))
    end
    function fparts(st, cp, sp, k, tab, u)
        tb, _ = F.build(fv, st, cp, sp, tab, 1, k)
        qa, Ja, s0, over, nz0, s0abs = fused_entries(tb, k; K=F.nroots(fv))
        return qa, Ja, s0, nz0, s0abs
    end
    cx = tcomp("XLA round", fx, args); cf = tcomp("fused round", ff, args); cd_ = tcomp("fused draw", fdraw, args)
    cdx = tcomp("XLA draw", fdrawx, args); cp_ = tcomp("parts", fparts, args); cd1 = tcomp("no tail", fdraw1, args); cd2 = tcomp("no tail no sort", fdraw2, args)
    od, odx = cd_(args...), cdx(args...)
    tk, tx = d64(od[1]), d64(odx[1])
    @printf("%s Float32 GPU draws: kernel vs XLA %.2e (>1e-4: %d, >1e-2: %d); kernel vs FP64 %.2e (>1e-4: %d, >1e-2: %d); XLA vs FP64 %.2e (>1e-4: %d, >1e-2: %d); kernel vs its array form %.2e, XLA vs its array form %.2e; guards %d / %d, flips %d\n",
            label, stats(tk, tx)..., stats(tk, tref)..., stats(tx, tref)..., stats(tk, Float64.(tf32[:, :, 1]))[1], stats(tx, Float64.(tj32[:, :, 1]))[1],
            count(Array(od[2])), count(Array(odx[2])), count(Array(od[2]) .!= Array(odx[2])))
    ox, of = cx(args...), cf(args...)
    @printf("%s Float32 GPU rounds: t kernel vs XLA %.2e (>1e-4: %d, >1e-2: %d); state: %s\n", label, stats(d64(of[2]), d64(ox[2]))..., state_line(ox, of))
    mx, mdx = timeit(cx, args); mf, mdf = timeit(cf, args); md, mdd = timeit(cd_, args); mxd, mdxd = timeit(cdx, args); mp, _ = timeit(cp_, args); m1, _ = timeit(cd1, args); m2, _ = timeit(cd2, args)
    @printf("%s Float32 timings (min/median of %d): XLA round %.4f / %.4f, fused round %.4f / %.4f; XLA draw %.4f / %.4f, fused draw %.4f / %.4f; parts %.4f, no tail %.4f, no sort %.4f (kernel tail %.4f, sort %.4f); speed-up round %.2fx, draw %.2fx\n",
            label, NREP, mx, mdx, mf, mdf, mxd, mdxd, md, mdd, mp, m1, m2, md - m1, m1 - m2, mx / mf, mxd / md)
    for (name, c) in (("XLA round", cx), ("fused round", cf))
        m = memstats(c)
        @printf("%s Float32 %-12s temporaries %.1f MB, arguments %.1f MB, peak %.1f MB\n", label, name, MB(m.temp_size_in_bytes), MB(m.argument_size_in_bytes), MB(m.peak_memory_in_bytes))
    end
    GC.gc()
end

# ---- soc
let tab = pls.dev[:soc], tabf = plfs.dev[:soc]
    B = size(tab.on, 1)
    U = reshape(weyl(C * B * 3, 0), C, B, 3); Uf = Float32.(U)
    tb, _ = D.build(fs, pls.st, pls.cp, pls.sp, tab, 1, pls.k)
    tj, _ = D.line_draw(fs, tb, pls.k, U[:, :, 1:1], U[:, :, 2:2])
    tref = tj[:, :, 1]
    tbf, _ = F.build(fs, plfs.st, plfs.cp, plfs.sp, tabf, 1, plfs.k)
    tj32, _ = F.line_draw(fs, tbf, plfs.k, Uf[:, :, 1:1], Uf[:, :, 2:2])
    tf32, _ = sample_soc_fused(tbf, plfs.k, Uf[:, :, 1:1], Uf[:, :, 2:2]; K=F.nroots(fs))
    @printf("soc Float32 arrays (%d sites): XLA form vs FP64 %.2e (>1e-4: %d, >1e-2: %d); kernel reference vs FP64 %.2e (>1e-4: %d, >1e-2: %d); kernel vs XLA form %.2e\n", C * B,
            stats(Float64.(tj32[:, :, 1]), tref)..., stats(Float64.(tf32[:, :, 1]), tref)..., stats(Float64.(tf32[:, :, 1]), Float64.(tj32[:, :, 1]))[1])
    args = (R_(plfs.st), R_(plfs.cp), R_(plfs.sp), R_(plfs.k), R_(tabf), R_(Uf))
    fx(st, cp, sp, k, tab, u) = F.do_round(fs, st, cp, sp, k, tab, 1, u)
    ff(st, cp, sp, k, tab, u) = do_round_soc(st, cp, sp, k, tab, 1, u)
    function fdraw(st, cp, sp, k, tab, u)
        tb, _ = F.build(fs, st, cp, sp, tab, 1, k)
        return sample_soc_fused(tb, k, u[:, :, 1:1], u[:, :, 2:2]; K=F.nroots(fs))
    end
    function fdrawx(st, cp, sp, k, tab, u)
        tb, _ = F.build(fs, st, cp, sp, tab, 1, k)
        return F.line_draw(fs, tb, k, u[:, :, 1:1], u[:, :, 2:2])
    end
    cx = tcomp("soc XLA round", fx, args); cf = tcomp("soc fused round", ff, args); cd_ = tcomp("soc fused draw", fdraw, args); cdx = tcomp("soc XLA draw", fdrawx, args)
    od, odx = cd_(args...), cdx(args...)
    tk, tx = d64(od[1]), d64(odx[1])
    @printf("soc Float32 GPU draws: kernel vs XLA %.2e (>1e-4: %d, >1e-2: %d); kernel vs FP64 %.2e (>1e-4: %d, >1e-2: %d); XLA vs FP64 %.2e (>1e-4: %d, >1e-2: %d); guards %d / %d, flips %d\n",
            stats(tk, tx)..., stats(tk, tref)..., stats(tx, tref)..., count(Array(od[2])), count(Array(odx[2])), count(Array(od[2]) .!= Array(odx[2])))
    ox, of = cx(args...), cf(args...)
    @printf("soc Float32 GPU rounds: t kernel vs XLA %.2e (>1e-4: %d, >1e-2: %d); state: %s\n", stats(d64(of[2]), d64(ox[2]))..., state_line(ox, of))
    mx, mdx = timeit(cx, args); mf, mdf = timeit(cf, args); md, mdd = timeit(cd_, args); mxd, mdxd = timeit(cdx, args)
    @printf("soc Float32 timings (min/median of %d), %d sites: XLA round %.5f / %.5f, fused round %.5f / %.5f; XLA draw %.5f / %.5f, fused draw %.5f / %.5f; speed-up round %.2fx, draw %.2fx\n",
            NREP, C * B, mx, mdx, mf, mdf, mxd, mdxd, md, mdd, mx / mf, mxd / md)
    for (name, c) in (("XLA round", cx), ("fused round", cf))
        m = memstats(c)
        @printf("soc Float32 %-12s temporaries %.1f MB, arguments %.1f MB, peak %.1f MB\n", name, MB(m.temp_size_in_bytes), MB(m.argument_size_in_bytes), MB(m.peak_memory_in_bytes))
    end
end
