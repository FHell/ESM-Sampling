# Cost of the joint z0 block moves at a 6-hourly finest level (complete40.toml, α = 0): a state
# after a short relaxation from the heuristic start, then the cost per block move of each length,
# and the fine step with and without the block moves. Usage:
#   nice -n 19 julia --project=. -t 4 scripts/mixing/wblock_cost6h.jl [config] [nrelax] [lengths...]
include(joinpath(@__DIR__, "common.jl"))
path = length(ARGS) >= 1 ? ARGS[1] : "configs/complete40.toml"
nrelax = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 15
lengths = length(ARGS) >= 3 ? Tuple(parse.(Int, ARGS[3:end])) : (16, 64, 365)
cfg, P, sL, info = mix_setup(path; use_bias=false, fine_weather_blocks=0.0)
L = ML.nlevels(P) - 1
W = P.W
println("finest level ", P.levels[end], ", T = ", P.lv[L+1].T, ", kernel ", cfg.fine_kernel, ", fine_weather ", cfg.fine_weather); flush(stdout)
S = ML.new_sampler(P, sL, cfg)
rng = MersenneTwister(1)
s = copy(sL)
for it in 1:nrelax
    a = time()
    global s = ML.fine_local_step(rng, P, s, S)
    @printf("relax %2d  %.1f s  L %.6g\n", it, time() - a, s.L); flush(stdout)
end
LM = ML.local_model(S.loc, P, L)
# fine steps without block moves (the current kernel), and the weather sweep alone
tf = Float64[]
for it in 1:4
    a = time(); global s = ML.fine_local_step(rng, P, s, S); push!(tf, time() - a)
end
tw = Float64[]
pools = Any[]
x = deepcopy(s)
x = ML.fine_weather_sweep!(rng, P, LM, x; frac=cfg.fine_weather, frac0=cfg.fine_weather, pools=pools)[1]
for it in 1:3
    a = time(); global x = ML.fine_weather_sweep!(rng, P, LM, x; frac=cfg.fine_weather, frac0=cfg.fine_weather, pools=pools)[1]; push!(tw, time() - a)
end
@printf("fine step %.1f s (mean of %d), weather sweep %.2f s\n", mean(tf), length(tf), mean(tw)); flush(stdout)
# block moves of each length at the relaxed state
V = ML.z0_block_directions(W)
T0 = W.lays[1].T
cb = Float64[]
for (B, nm) in zip(lengths, map(B -> B <= 16 ? 12 : (B <= 64 ? 6 : 3), lengths))
    x1 = deepcopy(s); LC = ML.LocalCost(P, LM, x1)
    bm = ML.BlockMove(ML.BlockPool(P, ML.z0_block_span(W, L, B)).wss, 0, Int[], P, LM, LC, Float64[], Float64[])
    jac, sp = NTuple{5,Float64}[], Float64[]
    ML.z0_block_move!(MersenneTwister(2), P, LM, LC, x1, bm, jac, sp, 1, min(4, T0), V[:, 1])     # compile
    ts = Float64[]
    r = MersenneTwister(3 + B)
    a = time()
    for _ in 1:nm
        st = rand(r, 1:T0)
        push!(ts, ML.z0_block_move!(r, P, LM, LC, x1, bm, jac, sp, st, min(st + B - 1, T0), V[:, rand(r, 1:size(V, 2))]))
    end
    push!(cb, (time() - a) / nm)
    @printf("B = %3d: %.2f s per move (window %d fine steps), accepted %d/%d\n", B, cb[end], ML.z0_block_span(W, L, B),
            count(!=(0.0), ts), nm); flush(stdout)
end
pB = [1 / B for B in lengths]; pB ./= sum(pB)
cmix = sum(pB .* cb)
@printf("mixture (p ∝ 1/B over %s): %.2f s per move = %.1f %% of a fine step (%.1f s)\n", string(lengths), cmix, 100 * cmix / mean(tf), mean(tf))
println("done")
