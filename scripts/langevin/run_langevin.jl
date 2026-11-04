# Record of the Langevin capacity and weather-refresh measurements on chains40 (run
# interactively in one session with these calls; outputs in results/langevin/, numbers in
# results/langevin/summary.txt). From the repository root:
#   nice -n 19 julia --project=. -t 4 scripts/langevin/run_langevin.jl
include(joinpath(@__DIR__, "setup40.jl"))
include(joinpath(@__DIR__, "refresh.jl"))
mkpath("results/langevin")
κs = [500.0, 1000.0, 200.0, 300.0, 100.0, 100.0]          # onwind, solar, offwind, gas, battery, H2 (MW)

# starts ×0.5 and ×2 of the heuristic on the set-up weather, 30 operation sweeps at fixed k
warm = Dict{String,Any}()
rngw = MersenneTwister(77)
for (lab, kind, seed) in (("low", "low", 1), ("high", "high", 2))
    s = ML.evaluate!(P, ML.chain_start(MersenneTwister(seed), P, sL, kind, false))
    for _ in 1:30
        s = lgv_op!(rngw, P, S.loc, LM, s)
    end
    warm[lab] = s
end
# A: ηw = 1e-3, one operation sweep per step, scores of the current state
A = Dict(lab => langevin_run!(MersenneTwister(seed), P, S, warm[lab], 600; ηw=fill(1e-3, 6), κs, inner=1,
                              log_every=50, label=lab, clip=0.05) for (lab, seed) in (("low", 21), ("high", 22)))
# B: ηw = 2.5e-4, three sweeps per step, scores averaged over them, reflection at the box
B = Dict(lab => langevin_run!(MersenneTwister(seed), P, S, A[lab][2], 250; ηw=fill(2.5e-4, 6), κs, inner=3,
                              log_every=25, label=lab * "-B", clip=0.02, box=true, avg=true, nodes=true)
         for (lab, seed) in (("low", 31), ("high", 32)))
# C: larger steps for gas (2e-3) and storage (5e-4)
C = Dict(lab => langevin_run!(MersenneTwister(seed), P, S, B[lab][2], 250; ηw=[2.5e-4, 2.5e-4, 2.5e-4, 2e-3, 5e-4, 5e-4], κs,
                              inner=3, log_every=25, label=lab * "-C", clip=0.02, box=true, avg=true, nodes=true, it0=250)
         for (lab, seed) in (("low", 51), ("high", 52)))
for (n, D) in (("A", A), ("B", B), ("C", C)), lab in ("low", "high")
    serialize("results/langevin/lgv_$(lab)_$n.jls", (; trace=D[lab][1], state=D[lab][2]))
end

# weather refresh from the C low end state after 40 operation sweeps at fixed (k, x)
s0 = copy(C["low"][2])
rngr = MersenneTwister(61)
for _ in 1:40
    global s0 = lgv_op!(rngr, P, S.loc, LM, s0)
end
ref = Dict{String,Any}()
ref["control"] = refresh_trace(MersenneTwister(70), P, S, LM, s0, 120; newweather=false)[1]
for d in 1:3
    ref["draw$d"] = refresh_trace(MersenneTwister(70 + d), P, S, LM, s0, 120; newweather=true, wrng=MersenneTwister(900 + d))[1]
    ref["comp$d"] = refresh_trace(MersenneTwister(70 + d), P, S, LM, refresh_weather(P, LM, s0, MersenneTwister(900 + d)), 120;
                                  newweather=false)[1]
end
ref["comp1_long"] = refresh_trace(MersenneTwister(71), P, S, LM, refresh_weather(P, LM, s0, MersenneTwister(901)), 320;
                                  newweather=false)[1]
serialize("results/langevin/refresh_daily.jls", ref)
