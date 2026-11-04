# Durable files of the CPU driver (docs/system/hpc.md §4, §7): atomic cache writes and the
# read-only mirror, the state snapshots with stop and resume (exit code 75), and SIGTERM
# ignored for the job wrapper. Deterministic checks only.
using Test, Random, HDF5
using ESMSampling
const MH = ESMSampling

@testset "atomic cache writes and the mirror" begin
    d = mktempdir()
    cache, mirror = joinpath(d, "cache"), joinpath(d, "mirror")
    mkpath(cache); mkpath(mirror)
    # an interrupted write leaves neither the entry nor a temporary file
    p = joinpath(cache, "lp_dead.h5")
    @test_throws ErrorException MH.atomic_write(p) do tmp
        h5open(tmp, "w") do f
            f["x"] = rand(10)
        end
        error("writer killed")
    end
    @test !isfile(p)
    @test isempty(readdir(cache))
    # a temporary file left by a killed writer is never taken for the entry
    write(MH.tmp_name(p), "half-written")
    @test MH.cache_lookup(p, mirror) === nothing
    @test MH.cache_read(f -> h5open(read, f), p, mirror) === nothing
    foreach(f -> rm(joinpath(cache, f)), readdir(cache))
    # a complete write appears under its name only
    x = randn(MersenneTwister(1), 50)
    MH.atomic_write(f -> h5open(g -> (g["x"] = x), f, "w"), p)
    @test readdir(cache) == ["lp_dead.h5"]
    @test h5open(f -> read(f["x"]), p) == x
    # an entry missing in the cache but present in the mirror is copied in, byte for byte
    q = joinpath(mirror, "weather_abc.jls")
    MH.atomic_open(io -> MH.serialize(io, (x, "model")), q)
    r = joinpath(cache, "weather_abc.jls")
    @test MH.cache_read(MH.deserialize, r, mirror) == (x, "model")
    @test isfile(r) && read(r) == read(q)
    @test sort(readdir(cache)) == ["lp_dead.h5", "weather_abc.jls"]
    @test readdir(mirror) == ["weather_abc.jls"]               # the mirror is only read
    # an unreadable entry is recomputed, not used
    write(joinpath(cache, "lp_bad.h5"), "not hdf5")
    @test (@test_logs (:warn,) match_mode=:any MH.cache_read(f -> h5open(read, f), joinpath(cache, "lp_bad.h5"))) === nothing
    # the environment overrides the keys
    cfg = Config(); cfg.out = joinpath(d, "run", "out.h5")
    @test MH.cache_dir(cfg) == joinpath(d, "run", "cache")
    cfg.cache_dir = cache
    withenv("ESM_CACHE_DIR" => nothing, "ESM_CACHE_MIRROR" => nothing) do
        @test MH.cache_dir(cfg) == cache && MH.cache_mirror(cfg) == ""
    end
    withenv("ESM_CACHE_DIR" => joinpath(d, "env"), "ESM_CACHE_MIRROR" => mirror) do
        @test MH.cache_dir(cfg) == joinpath(d, "env") && MH.cache_mirror(cfg) == mirror
    end
    withenv("ESM_CACHE_DIR" => "") do
        @test MH.cache_dir(cfg) == cache
    end
end

@testset "SIGTERM ignored, exit code passed on" begin
    proj = dirname(@__DIR__)
    code = "using ESMSampling; ESMSampling.ignore_sigterm(); println(\"ready\"); flush(stdout); sleep(4); exit(75)"
    p = open(`$(Base.julia_cmd()) --startup-file=no -t 1 --project=$proj -e $code`; read=true)
    @test readline(p) == "ready"
    kill(p, Base.SIGTERM)
    wait(p)
    @test p.exitcode == 75 && p.termsignal == 0
end

if isfile(joinpath(@__DIR__, "..", "data", "processed", "model.h5"))
    @testset "snapshots: stop in every phase, resume from the saved states" begin
        d = mktempdir()
        function hpccfg(cache; mirror="")
            cfg = load_config(joinpath(@__DIR__, "..", "configs", "haar_poc.toml"))
            cfg.data = joinpath(@__DIR__, "..", "data", "processed", "model.h5")
            cfg.out = joinpath(d, "run.h5")
            cfg.levels = [(10, 24), (40, 24)]
            cfg.use_bias, cfg.init = true, "heuristic"
            cfg.nchains, cfg.warmup0, cfg.mlda_warmup, cfg.nsamples = 2, 4, 3, 3
            cfg.bias_start, cfg.bias_every, cfg.bias_samples, cfg.bias_sample_steps = 2, 2, 4, 256
            cfg.descent_iters = 20
            cfg.cache_dir, cfg.cache_mirror = cache, mirror
            cfg.checkpoint, cfg.stop_file = joinpath(d, "snapshot.h5"), joinpath(d, "STOP")
            cfg.checkpoint_minutes = 1e6                    # only the forced and the stop snapshots
            cfg.verbose = false
            return cfg
        end
        stop_at(ph, n) = cfg -> ((p, it) -> (p == ph && it == n && touch(cfg.stop_file); nothing))
        same(a, b) = a.ℓ == b.ℓ && a.m == b.m && a.z0 == b.z0 && a.ws == b.ws && a.k == b.k && a.v == b.v
        snapinfo() = h5open(f -> (read(f["phase"]), read(f["it"]), read(f["restarts"])), joinpath(d, "snapshot.h5"))
        cA, cB = joinpath(d, "cacheA"), joinpath(d, "cacheB")

        # level-0 warmup (cache A computed, no mirror)
        cfg = hpccfg(cA)
        A = withenv(() -> MH.run_chains(cfg; hook=stop_at(:level0, 2)(cfg)), "ESM_CACHE_DIR" => nothing, "ESM_CACHE_MIRROR" => nothing)
        @test A.exitcode == MH.EXIT_CHECKPOINTED && A.status == :stopped
        @test snapinfo() == ("level0", 2, 0)
        @test !isfile(cfg.stop_file)                        # consumed
        @test A.resume_states === nothing && length(A.states) == 1 && A.states[1].ℓ == 0

        # fine warmup (cache B empty, filled from the mirror A)
        cfg = hpccfg(cB; mirror=cA)
        B = withenv(() -> MH.run_chains(cfg; hook=stop_at(:fine, 2)(cfg)), "ESM_CACHE_DIR" => nothing, "ESM_CACHE_MIRROR" => nothing)
        @test same(B.resume_states[1], A.states[1])
        @test B.exitcode == 75 && snapinfo() == ("fine", 2, 1)
        @test length(B.states) == 2 && all(s -> s.ℓ == 1, B.states)
        @test !isempty(readdir(cA)) && sort(readdir(cA)) == sort(readdir(cB))
        @test all(f -> read(joinpath(cA, f)) == read(joinpath(cB, f)), readdir(cA))
        @test !any(f -> occursin(".tmp-", f), vcat(readdir(cA), readdir(cB), readdir(d)))

        # production
        cfg = hpccfg(cB)
        C = withenv(() -> MH.run_chains(cfg; hook=stop_at(:fine, 4)(cfg)), "ESM_CACHE_DIR" => nothing, "ESM_CACHE_MIRROR" => nothing)
        @test all(same.(C.resume_states, B.states))
        @test C.exitcode == 75 && snapinfo() == ("fine", 4, 2)

        Qsnap = h5open(f -> [read(f["chains/$c/Q"]) for c in 1:2], joinpath(d, "snapshot.h5"))
        @test all(Q -> size(Q, 2) == 4, Qsnap)

        # to the end
        cfg = hpccfg(cB)
        D = withenv(() -> MH.run_chains(cfg), "ESM_CACHE_DIR" => nothing, "ESM_CACHE_MIRROR" => nothing)
        @test all(same.(D.resume_states, C.states))
        @test D.exitcode == 0 && D.status == :finished && D.restarts == 3
        h5open(cfg.out, "r") do f
            @test all(c -> size(read(f["chains/$c/Q"]), 2) == 6, 1:2)
            @test all(c -> read(f["chains/$c/Q"])[:, 1:4] == Qsnap[c], 1:2)   # the traces before the stop are kept
            @test read(f["run/restarts"]) == 3 && read(f["run/resumed_at_sweep"]) == 4
        end
    end
end
