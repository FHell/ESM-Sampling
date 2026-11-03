# Compare two outputs of `bench_perf.jl regress` (bitwise / relative differences of v).
using Serialization
a, b = deserialize(ARGS[1]), deserialize(ARGS[2])
for (j, (x, y)) in enumerate(zip(a, b))
    d = maximum(abs.(x .- y)); sc = maximum(abs.(x))
    println("stage $j: identical = $(x == y), max |Δv| = $d (max |v| = $sc), differing coords = $(count(x .!= y)) / $(length(x))")
end
