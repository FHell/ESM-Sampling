# test/test_exact.jl with the pre-fix kpeak slice width (w·k0 in δ), for comparison with the fixed
# version: the old method is re-evaluated into the module from the current source with the two
# lines swapped back, then the test file is included.
import ESMSampling
src = read(joinpath(@__DIR__, "..", "..", "src", "adaptive.jl"), String)
i = findfirst("function kpeak_sweep!(", src)
j = findnext("\nend\n", src, last(i))
f = src[first(i):last(j)]
new = "            lfu(u) = lf(k0 * expm1(u)) + u\n            δ = k0 * expm1(slice_scalar(rng, lfu, 0.0, w, log(κmin / k0)))"
@assert occursin(new, f)
f = replace(f, new => "            δ = slice_scalar(rng, lf, 0.0, w * k0, κmin - k0)")
Core.eval(ESMSampling, Meta.parse("begin\n" * f * "\nend"))
println("kpeak_sweep! replaced by the pre-fix version")
include(joinpath(@__DIR__, "..", "..", "test", "test_exact.jl"))
