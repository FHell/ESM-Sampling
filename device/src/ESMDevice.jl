# Device model and exact cost (docs/reactant_design.md, phase 1).
#
# Array-first code over a chain dimension: the same functions run on Julia Arrays
# (reference) and traced by Reactant (XLA on CPU or GPU). Scope as in the design §1:
# the complete level, storage = soc, renewables = load, regularized constraints. All
# model-completion terms are included (line losses and soft limits as hinge sums,
# lossy links, merged vintages, fleet unit commitment, CO₂ price through mc).
#
# Layout: Julia column-major [c, τ, i] (chain fastest). Gathers and scatters are
# dense contractions with static incidence matrices (exact, traceable; to be replaced
# by gathers/scatters for full Europe, design §4.5).
module ESMDevice

using ESMSampling
using LinearAlgebra
using Printf
using Random
using Reactant
using Reactant
using Reactant: @trace
const ML = ESMSampling

export DeviceModel, DeviceIndex, DeviceState, pack, to_device, to_host!, rebuild_r, cost_parts

"Float type of the device program of this module (ESMDevice: Float64; ESMDevice.F32: Float32)."
const FT = Float64
"Device inputs in the module's float type (identity in the Float64 build)."
ftconv(x) = x
"Float type of the host-side bookkeeping of the drivers (Float64 in both builds: the name is not mapped by `ft32`)."
const HostFloat = Float64
"""
    ftcompile(f)

Run `f` (a function that compiles device programs, `() -> @compile …`) under the compile settings
of the module's float type: identity in the Float64 build; in ESMDevice.F32 with
`dot_general_precision = HIGHEST`, so that the one-hot contractions (`tmul`, `tmulT`) run in full
Float32 and not in TF32 on GPUs.
"""
ftcompile(f) = f()

include("core.jl")

"""
    ft32(ex)

Expression map of the Float32 build (`include(ft32, file)`): Float64 literals become Float32
literals, the symbols `Float64`, `Inf`, `NaN` become `Float32`, `Inf32`, `NaN32`, `zeros`/`ones` without an element
type make Float32 arrays, and nested `include(file)` calls include with the same map.
"""
function ft32(ex)
    ex isa Float64 && return Float32(ex)
    ex === :Float64 && return :Float32
    ex === :Inf && return :Inf32
    ex === :NaN && return :NaN32
    if ex isa Expr
        if ex.head === :call && ex.args[1] === :include && length(ex.args) == 2
            return Expr(:call, :include, :ft32, ex.args[2])
        end
        if ex.head === :call && ex.args[1] in (:zeros, :ones)      # default element type: Float32
            return Expr(:call, ex.args[1] === :zeros ? :ftzeros : :ftones, map(ft32, ex.args[2:end])...)
        end
        return Expr(ex.head, map(ft32, ex.args)...)
    end
    return ex
end

"""
The whole device program in Float32 (state, caches, terms, line sampler, capacity pass, uniforms):
the same sources as ESMDevice with every literal mapped (`ft32`) and every device input converted
(`ftconv`). `ESMDevice.F32.run_device(cfg)` etc.; FP64 (ESMDevice) is the reference.
"""
module F32
using ESMSampling
using LinearAlgebra
using Printf
using Random
using Reactant
using Reactant: @trace
import ..ESMDevice: ft32, HostFloat
const ML = ESMSampling
const FT = Float32
"Float32 copy of a device input: Float64 numbers and arrays converted, structs of this module rebuilt."
ftzeros(T::Type, d...) = zeros(T, d...)
ftzeros(d...) = zeros(Float32, d...)
ftones(T::Type, d...) = ones(T, d...)
ftones(d...) = ones(Float32, d...)
ftconv(x::Float64) = Float32(x)
"Compiles under `dot_general_precision = HIGHEST` (no TF32 for the Float32 contractions; ESMDevice.ftcompile)."
ftcompile(f) = Reactant.with_config(f; dot_general_precision=Reactant.PrecisionConfig.HIGHEST)
ftconv(x::AbstractArray{Float64}) = Float32.(x)
ftconv(x::Union{Tuple,NamedTuple}) = map(ftconv, x)
ftconv(x::AbstractArray{<:Union{Tuple,NamedTuple}}) = map(ftconv, x)
function ftconv(x::T) where {T}
    (isstructtype(T) && parentmodule(T) === (@__MODULE__) && fieldcount(T) > 0) || return x
    return T.name.wrapper((ftconv(getfield(x, f)) for f in fieldnames(T))...)
end
include(ft32, "core.jl")
end

end
