"""
Near-optimal sampling of energy system decisions under weather uncertainty.

The CPU reference of the sampler (docs/system/): weather, capacities and dispatch
of a PyPSA-Eur network sampled jointly from a two-temperature ensemble with a
learned bias, on a graph-Haar hierarchy in space and block coarsening in time.
"""
module ESMSampling

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Printf
using TOML
using HDF5
using JuMP
using HiGHS
using Serialization
using SpecialFunctions: erfc, erfcx, erfcinv

include("spatial.jl")
include("layout.jl")
include("system.jl")
include("flowbasis.jl")
include("energy.jl")
include("lp.jl")
include("weather.jl")
include("bias.jl")
include("hierarchy.jl")
include("bias_online.jl")
include("bias_post.jl")
include("level0.jl")
include("local.jl")
include("weather_moves.jl")
include("weather_fine.jl")
include("weather_blocks.jl")
include("flowmoves.jl")
include("adaptive.jl")
include("kop.jl")
include("sampler.jl")
include("diagnostics.jl")
include("convergence.jl")
include("run.jl")
include("chains.jl")

export Config, load_config, run, setup

end
