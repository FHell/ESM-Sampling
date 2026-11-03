"""
Near-optimal sampling of energy system decisions under weather uncertainty.

Implements near_optimal_sampling_setup.md on a PyPSA-Eur network with a
multilevel hierarchy of Laplace modes (space) and block coarsening (time).
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
include("constraints.jl")
include("lp.jl")
include("weather.jl")
include("hmc.jl")
include("bias.jl")
include("hierarchy.jl")
include("bias_online.jl")
include("bias_post.jl")
include("mlda.jl")
include("gibbs.jl")
include("pairs.jl")
include("local.jl")
include("weather_moves.jl")
include("weather_fine.jl")
include("flowmoves.jl")
include("timehier.jl")
include("adaptive.jl")
include("decompose.jl")
include("diagnostics.jl")
include("convergence.jl")
include("run.jl")
include("chains.jl")

export Config, load_config, run, setup

end
