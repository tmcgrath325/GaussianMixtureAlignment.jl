module GOGMA

using StaticArrays
using LinearAlgebra
using DataStructures
# using Optim
using CUDA

export IsotropicGaussian, IsotropicGMM
export get_bounds
export subranges, Block
export branch_bound
export get_bounds_gpu

include("gmm.jl")
include("bounds.jl")
include("block.jl")
include("branchbound.jl")
include("gpubounds.jl")

end
