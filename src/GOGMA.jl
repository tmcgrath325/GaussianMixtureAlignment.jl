module GOGMA

using StaticArrays
using LinearAlgebra
using DataStructures

export IsotropicGaussian, IsotropicGMM
export get_bounds, rot
export subranges, Block, branch_bound

include("gmm.jl")
include("bounds.jl")
include("branchbound.jl")

end
