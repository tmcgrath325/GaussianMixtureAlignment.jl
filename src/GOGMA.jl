module GOGMA

using StaticArrays
using LinearAlgebra
using DataStructures
using Optim

export IsotropicGaussian, IsotropicGMM
export get_bounds, rot
export subranges, Block
export local_align
export branch_bound

include("gmm.jl")
include("bounds.jl")
include("block.jl")
include("localalign.jl")
include("branchbound.jl")

end
