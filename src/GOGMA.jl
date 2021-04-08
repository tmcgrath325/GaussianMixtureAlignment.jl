module GOGMA

using StaticArrays
using LinearAlgebra
using DataStructures
using Optim
using PlotlyJS

export IsotropicGaussian, IsotropicGMM
export get_bounds, rot
export subranges, fullBlock, rotBlock, trlBlock
export local_align
export branch_bound, rot_branch_bound, trl_branch_bound
export tivgmm, tiv_branch_bound
export draw3d

include("gmm.jl")
include("bounds.jl")
include("block.jl")
include("localalign.jl")
include("branchbound.jl")
include("tiv.jl")
include("draw.jl")

end
