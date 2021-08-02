module GOGMA

using StaticArrays
using LinearAlgebra
using DataStructures
using Optim

export IsotropicGaussian, IsotropicGMM, MultiGMM
export get_bounds, rot
export subranges, fullBlock, rotBlock, trlBlock
export local_align
export gogma_align, rot_gogma_align, trl_gogma_align
export tivgmm, tiv_gogma_align

include("gmm.jl")
include("bounds.jl")
include("block.jl")
include("localalign.jl")
include("branchbound.jl")
include("tiv.jl")
include("combine.jl")

end
