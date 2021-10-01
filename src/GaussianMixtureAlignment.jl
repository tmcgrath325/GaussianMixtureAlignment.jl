module GaussianMixtureAlignment

using StaticArrays
using LinearAlgebra
using DataStructures
using Rotations
using CoordinateTransformations
using Optim

export AbstractGMM, AbstractSingleGMM, AbstractMultiGMM
export IsotropicGaussian, IsotropicGMM, IsotropicMultiGMM
export rotmat
export overlap, distance, tanimoto
export local_align
export gogma_align, rot_gogma_align, trl_gogma_align
export tivgmm, tiv_gogma_align

include("gmm.jl")
include("transformation.jl")
include("overlap.jl")
include("bounds.jl")
include("block.jl")
include("localalign.jl")
include("branchbound.jl")
include("tiv.jl")
include("combine.jl")

end
