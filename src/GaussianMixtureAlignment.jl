module GaussianMixtureAlignment

using StaticArrays
using LinearAlgebra
using DataStructures
using Rotations
using CoordinateTransformations
using Optim

export AbstractGaussian, AbstractGMM
export IsotropicGaussian, IsotropicGMM, IsotropicMultiGMM
export rotmat
export overlap, distance, tanimoto
export local_align
export gogma_align, rot_gogma_align, trl_gogma_align
export tivgmm, tiv_gogma_align
export rocs_align
export rocs_gogma_align

include("gmm.jl")
include("transformation.jl")
include("overlap.jl")
include("bounds.jl")
include("block.jl")
include("localalign.jl")
include("branchbound.jl")
include("tiv.jl")
include("combine.jl")
include("rocsalign.jl")
include("rocsgogma.jl")

using Requires

function __init__()
    @require PlotlyJS="f0f68f2c-4968-5e81-91da-67840de0976a" include("draw.jl")
end

end
