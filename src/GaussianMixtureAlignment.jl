"""
GaussianMixtureAlignment.jl
===========================

GaussianMixtureAlignment.jl is a package used to align Gaussian mixture models. In particular, it uses an implementation 
of the [GOGMA algorithm (Campbell, 2016)](https://arxiv.org/abs/1603.00150) to find globally optimal alignments of mixtures of 
isotropic (spherical) Gaussian distributions.

REPL help
=========

? followed by an algorith or constructor name will print help to the terminal. See: \n
    \t?IsotropicGaussian \n
    \t?IsotropicGMM \n
    \t?IsotropicMultiGMM \n
    \t?gogma_align \n
    \t?tiv_gogma_align \n
    \t?rocs_align \n
"""
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
