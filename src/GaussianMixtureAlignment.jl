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

abstract type AbstractModel{N,T} end

using StaticArrays
using LinearAlgebra
using GenericLinearAlgebra
using PairedLinkedLists
using MutableConvexHulls
using Rotations
using CoordinateTransformations
using Distances
using NearestNeighbors
using Hungarian
using Optim

export AbstractGaussian, AbstractGMM
export IsotropicGaussian, IsotropicGMM, IsotropicMultiGMM
export overlap, gogma_align, rot_gogma_align, trl_gogma_align, tiv_gogma_align
export rocs_align
export PointSet, MultiPointSet
export kabsch, icp, iterative_hungarian, goicp_align, goih_align, tiv_goicp_align, tiv_goih_align

include("tforms.jl")

include("goicp/pointset.jl")
include("gogma/gmm.jl")

include("uncertaintyregion.jl")
include("distancebounds.jl")

include("gogma/combine.jl")
include("gogma/transformation.jl")
include("gogma/overlap.jl")
include("gogma/bounds.jl")

include("goicp/bounds.jl")
include("goicp/correspondence.jl")
include("goicp/kabsch.jl")
include("goicp/icp.jl")
include("goicp/rmsd.jl")
include("goicp/local.jl")

include("localalign.jl")
include("branchbound.jl")

include("gogma/tiv.jl")
include("gogma/align.jl")
include("goicp/tiv.jl")
include("goicp/align.jl")
include("rocs/rocsalign.jl")

using Requires

function __init__()
    @require PlotlyJS="f0f68f2c-4968-5e81-91da-67840de0976a" include("draw.jl")
end

end
