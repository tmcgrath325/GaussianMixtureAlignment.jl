"""
GaussianMixtureAlignment.jl
===========================

GaussianMixtureAlignment.jl is a package used to align Gaussian mixture models. In particular, it uses an implementation
of the [GOGMA algorithm (Campbell, 2016)](https://arxiv.org/abs/1603.00150) to find globally optimal alignments of mixtures of
isotropic (spherical) Gaussian distributions.

REPL help
=========

? followed by an algorithm or constructor name will print help to the terminal. See: \n
    \t?IsotropicGaussian \n
    \t?IsotropicGMM \n
    \t?IsotropicMultiGMM \n
    \t?gogma_align \n
    \t?tiv_gogma_align \n
    \t?rocs_align \n
"""
module GaussianMixtureAlignment

abstract type AbstractModel{N,T} end

using ADTypes: ADTypes, AutoForwardDiff
using CoordinateTransformations: CoordinateTransformations, AbstractAffineMap, AffineMap,
                                 LinearMap, Translation, kabsch
using CoordinateTransformations: kabsch_centered
using Distances: Distances, SqEuclidean, colwise
using GenericLinearAlgebra: GenericLinearAlgebra
using Hungarian: Hungarian, hungarian
using LinearAlgebra: LinearAlgebra, det, dot, eigvecs, norm, svd, tr
using MutableConvexHulls: MutableConvexHulls, CCW, ChanLowerConvexHull,
                          addpoint!, mergepoints!, removepoint!
using NearestNeighbors: NearestNeighbors, Euclidean, KDTree, nn
using Optim: Optim, Fminbox, LBFGS, optimize
using PairedLinkedLists: PairedLinkedLists, ListNodeIterator, deletenode!, getnode
using Rotations: Rotations, AngleAxis, RotationVec
using StaticArrays: StaticArrays, SMatrix, SVector

export AbstractGaussian, AbstractGMM
export IsotropicGaussian, IsotropicGMM, IsotropicMultiGMM
export overlap, force!, gogma_align, rot_gogma_align, trl_gogma_align, tiv_gogma_align
export rocs_align
export PointSet, MultiPointSet
export kabsch, icp, iterative_hungarian, goicp_align, goih_align, tiv_goicp_align, tiv_goih_align

"Visualize an `AbstractIsotropicGaussian` as a sphere. Requires a Makie backend; load one (e.g. `using GLMakie`) to see the full docstring."
function gaussiandisplay end
"Mutating form of `gaussiandisplay`. Requires a Makie backend."
function gaussiandisplay! end
"Visualize an `AbstractIsotropicGMM` as a collection of spheres. Requires a Makie backend; load one (e.g. `using GLMakie`) to see the full docstring."
function gmmdisplay end
"Mutating form of `gmmdisplay`. Requires a Makie backend."
function gmmdisplay! end
"Visualize an `AbstractIsotropicMultiGMM`, coloring each labeled sub-GMM differently. Requires a Makie backend; load one (e.g. `using GLMakie`) to see the full docstring."
function multigmmdisplay end
"Mutating form of `multigmmdisplay`. Requires a Makie backend."
function multigmmdisplay! end
export gmmdisplay, gmmdisplay!, multigmmdisplay, multigmmdisplay!, gaussiandisplay, gaussiandisplay!

include("tforms.jl")

include("goicp/pointset.jl")
include("gogma/gmm.jl")

include("utils.jl")
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

end
