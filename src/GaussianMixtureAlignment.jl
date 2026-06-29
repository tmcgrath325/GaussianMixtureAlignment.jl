"""
    GaussianMixtureAlignment

A Julia package for globally-optimal rigid alignment of point sets via Gaussian mixture
models (GMMs). Implements the [GOGMA algorithm (Campbell, 2016)](https://arxiv.org/abs/1603.00150)
and related methods.

Main entry points: [`gogma_align`](@ref), [`tiv_gogma_align`](@ref), [`goicp_align`](@ref).
"""
module GaussianMixtureAlignment

abstract type AbstractModel{N, T} end

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
using PairedLinkedLists: PairedLinkedLists, ListNodeIterator, deletenode!, getnode, target
using Rotations: Rotations, AngleAxis, RotationVec
using StaticArrays: StaticArrays, SMatrix, SVector

export AbstractGaussian, AbstractGMM
export IsotropicGaussian, IsotropicGMM, IsotropicMultiGMM
export overlap, force!, force, gogma_align, rot_gogma_align, trl_gogma_align, tiv_gogma_align
export rocs_align
export PointSet, MultiPointSet
export kabsch, goicp_align, goih_align, tiv_goicp_align, tiv_goih_align

# Semi-public interface: callable and supported, but not brought into the caller's namespace by
# `using`. `branchbound` and the search-region types are the low-level entry points beneath the
# `*_align` functions; `icp` and `iterative_hungarian` are correspondence primitives returning
# `Vector{Tuple{Int,Int}}` rather than an `AlignmentResults`, so they sit below the `*_align`
# surface; the rest are the `AlignmentResults` read interface. `tform` and `converged` are
# deliberately not exported to avoid colliding with `Optim.converged` and with the
# CoordinateTransformations idiom of applying an affine map as `tform(x)`.
# `public` is a parse error before Julia 1.11, so build the expression directly rather than
# writing the keyword.
@static if VERSION >= v"1.11"
    eval(
        Expr(
            :public,
            :branchbound, :UncertaintyRegion, :RotationRegion, :TranslationRegion,
            :icp, :iterative_hungarian,
            :converged, :tform, :upperbound, :lowerbound, :obj_calls,
            :num_splits, :num_blocks, :stagnant_splits, :progress
        )
    )
end

"""
    gaussiandisplay([fig_or_ax,] g; display=:wire, color=..., label="", alpha=1, transparency=false)

Visualize an `AbstractIsotropicGaussian` as a sphere centered at `g.μ` with radius `g.σ`.
Requires a Makie backend (e.g. `using GLMakie`). `display` selects `:wire` (wireframe,
default) or `:solid` (filled mesh); other attributes are forwarded to Makie.
"""
function gaussiandisplay end
"""
    gaussiandisplay!([fig_or_ax,] g; kwargs...)

Add a sphere visualization of an `AbstractIsotropicGaussian` to an existing Makie figure or
axis. Requires a Makie backend. See [`gaussiandisplay`](@ref) for keyword arguments.
"""
function gaussiandisplay! end
"""
    gmmdisplay([fig_or_ax,] g; display=:wire, palette=..., color=nothing, label="", alpha=1, transparency=false)

Visualize an `AbstractIsotropicGMM` or `AbstractIsotropicMultiGMM` as a collection of
spheres, one per Gaussian component. Requires a Makie backend (e.g. `using GLMakie`). In a
multi-GMM, each labeled sub-GMM is drawn in a distinct color from `palette`.
"""
function gmmdisplay end
"""
    gmmdisplay!([fig_or_ax,] g; kwargs...)

Add sphere visualizations of an `AbstractIsotropicGMM` or `AbstractIsotropicMultiGMM` to an
existing Makie figure or axis. Requires a Makie backend. See [`gmmdisplay`](@ref) for keyword
arguments.
"""
function gmmdisplay! end
export gmmdisplay, gmmdisplay!, gaussiandisplay, gaussiandisplay!

# `gmmdisplay` already renders an `AbstractIsotropicMultiGMM`, coloring each labeled sub-GMM
# distinctly, and (unlike the former `multigmmdisplay`) supports legends. These forward to it.
@deprecate multigmmdisplay gmmdisplay
@deprecate multigmmdisplay! gmmdisplay!

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
