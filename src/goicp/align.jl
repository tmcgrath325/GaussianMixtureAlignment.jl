"""
    result = goicp_align(x, y; kwargs...)

Find the globally optimal rigid transformation mapping point set `x` onto point set `y`
using globally-optimal ICP (GO-ICP) with branch-and-bound. Correspondences within each
candidate region are determined by nearest-neighbor search via a KD-tree.

Returns a `GlobalAlignmentResult`. Keyword arguments are forwarded to `branchbound`;
see `?branchbound` for the full list.
"""
function goicp_align(x::AbstractSinglePointSet, y::AbstractSinglePointSet; kwargs...)
    kdtree = KDTree(y.coords, Euclidean())
    correspondence(xx::AbstractMatrix, yy::AbstractMatrix) = closest_points(xx, kdtree)
    boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = correspondence, distance_bound_fun = tight_distance_bounds)
    localfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, block::SearchRegion) = local_icp(xx, yy, block; kdtree=kdtree)

    return branchbound(x, y; boundsfun=boundsfun, localfun=localfun, kwargs...)
end

function trl_goicp_align(x::AbstractSinglePointSet, y::AbstractSinglePointSet; kwargs...)
    return goicp_align(x, y; blockfun=TranslationRegion, tformfun=Translation, kwargs...)
end

"""
    result = goih_align(x, y; kwargs...)

Find the globally optimal rigid transformation mapping point set `x` onto point set `y`
using globally-optimal Iterative Hungarian (GO-IH) with branch-and-bound. Correspondences
within each candidate region are determined by solving the linear assignment problem.

Returns a `GlobalAlignmentResult`. Keyword arguments are forwarded to `branchbound`.
"""
function goih_align(x::AbstractPointSet, y::AbstractPointSet; kwargs...)
    boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = hungarian_assignment, distance_bound_fun = tight_distance_bounds)
    branchbound(x, y; boundsfun=boundsfun, localfun=local_iterative_hungarian, kwargs...)
end

"""
    result = tiv_goicp_align(x, y; cutoff_x=Inf, cutoff_y=Inf, kwargs...)

TIV (Translation-Invariant Vector) variant of `goicp_align`. Decomposes the 6-DOF search
into a rotation-only phase on TIV point sets (using nearest-neighbor correspondence),
followed by a translation-only phase on the original point sets.

`cutoff_x` and `cutoff_y` are radius cutoffs for TIV construction (default `Inf`). Returns a
`TIVAlignmentResult` whose `.rotation_result` and `.translation_result` fields hold the
individual `GlobalAlignmentResult`s.
"""
function tiv_goicp_align(x::AbstractSinglePointSet, y::AbstractSinglePointSet; cutoff_x=Inf, cutoff_y=Inf, kwargs...)
    tivx, tivy = tivpointset(x,cutoff_x), tivpointset(y,cutoff_y)
    kdtree = KDTree(y.coords, Euclidean())
    tiv_kdtree = KDTree(tivy.coords, Euclidean())
    correspondence(xx::AbstractMatrix, yy::AbstractMatrix) = closest_points(xx, kdtree)
    rot_correspondence(xx::AbstractMatrix, yy::AbstractMatrix) = closest_points(xx, tiv_kdtree)
    boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = correspondence, distance_bound_fun = tight_distance_bounds)
    rot_boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = rot_correspondence, distance_bound_fun = tight_distance_bounds)
    localfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, block::SearchRegion) = local_icp(xx, yy, block; kdtree=kdtree)
    objfun(X, x, y) = distanceobj(X, x, y; correspondence = correspondence);
    rot_objfun(X, x, y) = distanceobj(X, x, y; correspondence = rot_correspondence);
    rot_localfun(xx, yy, block; kwargs...) = iterate_local_alignment(xx, yy, block; correspondence = rot_correspondence, tformfun=LinearMap, kwargs...);
    trl_localfun(xx, yy, block; kwargs...) = iterate_local_alignment(xx, yy, block; correspondence = correspondence, tformfun=Translation, kwargs...);

    return tiv_branchbound(x, y, tivx, tivy; rot_boundsfun=rot_boundsfun, boundsfun=boundsfun, localfun=localfun, kwargs...)
end

function tiv_goicp_align(x::AbstractSinglePointSet, y::AbstractSinglePointSet, cutoff_x, cutoff_y=Inf; kwargs...)
    Base.depwarn("passing TIV radius cutoffs positionally to `tiv_goicp_align` is deprecated; pass them as the `cutoff_x` and `cutoff_y` keyword arguments instead.", :tiv_goicp_align)
    return tiv_goicp_align(x, y; cutoff_x, cutoff_y, kwargs...)
end

"""
    result = tiv_goih_align(x, y; cutoff_x=Inf, cutoff_y=Inf, kwargs...)

TIV (Translation-Invariant Vector) variant of `goih_align`. Decomposes the 6-DOF search
into a rotation-only phase on TIV point sets (using Hungarian assignment), followed by a
translation-only phase on the original point sets.

`cutoff_x` and `cutoff_y` are radius cutoffs for TIV construction (default `Inf`). Returns a
`TIVAlignmentResult`.
"""
function tiv_goih_align(x::AbstractPointSet, y::AbstractPointSet; cutoff_x=Inf, cutoff_y=Inf, kwargs...)
    boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = hungarian_assignment, distance_bound_fun = tight_distance_bounds)
    objfun(X, x, y) = distanceobj(X, x, y; correspondence = hungarian_assignment);
    rot_localfun(xx, yy, block; kwargs...) = iterate_local_alignment(xx, yy, block; correspondence = hungarian_assignment, tformfun=LinearMap, kwargs...);
    trl_localfun(xx, yy, block; kwargs...) = iterate_local_alignment(xx, yy, block; correspondence = hungarian_assignment, tformfun=Translation, kwargs...);
    return tiv_branchbound(x, y, tivpointset(x,cutoff_x), tivpointset(y,cutoff_y); boundsfun=squared_dist_bounds, localfun=local_iterative_hungarian, rot_localfun=rot_localfun, trl_localfun=rot_localfun, kwargs...)
end

function tiv_goih_align(x::AbstractPointSet, y::AbstractPointSet, cutoff_x, cutoff_y=Inf; kwargs...)
    Base.depwarn("passing TIV radius cutoffs positionally to `tiv_goih_align` is deprecated; pass them as the `cutoff_x` and `cutoff_y` keyword arguments instead.", :tiv_goih_align)
    return tiv_goih_align(x, y; cutoff_x, cutoff_y, kwargs...)
end
