function goicp_align(x::AbstractSinglePointSet, y::AbstractSinglePointSet; kwargs...)
    kdtree = KDTree(y.coords, Euclidean())
    correspondence(xx::AbstractMatrix, yy::AbstractMatrix) = closest_points(xx, kdtree)
    boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = correspondence, distance_bound_fun = tight_distance_bounds)
    localfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, block::SearchRegion) = local_icp(xx, yy, block; kdtree=kdtree)

    return branchbound(x, y; boundsfun=boundsfun, localfun=localfun, kwargs...)
end

function goih_align(x::AbstractPointSet, y::AbstractPointSet; kwargs...)
    boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = hungarian_assignment, distance_bound_fun = loose_distance_bounds)
    branchbound(x, y; boundsfun=boundsfun, localfun=local_iterative_hungarian, kwargs...)
end

function tiv_goicp_align(x::AbstractSinglePointSet, y::AbstractSinglePointSet, cx=Inf, cy=Inf; kwargs...)
    tivx, tivy = tivpointset(x,cx), tivpointset(y,cy)
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

function tiv_goih_align(x::AbstractPointSet, y::AbstractPointSet, cx=Inf, cy=Inf; kwargs...)
    boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = hungarian_assignment, distance_bound_fun = loose_distance_bounds)
    objfun(X, x, y) = distanceobj(X, x, y; correspondence = hungarian_assignment);
    rot_localfun(xx, yy, block; kwargs...) = iterate_local_alignment(xx, yy, block; correspondence = hungarian_assignment, tformfun=LinearMap, kwargs...);
    trl_localfun(xx, yy, block; kwargs...) = iterate_local_alignment(xx, yy, block; correspondence = hungarian_assignment, tformfun=Translation, kwargs...);
    return tiv_branchbound(x, y, tivpointset(x,cx), tivpointset(y,cy); boundsfun=squared_dist_bounds, localfun=local_iterative_hungarian, rot_localfun=rot_localfun, trl_localfun=rot_localfun, kwargs...)
end
