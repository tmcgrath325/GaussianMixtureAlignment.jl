function goicp_align(x::AbstractSinglePointSet, y::AbstractSinglePointSet; kwargs...)
    kdtree = KDTree(y.coords, Euclidean())
    correspondence(xx::AbstractMatrix, yy::AbstractMatrix) = closest_points(xx, kdtree)
    boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = correspondence, distance_bound_fun = tight_distance_bounds)
    localfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, block::SearchRegion) = local_icp(xx, yy, block; kdtree=kdtree)

    return branchbound(x, y; boundsfun=boundsfun, localfun=localfun, kwargs...)
end

goih_align(x::AbstractPointSet, y::AbstractPointSet; kwargs...) = branchbound(x, y; boundsfun=squared_dist_bounds, localfun=local_iterative_hungarian, kwargs...)

function tiv_goicp_align(x::AbstractSinglePointSet, y::AbstractSinglePointSet, cx=Inf, cy=Inf; kwargs...)
    tivx, tivy = tivpointset(x,cx), tivpointset(y,cy)
    kdtree = KDTree(y.coords, Euclidean())
    tiv_kdtree = KDTree(tivy.coords, Euclidean())
    correspondence(xx::AbstractMatrix, yy::AbstractMatrix) = closest_points(xx, kdtree)
    rot_correspondence(xx::AbstractMatrix, yy::AbstractMatrix) = closest_points(xx, tiv_kdtree)
    boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = correspondence, distance_bound_fun = tight_distance_bounds)
    rot_boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = rot_correspondence, distance_bound_fun = tight_distance_bounds)
    localfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, block::SearchRegion) = local_icp(xx, yy, block; kdtree=kdtree)

    return tiv_branchbound(x, y, tivx, tivy; rot_boundsfun=rot_boundsfun, boundsfun=boundsfun, localfun=localfun, kwargs...)
end

tiv_goih_align(x::AbstractPointSet, y::AbstractPointSet, cx=Inf, cy=Inf; kwargs...) = tiv_branchbound(x, y, tivpointset(x,cx), tivpointset(y,cy); boundsfun=squared_dist_bounds, localfun=local_iterative_hungarian, kwargs...)
