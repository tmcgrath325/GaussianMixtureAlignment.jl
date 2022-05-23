function goicp_align(x::AbstractSinglePointSet, y::AbstractSinglePointSet; kwargs...)
    kdtree = KDTree(y.coords, Euclidean())
    correspondence(xx::AbstractMatrix, yy::AbstractMatrix) = closest_points(xx, kdtree)
    boundsfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, sr::SearchRegion) = squared_dist_bounds(xx,yy,sr; correspondence = correspondence)
    localfun(xx::AbstractSinglePointSet, yy::AbstractSinglePointSet, block::SearchRegion) = local_icp(xx, yy, block; kdtree=kdtree)

    return branchbound(x, y; boundsfun=boundsfun, localfun=localfun, kwargs...)
end

goih_align(x::AbstractPointSet, y::AbstractPointSet; kwargs...) = branchbound(x, y; boundsfun=squared_dist_bounds, localfun=local_iterative_hungarian, kwargs...)
