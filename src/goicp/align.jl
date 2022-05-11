function goicp_align(x::AbstractPointSet, y::AbstractPointSet; kwargs...)
    kdtree = KDTree(y.coords, Euclidean())
    return branchbound(x, y, kdtree; boundsfun=squared_dist_bounds, localfun=local_icp, kwargs...)
end