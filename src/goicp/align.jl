function goicp_align(x::AbstractPointset, y::AbstractPointset; kwargs...)
    kdtree = KDTree(y.coords, Euclidean())
    return branchbound(x, y, kdtree; localfun=local_icp, kwargs...)
end