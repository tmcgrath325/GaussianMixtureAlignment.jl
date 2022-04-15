# closest points using a particular metric, using a K-D tree implemented by NearestNeighbors
function closest_points(P,Q; metric=Euclidean())
    kdtree = KDTree(Q, metric)
    nearestidx, dists = nn(kdtree, P)
    return [(i,nearestidx[i]) for i=1:size(P,2)]
end

# generate matrices for Kabsch from a list of correspondences
matched_points(P::AbstractMatrix, Q::AbstractMatrix, matches) = (hcat([P[:,i] for (i,j) in matches]...), hcat([Q[:,j] for (i,j) in matches]...))
matched_points(P, Q, matches) = ([P[i] for (i,j) in matches], [Q[j] for (i,j) in matches])
matched_points(P,Q; correspondence=closest_points, kwargs...) = matched_points(P, Q, correspondence(P,Q; kwargs...))
