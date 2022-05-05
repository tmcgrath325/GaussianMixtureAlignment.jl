# closest points using a particular metric, using a K-D tree implemented by NearestNeighbors
function closest_points(P, kdtree::KDTree); # kdtree=KDTree(Q, Euclidean())
    nearestidx, dists = nn(kdtree, P)
    return [(i,nearestidx[i]) for i=1:size(P,2)]
end

closest_points(P, Q) = closestPoints(P, KDTree(Q, Euclidean()))

# Hungarian algorithm for assignment
function hungarian_assignment(P,Q,metric=EuclideanSq())
    weights = pairwise(metric, P, Q; dims=2)
    assignment, cost = hungarian(weights)
    matches = Tuble{Int,Int}[]
    for (i,a) in enumerate(assignment)
        if a !== 0
            push!(matches((i,a)))
        end
    end
    return matches
end


# generate matrices for Kabsch from a list of correspondences
matched_points(P::AbstractMatrix, Q::AbstractMatrix, matches) = (hcat([P[:,i] for (i,j) in matches]...), hcat([Q[:,j] for (i,j) in matches]...))
matched_points(P, Q, matches) = ([P[i] for (i,j) in matches], [Q[j] for (i,j) in matches])
matched_points(P, Q; correspondence=closest_points, kwargs...) = matched_points(P, Q, correspondence(P,Q; kwargs...))

