# closest points using a particular metric, using a K-D tree implemented by NearestNeighbors
function closest_points(P, kdtree::KDTree); # kdtree=KDTree(Q, Euclidean())
    nearestidx, dists = nn(kdtree, P)
    return [(i,nearestidx[i]) for i=1:size(P,2)]
end

closest_points(P::AbstractMatrix, Q::AbstractMatrix) = closest_points(P, KDTree(Q, Euclidean()))
closest_points(P::AbstractSinglePointSet, Q::AbstractSinglePointSet) = closest_points(P.coords, Q.coords)

# Hungarian algorithm for assignment
function hungarian_assignment(P::AbstractMatrix,Q::AbstractMatrix,metric=SqEuclidean())
    weights = pairwise(metric, P, Q; dims=2)
    assignment, cost = hungarian(weights)
    matches = Tuple{Int,Int}[]
    for (i,a) in enumerate(assignment)
        if a !== 0
            push!(matches, (i,a))
        end
    end
    return matches
end

hungarian_assignment(P::AbstractSinglePointSet, Q::AbstractSinglePointSet, metric=SqEuclidean()) = hungarian_assignment(P.coords, Q.coords, metric)

function hungarian_assignment(P::AbstractMultiPointSet{N,T,K}, Q::AbstractMultiPointSet{N,T,K}, metric=SqEuclidean()) where {N,T,K}
    matchesdict = Dict{K, Vector{Tuple{Int,Int}}}();
    for (key, ps) in P.pointsets
        push!(matchesdict, key => hungarian_assignment(ps, Q.pointsets[key], metric));
    end
    return matchesdict
end


# generate matrices for Kabsch from a list of correspondences
matched_points(P::AbstractMatrix, Q::AbstractMatrix, matches) = (hcat([P[:,i] for (i,j) in matches]...), hcat([Q[:,j] for (i,j) in matches]...))
matched_points(P, Q, matches) = ([P[i] for (i,j) in matches], [Q[j] for (i,j) in matches])
matched_points(P, Q; correspondence=closest_points, kwargs...) = matched_points(P, Q, correspondence(P,Q; kwargs...))
matched_points(P::AbstractSinglePointSet, Q::AbstractSinglePointSet, args...; kwargs...) = matched_points(P.coords, Q.coords, args...; kwargs...)

function matched_points(P::AbstractMultiPointSet{N,T,K}, Q::AbstractMultiPointSet{N,T,K}, matchesdict::Dict{K, Vector{Tuple{Int,Int}}}) where {N,T,K}
    matchedP = Array{T}(undef, N, 0)
    matchedQ = Array{T}(undef, N, 0)
    for (key, matches) in matchesdict
        (mp, mq) = matched_points(P.pointsets[key], Q.pointsets[key], matches)
        matchedP = hcat(matchedP, mp)
        matchedQ = hcat(matchedQ, mq)
    end
    return matchedP, matchedQ
end

