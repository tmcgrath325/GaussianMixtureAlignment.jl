squared_deviation(P::AbstractMatrix, Q::AbstractMatrix, w::AbstractVector=ones(size(P,2))) = sum(w .* colwise(EuclideanSq(), P, Q))
squared_deviation(P, Q, matches::AbstractVector{<:Tuple{Int,Int}}, wp=ones(size(P,2)), wq=ones(size(Q,2))) = squared_deviation(matched_points(P, Q, matches)..., [wp[i]*wq[j] for (i,j) in matches])
squared_deviation(P::AbstractPointSet, Q::AbstractPointSet) = squared_deviation(P.coords, Q.coords, closestPoints(P.coords,Q.coords), P.weights, Q.weights)

rmsd(P, Q, args...) = sqrt(squared_deviation(P,Q,args...)/size(P,2))
