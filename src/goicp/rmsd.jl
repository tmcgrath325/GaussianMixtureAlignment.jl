squared_deviation(P::AbstractMatrix, Q::AbstractMatrix) = sum(colwise(EuclideanSq(), P, Q))
squared_deviation(P, Q, matches) = squared_deviation(matched_points(P, Q, matches)..., Euclidean())
squared_deviation(P::AbstractPointSet, Q::AbstractPointSet) = squared_deviation(P, Q, closest_points(P,Q))

rmsd(P, Q, args...) = sqrt(squared_deviation(P,Q,args...)/size(P,2))
