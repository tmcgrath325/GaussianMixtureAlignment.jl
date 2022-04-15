# perform point-to-point ICP with provided correspondence and distance score functions

squared_deviation(P, Q, metric) = sum(abs2, colwise(metric, P, Q))
squared_deviation(P, Q, metric, matches) = squared_deviation(matched_points(P, Q, matches)..., metric)

rmsd(P, Q, metric, n::Int) = sqrt(squared_deviation(P,Q,metric)/n)
rmsd(P::AbstractMatrix, Q::AbstractMatrix, metric) = rmsd(P, Q, metric, size(P,2))
rmsd(P, Q, metric) = rmsd(P, Q, metric, length(P))

rmsd(P, Q, metric, matches) = rmsd(matched_points(P, Q, matches)..., metric)

function iterate_kabsch(P, Q; iterations=1000, correspondence=closest_points, kwargs...)
    # initial correspondences
    matches = correspondence(P, Q; kwargs...)
    tform = identity

    # iterate until convergence
    it = 0
    while it < iterations
        it += 1
        matchedP, matchedQ = matched_points(P, Q, matches)
        tform = kabsch(matchedP, matchedQ)
        
        prevmatches = matches
        matches = correspondence(tform(P), Q; kwargs...)
        if matches == prevmatches
            break
        end
    end
    return matches
end