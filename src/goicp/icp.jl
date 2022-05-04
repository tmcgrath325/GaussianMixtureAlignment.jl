# perform point-to-point ICP with provided correspondence and distance score functions

function iterate_kabsch(P, Q; iterations=1000, correspondence=nothing)
    # default to nearest neighbor correspondence
    if correspondence === nothing
        kdtree = KDTree(Q, Euclidean())
        correspondence = (p) => closest_points(p, kdtree)
    end

    # initial correspondences
    matches = correspondence(P)
    tform = identity

    # iterate until convergence
    it = 0
    while it < iterations
        it += 1
        matchedP, matchedQ = matched_points(P, Q, matches)
        tform = kabsch(matchedP, matchedQ)
        
        prevmatches = matches
        matches = correspondence(tform(P))
        if matches == prevmatches
            break
        end
    end
    return matches
end