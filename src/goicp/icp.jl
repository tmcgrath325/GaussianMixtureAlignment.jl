# perform point-to-point ICP with provided correspondence and distance score functions

function iterate_kabsch(P, Q, w, args...; iterations=1000; correspondence = (p,q) => closest_points(p,q,args...))
    # initial correspondences
    matches = correspondence(P,Q)
    tform = identity

    # iterate until convergence
    it = 0
    while it < iterations
        it += 1
        matchedP, matchedQ = matched_points(P, Q, matches)
        tform = kabsch(matchedP, matchedQ, w)
        
        prevmatches = matches
        matches = correspondence(tform(P))
        if matches == prevmatches
            break
        end
    end
    return matches
end

iterate_kabsch(P, Q; kwargs...) = iterate_kabsch(P, KDTree(Q, Euclidean()))

function local_icp(x::AbstractPointSet, y::AbstractPointSet, block::SearchRegion, kdtree=KDTree(y.coords); kwargs...)
    ur = UncertaintyRegion(block)
    tformedx = tformwithparams(center(ur), x)
    matches = iterate_kabsch(tformedx.coords, kdtree; kwargs...)
    return squared_deviation(x, y, matches)
end