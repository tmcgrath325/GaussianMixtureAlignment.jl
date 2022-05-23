# perform point-to-point ICP with provided correspondence and distance score functions

function iterate_kabsch(P, Q, w=ones(size(P,2)); iterations=1000, correspondence = hungarian_assignment)
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
        matches = correspondence(tform(P),Q)
        if matches == prevmatches
            break
        end
    end
    return matches
end

iterate_kabsch(P::PointSet, Q::PointSet; kwargs...) = iterate_kabsch(P.coords, Q.coords, P.weights .* Q.weights; kwargs...);

function icp(P::AbstractMatrix, Q::AbstractMatrix, w=ones(size(P,2)); kdtree = KDTree(Q, Euclidean()), kwargs...)
    return iterate_kabsch(P, Q, w; correspondence = f(p,q) = closest_points(p, kdtree), kwargs...)
end
icp(P::PointSet, Q::PointSet; kwargs...) = icp(P.coords, Q.coords, P.weights .* Q.weights; kwargs...)

iterative_hungarian(args...; kwargs...) = iterate_kabsch(args...; correspondence = hungarian_assignment, kwargs...) 

function local_matching_alignment(x::AbstractPointSet, y::AbstractPointSet, block::SearchRegion; matching_fun = iterative_hungarian, kwargs...)
    ur = UncertaintyRegion(block)
    tformedx = tformwithparams(center(ur), x)
    matches = matching_fun(tformedx, y; kwargs...)
    tform = kabsch(x, y, matches)
    score = squared_deviation(tform(x), y, matches)
    R = RotationVec(tform.linear)
    params = (R.sx, R.sy, R.sz, tform.translation...)
    return (score, params)
end

local_icp(x, y, block; kwargs...) = local_matching_alignment(x, y, block; matching_fun = icp, kwargs...)
local_iterative_hungarian(args...; kwargs...) = local_matching_alignment(args...; matching_fun = iterative_hungarian, kwargs...)