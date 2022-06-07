# perform point-to-point ICP with provided correspondence and distance score functions
# TO DO: for MultiPointSets, choose the appropriate number of weights
function iterate_kabsch(P, Q, w=ones(size(P,2)); iterations=1000, correspondence = hungarian_assignment)
    # initial correspondences
    matches = correspondence(P,Q)
    tform = identity

    # iterate until convergence
    it = 0
    while it < iterations
        it += 1
        matchedP, matchedQ = matched_points(P, Q, matches)
        # TO DO: make sure that, for hungarian assignment, the proper weights are selected
        tform = kabsch(matchedP, matchedQ, w)
        
        prevmatches = matches
        matches = correspondence(tform(P),Q)
        if matches == prevmatches
            break
        end
    end
    return matches
end

iterate_kabsch(P::AbstractSinglePointSet, Q::AbstractSinglePointSet; kwargs...) = iterate_kabsch(P.coords, Q.coords, P.weights .* Q.weights; kwargs...);

function icp(P::AbstractMatrix, Q::AbstractMatrix, w=ones(size(P,2)); kdtree = KDTree(Q, Euclidean()), kwargs...)
    return iterate_kabsch(P, Q, w; correspondence = f(p,q) = closest_points(p, kdtree), kwargs...)
end
icp(P::AbstractSinglePointSet, Q::AbstractSinglePointSet; kwargs...) = icp(P.coords, Q.coords, P.weights .* Q.weights; kwargs...)

iterative_hungarian(args...; kwargs...) = iterate_kabsch(args...; correspondence = hungarian_assignment, kwargs...) 

function local_matching_alignment(x::AbstractPointSet, y::AbstractPointSet, block::SearchRegion; matching_fun = iterative_hungarian, kwargs...)
    tformedx = tformwithparams((block.R..., block.T...), x)
    matches = matching_fun(tformedx, y; kwargs...)
    tform = kabsch(x, y, matches)
    score = squared_deviation(tform(x), y, matches)
    R = RotationVec(tform.linear)
    params = (R.sx, R.sy, R.sz, tform.translation...)
    return (score, params)
end

 local_icp(x, y, block::UncertaintyRegion; kwargs...) = local_matching_alignment(x, y, block; matching_fun = icp, kwargs...)
function local_icp(x, y, block::RotationRegion; kwargs...)
    (score, params) = local_icp(x, y, UncertaintyRegion(block); kwargs...)
    return (score, params[1:3])
end
function local_icp(x, y, block::TranslationRegion; kwargs...)
    (score, params) = local_icp(x, y, UncertaintyRegion(block); kwargs...)
    return (score, params[4:6])
end

local_iterative_hungarian(x, y, block::UncertaintyRegion; kwargs...) = local_matching_alignment(x, y, block; matching_fun = iterative_hungarian, kwargs...)
function local_iterative_hungarian(x, y, block::RotationRegion; kwargs...)
    (score, params) = local_iterative_hungarian(x, y, UncertaintyRegion(block); kwargs...)
    return (score, params[1:3])
end
function local_iterative_hungarian(x, y, block::TranslationRegion; kwargs...) 
    (score, params) = local_iterative_hungarian(x, y, UncertaintyRegion(block); kwargs...)
    return (score, params[4:6])
end