# perform point-to-point ICP with provided correspondence and distance score functions
# TO DO: for MultiPointSets, choose the appropriate number of weights
function iterate_kabsch(P, Q, wp=ones(size(P,2)), wq=ones(size(Q,2)); iterations=1000, correspondence = hungarian_assignment)
    # initial correspondences
    matches = correspondence(P,Q)
    prevmatches = matches
    tform = identity
    # iterate until convergence
    it = 0
    score = Inf
    prevscore = Inf
    while it < iterations
        it += 1
        # TO DO: make sure that, for hungarian assignment, the proper weights are selected

        prevscore = score
        tform = kabsch_matches(P, Q, matches, wp, wq)
        Pt = transform_columns(tform, P)
        score = squared_deviation(Pt,Q,matches)
        if prevscore < score
            matches = prevmatches
            break
        end

        prevmatches = matches
        matches = correspondence(Pt,Q)
        if matches == prevmatches
            break
        end
    end
    return matches
end

iterate_kabsch(P::AbstractPointSet, Q::AbstractSet; kwargs...) = iterate_kabsch(P.coords, Q.coords, P.weights, Q.weights; kwargs...)
iterate_kabsch(P::AbstractMultiPointSet,  Q::AbstractMultiPointSet;  kwargs...) = iterate_kabsch(P, Q, weights(P), weights(Q); kwargs...)

"""
    matches = icp(P, Q; kdtree=KDTree(Q, Euclidean()), kwargs...)

Iterative Closest Point: iteratively assigns each point in `P` to its nearest neighbor in
`Q` via a KD-tree and refines the alignment with Kabsch until the assignment stabilizes.

Returns the converged correspondence as a vector of `(i, j)` index pairs, where point `i`
of `P` is matched to point `j` of `Q`. Pass a pre-built `kdtree` to avoid redundant
construction when calling repeatedly on the same `Q`.
"""
function icp(P::AbstractMatrix, Q::AbstractMatrix, wp=ones(size(P,2)), wq=ones(size(P,2)); kdtree = KDTree(Q, Euclidean()), kwargs...)
    return iterate_kabsch(P, Q, wp, wq; correspondence = f(p,q) = closest_points(p, kdtree), kwargs...)
end
icp(P::AbstractSinglePointSet, Q::AbstractSinglePointSet; kwargs...) = icp(P.coords, Q.coords, P.weights, Q.weights; kwargs...)

"""
    matches = iterative_hungarian(P, Q; kwargs...)

Iteratively solves the linear assignment problem between `P` and `Q` using the Hungarian
algorithm and refines the alignment with Kabsch until the assignment stabilizes.

Returns the converged correspondence as a vector of `(i, j)` index pairs. Accepts the same
keyword arguments as `icp`.
"""
iterative_hungarian(args...; kwargs...) = iterate_kabsch(args...; correspondence = hungarian_assignment, kwargs...)

function local_matching_alignment(x::AbstractPointSet, y::AbstractPointSet, block::SearchRegion; matching_fun = iterative_hungarian, kwargs...)
    tformedx = block.R*x + block.T
    matches = matching_fun(tformedx, y; kwargs...)
    tform = kabsch_matches(x, y, matches)
    score = squared_deviation(tform(x), y, matches)
    R = RotationVec(tform.linear)
    params = (R.sx, R.sy, R.sz, tform.translation...)
    return (score, params)
end

function local_matching_alignment(x::AbstractPointSet, y::AbstractPointSet, block::RotationRegion; matching_fun = iterative_hungarian, kwargs...)
    tformedx = block.R*x + block.T
    matches = matching_fun(tformedx, y; kwargs...)
    tform = kabsch_matches(x, y, matches)
    score = squared_deviation(tform(x), y, matches)
    R = RotationVec(tform.linear)
    params = (R.sx, R.sy, R.sz)
    return (score, params)
end

function local_matching_alignment(x::AbstractPointSet, y::AbstractPointSet, block::TranslationRegion; matching_fun = iterative_hungarian, kwargs...)
    tformedx = block.R*x + block.T
    matches = matching_fun(tformedx, y; kwargs...)
    tform = kabsch_matches(x, y, matches)
    score = squared_deviation(tform(x), y, matches)
    params = (tform.translation...,)
    return (score, params)
end

local_icp(x, y, block::SearchRegion; kwargs...) = local_matching_alignment(x, y, block; matching_fun = icp, kwargs...)

local_iterative_hungarian(x, y, block::SearchRegion; kwargs...) = local_matching_alignment(x, y, block; matching_fun = iterative_hungarian, kwargs...)
