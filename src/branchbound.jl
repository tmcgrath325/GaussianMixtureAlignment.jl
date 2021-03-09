struct Block{T<:Real, N}
    ranges::NTuple{N, Tuple{T,T}}
    center::Vector{T}
    lowerbound::T
    upperbound::T
end

"""
    sbrngs = subranges(ranges, nsplits)

Takes `ranges`, a nested array describing intervals for each dimension in rigid-rotation space
defining a hypercube, and splits the hypercube into `nsplits` even components along each dimension.
Since the space is 6-dimensional, the number of returned sub-cubes will be `nsplits^6`.
"""
function subranges(ranges, nsplits)
    if isodd(nsplits)
        throw(ArgumentError("`nsplits` must be even"))
    end
    ndims = length(ranges)
    t = eltype(eltype(ranges))

    # calculate even splititng points for each dimension
    splits = [range(r[1], stop=r[2], length=nsplits+1) |> collect for r in ranges]
    children =  NTuple{ndims,Tuple{t,t}}[]
    for I in CartesianIndices(fill(0, fill(nsplits, ndims)...))
        child = Array{Tuple{t,t}}(fill((NaN,NaN), 2, ndims))
        for (dim,j) in enumerate(Tuple(I))
            child[dim] = (splits[dim][j], splits[dim][j+1])
        end
        push!(children, NTuple{ndims,Tuple{t,t}}(child))
    end
    return children
end

function Block(gmmx::IsotropicGMM, gmmy::IsotropicGMM, ranges=nothing)
    # get center and uncertainty region
    if isnothing(ranges)
        t = promote_type(eltype(gmmx),eltype(gmmx))
        trlim = typemin(t)
        for gaussians in (gmmx.gaussians, gmmy.gaussians)
            if !isempty(gaussians)
                trlim = max(trlim, maximum(gaussians) do gauss
                        maximum(abs, gauss.μ) end)
                end
        end
        pie = t(π)
        ranges = ((-pie,pie), (-pie,pie), (-pie,pie), (-trlim,trlim), (-trlim,trlim), (-trlim,trlim))
    end
    center = [sum(dim)/2 for dim in ranges]
    rwidth = ranges[1][2] - ranges[1][1]      # TO DO: add check to make sure bounds form a cube?
    twidth = ranges[4][2] - ranges[4][1]

    # calculate objective function bounds for the block
    lb, ub = get_bounds(gmmx, gmmy, rwidth, twidth, center)

    return Block(ranges, center, lb, ub)
end

"""
    min, pos, n = branch_bound(gmmx, gmmy)

Finds the globally optimal minimum for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150).
"""
function branch_bound(gmmx::IsotropicGMM, gmmy::IsotropicGMM, nsplits=2; tol=1e-12, maxblocks=5e8)
    if isodd(nsplits)
        throw(ArgumentError("`nsplits` must be even"))
    end
    ndims = size(gmmx,2)
    if ndims != size(gmmy,2)
        throw(ArgumentError("Dimensionality of the GMMs must be equal"))
    end
    # initialization
    initblock = Block(gmmx, gmmy)
    ub = initblock.upperbound       # best-so-far objective value
    bestloc = initblock.center      # best-so-far transformation     
    t = promote_type(eltype(gmmx), eltype(gmmy))
    pq = PriorityQueue{Block{t, 2*ndims}, t}()
    enqueue!(pq, initblock, initblock.lowerbound)

    ndivisions = 0
    while !isempty(pq)
        if length(pq) - ndivisions > maxblocks
            break
        end
        ndivisions = ndivisions + 1
        # take the block with the lowest lower bound
        bl, lb = dequeue_pair!(pq)
        # update the best solution if this one is better
        if bl.upperbound < ub
            ub = bl.upperbound
            bestloc = bl.center
        end

        # # for testing
        # if ndivisions == 1
        #     @show ub
        # elseif ndivisions%1000 == 0 
        #     @show lb, ub
        # end

        # if the best solution so far is close enough to the best possible solution, end
        if abs((ub - lb)/lb) < tol
            return ub, bestloc, ndivisions
        end

        # split up the block into `nsplits` smaller blocks across each dimension
        subrngs = subranges(bl.ranges, nsplits)
        for srng in subrngs
            sblk = Block(gmmx, gmmy, srng)
            # only add sub-blocks to the queue if they present possibility for improvement
            if sblk.lowerbound < ub
                # TODO: local alignment with L-BFGS-B to reduce upperbounds in each box added to the queue
                enqueue!(pq, sblk, sblk.lowerbound)
            end
        end
    end
    # throw(ErrorException("Not supposed to empty the queue without finding the global min"))
    # return the best value so far, `ub`
    return ub, bestloc, ndivisions
end