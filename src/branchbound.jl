struct Block{T<:Real}
    ranges::Vector{Vector{T}}
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
    splits = [range(r[1], stop=r[2], length=nsplits+1) |> collect for r in ranges]
    if length(splits) != 6
        throw(ArgumentError("Need splits for exactly 6 dimensions"))
    end
    children = []
    for i=1:nsplits    # TO DO: find a cleaner way to do this, compatible with arbitrary # of dimensions (someone might want 2D)?
        for j=1:nsplits
            for k=1:nsplits
                for m=1:nsplits
                    for n=1:nsplits
                        for p=1:nsplits
                            child = [[splits[1][i], splits[1][i+1]],
                                     [splits[2][j], splits[2][j+1]],
                                     [splits[3][k], splits[3][k+1]],
                                     [splits[4][m], splits[4][m+1]],
                                     [splits[5][n], splits[5][n+1]],
                                     [splits[6][p], splits[6][p+1]]]
                            push!(children, child)
                        end
                    end
                end
            end
        end
    end
    return children
end

function Block(gmmx::IsotropicGMM, gmmy::IsotropicGMM, ranges=nothing)
    # get center and uncertainty region
    if isnothing(ranges)
        trlim = typemin(promote_type(eltype(gmmx),eltype(gmmy)))
        for gaussians in (gmmx.gaussians, gmmy.gaussians)
            if !isempty(gaussians)
                trlim = max(trlim, maximum(gaussians) do gauss
                        maximum(abs, gauss.μ) end)
                end
        end
        ranges = [[-π,π], [-π,π], [-π,π], [-trlim,trlim], [-trlim,trlim], [-trlim,trlim]]
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
    # initialization
    initblock = Block(gmmx, gmmy)
    ub = initblock.upperbound       # best-so-far objective value
    bestloc = initblock.center      # best-so-far transformation     
    pq = PriorityQueue{Block,Float64}()
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