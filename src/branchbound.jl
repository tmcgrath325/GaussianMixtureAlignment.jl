"""
    min, lb, pos, n = branch_bound(gmmx, gmmy)

Finds the globally optimal minimum for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150). Returns the value of the minimum,
the global lower bound, the position of the minimum, and the number of cube splits performed. 
"""
function branch_bound(gmmx::IsotropicGMM, gmmy::IsotropicGMM, nsplits=4; rtol=0.1, maxblocks=5e8, gpu=false)
    # if isodd(nsplits)
    #     throw(ArgumentError("`nsplits` must be even"))
    # end
    dims = 2*size(gmmx,2)
    if dims != 2*size(gmmy,2)
        throw(ArgumentError("Dimensionality of the GMMs must be equal"))
    end
    # initialization
    if gpu
        xgausses = CuArray(gmmx.gaussians)
        ygausses = CuArray(gmmy.gaussians)
    end
    initblock = Block(gmmx, gmmy)
    ub = initblock.upperbound       # best-so-far objective value
    bestloc = initblock.center      # best-so-far transformation     
    t = promote_type(eltype(gmmx), eltype(gmmy))
    pq = PriorityQueue{Block{t, dims}, Tuple{t,t}}()
    enqueue!(pq, initblock, (initblock.lowerbound, initblock.upperbound))

    ndivisions = 0
    while !isempty(pq)
        if length(pq) > maxblocks
            break
        end
        ndivisions = ndivisions + 1
        # take the block with the lowest lower bound
        bl, (lb, blub) = dequeue_pair!(pq)

        # get bounds for subcubes
        subcntrs = GOGMA.subcenters(bl.center, bl.rwidth, bl.twidth, nsplits)
        bounds = fill((zero(t),zero(t)), length(subcntrs))
        if gpu
            subcntrs = CuArray(subcntrs)
            bounds = CuArray(bounds)
            numblocks = ceil(Int, length(subcntrs)/256)
            CUDA.@sync begin
                @cuda threads=256 blocks=numblocks bounds_kernel!(bounds, xgausses, ygausses, bl.rwidth/nsplits, bl.twidth/nsplits, subcntrs)
            end
            subcntrs = Array(subcntrs)
            bounds = Array(bounds)
        else
            Threads.@threads for i=1:length(subcntrs)
                bounds[i] = get_bounds(gmmx, gmmy, bl.rwidth/nsplits, bl.twidth/nsplits, subcntrs[i])
            end
        end    

        # update the upper bound if appropriate
        sububs = [bds[2] for bds in bounds]
        minub, ubidx = findmin(sububs)
        if minub < ub
            ub = minub
            bestloc = subcntrs[ubidx]
        end

        # only add sub-blocks to the queue if they present possibility for improvement
        for (i,bds) in enumerate(bounds)
            if bds[1] < ub
                enqueue!(pq, Block{t,dims}(subcntrs[i], bl.rwidth/nsplits, bl.twidth/nsplits, bds...), bds)
            end
        end
    end

    # return details for the found minimum
    return ub, dequeue_pair!(pq)[2], bestloc, ndivisions
end