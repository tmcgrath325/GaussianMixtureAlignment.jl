"""
    min, pos, n = branch_bound(gmmx, gmmy)

Finds the globally optimal minimum for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150).
"""
function branch_bound(gmmx::IsotropicGMM, gmmy::IsotropicGMM, nsplits=2; rtol=1e-12, maxblocks=5e8)
    if isodd(nsplits)
        throw(ArgumentError("`nsplits` must be even"))
    end
    dims = 2*size(gmmx,2)
    if dims != 2*size(gmmy,2)
        throw(ArgumentError("Dimensionality of the GMMs must be equal"))
    end
    # initialization
    xgausses = CuArray(gmmx.gaussians)
    ygausses = CuArray(gmmy.gaussians)
    initblock = Block(gmmx, gmmy)
    ub = initblock.upperbound       # best-so-far objective value
    bestloc = initblock.center      # best-so-far transformation     
    t = promote_type(eltype(gmmx), eltype(gmmy))
    pq = PriorityQueue{Block{t, dims}, t}()
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

        # split into subcubes
        subcntrs = CuArray(subcenters(bl.center, bl.rwidth, bl.twidth, nsplits))
        bounds = CuArray(fill((zero(t),zero(t)), length(subcntrs)))
        numblocks = ceil(Int, length(subcntrs)/256)
        CUDA.@sync begin
            @cuda threads=256 blocks=numblocks bounds_kernel!(bounds, xgausses, ygausses, bl.rwidth, bl.twidth, subcntrs)
        end

        # only add sub-blocks to the queue if they present possibility for improvement
        # bounds = Array(bounds)
        # subcntrs = Array(subcntrs)
        for (i,bds) in enumerate(bounds)
            if bds[1] < ub
                enqueue!(pq, Block(subcntrs[i], bl.rwidth/2, bl.twidth/2, bds...), bds[1])                
            end
        end
    end
    # throw(ErrorException("Not supposed to empty the queue without finding the global min"))
    # return the best value so far, `ub`
    return ub, dequeue_pair!(pq)[2], bestloc, ndivisions
end