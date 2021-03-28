"""
    min, pos, n = branch_bound(gmmx, gmmy)

Finds the globally optimal minimum for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150).
"""
function branch_bound(gmmx::IsotropicGMM, gmmy::IsotropicGMM, nsplits=2; tol=0.05, maxblocks=5e8, threads=true)
    if isodd(nsplits)
        throw(ArgumentError("`nsplits` must be even"))
    end
    ndims = 2*size(gmmx,2)
    if ndims != 2*size(gmmy,2)
        throw(ArgumentError("Dimensionality of the GMMs must be equal"))
    end

    # prepare pairwise widths and weights
    pσ, pϕ = pairwise_consts(gmmx, gmmy)

    # initialization
    initblock = Block(gmmx, gmmy)
    ub, bestloc = local_align(gmmx, gmmy, initblock, pσ, pϕ)    # best-so-far objective value and transformation
    t = promote_type(eltype(gmmx), eltype(gmmy))
    pq = PriorityQueue{Block{t, ndims}, Tuple{t,t}}()
    enqueue!(pq, initblock, (initblock.lowerbound, initblock.upperbound))
    
    # split cubes until convergence
    ndivisions = 0
    while !isempty(pq)
        if length(pq) - ndivisions > maxblocks
            break
        end
        ndivisions = ndivisions + 1
        # take the block with the lowest lower bound
        bl, (lb, blub) = dequeue_pair!(pq)

        # display current 
        if mod(ndivisions, 100) == 0
            @show lb, ub, ndivisions
        end

        # if the best solution so far is close enough to the best possible solution, end
        if abs((ub - lb)/lb) < tol
            return ub, lb, bestloc, ndivisions
        end

        # split up the block into `nsplits` smaller blocks across each dimension
        subrngs = subranges(bl.ranges, nsplits)
        sblks = fill(Block{t, ndims}(), nsplits^ndims)
        if threads
            Threads.@threads for i=1:length(subrngs)
                sblks[i] = Block(gmmx, gmmy, subrngs[i], pσ, pϕ)
            end
        else
            for i=1:length(subrngs)
                # TODO: local alignment with L-BFGS-B to reduce upperbounds in each box added to the queue
                sblks[i] = Block(gmmx, gmmy, subrngs[i], pσ, pϕ)
            end
        end

        # reset the upper bound if appropriate
        subs = [sblk.upperbound for sblk in sblks]
        minub, ubidx = findmin(subs)
        if minub < ub
            ub, bestloc = local_align(gmmx, gmmy, sblks[ubidx], pσ, pϕ)
        end

        # only add sub-blocks to the queue if they present possibility for improvement
        for sblk in sblks
            if sblk.lowerbound < ub
                enqueue!(pq, sblk, (sblk.lowerbound, sblk.upperbound))
            end
        end
    end

    return ub, dequeue_pair!(pq)[2][1], bestloc, ndivisions
end