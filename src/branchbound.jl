"""
    min, lb, pos, n = branch_bound(gmmx, gmmy, nsplits=2, initblock=nothing, 
                                   rot=nothing, trl=nothing; blockfun=fullBlock, objfun=alignment_objective,
                                   rtol=0.01, maxblocks=5e8, maxevals=Inf, maxstagnant=Inf, threads=false)

Finds the globally optimal rigid transform for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150). `nsplits` is the number
of splits performed along each dimension during branching. Returns the overlap between the GMMs, the
lower bound on the overlap, and the transformation vector for the best transformation, as well as the 
number of objective evaluations.  
"""
function branch_bound(gmmx::Union{IsotropicGMM,MultiGMM}, gmmy::Union{IsotropicGMM,MultiGMM}, nsplits=2, initblock=nothing, 
                      rot=nothing, trl=nothing; blockfun=fullBlock, objfun=alignment_objective,
                      atol=0.1, rtol=0, maxblocks=5e8, maxevals=Inf, maxstagnant=Inf, threads=false)
    if isodd(nsplits)
        throw(ArgumentError("`nsplits` must be even"))
    end
    if size(gmmx,2) != size(gmmy,2)
        throw(ArgumentError("Dimensionality of the GMMs must be equal"))
    end

    # prepare pairwise widths and weights
    pσ, pϕ = pairwise_consts(gmmx, gmmy)

    # initialization
    if isnothing(initblock)
        initblock = blockfun(gmmx, gmmy, nothing, pσ, pϕ, rot, trl)
    end
    ndims = size(initblock)
    ub, bestloc = initblock.upperbound, initblock.center # local_align(gmmx, gmmy, initblock, pσ, pϕ)    # best-so-far objective value and transformation
    lb = Inf
    t = promote_type(eltype(gmmx), eltype(gmmy))
    pq = PriorityQueue{Block{t, ndims}, Tuple{t,t}}()
    enqueue!(pq, initblock, (initblock.lowerbound, initblock.upperbound))
    
    # split cubes until convergence
    ndivisions = 0
    sinceimprove = 0
    evalsperdiv = length(gmmx)*length(gmmy)*nsplits^ndims
    while !isempty(pq)
        if (length(pq) > maxblocks) || (ndivisions*evalsperdiv > maxevals) || (sinceimprove > maxstagnant)
            break
        end
        ndivisions += 1
        sinceimprove += 1

        # take the block with the lowest lower bound
        bl, (lb, blub) = dequeue_pair!(pq)

        # if the best solution so far is close enough to the best possible solution, end
        if abs((ub - lb)/lb) < rtol || abs(ub-lb) < atol
            return ub, lb, bestloc, ndivisions
        end

        # split up the block into `nsplits` smaller blocks across each dimension
        subrngs = subranges(bl.ranges, nsplits)
        sblks = fill(Block{t, ndims}(), nsplits^ndims)
        if threads
            Threads.@threads for i=1:length(subrngs)
                sblks[i] = blockfun(gmmx, gmmy, subrngs[i], pσ, pϕ, rot, trl)
            end
        else
            for i=1:length(subrngs)
                sblks[i] = blockfun(gmmx, gmmy, subrngs[i], pσ, pϕ, rot, trl)
            end
        end

        # reset the upper bound if appropriate
        subs = [sblk.upperbound for sblk in sblks]
        minub, ubidx = findmin(subs)
        if minub < ub
            ub, bestloc = local_align(gmmx, gmmy, sblks[ubidx], pσ, pϕ, objfun=objfun, rot=rot, trl=trl)
            sinceimprove = 0
        end

        # only add sub-blocks to the queue if they present possibility for improvement
        for sblk in sblks
            if sblk.lowerbound < ub
                enqueue!(pq, sblk, (sblk.lowerbound, sblk.upperbound))
            end
        end
    end
    if isempty(pq)
        return ub, lb, bestloc, ndivisions
    else
        return ub, dequeue_pair!(pq)[2][1], bestloc, ndivisions
    end
end

"""
    min, lb, pos, n = rot_branch_bound(gmmx, gmmy, nsplits=2, trl=nothing, initblock=nothing;
                                       rtol=0.01, maxblocks=5e8, maxevals=Inf, maxstagnant=Inf, threads=false)

Finds the globally optimal rotation for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150), for a given translation, `trl`.
`nsplits` is the number of splits performed along each dimension during branching. Returns the overlap
between the GMMs, the lower bound on the overlap, and the transformation vector for the best rotation,
as well as the number of objective evaluations. 
"""
function rot_branch_bound(gmmx::Union{IsotropicGMM,MultiGMM}, gmmy::Union{IsotropicGMM,MultiGMM}, nsplits=2, trl=nothing, initblock=nothing;
                          atol=0.1, rtol=0, maxblocks=5e8, maxevals=Inf, maxstagnant=Inf, threads=false)
    return branch_bound(gmmx, gmmy, nsplits, initblock, nothing, trl, blockfun=rotBlock, objfun=rot_alignment_objective,
                        atol=atol, rtol=rtol, maxblocks=maxblocks, maxevals=maxevals, maxstagnant=maxstagnant, threads=threads)
end

"""
    min, lb, pos, n = rot_branch_bound(gmmx, gmmy, nsplits=2, rot=nothing, initblock=nothing;
                                       rtol=0.01, maxblocks=5e8, maxevals=Inf, maxstagnant=Inf, threads=false)

Finds the globally optimal translation for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150), for a given rotation, `rot`.
`nsplits` is the number of splits performed along each dimension during branching. Returns the overlap
between the GMMs, the lower bound on the overlap, and the transformation vector for the best translation,
as well as the number of objective evaluations. 
"""
function trl_branch_bound(gmmx::Union{IsotropicGMM,MultiGMM}, gmmy::Union{IsotropicGMM,MultiGMM}, nsplits=2, rot=nothing, initblock=nothing;
                          atol=0.1, rtol=0, maxblocks=5e8, maxevals=Inf, maxstagnant=Inf, threads=false)
    return branch_bound(gmmx, gmmy, nsplits, initblock, rot, nothing, blockfun=trlBlock, objfun=trl_alignment_objective,
                        atol=atol, rtol=rtol, maxblocks=maxblocks, maxevals=maxevals, maxstagnant=maxstagnant, threads=threads)
end
