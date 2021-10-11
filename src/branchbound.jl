abstract type AlignmentResults end

struct GMAlignmentResult{T,D,N,F<:AbstractAffineMap,X<:AbstractGMM{D,T},Y<:AbstractGMM{D,T}} <: AlignmentResults
    x::X
    y::Y
    upperbound::T
    lowerbound::T
    tform::F
    tform_params::NTuple{N,T}
    obj_calls::Int
    num_splits::Int
    num_blocks::Int
    stagnant_splits::Int
end

# Keyword arguments:\n
#     nsplits     - an integer representing the number of splits that should be made along each dimension during branching\n
#     initblock   - a `Block` that defines the searchspace, which defaults to the smallest space gauranteed to contain the global minimum\n
#     rot         - a three-tuple containing a rotation position, which is passed to the `blockfun`\n
#     trl         - a three-tuple containing a translation position, which is passed to the `blockfun`\n
#     blockfun    - the function used for generating `Block`s that define search subspaces (i.e. fullBlock, rotBlock, trlBlock)\n
#     objfun      - the objective function used for local alignment\n
#     atol        - absolute tolerance. Search terminates when the upper bound is within `atol` of the lower bound\n
#     rtol        - relative tolerance. Search terminates when the upper bound is within `rtol*lb` of the lower bound `lb`\n
#     maxblocks   - the maximum number of `Block`s that can be held in the priority queue before search termination\n
#     maxsplits   - the maximum number of `Block` splits that are allowed before search termination\n
#     maxevals    - the maximum number of objective function evaluations allowed before search termination\n
#     maxstagnant - the maximum number of `Block` splits allowed without improvement before search termination\n
#     threads     - when true, utilizes multithreading for performing `Block` splitting. More useful when `nsplits` is large.\n
"""
    result = gogma_align(gmmx, gmmy; nsplits=2, initblock=nothing, 
                         rot=nothing, trl=nothing, blockfun=fullBlock, objfun=alignment_objective,
                         rtol=0.01, maxblocks=5e8, maxevals=Inf, maxstagnant=Inf, threads=false)

Finds the globally optimal rigid transform for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150).

Returns a `GMAlignmentResult` that contains the maximized overlap of the two GMMs (the upperbound on the objective function),
a lower bound on the alignment objective function, an `AffineMap` which aligns `gmmx` with `gmmy`, and information about the
number of evaluations during the alignment procedure. 
""" 
function gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM;
                     nsplits=2, initblock=nothing, rot=nothing, trl=nothing,
                     blockfun=fullBlock, objfun=alignment_objective, tformfun=AffineMap,
                     atol=0.1, rtol=0, maxblocks=5e8, maxsplits=Inf, maxevals=Inf, maxstagnant=Inf, threads=false)
    if isodd(nsplits)
        throw(ArgumentError("`nsplits` must be even"))
    end
    if dims(gmmx) != dims(gmmy)
        throw(ArgumentError("Dimensionality of the GMMs must be equal"))
    end

    # prepare pairwise widths and weights
    pσ, pϕ = pairwise_consts(gmmx, gmmy)

    # initialization
    if isnothing(initblock)
        initblock = blockfun(gmmx, gmmy, nothing, pσ, pϕ, rot, trl)
    end
    ndims = dims(initblock)
    ub, bestloc = initblock.upperbound, initblock.center    # best-so-far objective value and transformation
    lb = Inf
    t = promote_type(numbertype(gmmx), numbertype(gmmy))
    pq = PriorityQueue{Block{ndims, t}, Tuple{t,t}}()
    enqueue!(pq, initblock, (initblock.lowerbound, initblock.upperbound))
    
    # split cubes until convergence
    ndivisions = 0
    sinceimprove = 0
    evalsperdiv = length(gmmx)*length(gmmy)*nsplits^ndims
    while !isempty(pq)
        if (length(pq) > maxblocks) || (ndivisions*evalsperdiv > maxevals) || (sinceimprove > maxstagnant) || (ndivisions > maxsplits)
            break
        end
        ndivisions += 1
        sinceimprove += 1

        # take the block with the lowest lower bound
        bl, (lb, blub) = dequeue_pair!(pq)

        # if the best solution so far is close enough to the best possible solution, end
        if abs((ub - lb)/lb) < rtol || abs(ub-lb) < atol
            return GMAlignmentResult(gmmx, gmmy, ub, lb, tformfun(bestloc...), bestloc, ndivisions*evalsperdiv, ndivisions, length(pq), sinceimprove)
        end

        # split up the block into `nsplits` smaller blocks across each dimension
        subrngs = subranges(bl.ranges, nsplits)
        sblks = fill(Block{ndims,t}(), nsplits^ndims)
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
        minub, ubidx = findmin([sblk.upperbound for sblk in sblks])
        if minub < ub
            ub, bestloc = local_align(gmmx, gmmy, sblks[ubidx], pσ, pϕ; objfun=objfun, rot=rot, trl=trl)
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
        return GMAlignmentResult(gmmx, gmmy, ub, lb, tformfun(bestloc...), bestloc, ndivisions*evalsperdiv, ndivisions, length(pq), sinceimprove)
    else
        return GMAlignmentResult(gmmx, gmmy, ub, dequeue_pair!(pq)[2][1], tformfun(bestloc...), bestloc, ndivisions*evalsperdiv, ndivisions, length(pq), sinceimprove)
    end
end

"""
    result = rot_gogma_align(gmmx, gmmy; kwargs...)

Finds the globally optimal rotation for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150), for a given rotation, `rot`.
That is, only rigid translation is allowed.

For details about keyword arguments, see `gogma_align()`.
"""
function rot_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM; kwargs...)
    return gogma_align(gmmx, gmmy; blockfun=rotBlock, objfun=rot_alignment_objective, tformfun=LinearMap, kwargs...)
end

"""
    result = trl_gogma_align(gmmx, gmmy; kwargs...)

Finds the globally optimal translation for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150), for a given translation, `trl`.
That is, only rigid rotation is allowed.

For details about keyword arguments, see `gogma_align()`.
"""
function trl_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM;  kwargs...)
    return gogma_align(gmmx, gmmy; blockfun=trlBlock, objfun=trl_alignment_objective, tformfun=Translation, kwargs...)
end
