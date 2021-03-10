struct Block{T<:Real, N}
    ranges::NTuple{N, Tuple{T,T}}
    center::NTuple{N,T}
    lowerbound::T
    upperbound::T
end
Block{T, N}() where T where N = Block{T, N}(ntuple(x->(zero(T),zero(T)),N), ntuple(x->(zero(T)),N), typemax(T), typemax(T))

const hash_block_seed = UInt === UInt64 ? 0x03f7a7ad5ef46a89 : 0xa9bf8ce0
function hash(B::Block, h::UInt)
    h += hash_block_seed
    h = hash(B.center, h)
    return h
end

"""
    sbrngs = subranges(ranges, nsplits)

Takes `ranges`, a nested array describing intervals for each dimension in rigid-rotation space
defining a hypercube, and splits the hypercube into `nsplits` even components along each dimension.
Since the space is 6-dimensional, the number of returned sub-cubes will be `nsplits^6`.
"""
function subranges(ranges, nsplits::Int)
    if length(ranges) != 6
        throw(ArgumentError("`ranges` must have a length of 6"))
    end
    if isodd(nsplits)
        throw(ArgumentError("`nsplits` must be even"))
    end
    dims = length(ranges)
    t = eltype(eltype(ranges))

    # calculate even splititng points for each dimension
    splits = [range(r[1], stop=r[2], length=nsplits+1) |> collect for r in ranges]
    children = fill(ranges, nsplits^dims)
    idx = 1
    for i=1:nsplits
        for j=1:nsplits
            for k=1:nsplits
                for m=1:nsplits
                    for n=1:nsplits
                        for p=1:nsplits
                            child = ((splits[1][i], splits[1][i+1]),
                                     (splits[2][j], splits[2][j+1]),
                                     (splits[3][k], splits[3][k+1]),
                                     (splits[4][m], splits[4][m+1]),
                                     (splits[5][n], splits[5][n+1]),
                                     (splits[6][p], splits[6][p+1]))
                            children[idx] = child
                            idx += 1
                        end
                    end
                end
            end
        end
    end
    # # Arbitrary dimensionality, but somewhat slower
    # splitvals = [range(r[1], stop=r[2], length=nsplits+1) |> collect for r in ranges]
    # splits = [[(splitvals[i][j], splitvals[i][j+1]) for j=1:nsplits] for i=1:dims]
    # f(x) = splits[x[1]][x[2]]
    # children = fill(ranges, nsplits^dims)
    # for I in CartesianIndices(NTuple{dims,UnitRange{Int}}(fill(1:nsplits, dims)))
    #     push!(children, NTuple{dims,Tuple{t,t}}(map(x->f(x), enumerate(Tuple(I)))))
    # end
    return children
end

function Block(gmmx::IsotropicGMM, gmmy::IsotropicGMM, ranges=nothing)
    # get center and uncertainty region
    t = promote_type(eltype(gmmx),eltype(gmmx))
    if isnothing(ranges)
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
    center = NTuple{length(ranges),t}([sum(dim)/2 for dim in ranges])
    rwidth = ranges[1][2] - ranges[1][1]
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
    dims = 2*size(gmmx,2)
    if dims != 2*size(gmmy,2)
        throw(ArgumentError("Dimensionality of the GMMs must be equal"))
    end
    # initialization
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

        # if the best solution so far is close enough to the best possible solution, end
        if abs((ub - lb)/lb) < tol
            return ub, lb, bestloc, ndivisions
        end

        # split up the block into `nsplits` smaller blocks across each dimension
        subrngs = subranges(bl.ranges, nsplits)
        sblks = fill(Block{t, dims}(), nsplits^dims)
        Threads.@threads for i=1:length(subrngs)
            # TODO: local alignment with L-BFGS-B to reduce upperbounds in each box added to the queue
            sblks[i] = Block(gmmx, gmmy, subrngs[i])
        end

        # only add sub-blocks to the queue if they present possibility for improvement
        for sblk in sblks
            if sblk.lowerbound < ub
                enqueue!(pq, sblk, sblk.lowerbound)                
            end
        end
        # for srng in subrngs
        #     sblk = Block(gmmx, gmmy, srng)
        #     if sblk.lowerbound < ub
        #         enqueue!(pq, sblk, sblk.lowerbound)              
        #     end
        # end
    end
    # throw(ErrorException("Not supposed to empty the queue without finding the global min"))
    # return the best value so far, `ub`
    return ub, dequeue_pair!(pq)[2], bestloc, ndivisions
end