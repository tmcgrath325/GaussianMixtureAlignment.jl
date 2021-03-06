"""
A structure which defines a hypercube in `N`-dimensional searchspace, with a `center` location 
and conrners defined by `ranges`.
"""
struct Block{N,T<:Real}
    ranges::NTuple{N, Tuple{T,T}}
    center::NTuple{N,T}
    lowerbound::T
    upperbound::T
end
Block{N,T}() where {N,T} = Block{N,T}(ntuple(x->(zero(T),zero(T)),N), ntuple(x->(zero(T)),N), typemax(T), typemax(T))

numbertype(::Block{N,T}) where {N,T} = T
dims(::Block{N,T}) where {N,T} = N
Base.size(::Block{N,T}) where {N,T} = (N,)

# for speeding up hashing and performance of the priority queue in the branch and bound procedure
const hash_block_seed = UInt === UInt64 ? 0x03f7a7ad5ef46a89 : 0xa9bf8ce0
function hash(B::Block, h::UInt)
    h += hash_block_seed
    h = hash(B.ranges, h)
    return h
end

CoordinateTransformations.AffineMap(bl::Block) = AffineMap(bl.center)

# 
function boxranges(center::Union{Tuple, NTuple, AbstractArray}, widths::Union{Tuple, NTuple, AbstractArray})
    t = eltype(center)
    return NTuple{length(center),Tuple{t,t}}([(center[i]-widths[i], center[i]+widths[i]) for i=1:length(center)])
end

"""
    sbrngs = subranges(ranges, nsplits)

Takes `ranges`, a nested tuple describing intervals for each dimension in rigid-rotation space
defining a hypercube, and splits the hypercube into `nsplits` even components along each dimension.
Since the space is 6-dimensional, the number of returned sub-cubes will be `nsplits^6`.
"""
function subranges(ranges, nsplits::Int)
    ndims = length(ranges)
    t = eltype(eltype(ranges))

    # calculate even splititng points for each dimension
    splitvals = [range(r[1], stop=r[2], length=nsplits+1) |> collect for r in ranges]
    splits = [[(splitvals[i][j], splitvals[i][j+1]) for j=1:nsplits] for i=1:ndims]
    f(x) = splits[x[1]][x[2]]
    children = fill(ranges, nsplits^ndims)
    for (i,I) in enumerate(CartesianIndices(NTuple{ndims,UnitRange{Int}}(fill(1:nsplits, ndims))))
        children[i] = NTuple{ndims,Tuple{t,t}}(map(x->f(x), enumerate(Tuple(I))))
    end
    return children
end

"""
    lim = translation_limit(gmmx, gmmy)

Computes the largest translation needed to ensure that the searchspace contains the best alignment transformation.
"""
function translation_limit(gmmx::AbstractSingleGMM, gmmy::AbstractSingleGMM)
    trlim = typemin(promote_type(numbertype(gmmx),numbertype(gmmy)))
    for gaussians in (gmmx.gaussians, gmmy.gaussians)
        if !isempty(gaussians)
            trlim = max(trlim, maximum(gaussians) do gauss
                    maximum(abs, gauss.??) end)
        end
    end
    return trlim
end

function translation_limit(mgmmx::AbstractMultiGMM, mgmmy::AbstractMultiGMM)
    trlim = typemin(promote_type(numbertype(mgmmx),numbertype(mgmmy)))
    for key in keys(mgmmx.gmms) ??? keys(mgmmy.gmms)
        trlim = max(trlim, translation_limit(mgmmx.gmms[key], mgmmy.gmms[key]))
    end
    return trlim
end

# Block for GOGMA procedure in full transformational space
function fullBlock(gmmx::AbstractGMM, gmmy::AbstractGMM, ranges=nothing, p??=nothing, p??=nothing, rot=nothing, trl=nothing)
    # get center and uncertainty region
    t = promote_type(numbertype(gmmx),numbertype(gmmy))
    if isnothing(ranges)
        trlim = translation_limit(gmmx, gmmy)
        ranges = ((-t(??),t(??)), (-t(??),t(??)), (-t(??),t(??)), (-trlim,trlim), (-trlim,trlim), (-trlim,trlim))
    end
    center = NTuple{length(ranges),t}([sum(dim)/2 for dim in ranges])
    rwidth = ranges[1][2] - ranges[1][1]
    twidth = ranges[4][2] - ranges[4][1]

    # calculate objective function bounds for the block
    lb, ub = get_bounds(gmmx, gmmy, rwidth, twidth, center, p??, p??)

    return Block(ranges, center, lb, ub)
end

# Block for rotational space only (i.e. the first stage of TIV-GOGMA)
function rotBlock(gmmx::AbstractGMM, gmmy::AbstractGMM, ranges=nothing, p??=nothing, p??=nothing, rot=nothing, trl=nothing)
    # get center and uncertainty region
    t = promote_type(numbertype(gmmx),numbertype(gmmy))
    if isnothing(ranges)
        ranges = ((-t(??),t(??)), (-t(??),t(??)), (-t(??),t(??)))
    end
    center = NTuple{length(ranges),t}([sum(dim)/2 for dim in ranges])
    rwidth = ranges[1][2] - ranges[1][1]

    # calculate objective function bounds for the block
    if isnothing(trl)
        trl = (zero(t), zero(t), zero(t))
    end
    lb, ub = get_bounds(gmmx, gmmy, rwidth, zero(t), (center..., trl...), p??, p??)

    return Block(ranges, center, lb, ub)
end

# Block for translational space only (i.e. the second stage of TIV-GOGMA)
function trlBlock(gmmx::AbstractGMM, gmmy::AbstractGMM, ranges=nothing, p??=nothing, p??=nothing, rot=nothing, trl=nothing)
    # get center and uncertainty region
    t = promote_type(numbertype(gmmx),numbertype(gmmy))
    if isnothing(ranges)
        trlim = translation_limit(gmmx, gmmy)
        ranges = ((-trlim,trlim), (-trlim,trlim), (-trlim,trlim))
    end
    center = NTuple{length(ranges),t}([sum(dim)/2 for dim in ranges])
    twidth = ranges[1][2] - ranges[1][1]

    # calculate objective function bounds for the block
    if isnothing(rot)
        rot = (zero(t), zero(t), zero(t))
    end
    lb, ub = get_bounds(gmmx, gmmy, zero(t), twidth, (rot..., center...), p??, p??)

    return Block(ranges, center, lb, ub)
end