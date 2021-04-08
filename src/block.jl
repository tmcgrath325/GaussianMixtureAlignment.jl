struct Block{T<:Real, N}
    ranges::NTuple{N, Tuple{T,T}}
    center::NTuple{N,T}
    lowerbound::T
    upperbound::T
end
Block{T, N}() where T where N = Block{T, N}(ntuple(x->(zero(T),zero(T)),N), ntuple(x->(zero(T)),N), typemax(T), typemax(T))
Base.size(blk::Block{T,N}) where T where N = N

const hash_block_seed = UInt === UInt64 ? 0x03f7a7ad5ef46a89 : 0xa9bf8ce0
function hash(B::Block, h::UInt)
    h += hash_block_seed
    h = hash(B.ranges, h)
    return h
end

function boxranges(center::Union{Tuple, NTuple, AbstractArray}, widths::Union{Tuple, NTuple, AbstractArray})
    t = eltype(center)
    return NTuple{length(center),Tuple{t,t}}([(center[i]-widths[i], center[i]+widths[i]) for i=1:length(center)])
end

"""
    sbrngs = subranges(ranges, nsplits)

Takes `ranges`, a nested array describing intervals for each dimension in rigid-rotation space
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

function translation_limit(gmmx, gmmy)
    trlim = typemin(promote_type(eltype(gmmx),eltype(gmmy)))
    for gaussians in (gmmx.gaussians, gmmy.gaussians)
        if !isempty(gaussians)
            trlim = max(trlim, maximum(gaussians) do gauss
                    maximum(abs, gauss.μ) end)
        end
    end
    return trlim
end

function fullBlock(gmmx::IsotropicGMM, gmmy::IsotropicGMM, ranges=nothing, pσ=nothing, pϕ=nothing, rot=nothing, trl=nothing)
    # get center and uncertainty region
    t = promote_type(eltype(gmmx),eltype(gmmy))
    if isnothing(ranges)
        trlim = translation_limit(gmmx, gmmy)
        pie = t(π)
        ranges = ((-pie,pie), (-pie,pie), (-pie,pie), (-trlim,trlim), (-trlim,trlim), (-trlim,trlim))
    end
    center = NTuple{length(ranges),t}([sum(dim)/2 for dim in ranges])
    rwidth = ranges[1][2] - ranges[1][1]
    twidth = ranges[4][2] - ranges[4][1]

    # calculate objective function bounds for the block
    lb, ub = get_bounds(gmmx, gmmy, rwidth, twidth, center, pσ, pϕ)

    return Block(ranges, center, lb, ub)
end

function rotBlock(gmmx::IsotropicGMM, gmmy::IsotropicGMM, ranges=nothing, pσ=nothing, pϕ=nothing, rot=nothing, trl=nothing)
    # get center and uncertainty region
    t = promote_type(eltype(gmmx),eltype(gmmy))
    if isnothing(ranges)
        pie = t(π)
        ranges = ((-pie,pie), (-pie,pie), (-pie,pie))
    end
    center = NTuple{length(ranges),t}([sum(dim)/2 for dim in ranges])
    rwidth = ranges[1][2] - ranges[1][1]

    # calculate objective function bounds for the block
    zro = zero(t)
    if isnothing(trl)
        trl = (zro, zro, zro)
    end
    lb, ub = get_bounds(gmmx, gmmy, rwidth, zro, (center..., trl...), pσ, pϕ)

    return Block(ranges, center, lb, ub)
end

function trlBlock(gmmx::IsotropicGMM, gmmy::IsotropicGMM, ranges=nothing, pσ=nothing, pϕ=nothing, rot=nothing, trl=nothing)
    # get center and uncertainty region
    t = promote_type(eltype(gmmx),eltype(gmmy))
    if isnothing(ranges)
        trlim = translation_limit(gmmx, gmmy)
        ranges = ((-trlim,trlim), (-trlim,trlim), (-trlim,trlim))
    end
    center = NTuple{length(ranges),t}([sum(dim)/2 for dim in ranges])
    twidth = ranges[1][2] - ranges[1][1]

    # calculate objective function bounds for the block
    zro = zero(t)
    if isnothing(rot)
        rot = (zro, zro, zro)
    end
    zro = zero(t)
    lb, ub = get_bounds(gmmx, gmmy, zro, twidth, (rot..., center...), pσ, pϕ)

    return Block(ranges, center, lb, ub)
end