struct Block{T<:Real, N}
    center::NTuple{N,T}
    rwidth::T
    twidth::T
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
    scntrs = subcenters(center, rwidth, twidth, nsplits)

Takes `centers`, a Tuple describing the center of a hypercube, along with its rotational and translational
widths `rwidth` and `twidth`, and returns the centers of all subcubes after splitting the cube into
`nsplits` equal components along each dimension.
"""
function subcenters(center, rwidth, twidth, nsplits::Int)
    dims = length(center)÷2
    t = eltype(center)
    lowcorner = center .- (ntuple(x->rwidth/2, dims)..., ntuple(x->twidth/2, dims)...)
    # calculate even splititng points for each dimension
    rincr, tincr = t(rwidth/nsplits), t(twidth/nsplits)
    children = Array{NTuple{2*dims,t}}(fill(center, nsplits^(2*dims)))
    for (i,I) in enumerate(CartesianIndices(NTuple{2*dims,UnitRange{Int}}(fill(1:nsplits, 2*dims))))
        rchild = NTuple{dims,t}(rincr.*(Tuple(I)[1:dims].-0.5))
        tchild = NTuple{dims,t}(tincr.*(Tuple(I)[dims+1:2*dims].-0.5))
        children[i] = NTuple{2*dims,t}(lowcorner .+ (rchild..., tchild...))
    end
    return children
end

function Block(gmmx::IsotropicGMM, gmmy::IsotropicGMM, center=nothing, rwidth=nothing, twidth=nothing)
    # get center and uncertainty region
    if isnothing(center)
        t = promote_type(eltype(gmmx),eltype(gmmx))
        dims = size(gmmx,2)
        center = NTuple{2*dims,t}(zeros(t,2*dims))
        rwidth = t(2*π)
        trlim = typemin(t)
        for gaussians in (gmmx.gaussians, gmmy.gaussians)
            if !isempty(gaussians)
                trlim = max(trlim, maximum(gaussians) do gauss
                        maximum(abs, gauss.μ) end)
            end
        end
        twidth = 2*trlim
    end

    # calculate objective function bounds for the block
    lb, ub = get_bounds(gmmx, gmmy, rwidth, twidth, center)

    return Block(center, rwidth, twidth, lb, ub)
end