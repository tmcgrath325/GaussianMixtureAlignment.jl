"""
    tset = tivpointset(ps::PointSet, c=Inf)
    tset = tivpointset(mps::MultiPointSet, c=Inf)

Returns a new `PointSet` or `MultiPointSet` containing up to `c*length(gmm)` translation invariant vectors (TIVs) connecting point coordinates in the input model.
TIVs are chosen to maximize length multiplied by the weights of the connected distributions. 

See [Li et. al. (2019)](https://arxiv.org/abs/1812.11307) for a description of TIV construction.
"""
function tivpointset(x::AbstractSinglePointSet, c=Inf)
    t = numbertype(x)
    npts = length(x)
    n = ceil(c)*npts
    if npts^2 - npts < n
        n = npts^2 - npts
    end
    scores = fill(zero(t), npts, npts)
    for i=1:npts
        for j = i+1:npts
            scores[i,j] = scores[j,i] = norm(x[i].coords - x[j].coords) * √(x[i].weight * x[j].weight)
        end
    end
    
    tivcoords = SVector{3,t}[]
    tivweights = t[]
    order = sortperm(vec(scores), rev=true)
    for idx in order[1:Int(n)]
        i = Int(floor((idx-1)/npts)+1)
        j = mod(idx-1, npts)+1
        push!(tivcoords, x[i].coords - x[j].coords)
        push!(tivweights, √(x[i].weight * x[j].weight))
    end
    return PointSet(hcat(tivcoords...), tivweights)
end

function tivpointset(mps::AbstractMultiPointSet, c=Inf)
    pointsets = Dict{Symbol, PointSet{dims(mps),numbertype(mps)}}()
    for key in keys(mps.pointsets)
        push!(pointsets, Pair(key, tivpointset(mps.pointsets[key], c)))
    end
    return MultiPointSet(pointsets)
end
