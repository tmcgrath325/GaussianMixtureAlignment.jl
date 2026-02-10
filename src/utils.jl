# centroid of positions in A, weighted by weights in w (assumed to sum to 1)
centroid(A, w=fill(1/size(A,2), size(A,2))) = A*w
function centroid(m::AbstractModel)
    w = weights(m)
    return centroid(coords(m), w / sum(w))
end

# translation moving centroid to origin
center_translation(A, w=fill(1/size(A,2), size(A,2))) = Translation(-centroid(A,w))
center_translation(m::AbstractModel) = Translation(-centroid(m))


# convert between pointsets and GMMs
function IsotropicGMM(ps::AbstractSinglePointSet{N,T}, σs = ones(T, length(ps))) where {N,T}
    μs = ps.coords
    ϕs = ps.weights
    return IsotropicGMM{N,T}([IsotropicGaussian(μs[:,i], σs[i], ϕs[i]) for i=1:length(ps)])
end

PointSet(gmm::AbstractSingleGMM{N,T}) where {N,T} = PointSet{N,T}(coords(gmm), weights(gmm))

MultiPointSet(mgmm::AbstractMultiGMM{N,T,K}) where {N,T,K} = MultiPointSet{N,T,K}(Dict{K,PointSet{N,T}}([k => PointSet{N,T}(coords(gmm), weights(gmm)) for (k,gmm) in mgmm.gmms]...))


"""
    lim = translation_limit(gmmx, gmmy)

Computes the largest translation needed to ensure that the searchspace contains the best alignment transformation.
"""
translation_limit(xs::Vararg{<:AbstractMatrix}) = maximum(map(x -> maximum(abs.(x)), xs))
translation_limit(xs::Vararg{<:AbstractModel}) = translation_limit(coords.(xs)...)

lohi_interval(lo, hi) = lohi(Interval, lo, hi)