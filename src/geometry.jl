"""
    com = centroid(gmm)

Returns the center of mass of `gmm`, where its first order moments are equal to 0.
"""
function centroid(positions::AbstractMatrix{<:Real}, weights::AbstractVector{<:Real}=ones(eltype(positions),size(positions,2)))
    normweights = weights / sum(weights)
    return SVector{size(positions,1)}(positions * normweights)
end

centroid(gaussians::AbstractVector{<:AbstractIsotropicGaussian}) = 
    return centroid(hcat([g.μ for g in gaussians]...), [g.ϕ for g in gaussians])

centroid(x::AbstractPointSet) = centroid(x.coords, x.weights)
centroid(gmm::AbstractIsotropicGMM) = centroid(gmm.gaussians)
centroid(mgmm::AbstractMultiGMM) = centroid(collect(Iterators.flatten([gmm.gaussians for (k,gmm) in mgmm])))

