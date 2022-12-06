"""
    tgmm = tivgmm(gmm::IsotropicGMM, c=Inf)
    tgmm = tivgmm(mgmm::MultiGMM, c=Inf)

Returns a new `IsotropicGMM` or `MultiGMM` containing up to `c*length(gmm)` translation invariant vectors (TIVs) connecting Gaussian means in `gmm`.
TIVs are chosen to maximize length multiplied by the weights of the connected distributions. 

See [Li et. al. (2019)](https://arxiv.org/abs/1812.11307) for a description of TIV construction.
"""
function tivgmm(gmm::AbstractIsotropicGMM, c=Inf)
    t = numbertype(gmm)
    npts, ndims = size(gmm)
    n = ceil(c*npts)
    if npts^2 < n
        n = npts^2
    end
    scores = fill(zero(t), npts, npts)
    for i=1:npts
        for j = i:npts
            scores[i,j] = scores[j,i] = norm(gmm.gaussians[i].μ-gmm.gaussians[j].μ) * √(gmm.gaussians[i].ϕ * gmm.gaussians[j].ϕ)
        end
    end
    
    tivgaussians = IsotropicGaussian{ndims,t}[]
    order = sortperm(vec(scores), rev=true)
    for idx in order[1:Int(n)]
        i = Int(floor((idx-1)/npts)+1)
        j = mod(idx-1, npts)+1
        x, y = gmm.gaussians[i], gmm.gaussians[j]
        push!(tivgaussians, IsotropicGaussian(x.μ-y.μ, √(x.σ*y.σ), √(x.ϕ*y.ϕ)))
    end
    return IsotropicGMM(tivgaussians)
end

function tivgmm(mgmm::AbstractIsotropicMultiGMM, c=Inf)
    gmms = Dict{Symbol, IsotropicGMM{dims(mgmm),numbertype(mgmm)}}()
    for key in keys(mgmm.gmms)
        push!(gmms, Pair(key, tivgmm(mgmm.gmms[key], c)))
    end
    return IsotropicMultiGMM(gmms)
end
