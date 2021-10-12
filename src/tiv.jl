struct TIVAlignmentResult{T,D,N,M,F<:AffineMap,G<:AbstractAffineMap,H<:AbstractAffineMap,V<:AbstractGMM{D,T},W<:AbstractGMM{D,T},X<:AbstractGMM{D,T},Y<:AbstractGMM{D,T}} <: AlignmentResults
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
    rotation_result::GMAlignmentResult{T,D,M,G,V,W}
    translation_result::GMAlignmentResult{T,D,M,H,X,Y}
end


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
        for j = i+1:npts
            scores[i,j] = scores[j,i] = norm(gmm.gaussians[i].μ-gmm.gaussians[j].μ) * √(gmm.gaussians[i].ϕ * gmm.gaussians[j].ϕ)
        end
    end
    
    tivgaussians = IsotropicGaussian{ndims,t}[]
    order = sortperm(vec(scores), rev=true)
    for idx in order[1:Int(n)]
        i = Int(floor((idx-1)/npts)+1)
        j = mod(idx-1, npts)+1
        x, y = gmm.gaussians[i], gmm.gaussians[j]
        push!(tivgaussians, IsotropicGaussian(x.μ-y.μ, √(x.σ*y.σ), √(x.ϕ*y.ϕ), x.dirs))
    end
    return IsotropicGMM(tivgaussians)
end

function tivgmm(mgmm::AbstractIsotropicMultiGMM, c=Inf)
    gmms = Dict{Symbol, IsotropicGMM{dims(mgmm),numbertype(mgmm)}}()
    for key in keys(mgmm.gmms)
        push!(gmms, Pair(key, tivgmm(mgmm.gmms[key],c)))
    end
    return IsotropicMultiGMM(gmms)
end

# fit a plane to a set of points, returning the normal vector
function planefit(pts)
    decomp = svd(pts .- sum(pts, dims=2))
    dist, nvecidx = findmin(decomp.S)
    return decomp.U[:, nvecidx], dist
end

function planefit(gmm::AbstractIsotropicGMM, R)
    ptsmat = fill(zero(numbertype(gmm)), 3, length(gmm))
    for (i,gauss) in enumerate(gmm.gaussians)
        ptsmat[:,i] = gauss.μ
    end
    return planefit(R * ptsmat)
end

function planefit(mgmm::AbstractIsotropicMultiGMM, R)
    len = sum([length(gmm) for gmm in values(mgmm.gmms)])
    ptsmat = fill(zero(numbertype(mgmm)), 3, len)
    idx = 1
    for gmm in values(mgmm.gmms)
        for gauss in gmm.gaussians
            ptsmat[:,idx] = gauss.μ
            idx += 1
        end
    end
    return planefit(R * ptsmat)
end

"""
    result = tiv_gogma_align(gmmx, gmmy, cx, cy; kwargs...)

Finds the globally optimal translation for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using a modified GOGMA algorithm that performs rotational and translational optimization separately
by making use of translation invariant vectors (TIVs).

`cx` and `cy` are the ratios between number of TIVs used to represent a GMM and the number of Gaussians it contains, 
for `gmmx` and `gmmy` respectively, during rotational alignment. 

For details about keyword arguments, see `gogma_align()`.
"""
function tiv_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM, cx=Inf, cy=Inf; kwargs...)
    t = promote_type(numbertype(gmmx),numbertype(gmmy))
    p = t(π)
    z = zero(t)
    
    # align TIVs
    tivgmmx, tivgmmy = tivgmm(gmmx, cx), tivgmm(gmmy, cy)
    rot_res = rot_gogma_align(tivgmmx, tivgmmy; kwargs...)
    rotblock = Block(((-p,p), (-p,p), (-p,p)), rot_res.tform_params, z, z)
    rotscore, rotpos = local_align(tivgmmx, tivgmmy, rotblock, objfun=rot_alignment_objective)

    # spin the moving tivgmm around to check for a better rotation (helps when the Gaussians are largely coplanar)
    R = AngleAxis(rot_res.tform_params...)
    spinvec, dist = planefit(tivgmmx, R)
    spinblock = Block(((-p,p), (-p,p), (-p,p)), rot_to_axis(AngleAxis(π*spinvec...) * R), z, z)
    spinscore, spinrotpos = local_align(tivgmmx, tivgmmy, spinblock, objfun=rot_alignment_objective)
    if spinscore < rotscore
        rotpos = spinrotpos
    end

    # perform translation alignment
    trl_res = trl_gogma_align(gmmx, gmmy; rot=rotpos, kwargs...)

    # perform local alignment in the full transformation space
    pos = NTuple{6, t}([rotpos..., trl_res.tform_params...])
    trlim = translation_limit(gmmx, gmmy)
    localblock = Block(((-p,p), (-p,p), (-p,p), (-trlim,trlim), (-trlim,trlim), (-trlim,trlim)), pos, trl_res.upperbound, -Inf)
    min, bestpos = local_align(gmmx, gmmy, localblock)

    return TIVAlignmentResult(gmmx, gmmy, min, trl_res.lowerbound, AffineMap(bestpos...), bestpos, 
                             rot_res.obj_calls+trl_res.obj_calls, rot_res.num_splits+trl_res.num_splits,
                             rot_res.num_blocks+trl_res.num_blocks, rot_res.stagnant_splits+trl_res.stagnant_splits,
                             rot_res, trl_res)
end
