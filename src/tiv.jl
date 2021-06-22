"""
    tgmm = tivgmm(gmm, c)

Returns a new GMM containing `c*length(gmm)` translation invariant vectors (TIVs) connecting Gaussian means in `gmm`.
TIVs are chosen to maximize length multiplied by the weights of the connected distributions. 
"""
function tivgmm(gmm::IsotropicGMM, c=Inf)
    t = eltype(gmm)
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
    
    tivgaussians = IsotropicGaussian{t, ndims}[]
    order = sortperm(vec(scores), rev=true)
    for idx in order[1:Int(n)]
        i = Int(floor((idx-1)/npts)+1)
        j = mod(idx-1, npts)+1
        x, y = gmm.gaussians[i], gmm.gaussians[j]
        push!(tivgaussians, IsotropicGaussian(x.μ-y.μ, √(x.σ*y.σ), √(x.ϕ*y.ϕ), x.dirs))
    end
    return IsotropicGMM(tivgaussians)
end

function tivgmm(mgmm::MultiGMM, c=Inf)
    gmms = Dict{Symbol, IsotropicGMM{eltype(mgmm),size(mgmm,2)}}()
    for key in keys(mgmm.gmms)
        push!(gmms, Pair(key, tivgmm(mgmm.gmms[key],c)))
    end
    return MultiGMM(gmms)
end

# fit a plane to a set of points, returning the normal vector
function planefit(pts)
    decomp = svd(pts .- sum(pts, dims=2))
    dist, nvecidx = findmin(decomp.S)
    return decomp.U[:, nvecidx], dist
end

function planefit(gmm::IsotropicGMM, R)
    ptsmat = fill(zero(eltype(gmm)), 3, length(gmm))
    for (i,gauss) in enumerate(gmm.gaussians)
        ptsmat[:,i] = gauss.μ
    end
    return planefit(R * ptsmat)
end

function planefit(mgmm::MultiGMM, R)
    len = sum([length(gmm) for gmm in values(mgmm.gmms)])
    ptsmat = fill(zero(eltype(mgmm)), 3, len)
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
    min, lb, pos, n = tiv_branch_bound(gmmx, gmmy, cx, cy, nsplits=2, rot=nothing;
                                       atol=0.1, rtol=0, maxblocks=5e8, maxevals=Inf, maxstagnant=Inf, threads=false)

Finds the globally optimal translation for alignment between two isotropic Gaussian mixtures, `gmmx`
and `gmmy`, using a modified GOGMA algorithm that performs rotational and translational optimization separately
by making use of translation invariant vectors (TIVs).

`cx` and `cy` are the ratios between number of TIVs used to represent a GMM and the number of Gaussians it contains, 
for `gmmx` and `gmmy` respectively, during rotational alignment. 
`nsplits` is the number of splits performed along each dimension during branching. Returns the overlap
between the GMMs, the lower bound on the overlap, and the transformation vector for the best transformation,
as well as the number of objective evaluations. 
"""
function tiv_branch_bound(gmmx::Union{IsotropicGMM,MultiGMM}, gmmy::Union{IsotropicGMM,MultiGMM}, cx=Inf, cy=Inf, nsplits=2;
                          atol=0.1, rtol=0, maxblocks=5e8, maxevals=Inf, maxstagnant=Inf, threads=false)
    # align TIVs
    t = promote_type(eltype(gmmx),eltype(gmmy))
    pie = t(π)
    tivgmmx, tivgmmy = tivgmm(gmmx, cx), tivgmm(gmmy, cy)
    rotatn = rot_branch_bound(tivgmmx, tivgmmy, nsplits,
                              atol=atol, rtol=rtol, maxblocks=maxblocks, maxevals=maxevals, maxstagnant=maxstagnant, threads=threads)
    rotblock = Block(((-pie,pie), (-pie,pie), (-pie,pie)), rotatn[3], zero(t), zero(t))
    rotscore, rotpos = local_align(tivgmmx, tivgmmy, rotblock, objfun=rot_alignment_objective)
    # spin the moving tivgmm around to check for a better rotation
    R = rotmat(rotatn[3]...)
    spinvec, dist = planefit(tivgmmx, R)
    spinblock = Block(((-pie,pie), (-pie,pie), (-pie,pie)), rotmat_to_params(rotmat(π*spinvec...) * R), zero(t), zero(t))
    spinscore, spinrotpos = local_align(tivgmmx, tivgmmy, spinblock, objfun=rot_alignment_objective)
    if spinscore < rotscore
        rotpos = spinrotpos
    end

    # perform translation alignment
    transl = trl_branch_bound(gmmx, gmmy, nsplits, rotpos,
                              atol=atol, rtol=rtol, maxblocks=maxblocks, maxevals=maxevals, maxstagnant=maxstagnant, threads=threads)

    # perform local alignment in the full transformation space
    pos = NTuple{6, t}([rotpos..., transl[3]...])
    trlim = translation_limit(gmmx, gmmy)
    localblock = Block(((-pie,pie), (-pie,pie), (-pie,pie), (-trlim,trlim), (-trlim,trlim), (-trlim,trlim)), pos, zero(t), zero(t))
    localopt = local_align(gmmx, gmmy, localblock)

    return localopt[1], transl[2], localopt[2], transl[4]+rotatn[4]
end
