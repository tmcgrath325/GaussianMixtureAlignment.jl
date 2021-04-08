function tivgmm(gmm::IsotropicGMM, n)
    t = eltype(gmm)
    npts, ndims = size(gmm)
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
    for idx in order[1:n]
        i = Int(floor((idx-1)/npts)+1)
        j = mod(idx-1, npts)+1
        x, y = gmm.gaussians[i], gmm.gaussians[j]
        push!(tivgaussians, IsotropicGaussian(x.μ-y.μ, √(x.σ*y.σ), √(x.ϕ*y.ϕ)))
    end
    return IsotropicGMM(tivgaussians)
end

# fit a plane to a set of points, returning the normal vector
function planefit(pts)
    decomp = svd(pts .- sum(pts, dims=2))
    dist, nvecidx = findmin(decomp.S)
    return decomp.U[:, nvecidx], dist
end

function tiv_branch_bound(gmmx::IsotropicGMM, gmmy::IsotropicGMM, nx=length(gmmx), ny=length(gmmy), nsplits=2;
                          rtol=0.05, maxblocks=5e8, maxevals=Inf, maxstagnant=Inf, threads=false) #, szc=0.25)
    # align TIVs
    t = promote_type(eltype(gmmx),eltype(gmmy))
    pie = t(π)
    tivgmmx, tivgmmy = tivgmm(gmmx, nx), tivgmm(gmmy, ny)
    rotatn = rot_branch_bound(tivgmmx, tivgmmy, nsplits,
                           rtol=rtol, maxblocks=maxblocks, maxevals=maxevals, maxstagnant=maxstagnant, threads=threads)
    rotblock = Block(((-pie,pie), (-pie,pie), (-pie,pie)), rotatn[3], zero(t), zero(t))
    rotscore, rotpos = local_align(tivgmmx, tivgmmy, rotblock, objfun=rot_alignment_objective)

    # spin the moving tivgmm around to check for a better rotation
    R = rotmat(rotatn[3]...)
    ptsmat = fill(zero(t), 3, length(tivgmmx))
    for (i,gauss) in enumerate(tivgmmx.gaussians)
        ptsmat[:,i] = gauss.μ
    end
    spinvec, dist = planefit(R * ptsmat)
    spinblock = Block(((-pie,pie), (-pie,pie), (-pie,pie)), rotmat_to_params(rotmat(π*spinvec...) * R), zero(t), zero(t))
    spinscore, spinrotvec = local_align(tivgmmx, tivgmmy, spinblock, objfun=rot_alignment_objective)
    if spinscore < rotscore
        rotpos = spinrotvec
    end

    # perform translation alignment
    transl = trl_branch_bound(gmmx, gmmy, nsplits, rotpos,
                              rtol=rtol, maxblocks=maxblocks, maxevals=maxevals, maxstagnant=maxstagnant, threads=threads)

    # perform local alignment in the full transformation space
    pos = NTuple{6, t}([rotpos..., transl[3]...])
    trlim = translation_limit(gmmx, gmmy)
    localblock = Block(((-pie,pie), (-pie,pie), (-pie,pie), (-trlim,trlim), (-trlim,trlim), (-trlim,trlim)), pos, zero(t), zero(t))
    localopt = local_align(gmmx, gmmy, localblock)
   
    # figs = []
    # push!(figs, draw3d([gmmx, gmmy], sizecoef=szc))
    # push!(figs, draw3d([tivgmmx, tivgmmy], sizecoef=szc))
    # push!(figs, draw3d([tivgmmx, tivgmmy], [[rotatn[3]..., 0.,0.,0.], zeros(6)], sizecoef=szc))
    # push!(figs, draw3d([tivgmmx, tivgmmy], [[spinrotvec..., 0.,0.,0.], zeros(6)], sizecoef=szc))
    # push!(figs, draw3d([gmmx, gmmy],[[rotatn[3]..., 0.,0.,0.], zeros(6)], sizecoef=szc))
    # push!(figs, draw3d([gmmx, gmmy],[[spinrotvec..., 0.,0.,0.], zeros(6)], sizecoef=szc))
    # push!(figs, draw3d([gmmx, gmmy],[pos, zeros(6)], sizecoef=szc))
    # push!(figs, draw3d([gmmx, gmmy],[localopt[2], zeros(6)], sizecoef=szc))

    return localopt[1], transl[2], localopt[2], transl[4]+rotatn[4] # , figs
end
