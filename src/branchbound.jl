abstract type AlignmentResults end

struct GlobalAlignmentResult{D,S,T,N,F<:AbstractAffineMap,X<:AbstractModel{D,S},Y<:AbstractModel{D,T}} <: AlignmentResults
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
    progress::Vector{Tuple{Int,T,NTuple{N,T}}}
    terminated_by::String
end

struct TIVAlignmentResult{D,S,T,N,F<:AbstractAffineMap,X<:AbstractModel{D,S},Y<:AbstractModel{D,T},TD,TN,TF<:AbstractAffineMap,RD,RN,RF<:AbstractAffineMap,RX<:AbstractModel{TD,S},RY<:AbstractModel{TD,T}} <: AlignmentResults
    x::X
    y::Y
    upperbound::T
    lowerbound::T
    tform::F
    tform_params::NTuple{N,T}
    obj_calls::Int
    num_splits::Int
    num_blocks::Int
    rotation_result::GlobalAlignmentResult{RD,S,T,RN,RF,RX,RY}
    translation_result::GlobalAlignmentResult{TD,S,T,TN,TF,X,Y}
end

function lowestlbblock(hull::ChanLowerConvexHull{<:Tuple{T,T,<:SearchRegion}}, lb::T) where T
    lbnode = lowestlbnode(hull)
    (boxlb, boxub, bl) = lbnode.data
    lb = boxlb
    return lbnode, bl, boxlb, boxub, lb
end

function randomblock(hull::ChanLowerConvexHull{<:Tuple{T,T,<:SearchRegion}}, lb::T) where T
    randidx = rand(1:length(hull))
    lbnode = getnode(hull.hull, randidx)
    (boxlb, boxub, bl) = lbnode.data
    if boxlb == lb && !isempty(hull)
        lb = lowestlbnode(hull).data[1]
    end
    return lbnode, bl, boxlb, boxub, lb
end

function lowestlbnode(hull::ChanLowerConvexHull)
    node = PairedLinkedLists.head(hull.hull)
    for n in ListNodeIterator(hull.hull)
        n.data[1] != node.data[1] && break
        node = n
    end
    return node
end

# Keyword arguments:\n
#     nsplits       - an integer representing the number of splits that should be made along each dimension during branching
#     searchspace   - an `UncertaintyRegion` that defines the searchspace, which defaults to the smallest space gauranteed to contain the global minimum
#     R             - a `Rotationvec` containing a rotation position, which is passed to the `blockfun`
#     T             - an `SVector{3}` containing a translation position, which is passed to the `blockfun`
#     centerinputs  - a `Bool` indicating whether to center the input models (at their respective centroids) prior to starting the search
#     blockfun      - the function used for generating `SearchRegion`s that define search subspaces (i.e. UncertaintyRegion, TranslationRegion, RotationRegion)
#     nextblockfun  - the function used for selecting the next "block" to be investigated (i.e. randomblock, lowestlbblock)
#     localfun      - the function used for local alignment
#     boundsfun     - the function used to calculate the bounds on each `SearchRegion`
#     tformfun      - the function used to convert the center of a `SearchRegion` to a rigid transformation (i.e. AffinMap, LinearMap, Translation)
#     atol          - absolute tolerance. Search terminates when the upper bound is within `atol` of the lower bound
#     rtol          - relative tolerance. Search terminates when the upper bound is within `rtol*lb` of the lower bound `lb`
#     maxblocks     - the maximum number of `Block`s that can be held in the priority queue before search termination
#     maxsplits     - the maximum number of `Block` splits that are allowed before search termination
#     maxevals      - the maximum number of objective function evaluations allowed before search termination
#     maxstagnant   - the maximum number of `Block` splits allowed without improvement before search termination
"""
    result = branchbound(x, y; nsplits=2, searchspace=nothing, 
                         rot=nothing, trl=nothing, blockfun=fullBlock, objfun=alignment_objective,
                         rtol=0.01, maxblocks=5e8, maxeva ls=Inf, maxstagnant=Inf, threads=false)

Finds the globally optimal rigid transform for alignment between two isotropic Gaussian mixtures, `x`
and `y`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150).

Returns a `GlobalAlignmentResult` that contains the maximized overlap of the two GMMs (the upperbound on the objective function),
a lower bound on the alignment objective function, an `AffineMap` which aligns `x` with `y`, and information about the
number of evaluations during the alignment procedure. 
""" 
function branchbound(xinput::AbstractModel, yinput::AbstractModel;
                     nsplits=2, searchspace=nothing, blockfun=UncertaintyRegion, R=RotationVec(0.,0.,0.), T=SVector{3}(0.,0.,0.),
                     nextblockfun=lowestlbblock, centerinputs=false, boundsfun=tight_distance_bounds, localfun=local_align, tformfun=AffineMap,
                     atol=0.1, rtol=0, maxblocks=5e8, maxsplits=Inf, maxevals=Inf, maxstagnant=Inf)
    x = xinput
    y = yinput
    if isodd(nsplits)
        throw(ArgumentError("`nsplits` must be even"))
    end
    if dims(x) != dims(y)
        throw(ArgumentError("Dimensionality of the GMMs must be equal"))
    end
    t = promote_type(numbertype(x), numbertype(y))

    centerx_tform = Translation([0,0,0])
    centery_tform = Translation([0,0,0])
    if centerinputs
        centerx_tform = center_translation(x)
        centery_tform = center_translation(y)
        x = centerx_tform(x)
        y = centery_tform(y)
    end

    # initialization
    if isnothing(searchspace)
        searchspace = blockfun(x, y, R, T)
    end
    ndims = length(center(searchspace))
    nsblks = nsplits^ndims
    sblks = fill(searchspace, nsblks)

    lb, centerub = boundsfun(x, y, searchspace)
    hull = ChanLowerConvexHull{Tuple{t,t,typeof(searchspace)}}(CCW, true, x -> (x[1], -x[2]))
    addpoint!(hull, (lb, centerub, searchspace))

    sbnds = fill((lb, centerub), nsblks)
    ub, bestloc = localfun(x, y, searchspace)

    progress = [(0, ub, bestloc)]
    
    # split cubes until convergence
    ndivisions = 0
    sinceimprove = 0
    evalsperdiv = length(x)*length(y)*nsplits^ndims

    while !isempty(hull)
        if (length(hull) > maxblocks) || (ndivisions*evalsperdiv > maxevals) || (sinceimprove > maxstagnant) || (ndivisions > maxsplits)
            break
        end
        ndivisions += 1
        sinceimprove += 1

        # pick the next search region to subdivide
        lbnode, bl, boxlb, boxub, lb = nextblockfun(hull, lb)

        # delete the chosen search region from the convex hull
        subhull = getfirst(x -> x.points===lbnode.target.list, hull.subhulls)
        removepoint!(subhull, lbnode.target)
        deletenode!(lbnode)

        # if the best solution so far is close enough to the best possible solution, end
        if abs((ub - lb)/lb) < rtol || abs(ub-lb) < atol
            tform = tformfun(bestloc)
            if centerinputs
                tform = centerx_tform ∘ tform ∘  inv(centery_tform) 
            end
            return GlobalAlignmentResult(x, y, ub, lb, tform, bestloc, ndivisions*evalsperdiv, ndivisions, length(hull), sinceimprove, progress, "optimum within tolerance")
        end

        # split up the block into `nsplits` smaller blocks across each dimension
        subregions!(sblks, bl, nsplits)
        for i=1:nsblks
            sbnds[i] = boundsfun(x,y,sblks[i])
        end

        # reset the upper bound if appropriate
        minub, ubidx = findmin([sbnd[2] for sbnd in sbnds])
        if minub < centerub
            centerub = minub
            nextub, nextbestloc = localfun(x, y, sblks[ubidx])
            if minub < nextub
                if minub < ub
                    ub, bestloc = minub, center(sblks[ubidx])
                end
            else
                if nextub < ub
                    ub, bestloc = nextub, nextbestloc
                end
            end
            push!(progress, (ndivisions, ub, bestloc))
            sinceimprove = 0
        end

        addblks = eltype(hull)[]
        addbnds = eltype(sbnds)[]
        for i=1:length(sblks)
            diff = abs(sbnds[i][2] - sbnds[i][1])
            if sbnds[i][1] < ub && diff >= atol && abs(diff/sbnds[i][1]) >= rtol
                push!(addblks, (sbnds[i][1], sbnds[i][2], sblks[i]))
                push!(addbnds, sbnds[i])
            end
        end
        if isempty(hull)
            if !isempty(addbnds)
                lb = minimum(addbnds)[1]
            end
        end
        mergepoints!(hull, addblks)
    end
    if isempty(hull)
        tform = tformfun(bestloc)
        if centerinputs
            tform = centerx_tform ∘ tform ∘  inv(centery_tform)
        end
        return GlobalAlignmentResult(x, y, ub, lb, tformfun(bestloc), bestloc, ndivisions*evalsperdiv, ndivisions, length(hull), sinceimprove, progress, "priority queue empty")
    else
        tform = tformfun(bestloc)
        if centerinputs
            tform = centerx_tform ∘ tform ∘  inv(centery_tform)
        end
        return GlobalAlignmentResult(x, y, ub, lowestlbnode(hull).data[1], tformfun(bestloc), bestloc, ndivisions*evalsperdiv, ndivisions, length(hull), sinceimprove, progress, "terminated early")
    end
end


"""
    result = rot_gogma_align(x, y; kwargs...)

Finds the globally optimal rotation for alignment between two isotropic Gaussian mixtures, `x`
and `y`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150), for a given rotation, `rot`.
That is, only rigid translation is allowed.

For details about keyword arguments, see `gogma_align()`.
"""
function rot_branchbound(x::AbstractModel, y::AbstractModel; kwargs...)
    return branchbound(x, y; blockfun=RotationRegion, tformfun=LinearMap, kwargs...)
end

"""
    result = trl_gogma_align(x, y; kwargs...)

Finds the globally optimal translation for alignment between two isotropic Gaussian mixtures, `x`
and `y`, using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150), for a given translation, `trl`.
That is, only rigid rotation is allowed.

For details about keyword arguments, see `gogma_align()`.
"""
function trl_branchbound(x::AbstractModel, y::AbstractModel; kwargs...)
    return branchbound(x, y; blockfun=TranslationRegion, tformfun=Translation, kwargs...)
end



# fit a plane to a set of points, returning the normal vector
function planefit(pts)
    decomp = GenericLinearAlgebra.svd(pts .- sum(pts, dims=2))
    dist, nvecidx = findmin(decomp.S)
    return decomp.U[:, nvecidx], dist
end

planefit(ps::AbstractSinglePointSet) = planefit(ps.coords)
planefit(ps::AbstractSinglePointSet, R) = planefit(R*ps.coords)

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

function tiv_branchbound(   x::AbstractModel, 
                            y::AbstractModel, 
                            tivx::AbstractModel, 
                            tivy::AbstractModel; 
                            boundsfun=tight_distance_bounds, 
                            rot_boundsfun=boundsfun, 
                            trl_boundsfun=boundsfun, 
                            localfun=local_align, 
                            rot_localfun=localfun,
                            trl_localfun=localfun,
                            kwargs...)
    t = promote_type(numbertype(x),numbertype(y))
    p = t(π)
    z = zero(t)
    zeroTranslation = SVector{3}(z,z,z)
    
    rot_res = rot_branchbound(tivx, tivy; localfun=rot_localfun, boundsfun=rot_boundsfun, kwargs...)
    rotblock = RotationRegion(RotationVec(rot_res.tform_params...), zeroTranslation, p)
    rotscore, rotpos = localfun(tivx, tivy, rotblock)

    # spin the moving tivgmm around to check for a better rotation (helps when the Gaussians are largely coplanar)
    R = RotationVec(rot_res.tform_params...)
    spinvec, dist = planefit(tivx, R)
    spinblock = RotationRegion(RotationVec(RotationVec(π*spinvec...) * R), zeroTranslation, z)
    spinscore, spinrotpos = localfun(tivx, tivy, spinblock)
    if spinscore < rotscore
        rotpos = RotationVec(spinrotpos...)
    else
        rotpos = RotationVec(rotpos...)
    end

    # perform translation alignment of original models
    trl_res = trl_branchbound(x, y; R=rotpos, localfun=trl_localfun, boundsfun=trl_boundsfun, kwargs...)
    trlpos = SVector{3}(trl_res.tform_params)

    # perform local alignment in the full transformation space
    trlim = translation_limit(x, y)
    localblock = UncertaintyRegion(rotpos, trlpos, 2*p, trlim)
    min, bestpos = localfun(x, y, localblock)
    if trl_res.upperbound < min
        min = trl_res.upperbound
        bestpos = (rot_res.tform_params...,  trl_res.tform_params...)
    end
 
    return TIVAlignmentResult(x, y, min, trl_res.lowerbound, AffineMap(bestpos), bestpos, 
                              rot_res.obj_calls+trl_res.obj_calls, rot_res.num_splits+trl_res.num_splits,
                              rot_res.num_blocks+trl_res.num_blocks,
                              rot_res, trl_res)
end
