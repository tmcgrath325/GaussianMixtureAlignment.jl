"""
    AlignmentResults

Abstract supertype for the results returned by the alignment functions.

Every subtype carries the aligned models and the rigid transformation aligning the moving
model onto the fixed one, accessible with `tform`. The branch-and-bound subtypes
`GlobalAlignmentResult` and `TIVAlignmentResult` additionally support the accessors
`upperbound`, `lowerbound`, `obj_calls`, `num_splits`, `num_blocks`, `stagnant_splits`, and
`progress`, and the `converged` predicate.
"""
abstract type AlignmentResults end

"""
    GlobalAlignmentResult

Result of a branch-and-bound search (`branchbound` and the `*_gogma_align`, `*_goicp_align`,
and `*_goih_align` functions).

`tform` aligns the moving model `x` onto the fixed model `y`, with `tform_params` the
parameters generating it. `upperbound` and `lowerbound` bracket the optimal objective, and
`terminated_by` records why the search stopped. See `AlignmentResults` for the shared
accessor interface and `converged` for the convergence predicate.
"""
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

"""
    TIVAlignmentResult

Result of a translation-invariant-vector branch-and-bound alignment, which searches for
rotation and translation in two stages. `rotation_result` and `translation_result` hold the
two underlying `GlobalAlignmentResult`s; the top-level fields aggregate them. `progress`
traces the translation stage under the fixed rotation, ending at the reported optimum. See
`AlignmentResults` for the shared accessor interface.
"""
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
    stagnant_splits::Int
    progress::Vector{Tuple{Int,T,NTuple{N,T}}}
    rotation_result::GlobalAlignmentResult{RD,S,T,RN,RF,RX,RY}
    translation_result::GlobalAlignmentResult{TD,S,T,TN,TF,X,Y}
end

const BranchBoundResult = Union{GlobalAlignmentResult,TIVAlignmentResult}

"""
    tform(result)

Return the `AbstractAffineMap` that aligns the moving model onto the fixed model in an
alignment `result`.
"""
tform(r::AlignmentResults) = r.tform

"""
    upperbound(result)

Return the objective value at the best transformation found by a branch-and-bound search.
"""
upperbound(r::BranchBoundResult) = r.upperbound

"""
    lowerbound(result)

Return the lower bound on the objective at termination of a branch-and-bound search.
"""
lowerbound(r::BranchBoundResult) = r.lowerbound

"""
    obj_calls(result)

Return the number of objective evaluations performed during a branch-and-bound search.
"""
obj_calls(r::BranchBoundResult) = r.obj_calls

"""
    num_splits(result)

Return the number of branch-and-bound subdivisions performed during a search.
"""
num_splits(r::BranchBoundResult) = r.num_splits

"""
    num_blocks(result)

Return the number of search regions left unexplored when a branch-and-bound search stopped.
"""
num_blocks(r::BranchBoundResult) = r.num_blocks

"""
    stagnant_splits(result)

Return the number of subdivisions performed since the last improvement to the best objective
in a branch-and-bound search.
"""
stagnant_splits(r::BranchBoundResult) = r.stagnant_splits

"""
    progress(result)

Return the trace of accepted improvements during a branch-and-bound search, as a vector of
`(split_index, objective, transformation_parameters)` tuples.
"""
progress(r::BranchBoundResult) = r.progress

"""
    converged(result)

Return `true` if an alignment `result` certifies a globally optimal transformation within the
requested tolerance.

A `GlobalAlignmentResult` converges when the branch-and-bound search closes the gap between
its bounds or exhausts the search queue, and does not converge when a split, evaluation,
block, or stagnation limit halts the search first. A `TIVAlignmentResult` converges only if
both its rotation and translation sub-searches converge. A `ROCSAlignmentResult` never
converges in this sense, as ROCS is a local method carrying no global-optimality guarantee.
"""
converged(r::GlobalAlignmentResult) = r.terminated_by in ("optimum within tolerance", "priority queue empty")
converged(r::TIVAlignmentResult) = converged(r.rotation_result) && converged(r.translation_result)

function Base.show(io::IO, ::MIME"text/plain", r::GlobalAlignmentResult)
    println(io, "GlobalAlignmentResult:")
    println(io, "  objective (upper bound): ", r.upperbound)
    println(io, "  lower bound:             ", r.lowerbound)
    println(io, "  converged:               ", converged(r), " (", r.terminated_by, ")")
    println(io, "  splits:                  ", r.num_splits)
    println(io, "  blocks remaining:        ", r.num_blocks)
    println(io, "  objective evaluations:   ", r.obj_calls)
    print(io,   "  transform:               ", r.tform)
end

function Base.show(io::IO, ::MIME"text/plain", r::TIVAlignmentResult)
    println(io, "TIVAlignmentResult:")
    println(io, "  objective (upper bound): ", r.upperbound)
    println(io, "  lower bound:             ", r.lowerbound)
    println(io, "  converged:               ", converged(r))
    println(io, "  splits:                  ", r.num_splits)
    println(io, "  objective evaluations:   ", r.obj_calls)
    print(io,   "  transform:               ", r.tform)
end

function lowestlbblock(hull::ChanLowerConvexHull{<:Tuple{T,T,<:SearchRegion}}, lb::T) where T
    lbnode = lowestlbnode(hull)
    (boxlb, boxub, bl) = lbnode.data
    lb = boxlb
    return lbnode, bl, lb
end

function randomblock(hull::ChanLowerConvexHull{<:Tuple{T,T,<:SearchRegion}}, lb::T) where T
    randidx = rand(1:length(hull))
    lbnode = getnode(hull.hull, randidx)
    (boxlb, boxub, bl) = lbnode.data
    if boxlb == lb && !isempty(hull)
        lb = lowestlbnode(hull).data[1]
    end
    return lbnode, bl, lb
end

function lowestlbnode(hull::ChanLowerConvexHull)
    node = PairedLinkedLists.head(hull.hull)
    for n in ListNodeIterator(hull.hull)
        n.data[1] != node.data[1] && break
        node = n
    end
    return node
end

"""
    result = branchbound(x, y; kwargs...)

Finds the globally optimal rigid transform for alignment between two models, `x` and `y`, using
branch-and-bound search.

Returns a `GlobalAlignmentResult` containing the best transformation found, upper and lower bounds
on the objective, and convergence statistics.

# Keyword arguments

- `nsplits=2`: number of subdivisions per dimension at each branching step (must be even)
- `searchspace=nothing`: initial `UncertaintyRegion`; defaults to the smallest region guaranteed to contain the global optimum
- `initial_rotation=RotationVec(0,0,0)`: initial rotation hint passed to `blockfun`
- `initial_translation=SVector(0,0,0)`: initial translation hint passed to `blockfun`
- `centerinputs=false`: if `true`, center both models at their centroids before searching
- `blockfun=UncertaintyRegion`: constructs `SearchRegion`s for each subspace
- `nextblockfun=lowestlbblock`: selects which block to subdivide next (alternative: `randomblock`)
- `boundsfun=tight_distance_bounds`: computes `(lowerbound, upperbound)` for a `SearchRegion`
- `localfun=local_align`: called as `localfun(x, y, block)` to locally refine the upper bound within
  each candidate block. Supply a custom closure to control local-alignment behaviour, for example to
  use an alternative autodiff backend:
  ```julia
  using ADTypes, FiniteDifferences
  mylocal = (x, y, bl) -> local_align(x, y, bl; autodiff = AutoFiniteDifferences(fdm = central_fdm(5, 1)))
  branchbound(x, y; localfun = mylocal, boundsfun = ...)
  ```
- `tformfun=AffineMap`: converts a search-region centre to a rigid transformation
- `atol=0.1`: terminate when `upperbound - lowerbound < atol`
- `rtol=0`: terminate when `(upperbound - lowerbound) / lowerbound < rtol`
- `maxblocks=5e8`: terminate if the priority queue exceeds this many blocks
- `maxsplits=Inf`: terminate after this many branching steps
- `maxevals=Inf`: terminate after this many objective evaluations
- `maxstagnant=Inf`: terminate after this many splits without improvement
"""
function branchbound(xinput::AbstractModel, yinput::AbstractModel;
                     nsplits=2, searchspace=nothing, blockfun=UncertaintyRegion, initial_rotation=RotationVec(0.,0.,0.), initial_translation=SVector{3}(0.,0.,0.),
                     nextblockfun=lowestlbblock, centerinputs=false, boundsfun=tight_distance_bounds, localfun=local_align, tformfun::TF=AffineMap,
                     atol=0.1, rtol=0, maxblocks=5e8, maxsplits=Inf, maxevals=Inf, maxstagnant=Inf, separatesplit=false) where TF
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
        searchspace = blockfun(x, y, initial_rotation, initial_translation)
    end
    ndims = length(center(searchspace))
    rot_trl_split = separatesplit && typeof(searchspace) <: UncertaintyRegion
    nsblks = rot_trl_split ? nsplits^3 : nsplits^ndims
    sblks = fill(searchspace, nsblks)
    sblks2 = fill(searchspace, rot_trl_split ? nsblks : 0)

    lb, centerub = boundsfun(x, y, searchspace)
    hull = ChanLowerConvexHull{Tuple{t,t,typeof(searchspace)}}(; orientation = CCW, collinear = true, sortedby = x -> (x[1], -x[2]))
    addpoint!(hull, (lb, centerub, searchspace))

    sbnds = fill((lb, centerub), nsblks)
    sbnds2 = fill((lb, centerub), rot_trl_split ? nsblks : 0)
    ub, bestloc = localfun(x, y, searchspace)

    progress = [(0, ub, bestloc)]

    # split cubes until convergence
    ndivisions = 0
    sinceimprove = 0
    evalsperdiv = rot_trl_split ? length(x)*length(y)*2*nsplits^3 : length(x)*length(y)*nsplits^ndims

    while !isempty(hull)
        if (length(hull) > maxblocks) || (ndivisions*evalsperdiv > maxevals) || (sinceimprove > maxstagnant) || (ndivisions > maxsplits)
            break
        end
        ndivisions += 1
        sinceimprove += 1

        # pick the next search region to subdivide
        lbnode, bl, lb = nextblockfun(hull, lb)

        # delete the chosen search region from the convex hull.
        # `hull.subhulls` and the targeted node's `.list` backreference are internal
        # layout of MutableConvexHulls / PairedLinkedLists with no public accessor;
        # this coupling is deliberate and pinned by [compat] (all three packages share
        # an author). `target(node)` is the public accessor for the targeted node.
        subhull = first(x for x in hull.subhulls if x.points===target(lbnode).list)
        removepoint!(subhull, target(lbnode))
        deletenode!(lbnode)

        # if the best solution so far is close enough to the best possible solution, end
        if abs((ub - lb)/lb) < rtol || abs(ub-lb) < atol
            tform = build_tform(tformfun, bestloc)
            if centerinputs
                tform = centerx_tform ∘ tform ∘  inv(centery_tform)
            end
            return GlobalAlignmentResult(x, y, ub, lb, tform, bestloc, ndivisions*evalsperdiv, ndivisions, length(hull), sinceimprove, progress, "optimum within tolerance")
        end

        # split up the block into `nsplits` smaller blocks across each dimension
        if rot_trl_split # split rotation and translation separately
            rot_subregions!(sblks, bl, nsplits)
            for i=1:nsblks
                sbnds[i] = boundsfun(x,y,sblks[i])
            end
            trl_subregions!(sblks2, bl, nsplits)
            for i=1:nsblks
                sbnds2[i] = boundsfun(x,y,sblks2[i])
            end
            if sum(x -> x[1], sbnds) < sum(x -> x[1], sbnds2) # pick whichever maximizes summed lower bounds
                for i=1:nsblks
                    sblks[i] = sblks2[i]
                    sbnds[i] = sbnds2[i]
                end
            end
        else # split rotation and translation simultaneously
            subregions!(sblks, bl, nsplits)
            for i=1:nsblks
                sbnds[i] = boundsfun(x,y,sblks[i])
            end
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
        tform = build_tform(tformfun, bestloc)
        if centerinputs
            tform = centerx_tform ∘ tform ∘  inv(centery_tform)
        end
        return GlobalAlignmentResult(x, y, ub, lb, build_tform(tformfun, bestloc), bestloc, ndivisions*evalsperdiv, ndivisions, length(hull), sinceimprove, progress, "priority queue empty")
    else
        tform = build_tform(tformfun, bestloc)
        if centerinputs
            tform = centerx_tform ∘ tform ∘  inv(centery_tform)
        end
        return GlobalAlignmentResult(x, y, ub, lowestlbnode(hull).data[1], build_tform(tformfun, bestloc), bestloc, ndivisions*evalsperdiv, ndivisions, length(hull), sinceimprove, progress, "terminated early")
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
    decomp = svd(pts .- sum(pts, dims=2))
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
    rotscore, rotpos = rot_localfun(tivx, tivy, rotblock)

    # spin the moving tivgmm around to check for a better rotation (helps when the Gaussians are largely coplanar)
    R = RotationVec(rot_res.tform_params...)
    spinvec, dist = planefit(tivx, R)
    spinblock = RotationRegion(RotationVec(RotationVec(π*spinvec...) * R), zeroTranslation, z)
    spinscore, spinrotpos = rot_localfun(tivx, tivy, spinblock)
    if spinscore <= rotscore
        # TIV scores may be degenerate when the TIV set is coplanar: a 180° spin
        # about the plane normal leaves all pairwise differences unchanged, so both
        # candidates score identically on TIVs. Break the tie on the original points.
        origscore, _ = localfun(x, y, RotationRegion(RotationVec(rotpos...), zeroTranslation, z))
        spinscore_x, _ = localfun(x, y, RotationRegion(RotationVec(spinrotpos...), zeroTranslation, z))
        rotpos = spinscore_x < origscore ? RotationVec(spinrotpos...) : RotationVec(rotpos...)
    else
        rotpos = RotationVec(rotpos...)
    end

    # perform translation alignment of original models
    trl_res = trl_branchbound(x, y; initial_rotation=rotpos, localfun=trl_localfun, boundsfun=trl_boundsfun, kwargs...)
    trlpos = SVector{3}(trl_res.tform_params)

    # perform local alignment in the full transformation space
    trlim = translation_limit(x, y)
    localblock = UncertaintyRegion(rotpos, trlpos, 2*p, trlim)
    min, bestpos = localfun(x, y, localblock)
    if trl_res.upperbound < min
        min = trl_res.upperbound
        bestpos = (rot_res.tform_params...,  trl_res.tform_params...)
    end

    # `progress` traces the translation stage, whose objective is the model overlap that
    # the final result also reports. Each entry pairs its translation with the rotation
    # `rotpos` held fixed during that stage; split indices continue past the rotation stage,
    # and a final entry records the reported optimum.
    rotparams = (t(rotpos.sx), t(rotpos.sy), t(rotpos.sz))
    tiv_progress = [(rot_res.num_splits + s, obj, (rotparams..., trlp...))
                    for (s, obj, trlp) in trl_res.progress]
    push!(tiv_progress, (rot_res.num_splits + trl_res.num_splits, min, bestpos))
    return TIVAlignmentResult(x, y, min, trl_res.lowerbound, build_tform(AffineMap, bestpos), bestpos,
                              rot_res.obj_calls+trl_res.obj_calls, rot_res.num_splits+trl_res.num_splits,
                              rot_res.num_blocks+trl_res.num_blocks,
                              rot_res.stagnant_splits+trl_res.stagnant_splits, tiv_progress,
                              rot_res, trl_res)
end
