## Branch-and-bound alignment over the articulated search space `(R, T, φ₁…φ_K)`.
##
## Mirrors the rigid `branchbound`/`gogma_align` loop, but a search node is a `FlexibleRegion`
## split one coordinate group at a time (so a node has a variable number of children rather
## than the fixed `nsplits^ndims`), and the reported transform is a rigid pose plus a vector of
## joint angles rather than a single affine map.

"""
    tformedx = flex_pose(params, x)

Apply the articulated transform parameters `params = (sx, sy, sz, tx, ty, tz, φ₁…φ_K)` to model
`x`: flex it by the `K` joint angles, then apply the rigid rotation `RotationVec(sx, sy, sz)`
and translation `(tx, ty, tz)`.
"""
function flex_pose(params, x)
    K = njoints(x)
    R = RotationVec(params[1], params[2], params[3])
    T = SVector{3}(params[4], params[5], params[6])
    φ = ntuple(k -> params[6 + k], K)
    return R * flex(x, φ) + T
end

flex_overlapobj(params, x, y, args...) = -overlap(flex_pose(params, x), y, args...)

"""
    obj, params = flex_local_align(x, y, block::FlexibleRegion, pσ=nothing, pϕ=nothing; autodiff=AutoForwardDiff(), maxevals=100)

Locally refine the articulated transform within `block` by minimizing the negative overlap over
the `6 + K` parameters, starting from the block center. Returns the objective and the parameter
tuple, mirroring `local_align` for the rigid case.
"""
function flex_local_align(x, y, block::FlexibleRegion, args...; autodiff = AutoForwardDiff(), maxevals = 100)
    initial_X = [center(block)...]
    f(X) = flex_overlapobj(X, x, y, args...)
    res = optimize(f, initial_X, LBFGS(), Optim.Options(f_calls_limit = maxevals); autodiff)
    return res.minimum, tuple(res.minimizer...)
end

"""
    FlexibleAlignmentResult

Result of a flexible (articulated) branch-and-bound alignment. `tform` is the rigid pose and
`angles` the joint angles that together align the moving model `x` onto the fixed model `y`;
`tform_params` concatenates them. See [`aligned`](@ref) for the posed model and `AlignmentResults`
for the shared accessor interface.
"""
struct FlexibleAlignmentResult{T, N, K, F <: AbstractAffineMap, X, Y <: AbstractModel} <: AlignmentResults
    x::X
    y::Y
    upperbound::T
    lowerbound::T
    tform::F
    angles::NTuple{K, T}
    tform_params::NTuple{N, T}
    obj_calls::Int
    num_splits::Int
    num_blocks::Int
    stagnant_splits::Int
    progress::Vector{Tuple{Int, T, NTuple{N, T}}}
    terminated_by::String
end

upperbound(r::FlexibleAlignmentResult) = r.upperbound
lowerbound(r::FlexibleAlignmentResult) = r.lowerbound
obj_calls(r::FlexibleAlignmentResult) = r.obj_calls
num_splits(r::FlexibleAlignmentResult) = r.num_splits
num_blocks(r::FlexibleAlignmentResult) = r.num_blocks
stagnant_splits(r::FlexibleAlignmentResult) = r.stagnant_splits
progress(r::FlexibleAlignmentResult) = r.progress
converged(r::FlexibleAlignmentResult) = r.terminated_by in ("optimum within tolerance", "priority queue empty")

"""
    joint_angles(result)

Return the optimal joint angles found by a flexible alignment, as a tuple of length `K`.
"""
joint_angles(r::FlexibleAlignmentResult) = r.angles

"""
    aligned(result)

Return the moving model of a flexible alignment posed by its optimal transform, i.e.
`flex_pose(result.tform_params, result.x)`.
"""
aligned(r::FlexibleAlignmentResult) = flex_pose(r.tform_params, r.x)

function Base.show(io::IO, ::MIME"text/plain", r::FlexibleAlignmentResult)
    println(io, "FlexibleAlignmentResult:")
    println(io, "  objective (upper bound): ", r.upperbound)
    println(io, "  lower bound:             ", r.lowerbound)
    println(io, "  converged:               ", converged(r), " (", r.terminated_by, ")")
    println(io, "  splits:                  ", r.num_splits)
    println(io, "  blocks remaining:        ", r.num_blocks)
    println(io, "  joints:                  ", length(r.angles))
    println(io, "  rigid transform:         ", r.tform)
    return print(io, "  joint angles:            ", r.angles)
end

function flexible_result(x, y, ub, lb, params, obj_calls, num_splits, num_blocks, stagnant_splits, progress, terminated_by)
    K = njoints(x)
    R = RotationVec(params[1], params[2], params[3])
    T = SVector{3}(params[4], params[5], params[6])
    tform = AffineMap(R, T)
    angles = ntuple(k -> params[6 + k], K)
    return FlexibleAlignmentResult(x, y, ub, lb, tform, angles, Tuple(params), obj_calls, num_splits, num_blocks, stagnant_splits, progress, terminated_by)
end

# ordering weights for the splitter: convert each group's angular half-width to an approximate
# displacement so the widest group is split first. Correctness is unaffected; only search order.
function flex_split_scales(x)
    T = numbertype(x)
    rotscale = maximum((norm(g.μ) for g in x.gaussians); init = one(T))
    K = njoints(x)
    jointscales = ntuple(K) do k
        ax = joint_axis(x, k)
        o = joint_origin(x, k)
        r = zero(T)
        for g in joint_features(x, k)
            d = x.gaussians[g].μ - o
            r = max(r, norm(d - dot(d, ax) * ax))
        end
        r
    end
    return rotscale, one(T), jointscales
end

"""
    result = flex_branchbound(x, y; boundsfun, localfun, splitfun, kwargs...)

Branch-and-bound over the articulated search space for aligning `x` onto `y`. `boundsfun(x, y,
block)` and `localfun(x, y, block)` mirror their rigid counterparts; `splitfun(block)` returns
the children of a `FlexibleRegion`. Returns a [`FlexibleAlignmentResult`](@ref).

Keyword arguments follow `branchbound`: `searchspace`, `atol`, `rtol`, `maxblocks`, `maxsplits`,
`maxevals`, `maxstagnant`.
"""
function flex_branchbound(
        x, y;
        nsplits = 2, searchspace = nothing,
        boundsfun = flex_gauss_l2_bounds, localfun = flex_local_align, splitfun,
        atol = 0.1, rtol = 0, maxblocks = 5.0e8, maxsplits = Inf, maxevals = Inf, maxstagnant = Inf
    )
    t = promote_type(numbertype(x), numbertype(y))
    if isnothing(searchspace)
        searchspace = FlexibleRegion(UncertaintyRegion(x, y), njoints(x))
    end

    lb, centerub = boundsfun(x, y, searchspace)
    hull = ChanLowerConvexHull{Tuple{t, t, typeof(searchspace)}}(; orientation = CCW, collinear = true, sortedby = x -> (x[1], -x[2]))
    addpoint!(hull, (lb, centerub, searchspace))
    ub, bestloc = localfun(x, y, searchspace)
    progress = [(0, ub, bestloc)]

    ndivisions = 0
    sinceimprove = 0
    evalsperdiv = length(x) * length(y) * nsplits

    while !isempty(hull)
        if (length(hull) > maxblocks) || (ndivisions * evalsperdiv > maxevals) || (sinceimprove > maxstagnant) || (ndivisions > maxsplits)
            break
        end
        ndivisions += 1
        sinceimprove += 1

        lbnode, bl, lb = lowestlbblock(hull, lb)
        subhull = first(sh for sh in hull.subhulls if sh.points === target(lbnode).list)
        removepoint!(subhull, target(lbnode))
        deletenode!(lbnode)

        if abs((ub - lb) / lb) < rtol || abs(ub - lb) < atol
            return flexible_result(x, y, ub, lb, bestloc, ndivisions * evalsperdiv, ndivisions, length(hull), sinceimprove, progress, "optimum within tolerance")
        end

        children = splitfun(bl)
        sbnds = [boundsfun(x, y, c) for c in children]

        minub, ubidx = findmin([sbnd[2] for sbnd in sbnds])
        if minub < centerub
            centerub = minub
            nextub, nextbestloc = localfun(x, y, children[ubidx])
            if minub < nextub
                if minub < ub
                    ub, bestloc = minub, center(children[ubidx])
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
        for (i, c) in enumerate(children)
            diff = abs(sbnds[i][2] - sbnds[i][1])
            if sbnds[i][1] < ub && diff >= atol && abs(diff / sbnds[i][1]) >= rtol
                push!(addblks, (sbnds[i][1], sbnds[i][2], c))
            end
        end
        if isempty(hull) && !isempty(addblks)
            lb = minimum(b[1] for b in addblks)
        end
        mergepoints!(hull, addblks)
    end

    if isempty(hull)
        return flexible_result(x, y, ub, lb, bestloc, ndivisions * evalsperdiv, ndivisions, length(hull), sinceimprove, progress, "priority queue empty")
    else
        return flexible_result(x, y, ub, lowestlbnode(hull).data[1], bestloc, ndivisions * evalsperdiv, ndivisions, length(hull), sinceimprove, progress, "terminated early")
    end
end

"""
    result = flex_gogma_align(x, y; interactions=nothing, autodiff=AutoForwardDiff(), nsplits=2, kwargs...)

Find a globally optimal *flexible* transformation aligning the articulated model `x` onto the
fixed GMM `y`: a rigid rotation and translation plus one rotation angle per joint of `x`. With a
model carrying no joints this reduces to rigid GOGMA alignment.

Returns a [`FlexibleAlignmentResult`](@ref). Additional keyword arguments are forwarded to
`flex_branchbound` (tolerances and iteration limits); see `?flex_branchbound`.
"""
function flex_gogma_align(x, y; interactions = nothing, autodiff = AutoForwardDiff(), nsplits = 2, kwargs...)
    pσ, pϕ = pairwise_consts(x, y, interactions)
    boundsfun(a, b, block) = flex_gauss_l2_bounds(a, b, block, pσ, pϕ)
    localfun(a, b, block) = flex_local_align(a, b, block, pσ, pϕ; autodiff)
    rotscale, trlscale, jointscales = flex_split_scales(x)
    splitfun(block) = subregions(block, nsplits; rotscale, trlscale, jointscales)
    return flex_branchbound(x, y; nsplits, boundsfun, localfun, splitfun, kwargs...)
end
