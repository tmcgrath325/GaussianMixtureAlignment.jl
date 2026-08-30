## Branch-and-bound alignment over the articulated search space `(R, T, φ₁…φ_K)`.
##
## Mirrors the rigid `branchbound`/`gogma_align` loop, but a search node is a `FlexibleRegion`
## split one coordinate group at a time (so a node has a variable number of children rather
## than the fixed `nsplits^ndims`), and the reported transform is a rigid pose plus joint
## angles rather than a single affine map. The target may itself be articulated: its joint
## angles follow the moving model's in the parameter vector and the search region, and it is
## flexed in place (never rigidly transformed).

"""
    tformedx = flex_pose(params, x)

Apply the articulated transform parameters `params = (sx, sy, sz, tx, ty, tz, φ₁…φ_K, …)` to
the moving model `x`: flex it by the first `K = njoints(x)` joint angles, then apply the rigid
rotation `RotationVec(sx, sy, sz)` and translation `(tx, ty, tz)`. Any further entries belong
to the target (see [`flex_target`](@ref)) and are ignored.
"""
function flex_pose(params, x)
    K = njoints(x)
    R = RotationVec(params[1], params[2], params[3])
    T = SVector{3}(params[4], params[5], params[6])
    φ = ntuple(k -> params[6 + k], K)
    return R * flex(x, φ) + T
end

"""
    flexedy = flex_target(params, x, y)

Flex the target `y` by its joint angles, the `njoints(y)` entries of `params` that follow the
`6 + njoints(x)` entries used by [`flex_pose`](@ref). With no such entries the target is held
in its base conformation; a rigid target is returned unchanged either way.
"""
function flex_target(params, x, y)
    Kx, Ky = njoints(x), njoints(y)
    n = length(params)
    n == 6 + Kx && return flex(y, ntuple(_ -> zero(eltype(params)), Ky))
    n == 6 + Kx + Ky || throw(DimensionMismatch("expected $(6 + Kx) or $(6 + Kx + Ky) parameters, got $n"))
    ψ = ntuple(k -> params[6 + Kx + k], Ky)
    return flex(y, ψ)
end

function flex_overlapobj(params, x, y, args...; penalties = (nothing, nothing))
    xt = flex_pose(params, x)
    yt = flex_target(params, x, y)
    px, py = penalties
    return -overlap(xt, yt, args...) + penalty(px, xt) + penalty(py, yt)
end

"""
    obj, params = flex_local_align(x, y, block::FlexibleRegion, pσ=nothing, pϕ=nothing; autodiff=AutoForwardDiff(), maxevals=100, penalties=(nothing, nothing))

Locally refine the articulated transform within `block` by minimizing the negative overlap
(plus any [`SelfOverlap`](@ref) `penalties`, one per model) over the `6 + njoints(x) +
njoints(y)` parameters, starting from the block center. Returns the objective and the
parameter tuple, mirroring `local_align` for the rigid case.
"""
function flex_local_align(x, y, block::FlexibleRegion, args...; autodiff = AutoForwardDiff(), maxevals = 100, penalties = (nothing, nothing))
    initial_X = [center(block)...]
    f(X) = flex_overlapobj(X, x, y, args...; penalties)
    res = optimize(f, initial_X, LBFGS(), Optim.Options(f_calls_limit = maxevals); autodiff)
    return res.minimum, tuple(res.minimizer...)
end

"""
    FlexibleAlignmentResult

Result of a flexible (articulated) branch-and-bound alignment. `tform` is the rigid pose and
`angles` the joint angles that together align the moving model `x` onto the target `y`, itself
flexed by `target_angles` (empty for a rigid target); `tform_params` concatenates all three.
See [`aligned`](@ref) and [`aligned_target`](@ref) for the two posed models and
`AlignmentResults` for the shared accessor interface.
"""
struct FlexibleAlignmentResult{T, N, K, L, F <: AbstractAffineMap, X, Y <: AbstractModel} <: AlignmentResults
    x::X
    y::Y
    upperbound::T
    lowerbound::T
    tform::F
    angles::NTuple{K, T}
    target_angles::NTuple{L, T}
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

Return the optimal joint angles of the moving model found by a flexible alignment, as a
tuple of length `njoints(result.x)`.
"""
joint_angles(r::FlexibleAlignmentResult) = r.angles

"""
    target_joint_angles(result)

Return the optimal joint angles of the target found by a flexible alignment, as a tuple of
length `njoints(result.y)` (empty for a rigid target).
"""
target_joint_angles(r::FlexibleAlignmentResult) = r.target_angles

"""
    aligned(result)

Return the moving model of a flexible alignment posed by its optimal transform, i.e.
`flex_pose(result.tform_params, result.x)`.
"""
aligned(r::FlexibleAlignmentResult) = flex_pose(r.tform_params, r.x)

"""
    aligned_target(result)

Return the target of a flexible alignment flexed by its optimal joint angles, i.e.
`flex(result.y, target_joint_angles(result))`. A target whose joints were not searched is
returned in its base conformation, and a rigid target is `result.y` itself.
"""
function aligned_target(r::FlexibleAlignmentResult)
    Ky = njoints(r.y)
    ψ = length(r.target_angles) == Ky ? r.target_angles : ntuple(_ -> zero(eltype(r.tform_params)), Ky)
    return flex(r.y, ψ)
end

function Base.show(io::IO, ::MIME"text/plain", r::FlexibleAlignmentResult)
    println(io, "FlexibleAlignmentResult:")
    println(io, "  objective (upper bound): ", r.upperbound)
    println(io, "  lower bound:             ", r.lowerbound)
    println(io, "  converged:               ", converged(r), " (", r.terminated_by, ")")
    println(io, "  splits:                  ", r.num_splits)
    println(io, "  blocks remaining:        ", r.num_blocks)
    println(io, "  joints:                  ", length(r.angles), " (model), ", length(r.target_angles), " (target)")
    println(io, "  rigid transform:         ", r.tform)
    println(io, "  joint angles:            ", r.angles)
    return print(io, "  target joint angles:     ", r.target_angles)
end

function flexible_result(x, y, ub, lb, params, obj_calls, num_splits, num_blocks, stagnant_splits, progress, terminated_by)
    Kx = njoints(x)
    Ky = length(params) - 6 - Kx     # zero when the target was held rigid
    R = RotationVec(params[1], params[2], params[3])
    T = SVector{3}(params[4], params[5], params[6])
    tform = AffineMap(R, T)
    angles = ntuple(k -> params[6 + k], Kx)
    target_angles = ntuple(k -> params[6 + Kx + k], Ky)
    return FlexibleAlignmentResult(x, y, ub, lb, tform, angles, target_angles, Tuple(params), obj_calls, num_splits, num_blocks, stagnant_splits, progress, terminated_by)
end

# largest perpendicular distance of a joint's features from its axis, per joint: the radius
# converting the joint's angular half-width to a displacement
function joint_radii(m)
    T = numbertype(m)
    return ntuple(njoints(m)) do k
        ax = joint_axis(m, k)
        o = joint_origin(m, k)
        r = zero(T)
        for g in joint_features(m, k)
            d = m.gaussians[g].μ - o
            r = max(r, norm(d - dot(d, ax) * ax))
        end
        r
    end
end

# ordering weights for the splitter: convert each group's angular half-width to an approximate
# displacement so the widest group is split first. Correctness is unaffected; only search order.
function flex_split_scales(x, y, flextarget::Bool)
    T = promote_type(numbertype(x), numbertype(y))
    rotscale = maximum((norm(g.μ) for g in x.gaussians); init = one(T))
    jointscales = flextarget ? (joint_radii(x)..., joint_radii(y)...) : joint_radii(x)
    return rotscale, one(T), jointscales
end

"""
    result = flex_branchbound(x, y; boundsfun, localfun, splitfun, flextarget=false, kwargs...)

Branch-and-bound over the articulated search space for aligning `x` onto `y`, either or both
of which may be articulated. `boundsfun(x, y, block)` and `localfun(x, y, block)` mirror their
rigid counterparts; `splitfun(block)` returns the children of a `FlexibleRegion`. Returns a
[`FlexibleAlignmentResult`](@ref).

Which joints are searched is fixed by the search region: one with `njoints(x)` joint
intervals holds an articulated target in its base conformation, one with `njoints(x) +
njoints(y)` searches the target's joints too. The default `searchspace` is built accordingly
from `flextarget`, covers every searched joint's full angular range, and takes its rigid box
from `UncertaintyRegion(x, y)` as the rigid searches do.

Keyword arguments follow `branchbound`: `searchspace`, `atol`, `rtol`, `maxblocks`, `maxsplits`,
`maxevals`, `maxstagnant`.
"""
function flex_branchbound(
        x, y;
        nsplits = 2, searchspace = nothing, flextarget::Bool = false,
        boundsfun = flex_gauss_l2_bounds, localfun = flex_local_align, splitfun,
        atol = 0.1, rtol = 0, maxblocks = 5.0e8, maxsplits = Inf, maxevals = Inf, maxstagnant = Inf
    )
    t = promote_type(numbertype(x), numbertype(y))
    if isnothing(searchspace)
        searchspace = FlexibleRegion(UncertaintyRegion(x, y), njoints(x) + (flextarget ? njoints(y) : 0))
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
    result = flex_gogma_align(x, y; interactions=nothing, selfoverlap=0, selfoverlap_interactions=interactions,
                              autodiff=AutoForwardDiff(), nsplits=2, kwargs...)

Find a globally optimal *flexible* transformation aligning the model `x` onto the target `y`:
a rigid rotation and translation plus one rotation angle per joint of `x` and, if the target
is articulated too and `flextarget = true`, one per joint of `y`. The target is flexed in
place; only `x` is rigidly transformed. By default an articulated target is held in its base
conformation. With no joints searched this reduces to rigid GOGMA alignment.

`selfoverlap > 0` adds a [`SelfOverlap`](@ref) penalty of that weight to each articulated
model, charging the overlap its joints let it acquire with itself. The penalty is in the
units of the objective: with `selfoverlap = 1`, stacking two features costs as much as the
target overlap it would gain. Without it, a model can raise its score by folding onto the
target's dense regions.

`selfoverlap_interactions` sets the label-interaction coefficients used inside the penalty,
independently of the `interactions` that score the models against each other. The two
legitimately differ: a label pair weighted negatively between models (a steric clash term)
must still be charged positively within a model, or folding into self-clash would lower the
penalty instead of raising it.

Returns a [`FlexibleAlignmentResult`](@ref) whose `upperbound` is the penalized objective.
Additional keyword arguments are forwarded to `flex_branchbound` (tolerances and iteration
limits); see `?flex_branchbound`.
"""
function flex_gogma_align(
        x, y; interactions = nothing, selfoverlap = 0, selfoverlap_interactions = interactions,
        flextarget::Bool = false, autodiff = AutoForwardDiff(), nsplits = 2, kwargs...
    )
    pσ, pϕ = pairwise_consts(x, y, interactions)
    selfoverlap >= 0 || throw(ArgumentError("selfoverlap weight must be nonnegative; got $selfoverlap"))
    mkpenalty(m) = (selfoverlap > 0 && njoints(m) > 0) ? SelfOverlap(m; weight = selfoverlap, interactions = selfoverlap_interactions) : nothing
    # a target held rigid has a constant self-overlap, which is left out of the objective
    penalties = (mkpenalty(x), flextarget ? mkpenalty(y) : nothing)
    boundsfun(a, b, block) = flex_gauss_l2_bounds(a, b, block, pσ, pϕ; penalties)
    localfun(a, b, block) = flex_local_align(a, b, block, pσ, pϕ; autodiff, penalties)
    rotscale, trlscale, jointscales = flex_split_scales(x, y, flextarget)
    splitfun(block) = subregions(block, nsplits; rotscale, trlscale, jointscales)
    return flex_branchbound(x, y; nsplits, flextarget, boundsfun, localfun, splitfun, kwargs...)
end
