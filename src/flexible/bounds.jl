## Flexible (articulated) bounds over a `FlexibleRegion`.
##
## Over the region, the transformed position of feature g of the moving model lies within δ_g
## of the rigidly-transformed center-conformation point R·c_g + T, where c_g is g's position at
## the block-center joint angles, and feature h of the target lies within ε_h of its own
## center-conformation point d_h (the target is never rigidly transformed). The reachable sets
## are thus a rigid-uncertainty image of the ball B(c_g, δ_g) and the ball B(d_h, ε_h): the
## rigid lower distance bound on (c_g, d_h), loosened by δ_g + ε_h, lower-bounds the true
## distance, while the center distance stays a valid upper bound. Only δ and ε are new; the
## rest reuses the rigid `distance_bound_fun` and `overlap`.

"""
    xc, δ = flex_displacements(x, φ, σφ)
    xc, δ = flex_displacements(x, block::FlexibleRegion)

Return the center-conformation model `xc = flex(x, φ)` and a vector `δ` of per-feature
body-frame displacement radii: `δ[g]` upper-bounds how far feature `g` can move, in the
model's frame, as the joint angles range over the box of centers `φ` and half-widths `σφ`
(one entry per joint of `x`). The `block` form uses the block's joint intervals, which must
all belong to `x`.

`δ[g]` sums one chord per joint on `g`'s root-to-feature path. A joint `b` of angular
half-width `σ_b` rotating a point at perpendicular distance `ρ` from its axis moves it by at
most `2·sin(σ_b/2)·ρ`, and `ρ` is the perpendicular distance from `g`'s center-conformation
position to the joint's center-conformation axis. No term inflates another: see
[`chord_sum`](@ref) for why the plain sum bounds the composed motion.
"""
function flex_displacements(x, φ, σφ)
    K = njoints(x)
    length(φ) == K || throw(DimensionMismatch("model has $K joints but $(length(φ)) angle centers were given"))
    length(σφ) == K || throw(DimensionMismatch("model has $K joints but $(length(σφ)) half-widths were given"))
    xc = flex(x, φ)
    n = length(xc)
    S = promote_type(numbertype(xc), eltype(φ), eltype(σφ))
    joints_of = feature_joints(x)
    δ = zeros(S, n)
    for g in 1:n
        δ[g] = chord_sum(xc, xc.gaussians[g].μ, joints_of[g], σφ, S)
    end
    return xc, δ
end

"""
    joints_of = feature_joints(x)

For each feature of `x`, the indices of the joints that move it, in ascending (root→leaf)
order. Features moved by the same joints lie on the same rigid fragment of the model.
"""
function feature_joints(x)
    joints_of = [Int[] for _ in 1:length(x)]
    for b in 1:njoints(x)
        for g in joint_features(x, b)
            push!(joints_of[g], b)
        end
    end
    return joints_of
end

"""
    chord_sum(xc, μ, path, σφ, S) -> S

Bound on how far the point `μ`, a feature position in the center conformation `xc`, moves
when each joint in `path` turns by up to its half-width in `σφ`: the sum over the path of
`2·sin(σ_b/2)·ρ_b`, with `ρ_b` the perpendicular distance from `μ` to joint `b`'s axis as
`xc` carries it. The order of `path` does not matter.

The plain sum is a bound, with no term inflating another, because the flexed position is a
composition of rotations about fixed axes and the deviation of each from its center value can
be taken against the *center* conformation of the joints beyond it. Writing the composition
`A₁ ∘ ⋯ ∘ Aₘ` and its center `B₁ ∘ ⋯ ∘ Bₘ`, the difference telescopes through the hybrids
`A₁ ∘ ⋯ ∘ A_j ∘ B_{j+1} ∘ ⋯ ∘ Bₘ`; the `j`-th step is `A₁ ∘ ⋯ ∘ A_{j-1}`, an isometry, applied
to `A_j(z) - B_j(z)` with `z` the center position after the joints beyond `j`, and `A_j` and
`B_j` turn about one axis, so that difference is a chord of `z` about it of angle at most
`σ_j`. Its lever arm, the perpendicular distance of `z` to the base-frame axis, equals the
perpendicular distance of the fully flexed center point to the axis as `xc` carries it,
which is what is measured here. The same holds for a pair path through a shared ancestor,
where the joints on one side are inverted: an inverted rotation turns about the same axis
and preserves the distance to it.

Accumulating the chords from the feature inward and inflating each lever arm by the
displacement already accumulated is also a bound, but it compounds as `∏(1 + 2·sin(σ_b/2))`
along the path and at full angular range exceeds the reachable displacement by `3^K` on a
`K`-joint chain.
"""
function chord_sum(xc, μ, path, σφ, ::Type{S}) where {S}
    acc = zero(S)
    for b in path
        ax = joint_axis(xc, b)
        o = joint_origin(xc, b)
        d = μ - o
        ρ = norm(d - dot(d, ax) * ax)
        acc += 2 * sin(min(σφ[b], S(π)) / 2) * ρ
    end
    return acc
end

function flex_displacements(x, block::FlexibleRegion{T, K}) where {T, K}
    njoints(x) == K || throw(DimensionMismatch("model has $(njoints(x)) joints but region has $K"))
    return flex_displacements(x, block.φ, block.σφ)
end

"""
    xφ, xσφ, yφ, yσφ = joint_intervals(x, y, block::FlexibleRegion)

Split the joint intervals of `block` between the moving model `x` (its first `njoints(x)`
entries) and the target `y` (the remaining `njoints(y)`), as static vectors. A block with
only `njoints(x)` intervals holds the target in its base conformation: its angle centers and
half-widths are zero.
"""
function joint_intervals(x, y, block::FlexibleRegion{T, K}) where {T, K}
    Kx, Ky = njoints(x), njoints(y)
    K == Kx || K == Kx + Ky || throw(DimensionMismatch("models have $Kx + $Ky joints but region has $K; expected $Kx (target held rigid) or $(Kx + Ky)"))
    xφ = SVector{Kx, T}(ntuple(k -> block.φ[k], Kx))
    xσφ = SVector{Kx, T}(ntuple(k -> block.σφ[k], Kx))
    if K == Kx
        return xφ, xσφ, zero(SVector{Ky, T}), zero(SVector{Ky, T})
    end
    yφ = SVector{Ky, T}(ntuple(k -> block.φ[Kx + k], Ky))
    yσφ = SVector{Ky, T}(ntuple(k -> block.σφ[Kx + k], Ky))
    return xφ, xσφ, yφ, yσφ
end

"""
    lowerbound, upperbound = flex_gauss_l2_bounds(x, y, block::FlexibleRegion, pσ, pϕ; distance_bound_fun=tight_distance_bounds)

Bounds on the negative-overlap objective between an articulated moving model `x` and a target
GMM `y` over the search region `block`, whose `njoints(x) + njoints(y)` joint intervals are
those of `x` followed by those of `y` (a rigid target has none). `pσ` and `pϕ` are the
transform-invariant pairwise constants from [`pairwise_consts`](@ref)`(x, y)`.

The rigid distance bounds are evaluated between the block-center conformations `flex(x, xφ)`
and `flex(y, yφ)` and loosened by the per-feature displacement radii of
[`flex_displacements`](@ref) for both models: the lower distance bound is reduced by `δ + ε`
(increased, for repulsive `w < 0` pairs), while the upper bound — the distance at the block
center, a feasible configuration — is unchanged.

`penalties = (px, py)` adds the bounds of a [`SelfOverlap`](@ref) penalty on each model
(`nothing` for none); see [`penalty_bounds`](@ref).
"""
function flex_gauss_l2_bounds(
        x, y::AbstractSingleGMM, block::FlexibleRegion, pσ, pϕ;
        distance_bound_fun = tight_distance_bounds, penalties = (nothing, nothing)
    )
    xφ, xσφ, yφ, yσφ = joint_intervals(x, y, block)
    px, py = penalties
    lbpx, ubpx = penalty_bounds(px, x, xφ, xσφ)
    lbpy, ubpy = penalty_bounds(py, y, yφ, yσφ)
    xc, δ = flex_displacements(x, xφ, xσφ)
    yc, ε = flex_displacements(y, yφ, yσφ)
    R, Tr, σᵣ, σₜ = block.rigid.R, block.rigid.T, block.rigid.σᵣ, block.rigid.σₜ
    Base.require_one_based_indexing(xc.gaussians, yc.gaussians, pσ, pϕ)
    lb = 0.0
    ub = 0.0
    for (i, gx) in enumerate(xc.gaussians)
        for (j, gy) in enumerate(yc.gaussians)
            w = pϕ[i, j]
            iszero(w) && continue
            # apply the block-center rigid transform, then bound over the residual box — the
            # same pre-transform the rigid `gauss_l2_bounds` uses
            lb, ub = (lb, ub) .+ flex_pair_bounds(R * gx.μ, gy.μ - Tr, σᵣ, σₜ, δ[i] + ε[j], pσ[i, j], w, distance_bound_fun)
        end
    end
    return lb + lbpx + lbpy, ub + ubpx + ubpy
end

# Rigid distance bounds for one Gaussian pair, loosened on the lower side by `slack` (the
# pair's combined joint displacement), then turned into negative-overlap bounds. The
# multi-term form serves stacked constants, where a pair carries one `(s, w)` per slot
# pairing sharing the same distance; terms of either sign pick the matching distance bound.
function flex_pair_bounds(xμ, yμ, σᵣ, σₜ, slack, s::Real, w::Real, distance_bound_fun)
    (lbdist, ubdist) = distance_bound_fun(xμ, yμ, σᵣ, σₜ, w < 0)
    lbdist = w < 0 ? lbdist + slack : max(lbdist - slack, zero(lbdist))
    return -overlap(lbdist^2, s, w), -overlap(ubdist^2, s, w)
end

function flex_pair_bounds(xμ, yμ, σᵣ, σₜ, slack, s::AbstractVector, w::AbstractVector, distance_bound_fun)
    (lbmin, ubdist) = distance_bound_fun(xμ, yμ, σᵣ, σₜ, false)
    lbmin = max(lbmin - slack, zero(lbmin))
    lbmax = any(wk -> wk < 0, w) ? distance_bound_fun(xμ, yμ, σᵣ, σₜ, true)[1] + slack : lbmin
    lb = ub = zero(promote_type(eltype(s), eltype(w), typeof(ubdist)))
    for k in eachindex(s, w)
        wk = w[k]
        iszero(wk) && continue
        lbdist = wk < 0 ? lbmax : lbmin
        lb -= overlap(lbdist^2, s[k], wk)
        ub -= overlap(ubdist^2, s[k], wk)
    end
    return lb, ub
end
