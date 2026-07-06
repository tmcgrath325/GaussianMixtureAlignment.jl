## Flexible (articulated) bounds over a `FlexibleRegion`.
##
## Over the region, the transformed position of feature g lies within δ_g of the
## rigidly-transformed center-conformation point R·c_g + T, where c_g is g's position at the
## block-center joint angles. The reachable set is thus a rigid-uncertainty image of the ball
## B(c_g, δ_g): the rigid lower distance bound on c_g, loosened by δ_g, lower-bounds the true
## distance, while the center distance stays a valid upper bound. Only δ_g is new; the rest
## reuses the rigid `distance_bound_fun` and `overlap`.

"""
    xc, δ = flex_displacements(x, block::FlexibleRegion)

Return the center-conformation model `xc = flex(x, block.φ)` and a vector `δ` of per-feature
body-frame displacement radii: `δ[g]` upper-bounds how far feature `g` can move, in the
model's frame, as the joint angles range over `block`.

`δ[g]` accumulates one chord per joint on `g`'s root-to-feature path. A joint `b` of angular
half-width `σ_b` rotating a point at perpendicular distance `ρ` from its axis moves it by at
most `2·sin(σ_b/2)·ρ`. The radius `ρ` is the center-conformation perpendicular distance from
`g` to the joint's axis, inflated by the displacement the joints distal to `b` (nearer the
feature) can already impart — so the sum is taken from the feature inward, carrying that
displacement toward the root.
"""
function flex_displacements(x, block::FlexibleRegion{T, K}) where {T, K}
    njoints(x) == K || throw(DimensionMismatch("model has $(njoints(x)) joints but region has $K"))
    xc = flex(x, block.φ)
    n = length(xc)
    S = promote_type(numbertype(xc), T)
    # features moved by each joint, gathered per feature in ascending (root→leaf) joint order
    joints_of = [Int[] for _ in 1:n]
    for b in 1:K
        for g in joint_features(x, b)
            push!(joints_of[g], b)
        end
    end
    δ = zeros(S, n)
    for g in 1:n
        μg = xc.gaussians[g].μ
        acc = zero(S)
        # inward sweep: a distal joint's chord inflates the rotation radius of the joints
        # closer to the root, so accumulate from the feature toward the root
        for b in Iterators.reverse(joints_of[g])
            ax = joint_axis(xc, b)
            o = joint_origin(xc, b)
            d = μg - o
            ρ = norm(d - dot(d, ax) * ax) + acc
            acc += 2 * sin(min(block.σφ[b], S(π)) / 2) * ρ
        end
        δ[g] = acc
    end
    return xc, δ
end

"""
    lowerbound, upperbound = flex_gauss_l2_bounds(x, y, block::FlexibleRegion, pσ, pϕ; distance_bound_fun=tight_distance_bounds)

Bounds on the negative-overlap objective between an articulated model `x` and a fixed single
GMM `y` over the flexible search region `block`. `pσ` and `pϕ` are the transform-invariant
pairwise constants from [`pairwise_consts`](@ref)`(x, y)`.

The rigid distance bounds are evaluated at the block-center conformation `flex(x, block.φ)`
and loosened by the per-feature displacement radii of [`flex_displacements`](@ref): the lower
distance bound is reduced by `δ` (increased, for repulsive `w < 0` pairs), while the upper
bound — the distance at the block center, a feasible configuration — is unchanged.
"""
function flex_gauss_l2_bounds(x, y::AbstractSingleGMM, block::FlexibleRegion, pσ, pϕ; distance_bound_fun = tight_distance_bounds)
    xc, δ = flex_displacements(x, block)
    R, Tr, σᵣ, σₜ = block.rigid.R, block.rigid.T, block.rigid.σᵣ, block.rigid.σₜ
    lb = 0.0
    ub = 0.0
    for (i, gx) in enumerate(xc.gaussians)
        for (j, gy) in enumerate(y.gaussians)
            s = pσ[i, j]
            w = pϕ[i, j]
            (lbdist, ubdist) = distance_bound_fun(gx.μ, gy.μ, R, Tr, σᵣ, σₜ, w < 0)
            lbdist = w < 0 ? lbdist + δ[i] : max(lbdist - δ[i], zero(lbdist))
            lb += -overlap(lbdist^2, s, w)
            ub += -overlap(ubdist^2, s, w)
        end
    end
    return lb, ub
end
