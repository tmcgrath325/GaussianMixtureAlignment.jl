## Self-overlap penalty for articulated models.
##
## Without it, the overlap objective rewards folding a flexible model so that its own features
## stack on top of one another wherever the target is dense. The penalty charges the overlap a
## model has with itself, restricted to pairs of features that its joints can move relative to
## each other; pairs on the same rigid fragment keep a constant distance, contribute a constant,
## and are omitted. The penalty is invariant to the rigid pose, so its bounds over a search
## region depend only on the joint intervals.

"""
    SelfOverlap(x; weight=1, interactions=nothing)

Penalty `weight · Σ overlap(|μ_g − μ_h|², s_gh, w_gh)` over the pairs `(g, h)` of features of
the articulated model `x` that lie on different rigid fragments, i.e. that at least one joint
moves relative to each other. Evaluate it at a conformation with [`penalty`](@ref) and bound
it over a box of joint angles with [`penalty_bounds`](@ref).

The pair constants follow the model's own overlap convention: for plain isotropic Gaussians,
`s = σ_g² + σ_h²` and `w = ϕ_g·ϕ_h`; for stacked labeled Gaussians, one term per slot
pairing, weighted by the labels' interaction coefficient (`interactions`, as in
`pairwise_consts`; by default only equal labels interact, so features of different kinds may
occupy the same place without charge).

For each pair the joints on the tree path between `g` and `h` are stored in the order that
runs from `h` toward `g`, which is the order [`penalty_bounds`](@ref) accumulates chords in.
"""
struct SelfOverlap{T, S, W}
    weight::T
    pairs::Vector{Tuple{Int, Int}}
    s::Vector{S}
    w::Vector{W}
    paths::Vector{Vector{Int}}
    function SelfOverlap{T, S, W}(weight, pairs, s, w, paths) where {T, S, W}
        length(pairs) == length(s) == length(w) == length(paths) || throw(DimensionMismatch("per-pair vectors must share length"))
        return new{T, S, W}(weight, pairs, s, w, paths)
    end
end

function SelfOverlap(x; weight = 1, interactions = nothing)
    T = promote_type(numbertype(x), typeof(weight))
    joints_of = feature_joints(x)
    pairs = Tuple{Int, Int}[]
    paths = Vector{Int}[]
    consts = []
    for h in 1:length(x), g in 1:(h - 1)
        jg, jh = joints_of[g], joints_of[h]
        jg == jh && continue
        # the path h → g runs up h's branch to the fragments' common ancestor and down g's
        # branch: joints unique to h in leaf→root order, then joints unique to g in root→leaf
        # order. Shared joints move both features together and drop out.
        gonly = setdiff(jg, jh)
        honly = setdiff(jh, jg)
        push!(pairs, (g, h))
        push!(paths, vcat(reverse(honly), gonly))
        push!(consts, self_pair_consts(x.gaussians[g], x.gaussians[h], interactions))
    end
    s = [c[1] for c in consts]
    w = [c[2] for c in consts]
    S = isempty(s) ? T : typeof(first(s))
    W = isempty(w) ? T : typeof(first(w))
    return SelfOverlap{T, S, W}(T(weight), pairs, convert(Vector{S}, s), convert(Vector{W}, w), paths)
end

# pair constants in the convention of the model's own overlap: scalar for isotropic
# Gaussians, one term per slot pairing for stacked ones
self_pair_consts(g::AbstractIsotropicGaussian, h::AbstractIsotropicGaussian, ::Nothing) = (g.σ^2 + h.σ^2, g.ϕ * h.ϕ)
self_pair_consts(g::AbstractIsotropicGaussian, h::AbstractIsotropicGaussian, interactions) = throw(ArgumentError("interaction weights apply only to labeled (stacked) Gaussians; got $(typeof(g))"))
self_pair_consts(g::StackedLabeledGaussian, h::StackedLabeledGaussian, ::Nothing) = stacked_pair_consts(g, h, nothing)
self_pair_consts(g::StackedLabeledGaussian, h::StackedLabeledGaussian, interactions::Dict) = stacked_pair_consts(g, h, interactions)

Base.length(p::SelfOverlap) = length(p.pairs)

"""
    penalty(p::SelfOverlap, xc)

Evaluate the self-overlap penalty at the conformation `xc` (a flexed model).
"""
function penalty(p::SelfOverlap, xc)
    tot = zero(promote_type(typeof(p.weight), numbertype(xc)))
    for (k, (g, h)) in enumerate(p.pairs)
        d2 = sum(abs2, xc.gaussians[g].μ - xc.gaussians[h].μ)
        tot += overlap(d2, p.s[k], p.w[k])
    end
    return p.weight * tot
end

"""
    lowerbound, upperbound = penalty_bounds(p::SelfOverlap, x, φ, σφ)

Bounds on the self-overlap penalty of `x` as its joint angles range over the box of centers
`φ` and half-widths `σφ`. Over the box the distance between features `g` and `h` stays
within `δ_gh` of its center-conformation value, where `δ_gh` is the chord sum over the joints
on the path between them (evaluated in `g`'s frame, where the joints shared by both features
have no effect). The overlap of an attractive term falls with distance, so its lower bound
is taken at distance `d + δ_gh` (at `max(d − δ_gh, 0)` for a repulsive `w < 0` term); the
upper bound is the value at the center, a feasible conformation.
"""
function penalty_bounds(p::SelfOverlap, x, φ, σφ)
    xc = flex(x, φ)
    S = promote_type(typeof(p.weight), numbertype(xc), eltype(φ), eltype(σφ))
    lb = zero(S)
    ub = zero(S)
    for (k, (g, h)) in enumerate(p.pairs)
        μg, μh = xc.gaussians[g].μ, xc.gaussians[h].μ
        d = norm(μh - μg)
        δ = chord_sum(xc, μh, p.paths[k], σφ, S)
        lb, ub = (lb, ub) .+ penalty_term_bounds(d, δ, p.s[k], p.w[k])
    end
    return p.weight * lb, p.weight * ub
end

function penalty_term_bounds(d, δ, s::Real, w::Real)
    dlb = w < 0 ? max(d - δ, zero(d)) : d + δ
    return overlap(dlb^2, s, w), overlap(d^2, s, w)
end

function penalty_term_bounds(d, δ, s::AbstractVector, w::AbstractVector)
    lb = ub = zero(promote_type(eltype(s), eltype(w), typeof(d)))
    for k in eachindex(s, w)
        l, u = penalty_term_bounds(d, δ, s[k], w[k])
        lb += l
        ub += u
    end
    return lb, ub
end

# a `nothing` penalty contributes nothing
penalty(::Nothing, xc) = false
penalty_bounds(::Nothing, x, φ, σφ) = (false, false)
