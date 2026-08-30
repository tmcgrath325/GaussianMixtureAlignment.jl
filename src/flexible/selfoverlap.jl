## Self-overlap penalty for articulated models.
##
## Without it, the overlap objective rewards folding a flexible model so that its own features
## stack on top of one another wherever the target is dense. The penalty charges the overlap a
## model has with itself, restricted to pairs of features that its joints can move relative to
## each other; pairs on the same rigid fragment keep a constant distance, contribute a constant,
## and are omitted. The penalty is invariant to the rigid pose, so its bounds over a search
## region depend only on the joint intervals.

"""
    SelfOverlap(x; weight=1)

Penalty `weight · Σ overlap(|μ_g − μ_h|², σ_g² + σ_h², ϕ_g·ϕ_h)` over the pairs `(g, h)` of
features of the articulated model `x` that lie on different rigid fragments, i.e. that at
least one joint moves relative to each other. Evaluate it at a conformation with
[`penalty`](@ref) and bound it over a box of joint angles with [`penalty_bounds`](@ref).

For each pair the joints on the tree path between `g` and `h` are stored in the order that
runs from `h` toward `g`, which is the order [`penalty_bounds`](@ref) accumulates chords in.
"""
struct SelfOverlap{T}
    weight::T
    pairs::Vector{Tuple{Int, Int}}
    s::Vector{T}
    w::Vector{T}
    paths::Vector{Vector{Int}}
    function SelfOverlap{T}(weight, pairs, s, w, paths) where {T}
        length(pairs) == length(s) == length(w) == length(paths) || throw(DimensionMismatch("per-pair vectors must share length"))
        return new{T}(weight, pairs, s, w, paths)
    end
end

function SelfOverlap(x; weight = 1)
    T = promote_type(numbertype(x), typeof(weight))
    joints_of = feature_joints(x)
    pairs = Tuple{Int, Int}[]
    s = T[]
    w = T[]
    paths = Vector{Int}[]
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
        gg, gh = x.gaussians[g], x.gaussians[h]
        push!(s, gg.σ^2 + gh.σ^2)
        push!(w, gg.ϕ * gh.ϕ)
    end
    return SelfOverlap{T}(T(weight), pairs, s, w, paths)
end

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
have no effect). The overlap of an attractive pair falls with distance, so its lower bound
is taken at distance `d + δ_gh` (at `max(d − δ_gh, 0)` for a repulsive `w < 0` pair); the
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
        w = p.w[k]
        dlb = w < 0 ? max(d - δ, zero(d)) : d + δ
        lb += overlap(dlb^2, p.s[k], w)
        ub += overlap(d^2, p.s[k], w)
    end
    return p.weight * lb, p.weight * ub
end

# a `nothing` penalty contributes nothing
penalty(::Nothing, xc) = false
penalty_bounds(::Nothing, x, φ, σφ) = (false, false)
