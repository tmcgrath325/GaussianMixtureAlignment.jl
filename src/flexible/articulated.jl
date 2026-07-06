## Articulated (flexible) models: a base GMM plus a tree of rotatable joints.
##
## A joint rotates the features "distal" to it about a fixed axis by one angle. Stacking
## `njoints` such angles onto the six rigid parameters gives the articulated search space
## `(R, T, φ₁…φ_K)`. The interface below is what the flexible bounds and search need from a
## model; `ArticulatedGMM` is the in-package implementation, and external models
## (e.g. MolecularGaussians' `PharmacophoreGMM`) supply their own methods.

"""
    Joint(axis, origin, features, children)

One rotatable degree of freedom of an articulated model. `axis` (a unit direction) and
`origin` (a point on it) define the rotation axis; `features` lists the indices of the base
Gaussians the joint moves and `children` the indices of the joints distal to it.

A model's joints are ordered so that every joint precedes its descendants: each index in
`children` is strictly greater than the joint's own index. Applying the joints in order is
then a valid forward-kinematic sweep from the root outward (see [`flex`](@ref)).
"""
struct Joint{N, T}
    axis::SVector{N, T}
    origin::SVector{N, T}
    features::Vector{Int}
    children::Vector{Int}
    function Joint{N, T}(axis, origin, features, children) where {N, T}
        return new{N, T}(axis, origin, features, children)
    end
end

function Joint(axis::AbstractVector, origin::AbstractVector, features, children)
    n = length(axis)
    length(origin) == n || throw(DimensionMismatch("axis and origin must share length; got $(length(axis)) and $(length(origin))"))
    t = promote_type(eltype(axis), eltype(origin))
    a = SVector{n, t}(axis)
    nrm = norm(a)
    nrm > 0 || throw(ArgumentError("joint axis must be nonzero"))
    return Joint{n, t}(a / nrm, SVector{n, t}(origin), collect(Int, features), collect(Int, children))
end

"""
    ArticulatedGMM(gaussians, joints)

An isotropic GMM whose Gaussians are organized into a kinematic tree of rotatable
[`Joint`](@ref)s. It is an ordinary `AbstractIsotropicGMM` in its base (unflexed)
conformation — every rigid method applies unchanged — and additionally satisfies the
articulated-model interface (`njoints`, `joint_axis`, `joint_origin`, `joint_features`,
`joint_children`, `flex`) used by the flexible search.

`joints` must be ordered with each joint before its descendants (see [`Joint`](@ref)); the
constructor checks this and that every referenced feature and child index is in range.
"""
struct ArticulatedGMM{N, T} <: AbstractIsotropicGMM{N, T}
    gaussians::Vector{IsotropicGaussian{N, T}}
    joints::Vector{Joint{N, T}}
    function ArticulatedGMM{N, T}(gaussians, joints) where {N, T}
        g = convert(Vector{IsotropicGaussian{N, T}}, gaussians)
        j = convert(Vector{Joint{N, T}}, joints)
        ng = length(g)
        for (b, joint) in enumerate(j)
            for f in joint.features
                1 <= f <= ng || throw(ArgumentError("joint $b moves feature $f, outside 1:$ng"))
            end
            for c in joint.children
                c > b || throw(ArgumentError("joint $b lists descendant joint $c; joints must precede their descendants (child index > parent index)"))
                c <= length(j) || throw(ArgumentError("joint $b lists descendant joint $c, outside 1:$(length(j))"))
            end
        end
        return new{N, T}(g, j)
    end
end

ArticulatedGMM(gaussians::AbstractVector{IsotropicGaussian{N, T}}, joints::AbstractVector{Joint{N, T}}) where {N, T} = ArticulatedGMM{N, T}(gaussians, joints)

"""
    njoints(model)

Return the number of rotatable joints in an articulated `model`.
"""
function njoints end

"""
    joint_axis(model, b)

Return the unit direction of the `b`-th joint's rotation axis.
"""
function joint_axis end

"""
    joint_origin(model, b)

Return a point on the `b`-th joint's rotation axis.
"""
function joint_origin end

"""
    joint_features(model, b)

Return the indices of the base Gaussians moved by the `b`-th joint.
"""
function joint_features end

"""
    joint_children(model, b)

Return the indices of the joints distal to the `b`-th joint.
"""
function joint_children end

njoints(model::ArticulatedGMM) = length(model.joints)
joint_axis(model::ArticulatedGMM, b) = model.joints[b].axis
joint_origin(model::ArticulatedGMM, b) = model.joints[b].origin
joint_features(model::ArticulatedGMM, b) = model.joints[b].features
joint_children(model::ArticulatedGMM, b) = model.joints[b].children

"""
    flex(model, φ)

Apply the joint angles `φ` (one per joint, in radians) to an articulated `model`, returning
a new model of the same type in the flexed conformation. Each joint `b` rotates its features
and the frames of its descendant joints about the joint's current axis by `φ[b]`; joints are
applied in stored order, so an ancestor's rotation carries its descendants' axes along before
those are used.

The base (unflexed) model is recovered by `φ = zeros`.
"""
function flex end

function flex(model::ArticulatedGMM{N, T}, φ) where {N, T}
    K = njoints(model)
    length(φ) == K || throw(DimensionMismatch("expected $K joint angles, got $(length(φ))"))
    S = promote_type(T, eltype(φ))
    gaussians = IsotropicGaussian{N, S}[IsotropicGaussian{N, S}(g.μ, g.σ, g.ϕ) for g in model.gaussians]
    joints = Joint{N, S}[Joint{N, S}(j.axis, j.origin, j.features, j.children) for j in model.joints]
    for b in 1:K
        j = joints[b]
        R = AngleAxis(φ[b], j.axis...)
        o = j.origin
        for f in j.features
            g = gaussians[f]
            gaussians[f] = IsotropicGaussian{N, S}(R * (g.μ - o) + o, g.σ, g.ϕ)
        end
        for c in j.children
            child = joints[c]
            joints[c] = Joint{N, S}(R * child.axis, R * (child.origin - o) + o, child.features, child.children)
        end
    end
    return ArticulatedGMM{N, S}(gaussians, joints)
end

# Rigid transforms carry the joints along with the Gaussians (axes are directions, origins are
# points), keeping the `ArticulatedGMM` type so a flexed model can be posed with `R * m + T`.
function Base.:*(R::AbstractMatrix{W}, m::ArticulatedGMM{N, V}) where {N, V, W}
    S = promote_type(V, W)
    gaussians = IsotropicGaussian{N, S}[R * g for g in m.gaussians]
    joints = Joint{N, S}[Joint{N, S}(R * j.axis, R * j.origin, j.features, j.children) for j in m.joints]
    return ArticulatedGMM{N, S}(gaussians, joints)
end

function Base.:+(m::ArticulatedGMM{N, V}, T::AbstractVector{W}) where {N, V, W}
    S = promote_type(V, W)
    gaussians = IsotropicGaussian{N, S}[g + T for g in m.gaussians]
    joints = Joint{N, S}[Joint{N, S}(j.axis, j.origin + T, j.features, j.children) for j in m.joints]
    return ArticulatedGMM{N, S}(gaussians, joints)
end

Base.:-(m::ArticulatedGMM, T::AbstractVector) = m + (-T)
