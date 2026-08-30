## Articulated (flexible) models: a base GMM plus a tree of rotatable joints.
##
## A joint rotates the features "distal" to it about a fixed axis by one angle. Stacking
## `njoints` such angles onto the six rigid parameters gives the articulated search space
## `(R, T, φ₁…φ_K)`. The interface below is what the flexible bounds and search need from a
## model; `ArticulatedGMM` (plain isotropic Gaussians) and `ArticulatedStackedGMM` (stacked
## labeled Gaussians) are the in-package implementations, and external models
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

# Every joint must move features that exist and list only descendants that follow it.
function validate_joints(ngaussians, joints)
    for (b, joint) in enumerate(joints)
        for f in joint.features
            1 <= f <= ngaussians || throw(ArgumentError("joint $b moves feature $f, outside 1:$ngaussians"))
        end
        for c in joint.children
            c > b || throw(ArgumentError("joint $b lists descendant joint $c; joints must precede their descendants (child index > parent index)"))
            c <= length(joints) || throw(ArgumentError("joint $b lists descendant joint $c, outside 1:$(length(joints))"))
        end
    end
    return nothing
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
See [`ArticulatedStackedGMM`](@ref) for the same tree over stacked labeled Gaussians.
"""
struct ArticulatedGMM{N, T} <: AbstractIsotropicGMM{N, T}
    gaussians::Vector{IsotropicGaussian{N, T}}
    joints::Vector{Joint{N, T}}
    function ArticulatedGMM{N, T}(gaussians, joints) where {N, T}
        g = convert(Vector{IsotropicGaussian{N, T}}, gaussians)
        j = convert(Vector{Joint{N, T}}, joints)
        validate_joints(length(g), j)
        return new{N, T}(g, j)
    end
end

ArticulatedGMM(gaussians::AbstractVector{IsotropicGaussian{N, T}}, joints::AbstractVector{Joint{N, T}}) where {N, T} = ArticulatedGMM{N, T}(gaussians, joints)

"""
    ArticulatedStackedGMM(gaussians, joints)

A stacked labeled GMM (see `StackedLabeledIsotropicGMM`) whose Gaussians are organized into a
kinematic tree of rotatable [`Joint`](@ref)s, exactly as [`ArticulatedGMM`](@ref) does for
plain isotropic Gaussians. It is an `AbstractStackedLabeledIsotropicGMM` in its base
conformation, so label-aware overlaps and interaction weights apply unchanged, and it
satisfies the articulated-model interface used by the flexible search.
"""
struct ArticulatedStackedGMM{N, T, L, K} <: AbstractStackedLabeledIsotropicGMM{N, T, L, K}
    gaussians::Vector{StackedLabeledGaussian{N, T, L, K}}
    joints::Vector{Joint{N, T}}
    function ArticulatedStackedGMM{N, T, L, K}(gaussians, joints) where {N, T, L, K}
        g = convert(Vector{StackedLabeledGaussian{N, T, L, K}}, gaussians)
        j = convert(Vector{Joint{N, T}}, joints)
        validate_joints(length(g), j)
        return new{N, T, L, K}(g, j)
    end
end

ArticulatedStackedGMM(gaussians::AbstractVector{StackedLabeledGaussian{N, T, L, K}}, joints::AbstractVector{Joint{N, T}}) where {N, T, L, K} = ArticulatedStackedGMM{N, T, L, K}(gaussians, joints)

const InPackageArticulated = Union{ArticulatedGMM, ArticulatedStackedGMM}

"""
    njoints(model)

Return the number of rotatable joints in an articulated `model`. A model that does not
implement the articulated interface is rigid and has zero joints.
"""
njoints(::AbstractModel) = 0

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

njoints(model::InPackageArticulated) = length(model.joints)
joint_axis(model::InPackageArticulated, b) = model.joints[b].axis
joint_origin(model::InPackageArticulated, b) = model.joints[b].origin
joint_features(model::InPackageArticulated, b) = model.joints[b].features
joint_children(model::InPackageArticulated, b) = model.joints[b].children

"""
    flex(model, φ)

Apply the joint angles `φ` (one per joint, in radians) to an articulated `model`, returning
a new model of the same type in the flexed conformation. Each joint `b` rotates its features
and the frames of its descendant joints about the joint's current axis by `φ[b]`; joints are
applied in stored order, so an ancestor's rotation carries its descendants' axes along before
those are used.

The base (unflexed) model is recovered by `φ = zeros`. A rigid model (`njoints(model) == 0`)
accepts only an empty `φ` and is returned unchanged.
"""
function flex(model::AbstractModel, φ)
    K = njoints(model)
    K == 0 || throw(MethodError(flex, (model, φ)))
    length(φ) == 0 || throw(DimensionMismatch("expected 0 joint angles, got $(length(φ))"))
    return model
end

# Forward kinematics shared by the in-package articulated models: rotate each joint's
# features and descendant frames about its current axis, root outward. Gaussians are moved
# with the rigid-transform arithmetic their type defines (`R * g`, `g + T`), so `gaussians`
# may hold any Gaussian type providing it; its element type must already accommodate the
# angles' number type.
function flex_kinematics(gaussians::AbstractVector, joints::AbstractVector{Joint{N, T}}, φ) where {N, T}
    K = length(joints)
    length(φ) == K || throw(DimensionMismatch("expected $K joint angles, got $(length(φ))"))
    S = promote_type(T, eltype(φ))
    gs = copy(gaussians)
    js = Joint{N, S}[Joint{N, S}(j.axis, j.origin, j.features, j.children) for j in joints]
    for b in 1:K
        j = js[b]
        R = AngleAxis(φ[b], j.axis...)
        o = j.origin
        for f in j.features
            gs[f] = R * (gs[f] - o) + o
        end
        for c in j.children
            child = js[c]
            js[c] = Joint{N, S}(R * child.axis, R * (child.origin - o) + o, child.features, child.children)
        end
    end
    return gs, js
end

function flex(model::ArticulatedGMM{N, T}, φ) where {N, T}
    S = promote_type(T, eltype(φ))
    gaussians = IsotropicGaussian{N, S}[IsotropicGaussian{N, S}(g.μ, g.σ, g.ϕ) for g in model.gaussians]
    gs, js = flex_kinematics(gaussians, model.joints, φ)
    return ArticulatedGMM{N, S}(gs, js)
end

function flex(model::ArticulatedStackedGMM{N, T, L, K}, φ) where {N, T, L, K}
    S = promote_type(T, eltype(φ))
    gaussians = StackedLabeledGaussian{N, S, L, K}[StackedLabeledGaussian{N, S, L, K}(g.μ, g.σ, g.ϕ, g.labels) for g in model.gaussians]
    gs, js = flex_kinematics(gaussians, model.joints, φ)
    return ArticulatedStackedGMM{N, S, L, K}(gs, js)
end

# Rigid transforms carry the joints along with the Gaussians (axes are directions, origins are
# points), keeping the articulated type so a flexed model can be posed with `R * m + T`.
rotate_joints(R, joints::AbstractVector{Joint{N, T}}, ::Type{S}) where {N, T, S} = Joint{N, S}[Joint{N, S}(R * j.axis, R * j.origin, j.features, j.children) for j in joints]
shift_joints(T, joints::AbstractVector{Joint{N, V}}, ::Type{S}) where {N, V, S} = Joint{N, S}[Joint{N, S}(j.axis, j.origin + T, j.features, j.children) for j in joints]

function Base.:*(R::AbstractMatrix{W}, m::ArticulatedGMM{N, V}) where {N, V, W}
    S = promote_type(V, W)
    return ArticulatedGMM{N, S}(IsotropicGaussian{N, S}[R * g for g in m.gaussians], rotate_joints(R, m.joints, S))
end

function Base.:+(m::ArticulatedGMM{N, V}, T::AbstractVector{W}) where {N, V, W}
    S = promote_type(V, W)
    return ArticulatedGMM{N, S}(IsotropicGaussian{N, S}[g + T for g in m.gaussians], shift_joints(T, m.joints, S))
end

function Base.:*(R::AbstractMatrix{W}, m::ArticulatedStackedGMM{N, V, L, K}) where {N, V, L, K, W}
    S = promote_type(V, W)
    return ArticulatedStackedGMM{N, S, L, K}(StackedLabeledGaussian{N, S, L, K}[R * g for g in m.gaussians], rotate_joints(R, m.joints, S))
end

function Base.:+(m::ArticulatedStackedGMM{N, V, L, K}, T::AbstractVector{W}) where {N, V, L, K, W}
    S = promote_type(V, W)
    return ArticulatedStackedGMM{N, S, L, K}(StackedLabeledGaussian{N, S, L, K}[g + T for g in m.gaussians], shift_joints(T, m.joints, S))
end

Base.:-(m::InPackageArticulated, T::AbstractVector) = m + (-T)
