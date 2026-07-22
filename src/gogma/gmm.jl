import Base: eltype, keytype, valtype, length, size, getindex, iterate, convert, promote_rule,
    keys, values, push!, pop!, empty!, haskey, get, get!, delete!

# Type structure: leaving things open for adding anisotropic Gaussians and GMMs

"""
Abstract base type for `N`-dimensional Gaussian distributions with numeric type `T`.
Subtypes include `AbstractIsotropicGaussian` and its concrete implementations such as
`IsotropicGaussian`. Extended by external packages (e.g., MolecularGaussians.jl) to add
anisotropic or feature-labeled Gaussians.
"""
abstract type AbstractGaussian{N, T} end
abstract type AbstractIsotropicGaussian{N, T} <: AbstractGaussian{N, T} end
# concrete subtypes:
#   IsotropicGaussian
#   AtomGaussian (MolecularGaussians.jl)
#   FeatureGaussian (MolecularGaussians.jl)

"""
Abstract base type for Gaussian Mixture Models in `N` dimensions with numeric type `T`.
The hierarchy includes `AbstractSingleGMM` (one component set) and `AbstractMultiGMM`
(multiple labeled component sets). All subtypes are also `AbstractModel{N,T}`.
"""
abstract type AbstractGMM{N, T} <: AbstractModel{N, T} end

abstract type AbstractSingleGMM{N, T} <: AbstractGMM{N, T} end
abstract type AbstractIsotropicGMM{N, T} <: AbstractSingleGMM{N, T} end
# concrete subtypes:
#   IsotropicGMM
#   MolGMM (MolecularGaussians.jl)

"""
Abstract base type for single isotropic GMMs whose Gaussians each carry a label of type `K`.
Overlap between two such GMMs is restricted to Gaussian pairs whose labels interact, as
specified by an `interactions` dictionary. Concrete implementation: `LabeledIsotropicGMM`.
"""
abstract type AbstractLabeledIsotropicGMM{N, T, K} <: AbstractIsotropicGMM{N, T} end
# concrete subtypes:
#   LabeledIsotropicGMM

"""
Abstract base type for single isotropic GMMs whose Gaussians each stack `L` labeled features
on a shared mean, with slot-wise widths, amplitudes, and labels (see
`StackedLabeledGaussian`). Concrete implementation: `StackedLabeledIsotropicGMM`.
"""
abstract type AbstractStackedLabeledIsotropicGMM{N, T, L, K} <: AbstractIsotropicGMM{N, T} end
# concrete subtypes:
#   StackedLabeledIsotropicGMM

abstract type AbstractMultiGMM{N, T, K} <: AbstractGMM{N, T} end
abstract type AbstractIsotropicMultiGMM{N, T, K} <: AbstractMultiGMM{N, T, K} end
# concrete subtypes:
#   IsotropicMultiGMM
#   FeatureMolGMM (MolecularGaussians.jl)


# # Base methods for Gaussians
# numbertype(::AbstractGaussian{N,T}) where {N,T} = T
# dims(::AbstractGaussian{N,T}) where {N,T} = N
# length(::AbstractGaussian{N,T}) where {N,T} = N
# size(::AbstractGaussian{N,T}) where {N,T} = (N,)
# size(::AbstractGaussian{N,T}, idx::Int) where {N,T} = (N,)[idx]

# Base methods for GMMs
numbertype(::AbstractGMM{N, T}) where {N, T} = T
dims(::AbstractGMM{N, T}) where {N, T} = N

length(gmm::AbstractSingleGMM) = length(gmm.gaussians)
getindex(gmm::AbstractSingleGMM, idx) = gmm.gaussians[idx]
iterate(gmm::AbstractSingleGMM) = iterate(gmm.gaussians)
iterate(gmm::AbstractSingleGMM, i) = iterate(gmm.gaussians, i)
size(gmm::AbstractSingleGMM{N, T}) where {N, T} = (length(gmm.gaussians), N)
size(gmm::AbstractSingleGMM{N, T}, idx::Int) where {N, T} = (length(gmm.gaussians), N)[idx]
eltype(gmm::AbstractSingleGMM) = eltype(gmm.gaussians)
push!(gmm::AbstractSingleGMM, g::AbstractGaussian) = push!(gmm.gaussians, g)
pop!(gmm::AbstractSingleGMM) = pop!(gmm.gaussians)
empty!(gmm::AbstractSingleGMM) = empty!(gmm.gaussians)

coords(gmm::AbstractSingleGMM) = reduce(hcat, [g.μ for g in gmm.gaussians])
weights(gmm::AbstractSingleGMM) = [g.ϕ for g in gmm.gaussians]
widths(gmm::AbstractSingleGMM) = [g.σ for g in gmm.gaussians]

length(mgmm::AbstractMultiGMM) = length(mgmm.gmms)
getindex(mgmm::AbstractMultiGMM, k) = mgmm.gmms[k]
keys(mgmm::AbstractMultiGMM) = keys(mgmm.gmms)
iterate(mgmm::AbstractMultiGMM) = iterate(mgmm.gmms)
iterate(mgmm::AbstractMultiGMM, i) = iterate(mgmm.gmms, i)
size(mgmm::AbstractMultiGMM{N, T, K}) where {N, T, K} = (length(mgmm.gmms), N)
size(mgmm::AbstractMultiGMM{N, T, K}, idx::Int) where {N, T, K} = (length(mgmm.gmms), N)[idx]
eltype(mgmm::AbstractMultiGMM) = eltype(mgmm.gmms)
eltype(::Type{MGMM}) where {MGMM <: AbstractMultiGMM} = Pair{keytype(MGMM), valtype(MGMM)}
keytype(mgmm::AbstractMultiGMM) = keytype(typeof(mgmm))
keytype(::Type{<:AbstractMultiGMM{N, T, K}}) where {N, T, K} = K
valtype(mgmm::AbstractMultiGMM) = valtype(mgmm.gmms)
haskey(mgmm::AbstractMultiGMM, k) = haskey(mgmm.gmms, k)
get(mgmm::AbstractMultiGMM, k, default) = get(mgmm.gmms, k, default)
get!(::Type{V}, mgmm::AbstractMultiGMM, k) where {V} = get!(V, mgmm.gmms, k)
delete!(mgmm::AbstractMultiGMM, k) = delete!(mgmm.gmms, k)
empty!(mgmm::AbstractMultiGMM) = empty!(mgmm.gmms)

coords(mgmm::AbstractMultiGMM) = reduce(hcat, [coords(gmm) for (k, gmm) in mgmm.gmms])
weights(mgmm::AbstractMultiGMM) = reduce(vcat, [weights(gmm) for (k, gmm) in mgmm.gmms])
widths(mgmm::AbstractMultiGMM) = reduce(vcat, [widths(gmm) for (k, gmm) in mgmm.gmms])

"""
    IsotropicGaussian(μ, σ, ϕ)

Isotropic Gaussian distribution in `N` dimensions with mean `μ`, standard deviation `σ`,
and scaling factor `ϕ`.
"""
struct IsotropicGaussian{N, T} <: AbstractIsotropicGaussian{N, T}
    μ::SVector{N, T}
    σ::T
    ϕ::T
end
IsotropicGaussian(μ::SVector{N, T}, σ::T, ϕ::T) where {N, T <: Real} = IsotropicGaussian{N, T}(μ, σ, ϕ)

function IsotropicGaussian(μ::AbstractArray, σ::Real, ϕ::Real)
    t = promote_type(eltype(μ), typeof(σ), typeof(ϕ))
    return IsotropicGaussian{length(μ), t}(SVector{length(μ), t}(μ), t(σ), t(ϕ))
end

IsotropicGaussian(g::AbstractIsotropicGaussian) = IsotropicGaussian(g.μ, g.σ, g.ϕ)

convert(::Type{IsotropicGaussian{N, T}}, g::AbstractIsotropicGaussian) where {N, T} = IsotropicGaussian{N, T}(g.μ, g.σ, g.ϕ)
promote_rule(::Type{IsotropicGaussian{N, T}}, ::Type{IsotropicGaussian{N, S}}) where {N, T <: Real, S <: Real} = IsotropicGaussian{N, promote_type(T, S)}

(g::IsotropicGaussian)(pos::AbstractVector) = exp(-sum(abs2, pos - g.μ) / (2 * g.σ^2)) * g.ϕ

"""
    IsotropicGMM(gaussians)

Gaussian Mixture Model in `N` dimensions, consisting of a vector of `IsotropicGaussian{N,T}`
components stored in the `.gaussians` field.
"""
struct IsotropicGMM{N, T} <: AbstractIsotropicGMM{N, T}
    gaussians::Vector{IsotropicGaussian{N, T}}
end

IsotropicGMM(gmm::AbstractIsotropicGMM) = IsotropicGMM(gmm.gaussians)
IsotropicGMM{N, T}() where {N, T} = IsotropicGMM{N, T}(IsotropicGaussian{N, T}[])

convert(::Type{GMM}, gmm::AbstractIsotropicGMM) where {GMM <: IsotropicGMM} = GMM(gmm.gaussians)
promote_rule(::Type{IsotropicGMM{N, T}}, ::Type{IsotropicGMM{N, S}}) where {T, S, N} = IsotropicGMM{N, promote_type(T, S)}
eltype(::Type{IsotropicGMM{N, T}}) where {N, T} = IsotropicGaussian{N, T}

(gmm::IsotropicGMM)(pos::AbstractVector) = sum(g(pos) for g in gmm)

"""
    LabeledIsotropicGMM(gaussians, labels)

Gaussian Mixture Model in `N` dimensions pairing a vector of `IsotropicGaussian{N,T}`
components (in `.gaussians`) with a vector of per-Gaussian labels of type `K` (in `.labels`).
The two vectors must have equal length. Unlike `IsotropicMultiGMM`, which groups Gaussians
into keyed sub-GMMs, every Gaussian carries its own label; an `interactions` dictionary then
selects which label pairs contribute to overlap (see [`overlap`](@ref) and `pairwise_consts`).
"""
struct LabeledIsotropicGMM{N, T, K} <: AbstractLabeledIsotropicGMM{N, T, K}
    gaussians::Vector{IsotropicGaussian{N, T}}
    labels::Vector{K}
    function LabeledIsotropicGMM{N, T, K}(gaussians, labels) where {N, T, K}
        length(gaussians) == length(labels) ||
            throw(DimensionMismatch("number of Gaussians ($(length(gaussians))) must match number of labels ($(length(labels)))"))
        return new{N, T, K}(gaussians, labels)
    end
end

LabeledIsotropicGMM(gaussians::AbstractVector{IsotropicGaussian{N, T}}, labels::AbstractVector{K}) where {N, T, K} = LabeledIsotropicGMM{N, T, K}(gaussians, labels)
LabeledIsotropicGMM(gmm::AbstractLabeledIsotropicGMM) = LabeledIsotropicGMM(gmm.gaussians, gmm.labels)
LabeledIsotropicGMM{N, T, K}() where {N, T, K} = LabeledIsotropicGMM{N, T, K}(IsotropicGaussian{N, T}[], K[])

convert(::Type{GMM}, gmm::AbstractLabeledIsotropicGMM) where {GMM <: LabeledIsotropicGMM} = GMM(gmm.gaussians, gmm.labels)
promote_rule(::Type{LabeledIsotropicGMM{N, T, K}}, ::Type{LabeledIsotropicGMM{N, S, K}}) where {N, T, S, K} = LabeledIsotropicGMM{N, promote_type(T, S), K}
eltype(::Type{LabeledIsotropicGMM{N, T, K}}) where {N, T, K} = IsotropicGaussian{N, T}

(gmm::LabeledIsotropicGMM)(pos::AbstractVector) = sum(g(pos) for g in gmm)

"""
    TIVGMM(gaussians, headσ, headϕ, headlabels, tailσ, tailϕ, taillabels)

Translation-invariant-vector model built from a `LabeledIsotropicGMM` by [`tivgmm`](@ref).
Each Gaussian's mean is the difference vector `μ_head - μ_tail` between two features of the
source model; the widths, weights, and labels of both endpoint features are kept in the
parallel `head*`/`tail*` vectors so that a TIV pair can be scored as the sum of a head-head
and a tail-tail feature overlap (see `tiv_pairwise_consts`). Endpoint positions are not
stored: they are not translation invariant, so they would go stale under the rigid
transformations applied during rotational search.

The `σ` and `ϕ` stored on the Gaussians themselves are the geometric means of the endpoint
values, used only where a TIV must be treated as a single feature (e.g. the generic
`pairwise_consts` fallback).
"""
struct TIVGMM{N, T, K} <: AbstractIsotropicGMM{N, T}
    gaussians::Vector{IsotropicGaussian{N, T}}
    headσ::Vector{T}
    headϕ::Vector{T}
    headlabels::Vector{K}
    tailσ::Vector{T}
    tailϕ::Vector{T}
    taillabels::Vector{K}
    function TIVGMM{N, T, K}(gaussians, headσ, headϕ, headlabels, tailσ, tailϕ, taillabels) where {N, T, K}
        n = length(gaussians)
        for v in (headσ, headϕ, headlabels, tailσ, tailϕ, taillabels)
            length(v) == n ||
                throw(DimensionMismatch("each endpoint vector must match the number of Gaussians ($n); got length $(length(v))"))
        end
        return new{N, T, K}(gaussians, headσ, headϕ, headlabels, tailσ, tailϕ, taillabels)
    end
end

function TIVGMM(
        gaussians::AbstractVector{IsotropicGaussian{N, T}}, headσ, headϕ, headlabels::AbstractVector{K},
        tailσ, tailϕ, taillabels::AbstractVector{K}
    ) where {N, T, K}
    return TIVGMM{N, T, K}(gaussians, headσ, headϕ, headlabels, tailσ, tailϕ, taillabels)
end

eltype(::Type{TIVGMM{N, T, K}}) where {N, T, K} = IsotropicGaussian{N, T}

"""
    StackedLabeledGaussian(μ, σ, ϕ, labels)

`L` isotropic Gaussian features stacked on a single point: one mean `μ` shared by all slots,
with slot-wise widths `σ`, amplitudes `ϕ`, and labels stored as length-`L` `SVector`s. Slot
`k` represents an isotropic Gaussian with mean `μ`, width `σ[k]`, amplitude `ϕ[k]`, and
label `labels[k]`.

Slots with `ϕ = 0` contribute nothing to overlaps, so a point with fewer than `L` features
is padded with amplitude-zero slots. A padded slot's width is arbitrary but must be positive
(conventionally `σ = 1`), and its label may repeat any valid label.
"""
struct StackedLabeledGaussian{N, T, L, K} <: AbstractIsotropicGaussian{N, T}
    μ::SVector{N, T}
    σ::SVector{L, T}
    ϕ::SVector{L, T}
    labels::SVector{L, K}
    function StackedLabeledGaussian{N, T, L, K}(μ, σ, ϕ, labels) where {N, T, L, K}
        g = new{N, T, L, K}(μ, σ, ϕ, labels)
        all(>(0), g.σ) || throw(ArgumentError("all slot widths must be positive, including padded slots (pad with σ = 1, ϕ = 0); got σ = $(g.σ)"))
        return g
    end
end

function StackedLabeledGaussian(μ::AbstractVector, σ::AbstractVector, ϕ::AbstractVector, labels::AbstractVector)
    length(σ) == length(ϕ) == length(labels) ||
        throw(DimensionMismatch("σ, ϕ, and labels must have equal lengths; got $(length(σ)), $(length(ϕ)), and $(length(labels))"))
    t = promote_type(eltype(μ), eltype(σ), eltype(ϕ))
    return StackedLabeledGaussian{length(μ), t, length(σ), eltype(labels)}(μ, σ, ϕ, labels)
end

StackedLabeledGaussian(g::StackedLabeledGaussian) = StackedLabeledGaussian(g.μ, g.σ, g.ϕ, g.labels)
StackedLabeledGaussian(g::AbstractIsotropicGaussian, label) =
    StackedLabeledGaussian(g.μ, SVector(g.σ), SVector(g.ϕ), SVector(label))

convert(::Type{StackedLabeledGaussian{N, T, L, K}}, g::StackedLabeledGaussian) where {N, T, L, K} =
    StackedLabeledGaussian{N, T, L, K}(g.μ, g.σ, g.ϕ, g.labels)
promote_rule(::Type{StackedLabeledGaussian{N, T, L, K}}, ::Type{StackedLabeledGaussian{N, S, L, K}}) where {N, T, S, L, K} =
    StackedLabeledGaussian{N, promote_type(T, S), L, K}

function (g::StackedLabeledGaussian)(pos::AbstractVector)
    distsq = sum(abs2, pos - g.μ)
    return sum(g.ϕ[k] * exp(-distsq / (2 * g.σ[k]^2)) for k in eachindex(g.σ, g.ϕ))
end

"""
    StackedLabeledIsotropicGMM(gaussians)
    StackedLabeledIsotropicGMM(μs, σs, ϕs, labelss; padσ = 1)
    StackedLabeledIsotropicGMM(gmm::AbstractLabeledIsotropicGMM)

Gaussian Mixture Model in `N` dimensions whose components are
`StackedLabeledGaussian{N,T,L,K}`s, each stacking `L` labeled features on a single mean.
The stacking degree `L` is uniform across the model — enforced by the concrete element type
— which keeps the pairwise constants type-stable; points with fewer features are padded with
amplitude-zero slots.

Compared to a `LabeledIsotropicGMM` that repeats a mean once per feature, aligning stacked
models computes distances and distance bounds once per pair of means rather than once per
pair of features, while producing identical overlaps and bounds (see [`stackedgmm`](@ref)).

The four-vector form builds the model from per-point feature data: point `i` has mean
`μs[i]` and features with widths `σs[i]`, amplitudes `ϕs[i]`, and labels `labelss[i]`. `L`
is the maximum feature count; points with fewer features are padded with amplitude-zero
slots of width `padσ`, repeating the point's first label. Every point must have at least one
feature.

Constructing from an `AbstractLabeledIsotropicGMM` lifts each labeled Gaussian to a
single-slot (`L = 1`) stacked Gaussian without any grouping of equal means.
"""
struct StackedLabeledIsotropicGMM{N, T, L, K} <: AbstractStackedLabeledIsotropicGMM{N, T, L, K}
    gaussians::Vector{StackedLabeledGaussian{N, T, L, K}}
end

StackedLabeledIsotropicGMM(gmm::AbstractStackedLabeledIsotropicGMM) = StackedLabeledIsotropicGMM(gmm.gaussians)
StackedLabeledIsotropicGMM{N, T, L, K}() where {N, T, L, K} = StackedLabeledIsotropicGMM{N, T, L, K}(StackedLabeledGaussian{N, T, L, K}[])

function StackedLabeledIsotropicGMM(gmm::AbstractLabeledIsotropicGMM{N, T, K}) where {N, T, K}
    gaussians = [
        StackedLabeledGaussian{N, T, 1, K}(g.μ, SVector(g.σ), SVector(g.ϕ), SVector(l))
            for (g, l) in zip(gmm.gaussians, gmm.labels)
    ]
    return StackedLabeledIsotropicGMM{N, T, 1, K}(gaussians)
end

function StackedLabeledIsotropicGMM(μs::AbstractVector, σs::AbstractVector, ϕs::AbstractVector, labelss::AbstractVector; padσ = 1)
    axes(μs) == axes(σs) == axes(ϕs) == axes(labelss) ||
        throw(DimensionMismatch("μs, σs, ϕs, and labelss must share axes; got $(axes(μs)), $(axes(σs)), $(axes(ϕs)), and $(axes(labelss))"))
    isempty(μs) && throw(ArgumentError("at least one point is required"))
    for i in eachindex(σs, ϕs, labelss)
        length(σs[i]) == length(ϕs[i]) == length(labelss[i]) ||
            throw(DimensionMismatch("point $i has mismatched feature counts: $(length(σs[i])) widths, $(length(ϕs[i])) amplitudes, and $(length(labelss[i])) labels"))
        isempty(σs[i]) && throw(ArgumentError("point $i has no features; every point needs at least one to supply a label for padded slots"))
    end
    N = length(first(μs))
    T = promote_type(
        mapreduce(eltype, promote_type, μs), mapreduce(eltype, promote_type, σs),
        mapreduce(eltype, promote_type, ϕs), typeof(padσ)
    )
    K = mapreduce(eltype, promote_type, labelss)
    L = maximum(length, σs)
    gaussians = map(eachindex(μs, σs, ϕs, labelss)) do i
        npad = L - length(σs[i])
        σ = [σs[i]; fill(padσ, npad)]
        ϕ = [ϕs[i]; zeros(npad)]
        labels = [labelss[i]; fill(first(labelss[i]), npad)]
        StackedLabeledGaussian{N, T, L, K}(μs[i], σ, ϕ, labels)
    end
    return StackedLabeledIsotropicGMM{N, T, L, K}(gaussians)
end

convert(::Type{GMM}, gmm::AbstractStackedLabeledIsotropicGMM) where {GMM <: StackedLabeledIsotropicGMM} = GMM(gmm.gaussians)
promote_rule(::Type{StackedLabeledIsotropicGMM{N, T, L, K}}, ::Type{StackedLabeledIsotropicGMM{N, S, L, K}}) where {N, T, S, L, K} =
    StackedLabeledIsotropicGMM{N, promote_type(T, S), L, K}
eltype(::Type{StackedLabeledIsotropicGMM{N, T, L, K}}) where {N, T, L, K} = StackedLabeledGaussian{N, T, L, K}

(gmm::StackedLabeledIsotropicGMM)(pos::AbstractVector) = sum(g(pos) for g in gmm)

weights(gmm::AbstractStackedLabeledIsotropicGMM) = [sum(g.ϕ) for g in gmm.gaussians]
# amplitude-weighted RMS width: reproduces the stack's total second moment Σₖ ϕₖσₖ² in the
# `weights .* widths .^ 2` products consumed by `inertial_transforms`
function widths(gmm::AbstractStackedLabeledIsotropicGMM{N, T}) where {N, T}
    z = zero(sqrt(one(T)))
    return map(gmm.gaussians) do g
        ϕtot = sum(g.ϕ)
        iszero(ϕtot) ? z : sqrt(sum(g.ϕ .* g.σ .^ 2) / ϕtot)
    end
end

"""
    sgmm = stackedgmm(gmm::AbstractLabeledIsotropicGMM)

Convert a labeled GMM to a [`StackedLabeledIsotropicGMM`](@ref) by grouping Gaussians with
exactly equal means into stacked components. The stacking degree is the largest group size;
smaller groups are padded with amplitude-zero slots. Groups are ordered by first appearance,
and features within a group keep their order in `gmm`.

The result's stacking degree depends on run-time values, so this function is not type
stable; the `*_align` entry points act as function barriers, so this does not affect
alignment performance.
"""
function stackedgmm(gmm::AbstractLabeledIsotropicGMM{N, T, K}) where {N, T, K}
    idxof = Dict{SVector{N, T}, Int}()
    groups = Vector{Int}[]
    for i in eachindex(gmm.gaussians)
        gi = get!(idxof, gmm.gaussians[i].μ) do
            push!(groups, Int[])
            length(groups)
        end
        push!(groups[gi], i)
    end
    μs = [gmm.gaussians[first(group)].μ for group in groups]
    σs = [[gmm.gaussians[i].σ for i in group] for group in groups]
    ϕs = [[gmm.gaussians[i].ϕ for i in group] for group in groups]
    labelss = [[gmm.labels[i] for i in group] for group in groups]
    return StackedLabeledIsotropicGMM(μs, σs, ϕs, labelss; padσ = one(T))
end

"""
    StackedTIVGMM(gaussians, headσ, headϕ, headlabels, tailσ, tailϕ, taillabels)

Translation-invariant-vector model built from a `StackedLabeledIsotropicGMM` by
[`tivgmm`](@ref). Each Gaussian's mean is the difference vector between two stacked points
of the source model; the slot-wise widths, amplitudes, and labels of both endpoint stacks
are kept in the parallel `head*`/`tail*` vectors so that a TIV pair can be scored as the sum
of head-head and tail-tail feature overlaps over all pairings of endpoint slots (see
`tiv_pairwise_consts`). Endpoint positions are not stored: they are not translation
invariant, so they would go stale under the rigid transformations applied during rotational
search.

The `σ` and `ϕ` stored on the Gaussians themselves are geometric means of aggregate endpoint
values (total amplitude and amplitude-weighted RMS width), used only where a TIV must be
treated as a single unlabeled feature.
"""
struct StackedTIVGMM{N, T, L, K} <: AbstractIsotropicGMM{N, T}
    gaussians::Vector{IsotropicGaussian{N, T}}
    headσ::Vector{SVector{L, T}}
    headϕ::Vector{SVector{L, T}}
    headlabels::Vector{SVector{L, K}}
    tailσ::Vector{SVector{L, T}}
    tailϕ::Vector{SVector{L, T}}
    taillabels::Vector{SVector{L, K}}
    function StackedTIVGMM{N, T, L, K}(gaussians, headσ, headϕ, headlabels, tailσ, tailϕ, taillabels) where {N, T, L, K}
        n = length(gaussians)
        for v in (headσ, headϕ, headlabels, tailσ, tailϕ, taillabels)
            length(v) == n ||
                throw(DimensionMismatch("each endpoint vector must match the number of Gaussians ($n); got length $(length(v))"))
        end
        return new{N, T, L, K}(gaussians, headσ, headϕ, headlabels, tailσ, tailϕ, taillabels)
    end
end

function StackedTIVGMM(
        gaussians::AbstractVector{IsotropicGaussian{N, T}}, headσ, headϕ, headlabels::AbstractVector{SVector{L, K}},
        tailσ, tailϕ, taillabels::AbstractVector{SVector{L, K}}
    ) where {N, T, L, K}
    return StackedTIVGMM{N, T, L, K}(gaussians, headσ, headϕ, headlabels, tailσ, tailϕ, taillabels)
end

StackedTIVGMM(tiv::TIVGMM{N, T, K}) where {N, T, K} = StackedTIVGMM{N, T, 1, K}(
    tiv.gaussians, SVector.(tiv.headσ), SVector.(tiv.headϕ), SVector.(tiv.headlabels),
    SVector.(tiv.tailσ), SVector.(tiv.tailϕ), SVector.(tiv.taillabels)
)

eltype(::Type{StackedTIVGMM{N, T, L, K}}) where {N, T, L, K} = IsotropicGaussian{N, T}

"""
    IsotropicMultiGMM(gmms)

A keyed collection of `IsotropicGMM`s, each considered separately during alignment.
Only overlap scores between `IsotropicGMM`s sharing the same key contribute when aligning
two `IsotropicMultiGMM`s. `gmms` is a `Dict{K, IsotropicGMM{N,T}}`.
"""
struct IsotropicMultiGMM{N, T, K} <: AbstractIsotropicMultiGMM{N, T, K}
    gmms::Dict{K, IsotropicGMM{N, T}}
end

IsotropicMultiGMM(gmm::AbstractIsotropicMultiGMM) = IsotropicMultiGMM(gmm.gmms)

convert(t::Type{IsotropicMultiGMM}, mgmm::AbstractIsotropicMultiGMM) = t(mgmm.gmms)
promote_rule(::Type{IsotropicMultiGMM{N, T, K}}, ::Type{IsotropicMultiGMM{N, S, K}}) where {N, T, S, K} = IsotropicMultiGMM{N, promote_type(T, S), K}
valtype(::Type{IsotropicMultiGMM{N, T, K}}) where {N, T, K} = IsotropicGMM{N, T}

# descriptive display
# TODO update to display type parameters, make use of supertypes, etc

Base.show(io::IO, g::AbstractIsotropicGaussian) = println(
    io,
    summary(g),
    " with μ = $(g.μ), σ = $(g.σ), and ϕ = $(g.ϕ).\n"

)

Base.show(io::IO, g::StackedLabeledGaussian) = println(
    io,
    summary(g),
    " with μ = $(g.μ), σ = $(g.σ), ϕ = $(g.ϕ), and labels = $(g.labels).\n"
)

Base.show(io::IO, gmm::AbstractSingleGMM) = println(
    io,
    summary(gmm),
    " with $(length(gmm)) $(eltype(gmm.gaussians)) distributions."
)

Base.show(io::IO, mgmm::AbstractMultiGMM) = println(
    io,
    summary(mgmm),
    " with $(length(mgmm)) labeled $(valtype(mgmm)) models made up of a total of $(sum([length(gmm) for (key, gmm) in mgmm.gmms])) $(valtype(mgmm)) distributions."
)
