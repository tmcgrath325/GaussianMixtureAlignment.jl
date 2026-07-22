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
