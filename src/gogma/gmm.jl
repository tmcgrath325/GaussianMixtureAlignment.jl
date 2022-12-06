import Base: eltype, length, size, getindex, iterate, convert, promote_rule, keys

# Type structure: leaving things open for adding anisotropic Gaussians and GMMs

abstract type AbstractGaussian{N,T} end
abstract type AbstractIsotropicGaussian{N,T} <: AbstractGaussian{N,T} end
    # concrete subtypes:
    #   IsotropicGaussian
    #   AtomGaussian (MolecularGaussians.jl)
    #   FeatureGaussian (MolecularGaussians.jl)

abstract type AbstractGMM{N,T} <: AbstractModel{N,T} end

abstract type AbstractSingleGMM{N,T} <: AbstractGMM{N,T} end
abstract type AbstractIsotropicGMM{N,T} <: AbstractSingleGMM{N,T} end
    # concrete subtypes:
    #   IsotropicGMM
    #   MolGMM (MolecularGaussians.jl)

abstract type AbstractMultiGMM{N,T,K} <: AbstractGMM{N,T} end
abstract type AbstractIsotropicMultiGMM{N,T,K} <: AbstractMultiGMM{N,T,K} end
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
numbertype(::AbstractGMM{N,T}) where {N,T} = T
dims(::AbstractGMM{N,T}) where {N,T} = N

length(gmm::AbstractSingleGMM) = length(gmm.gaussians)
getindex(gmm::AbstractSingleGMM, idx) = gmm.gaussians[idx]
iterate(gmm::AbstractSingleGMM) = iterate(gmm.gaussians)
iterate(gmm::AbstractSingleGMM, i) = iterate(gmm.gaussians, i)
size(gmm::AbstractSingleGMM{N,T}) where {N,T} = (length(gmm.gaussians), N)
size(gmm::AbstractSingleGMM{N,T}, idx::Int) where {N,T} = (length(gmm.gaussians), N)[idx]
eltype(gmm::AbstractSingleGMM) = eltype(gmm.gaussians);

length(mgmm::AbstractMultiGMM) = length(mgmm.gmms)
getindex(mgmm::AbstractMultiGMM, k) = mgmm.gmms[k]
keys(mgmm::AbstractMultiGMM) = keys(mgmm.gmms)
iterate(mgmm::AbstractMultiGMM) = iterate(mgmm.gmms)
iterate(mgmm::AbstractMultiGMM, i) = iterate(mgmm.gmms, i)
size(mgmm::AbstractMultiGMM{N,T,K}) where {N,T,K} = (length(mgmm.gmms), N)
size(mgmm::AbstractMultiGMM{N,T,K}, idx::Int) where {N,T,K} = (length(mgmm.gmms), N)[idx]
eltype(mgmm::AbstractMultiGMM) = eltype(mgmm.gmms);



"""
A structure that defines an isotropic Gaussian distribution with the location of the mean, `μ`, standard deviation `σ`, 
and scaling factor `ϕ`. 

"""
struct IsotropicGaussian{N,T} <: AbstractIsotropicGaussian{N,T}
    μ::SVector{N,T}
    σ::T
    ϕ::T
end
IsotropicGaussian(μ::SVector{N,T},σ::T,ϕ::T) where {N,T<:Real} = IsotropicGaussian{N,T}(μ,σ,ϕ)

function IsotropicGaussian(μ::AbstractArray, σ::Real, ϕ::Real)
    t = promote_type(eltype(μ), typeof(σ), typeof(ϕ))
    return IsotropicGaussian{length(μ),t}(SVector{length(μ),t}(μ), t(σ), t(ϕ))
end

IsotropicGaussian(g::AbstractIsotropicGaussian) = IsotropicGaussian(g.μ, g.σ, g.ϕ)

convert(::Type{IsotropicGaussian{N,T}}, g::AbstractIsotropicGaussian) where {N,T} = IsotropicGaussian(SVector{N,T}(g.μ), T(g.σ), T(g.ϕ))
promote_rule(::Type{IsotropicGaussian{N,T}}, ::Type{IsotropicGaussian{N,S}}) where {N,T<:Real,S<:Real} = IsotropicGaussian{N,promote_type(T,S)} 


"""
A collection of `IsotropicGaussian`s, making up a Gaussian Mixture Model (GMM).
"""
struct IsotropicGMM{N,T} <: AbstractIsotropicGMM{N,T}
    gaussians::Vector{IsotropicGaussian{N,T}}
end

IsotropicGMM(gmm::AbstractIsotropicGMM) = IsotropicGMM(gmm.gaussians)

convert(t::Type{IsotropicGMM}, gmm::AbstractIsotropicGMM) = t(gmm.gaussians)
promote_rule(::Type{IsotropicGMM{N,T}}, ::Type{IsotropicGMM{N,S}}) where {T,S,N} = IsotropicGMM{N,promote_type(T,S)}

"""
A collection of labeled `IsotropicGMM`s, to each be considered separately during an alignment procedure. That is, 
only alignment scores between `IsotropicGMM`s with the same key are considered when aligning two `MultiGMM`s. 
"""
struct IsotropicMultiGMM{N,T,K} <: AbstractIsotropicMultiGMM{N,T,K}
    gmms::Dict{K, <:AbstractIsotropicGMM{N,T}}
end

IsotropicMultiGMM(gmm::AbstractIsotropicMultiGMM) = IsotropicMultiGMM(gmm.gmms)

convert(t::Type{IsotropicMultiGMM}, mgmm::AbstractIsotropicMultiGMM) = t(mgmm.gmms)
promote_rule(::Type{IsotropicMultiGMM{N,T,K}}, ::Type{IsotropicMultiGMM{N,S,K}}) where {N,T,S,K} = IsotropicMultiGMM{N,promote_type(T,S),K}

coords(gmm::AbstractSingleGMM) = hcat([g.μ for g in gmm.gaussians]...)
weights(gmm::AbstractSingleGMM) = [g.ϕ for g in gmm.gaussians]

# descriptive display
# TODO update to display type parameters, make use of supertypes, etc

Base.show(io::IO, g::AbstractIsotropicGaussian) = println(io,
    summary(g),
    " with mean $(g.μ), standard deviation $(g.σ), and amplitude $(g.ϕ).\n"
)

Base.show(io::IO, gmm::AbstractIsotropicGMM) = println(io,
    summary(gmm),
    " with $(length(gmm)) $(eltype(gmm.gaussians)) distributions."
)

Base.show(io::IO, mgmm::AbstractIsotropicMultiGMM) = println(io,
    summary(mgmm),
    " with $(length(mgmm)) labeled $(eltype(mgmm.gmms).parameters[2])s and a total of $(sum([length(gmm) for (key,gmm) in mgmm.gmms])) Gaussians."
)