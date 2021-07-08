import Base: eltype, length, size, convert, promote_rule

struct IsotropicGaussian{T<:Real,N}
    μ::SVector{N,T}
    σ::T
    ϕ::T
    dirs::Vector{SVector{N,T}}
end
eltype(ig::IsotropicGaussian{T,N}) where T where N = T
length(ig::IsotropicGaussian{T,N}) where T where N = N
size(ig::IsotropicGaussian{T,N}) where T where N = (N,)
size(gmm::IsotropicGaussian{T,N}, idx::Int) where T where N = (N,)[idx]

function IsotropicGaussian(μ::AbstractArray, σ::Real, ϕ::Real, dirs::AbstractArray=SVector{length(μ),eltype(μ)}[])
    t = promote_type(eltype(μ), typeof(σ), typeof(ϕ), eltype(eltype(dirs)))
    return IsotropicGaussian(SVector{length(μ),t}(μ), t(σ), t(ϕ), SVector{length(μ),t}[SVector{length(μ),t}(dir/norm(dir)) for dir in dirs])
end

convert(t::Type{IsotropicGaussian{T,N}}, g::IsotropicGaussian) where T<:Real where N = t(g.μ, g.σ, g.ϕ, g.dirs)
IsotropicGaussian{T,N}(g::IsotropicGaussian) where T<:Real where N = convert(IsotropicGaussian{T,N}, g)
promote_rule(::Type{IsotropicGaussian{T,N}}, ::Type{IsotropicGaussian{S,N}}) where {T<:Real,S<:Real,N} = IsotropicGaussian{promote_type(T,S),N}

struct IsotropicGMM{T<:Real,N}
    gaussians::Vector{IsotropicGaussian{T,N}}
end
eltype(gmm::IsotropicGMM{T,N}) where T where N = T
length(gmm::IsotropicGMM) = length(gmm.gaussians)
size(gmm::IsotropicGMM{T,N}) where T where N = (length(gmm.gaussians), N)
size(gmm::IsotropicGMM{T,N}, idx::Int) where T where N = (length(gmm.gaussians), N)[idx]

convert(t::Type{IsotropicGMM{T,N}}, gmm::IsotropicGMM) where T<:Real where N = t(gmm.gaussians)
IsotropicGMM{T,N}(gmm::IsotropicGMM) where T<:Real where N = convert(IsotropicGMM{T,N}, gmm)
promote_rule(::Type{IsotropicGMM{T,N}}, ::Type{IsotropicGMM{S,N}}) where {T<:Real,S<:Real,N} = IsotropicGMM{promote_type(T,S),N}

struct MultiGMM{T<:Real,N}
    gmms::Dict{Symbol, IsotropicGMM{T,N}}
end
eltype(mgmm::MultiGMM{T,N}) where T where N = T
length(mgmm::MultiGMM) = length(mgmm.gmms)
size(mgmm::MultiGMM{T,N}) where T where N = (length(mgmm.gmms), N)
size(mgmm::MultiGMM{T,N}, idx::Int) where T where N = (length(mgmm.gmms), N)[idx]

convert(t::Type{MultiGMM{T,N}}, mgmm::MultiGMM) where T<:Real where N = t(mgmm.gmms)
MultiGMM{T,N}(mgmm::MultiGMM) where T<:Real where N = convert(MultiGMM{T,N}, mgmm)
promote_rule(::Type{MultiGMM{T,N}}, ::Type{MultiGMM{S,N}}) where {T<:Real,S<:Real,N} = MultiGMM{promote_type(T,S),N}