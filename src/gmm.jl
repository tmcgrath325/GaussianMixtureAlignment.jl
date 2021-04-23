struct IsotropicGaussian{T<:Real,N}
    μ::SVector{N,T}
    σ::T
    ϕ::T
end
Base.eltype(ig::IsotropicGaussian{T,N}) where T where N = T
Base.length(ig::IsotropicGaussian{T,N}) where T where N = N
Base.size(ig::IsotropicGaussian{T,N}) where T where N = (N,)
Base.size(gmm::IsotropicGaussian{T,N}, idx::Int) where T where N = (N,)[idx]

function IsotropicGaussian(μ::AbstractArray, σ::Real, ϕ::Real)
    t = promote_type(eltype(μ), typeof(σ), typeof(ϕ))
    return IsotropicGaussian(SVector{length(μ),t}(μ), t(σ), t(ϕ))
end

struct IsotropicGMM{T<:Real,N}
    gaussians::Vector{IsotropicGaussian{T,N}}
end
Base.eltype(gmm::IsotropicGMM{T,N}) where T where N = T
Base.length(gmm::IsotropicGMM) = length(gmm.gaussians)
Base.size(gmm::IsotropicGMM{T,N}) where T where N = (length(gmm.gaussians), N)
Base.size(gmm::IsotropicGMM{T,N}, idx::Int) where T where N = (length(gmm.gaussians), N)[idx]

struct MultiGMM{T<:Real,N}
    gmms::Dict{Symbol, IsotropicGMM{T,N}}
end
Base.eltype(mgmm::MultiGMM{T,N}) where T where N = T
Base.length(mgmm::MultiGMM) = length(mgmm.gmms)
Base.size(mgmm::MultiGMM{T,N}) where T where N = (length(mgmm.gmms), N)
Base.size(mgmm::MultiGMM{T,N}, idx::Int) where T where N = (length(mgmm.gmms), N)[idx]
