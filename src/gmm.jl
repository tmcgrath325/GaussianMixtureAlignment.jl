struct IsotropicGaussian{T<:Real,N}
    μ::SVector{N,T}
    σ::T
    ϕ::T
end
Base.eltype(::Type{IsotropicGaussian{T,N}}) where T where N = T 

function IsotropicGaussian(μ::AbstractArray, σ::Real, ϕ::Real)
    t = promote_type(eltype(μ), typeof(σ), typeof(ϕ))
    return IsotropicGaussian(SVector{length(μ),t}(μ), t(σ), t(ϕ))
end

struct IsotropicGMM{T<:Real,N}
    gaussians::Vector{IsotropicGaussian{T,N}}
end
Base.eltype(::Type{IsotropicGMM{T,N}}) where T where N = T