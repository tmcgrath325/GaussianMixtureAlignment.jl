abstract type AbstractPoint{N,T} end

"""
A Point contains a coordinate position μ and a weight ϕ.
"""
struct Point{N,T} <: AbstractPoint{N,T} 
    μ::SVector{N,T}
    ϕ::T
end

Point(μ<:AbstractVector{T}, ϕ<:T) where T = Point{length(μ),T}(SVector{length(μ),T}(μ), ϕ)
Point(μ) = Point(μ, one(eltype(μ)))

abstract type AbstractPointSet{N,T} end
abstract type AbstractSinglePointSet{N,T} <: AbstractPointSet{N,T} end
abstract type AbstractMultiPointSet{N,T,K} <: AbstractPointSet{N,T} end


"""
A Pointset is a vector of Point structures.
"""
struct PointSet{N,T} <: AbstractSinglePointSet{N,T}
    points::Vector{Point{N,T}}
end

getindex(ps::PointSet, idx::Int) = getindex(ps.points, idx)
length(ps::PointSet) = length(ps.points)