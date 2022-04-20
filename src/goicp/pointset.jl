abstract type AbstractPoint{N,T} end
abstract type AbstractPointSet{N,T} end
abstract type AbstractSinglePointSet{N,T} <: AbstractPointSet{N,T} end
abstract type AbstractMultiPointSet{N,T,K} <: AbstractPointSet{N,T} end

# Base methods for Points
numbertype(::AbstractPoint{N,T}) where {N,T} = T
dims(::AbstractPoint{N,T}) where {N,T} = N
length(::AbstractPoint{N,T}) where {N,T} = N
size(::AbstractPoint{N,T}) where {N,T} = (N,)
size(::AbstractPoint{N,T}, idx::Int) where {N,T} = (N,)[idx]

# Base methods for point sets
numbertype(::AAbstractPointSet{N,T}) where {N,T} = T
dims(::AbstractPointSet{N,T}) where {N,T} = N

length(x::AbstractSinglePointSet) = size(x.coords, 2)
getindex(x::AbstractSinglePointSet, idx) = getpoint(x, idx)
iterate(x::AbstractSinglePointSet) = iterate(x, 1)
iterate(x::AbstractSinglePointSet, i) = i > length(x) ? nothing : (getindex(x, i), i+1)
size(x::AbstractSinglePointSet{N,T}) where {N,T} = (length(x), N)
size(x::AbstractSinglePointSet{N,T}, idx::Int) where {N,T} = (length(x), N)[idx]

length(x::AbstractMultiPointSet) = length(x.pointsets)
getindex(x::AbstractMultiPointSet, key) = x.pointsets[key]
keys(x::AbstractMultiPointSet) = keys(x.pointsets)
iterate(x::AbstractMultiPointSet) = iterate(x.pointsets)
iterate(x::AbstractMultiPointSet, i) = iterate(x.pointsets, i)
size(x::AbstractMultiPointSet{N,T,K}) where {N,T,K} = (length(x.pointsets), N)
size(x::AbstractMultiPointSet{N,T,K}, idx::Int) where {N,T,K} = (length(x.pointsets), N)[idx]

"""
A coordinate position and a weight, to be used as part of a point set.
"""
struct Point{N,T}
    coords::SVector{N,T}
    weight::T
end

convert(::Type{Point{N,T}}, p::AbstractPoint) where {N,T} = Point(SVector{N,T}(p.coords), T(p.weight))
promote_rule(::Type{Point{N,T}}, ::Type{Point{N,S}}) where {N,T<:Real,S<:Real} = Point{N,promote_type(T,S)} 

function Base.:*(R::AbstractMatrix, p::Point)
    return Point(R*p.coords, p.weight)
end

function Base.:-(p::Point, T::AbstractVector)
    return Point(p.coords-T, p.weight)
end


"""
A point set made consisting of a matrix of coordinate positions with corresponding weights.
"""
struct PointSet{N,T} <: AbstractSinglePointSet{N,T}
    coords::SMatrix{N,N,T}
    weights::SVector{N,T}
end

getpoint(x::PointSet{N,T}, idx::Int) where {N,T} = Point{N,T}(x.coords[:,idx], weights[idx])

eltype(::Type{PointSet{N,T}}) where {N,T} = Point{N,T}
convert(t::Type{PointSet}, p::AbstractPointSet) = t(p.coords, p.weights)
promote_rule(::Type{PointSet{N,T}}, ::Type{PointSet{N,S}}) where {T,S,N} = PointSet{N,promote_type(T,S)}

function Base.:*(R::AbstractMatrix, p::PointSet)
    return PointSet(R*p.coords, p.weights)
end

function Base.:-(p::PointSet, T::AbstractVector)
    return PointSet(p.coords.-T, p.weights)
end


"""
A collection of labeled point sets, to each be considered separately during an alignment procedure. That is, 
only alignment scores between point sets with the same key are considered when aligning two `MultiPointSet`s. 
"""
struct MultiPointSet{N,T,K} <: AbstractMultiPointSet{N,T,K}
    pointsets::Dict{K, <:AbstractSinglePointSet{N,T}}
end

MultiPointSet(x::AbstractMultiPointSet) = MultiPointSet(x.pointsets)

eltype(::Type{MultiPointSet{N,T,K}}) where {N,T,K} = Pair{K, PointSet{N,T}}
convert(t::Type{MultiPointSet}, x::AbstractMultiPointSet) = t(x.pointsets)
promote_rule(::Type{MultiPointSet{N,T,K}}, ::Type{MultiPointSet{N,S,L}}) where {N,T,S,K,L} = MultiPointSet{N,promote_type(T,S), promote_type(K,L)}

function Base.:*(R::AbstractMatrix, p::MultiPointSet)
    return MultiPointSet([R*p[key] for key in keys(p)])
end

function Base.:-(p::MultiPointSet, T::AbstractVector)
    return MultiPointSet([p[key]-T for key in keys(p)])
end