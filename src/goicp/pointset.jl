import Base: eltype, length, size, getindex, iterate, convert, promote_rule, keys

abstract type AbstractPoint{N,T} end
abstract type AbstractPointSet{N,T} <: AbstractModel{N,T} end
abstract type AbstractSinglePointSet{N,T} <: AbstractPointSet{N,T} end
abstract type AbstractMultiPointSet{N,T,K} <: AbstractPointSet{N,T} end

# Base methods for Points
numbertype(::AbstractPoint{N,T}) where {N,T} = T
dims(::AbstractPoint{N,T}) where {N,T} = N
length(::AbstractPoint{N,T}) where {N,T} = N
size(::AbstractPoint{N,T}) where {N,T} = (N,)
size(::AbstractPoint{N,T}, idx::Int) where {N,T} = (N,)[idx]

# Base methods for point sets
numbertype(::AbstractPointSet{N,T}) where {N,T} = T
dims(::AbstractPointSet{N,T}) where {N,T} = N

length(x::AbstractSinglePointSet{N,T}) where {N,T} = size(x.coords,2)
getindex(x::AbstractSinglePointSet, idx) = getpoint(x, idx)
iterate(x::AbstractSinglePointSet) = iterate(x, 1)
iterate(x::AbstractSinglePointSet, i) = i > length(x) ? nothing : (getindex(x, i), i+1)
size(x::AbstractSinglePointSet{N,T}) where {N,T} = (N,length(x))
size(x::AbstractSinglePointSet{N,T}, idx::Int) where {N,T} = size(x)[idx]

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
struct Point{N,T} <: AbstractPoint{N,T}
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
    coords::Matrix{T}
    weights::Vector{T}
end

getpoint(x::PointSet{N,T}, idx::Int) where {N,T} = Point{N,T}(x.coords[:,idx], x.weights[idx])

eltype(::Type{PointSet{N,T}}) where {N,T} = Point{N,T}
convert(t::Type{PointSet}, p::AbstractPointSet) = t(p.coords, p.weights)
promote_rule(::Type{PointSet{N,T}}, ::Type{PointSet{N,S}}) where {T,S,N} = PointSet{N,promote_type(T,S)}

weights(ps::PointSet) = ps.weights

function Base.:*(R::AbstractMatrix, p::PointSet)
    return PointSet(R*p.coords, p.weights)
end

function Base.:+(p::PointSet, T::AbstractVector)
    return PointSet(p.coords.+T, p.weights)
end

function Base.:-(p::PointSet, T::AbstractVector)
    return PointSet(p.coords.-T, p.weights)
end

function PointSet(coords::AbstractMatrix{S}, weights::AbstractVector{T} = ones(T, size(coords,2))) where {S,T}
    dim = size(coords,1)
    numtype = promote_type(S,T)
    return PointSet{dim,numtype}(Matrix{numtype}(coords), Vector{numtype}(weights))
end

PointSet(coords::AbstractVector{<:AbstractVector{T}}, weights::AbstractVector{T} = ones(T, length(coords))) where T = PointSet(hcat(coords...), weights)

"""
A collection of labeled point sets, to each be considered separately during an alignment procedure. That is, 
only alignment scores between point sets with the same key are considered when aligning two `MultiPointSet`s. 
"""
struct MultiPointSet{N,T,K} <: AbstractMultiPointSet{N,T,K}
    pointsets::Dict{K, <:AbstractSinglePointSet{N,T}}
end

MultiPointSet(x::AbstractMultiPointSet) = MultiPointSet(x.pointsets)
function MultiPointSet(pointsetpairs::AbstractVector{<:Pair{K, S}}) where {K, S<:AbstractSinglePointSet}
    psdict = Dict{K,S}()
    for pair in pointsetpairs
        push!(psdict, pair)
    end
    return MultiPointSet(psdict)
end

eltype(::Type{MultiPointSet{N,T,K}}) where {N,T,K} = Pair{K, PointSet{N,T}}
convert(t::Type{MultiPointSet}, x::AbstractMultiPointSet) = t(x.pointsets)
promote_rule(::Type{MultiPointSet{N,T,K}}, ::Type{MultiPointSet{N,S,L}}) where {N,T,S,K,L} = MultiPointSet{N,promote_type(T,S), promote_type(K,L)}

function weights(mps::MultiPointSet{N,T,K}) where {N,T,K}
    w = Dict{K, Vector{T}}()
    for (key, ps) in mps.pointsets
        push!(w, key => ps.weights)
    end
    return w
end

function Base.:*(R::AbstractMatrix, p::MultiPointSet)
    return MultiPointSet([key => R*p[key] for key in keys(p)])
end

function Base.:+(p::MultiPointSet, T::AbstractVector)
    return MultiPointSet([key => p[key]+T for key in keys(p)])
end

function Base.:-(p::MultiPointSet, T::AbstractVector)
    return MultiPointSet([key => p[key]-T for key in keys(p)])
end