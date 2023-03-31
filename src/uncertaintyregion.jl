## splitting up searchcubes 

function cuberanges(center::NTuple{N,T}, widths) where {N,T}
    return  NTuple{N,Tuple{T,T}}([(center[i]-widths[i], center[i]+widths[i]) for i=1:length(center)])
end

cuberanges(R::RotationVec, T::SVector{3}, σᵣ, σₜ) = cuberanges((R.sx,R.sy,R.sz,T[1],T[2],T[3]), (σᵣ,σᵣ,σᵣ,σₜ,σₜ,σₜ))
cuberanges(R::RotationVec, σᵣ::Number) = cuberanges((R.sx,R.sy,R.sz), (σᵣ,σᵣ,σᵣ))
cuberanges(T::SVector{3}, σₜ::Number) = cuberanges((T[1],T[2],T[3]), (σₜ,σₜ,σₜ))


"""
    sbrngs = subranges(ranges, nsplits)

Takes `ranges`, a nested tuple describing intervals for each dimension in rigid-transformation space
defining a hypercube, and splits the hypercube into `nsplits` even components along each dimension.
If the cube is N-dimensional, the number of returned sub-cubes will be `nsplits^N`.
"""
function subranges(ranges, nsplits::Int=2)
    t = eltype(eltype(ranges))
    len = length(ranges)

    # calculate even splititng points for each dimension
    splitvals = [range(r[1], stop=r[2], length=nsplits+1) |> collect for r in ranges]
    splits = [[(splitvals[i][j], splitvals[i][j+1]) for j=1:nsplits] for i=1:len]
    f(x) = splits[x[1]][x[2]]
    children = fill(ranges, nsplits^len)
    for (i,I) in enumerate(CartesianIndices(NTuple{len,UnitRange{Int}}(fill(1:nsplits, len))))
        children[i] = NTuple{len,Tuple{t,t}}(map(x->f(x), enumerate(Tuple(I))))
    end
    return children
end

function center(ranges::NTuple{N,Tuple{T,T}}) where {N,T}
    return NTuple{N,T}([sum(dim)/2 for dim in ranges])
end

## supertype for search regions
abstract type SearchRegion{T} end

## rigid transformation
AffineMap(sr::SearchRegion) = AffineMap(sr.R, sr.T)

"""
Describes an transformation uncertainty region centered at rotation R and translation T, with rotation and translation half-widths of σᵣ and σₜ respectively
"""
struct UncertaintyRegion{N<:Real} <: SearchRegion{N}
    R::RotationVec{N}
    T::SVector{3,N}
    σᵣ::N
    σₜ::N
end

function UncertaintyRegion(R::RotationVec,T::SVector{3},σᵣ::Number,σₜ::Number)
    t = promote_type(eltype(R), eltype(T), typeof(σᵣ), typeof(σₜ))
    return UncertaintyRegion{t}(RotationVec{t}(R), SVector{3,t}(T), t(σᵣ), t(σₜ))
end
UncertaintyRegion(σᵣ::Number, σₜ::Number) = UncertaintyRegion(one(RotationVec), zero(SVector{3}), σᵣ, σₜ)    
UncertaintyRegion(σₜ::Number) = UncertaintyRegion(one(RotationVec), zero(SVector{3}), π, σₜ)
UncertaintyRegion() = UncertaintyRegion(one(RotationVec), zero(SVector{3}), π, 1.0)
UncertaintyRegion(block::UncertaintyRegion) = block;

center(ur::UncertaintyRegion) = (ur.R.sx, ur.R.sy, ur.R.sz, ur.T...);

# for speeding up hashing and performance of the priority queue in the branch and bound procedure
const hash_UncertaintyRegion_seed = UInt === UInt64 ? 0x4de49213ae1a23bf : 0xef78ce68
function Base.hash(B::UncertaintyRegion, h::UInt)
    h += hash_UncertaintyRegion_seed
    h = Base.hash(center(B), h)
    return h
end

## only rotation
struct RotationRegion{N<:Real} <: SearchRegion{N}
    R::RotationVec{N}
    T::SVector{3,N}
    σᵣ::N
end
function RotationRegion(R::RotationVec,T::SVector{3},σᵣ::Number)
    t = promote_type(eltype(R), eltype(T), typeof(σᵣ))
    return RotationRegion{t}(RotationVec{t}(R), SVector{3,t}(T), t(σᵣ))
end
RotationRegion(R,T,σᵣ::Number) = RotationRegion(R, T, σᵣ)
RotationRegion(σᵣ::Number) = RotationRegion(one(RotationVec), zero(SVector{3}), σᵣ)
RotationRegion() = RotationRegion(Float64(π))

center(rr::RotationRegion) = (rr.R.sx, rr.R.sy, rr.R.sz);

UncertaintyRegion(rr::RotationRegion{T}) where T = UncertaintyRegion(rr.R, rr.T, rr.σᵣ, zero(T))
RotationRegion(ur::UncertaintyRegion) = RotationRegion(ur.R, ur.T, ur.σᵣ)

# for speeding up hashing and performance of the priority queue in the branch and bound procedure
const hash_RotationRegion_seed = UInt === UInt64 ? 0xee63e114344da2b9 : 0xe6cb1eb7
function Base.hash(B::RotationRegion, h::UInt)
    h += hash_RotationRegion_seed
    h = Base.hash(center(B), h)
    return h
end


## only translation
struct TranslationRegion{N<:Real} <: SearchRegion{N}
    R::RotationVec{N}
    T::SVector{3,N}
    σₜ::N
end
function TranslationRegion(R::RotationVec,T::SVector{3},σₜ::Number)
    t = promote_type(eltype(R), eltype(T), typeof(σₜ))
    return TranslationRegion{t}(RotationVec{t}(R), SVector{3,t}(T), t(σₜ))
end
TranslationRegion(R,T,σₜ)   = TranslationRegion(R, T, σₜ)
TranslationRegion(σₜ)       = TranslationRegion(one(RotationVec{typeof(σₜ)}), zero(SVector{3, typeof(σₜ)}), σₜ)
TranslationRegion()         = TranslationRegion(1.0)

center(tr::TranslationRegion) = (tr.T...,);

UncertaintyRegion(tr::TranslationRegion{T}) where T = UncertaintyRegion(tr.R, tr.T, zero(T), tr.σₜ)
TranslationRegion(ur::UncertaintyRegion) = TranslationRegion(ur.R, ur.T, ur.σₜ)

# for speeding up hashing and performance of the priority queue in the branch and bound procedure
const hash_TranslationRegion_seed = UInt === UInt64 ? 0x24f59aedb6bf903f : 0x76f5f734
function Base.hash(B::TranslationRegion, h::UInt)
    h += hash_TranslationRegion_seed
    h = Base.hash(center(B), h)
    return h
end

# Split SearchRegion
function subregions!(subregionvec::Vector{S}, ur::S, nsplits=2) where S<:UncertaintyRegion 
    σᵣ = ur.σᵣ / nsplits
    σₜ = ur.σₜ / nsplits
    lowercorner = center(ur) .- (ur.σᵣ, ur.σᵣ, ur.σᵣ, ur.σₜ, ur.σₜ, ur.σₜ) .+ (σᵣ, σᵣ, σᵣ, σₜ, σₜ, σₜ)
    for (i,I) in enumerate(CartesianIndices(NTuple{6,UnitRange{Int}}(fill(0:nsplits-1, 6))))
        idxs = Tuple(I)
        c = lowercorner .+ (2*idxs[1]*σᵣ, 2*idxs[2]*σᵣ, 2*idxs[3]*σᵣ, 2*idxs[4]*σₜ, 2*idxs[5]*σₜ, 2*idxs[6]*σₜ)
        R = RotationVec(c[1], c[2], c[3])
        T = SVector{3}(c[4], c[5], c[6])
        subregionvec[i] = UncertaintyRegion(R,T,σᵣ,σₜ)
    end
end
function subregions!(subregionvec::Vector{S}, rr::S, nsplits=2) where S<:RotationRegion 
    σᵣ = rr.σᵣ / nsplits
    lowercorner = (center(rr) .- rr.σᵣ) .+ σᵣ
    for (i,I) in enumerate(CartesianIndices(NTuple{3,UnitRange{Int}}(fill(0:nsplits-1, 3))))
        idxs = Tuple(I)
        c = lowercorner .+ (2*idxs[1]*σᵣ, 2*idxs[2]*σᵣ, 2*idxs[3]*σᵣ)
        R = RotationVec(c[1], c[2], c[3])
        subregionvec[i] = RotationRegion(R,rr.T,σᵣ)
    end
end
function subregions!(subregionvec::Vector{S}, tr::S, nsplits=2) where S<:TranslationRegion 
    σₜ = tr.σₜ / nsplits
    lowercorner = (center(tr) .- tr.σₜ) .+ σₜ
    for (i,I) in enumerate(CartesianIndices(NTuple{3,UnitRange{Int}}(fill(0:nsplits-1, 3))))
        idxs = Tuple(I)
        c = lowercorner .+ (2*idxs[1]*σₜ, 2*idxs[2]*σₜ, 2*idxs[3]*σₜ)
        T = SVector{3}(c[1], c[2], c[3])
        subregionvec[i] = TranslationRegion(tr.R,T,σₜ)
    end
end

function subregions(sr::SearchRegion, nsplits=2)
    subregionvec = fill(sr, nsplits^length(center(sr)))
    subregions!(subregionvec, sr, nsplits)
    return subregionvec
end

# split only along translation or rotation axes
function rot_subregions!(subregionvec::Vector{S}, ur::S, nsplits=2) where S<:UncertaintyRegion
    σᵣ = ur.σᵣ / nsplits
    lowercorner = (ur.R.sx, ur.R.sy, ur.R.sz) .- (ur.σᵣ, ur.σᵣ, ur.σᵣ) .+ (σᵣ, σᵣ, σᵣ)
    for (i,I) in enumerate(CartesianIndices(NTuple{3,UnitRange{Int}}(fill(0:nsplits-1, 3))))
        idxs = Tuple(I)
        R = RotationVec((lowercorner .+ (2*idxs[1]*σᵣ, 2*idxs[2]*σᵣ, 2*idxs[3]*σᵣ))...)
        subregionvec[i] = UncertaintyRegion(R, ur.T, σᵣ, ur.σₜ)
    end
end

function trl_subregions!(subregionvec::Vector{S}, ur::S, nsplits=2) where S<:UncertaintyRegion
    σₜ = ur.σₜ / nsplits
    lowercorner = ur.T .- (ur.σₜ, ur.σₜ, ur.σₜ) .+ (σₜ, σₜ, σₜ)
    for (i,I) in enumerate(CartesianIndices(NTuple{3,UnitRange{Int}}(fill(0:nsplits-1, 3))))
        idxs = Tuple(I)
        T = lowercorner .+ (2*idxs[1]*σₜ, 2*idxs[2]*σₜ, 2*idxs[3]*σₜ)
        subregionvec[i] = UncertaintyRegion(ur.R, T, ur.σᵣ, σₜ)
    end
end

function rot_subregions(sr::UncertaintyRegion, nsplits=2)
    subregionvec = fill(sr, nsplits^3)
    rot_subregions!(subregionvec, sr, nsplits)
    return subregionvec
end

function trl_subregions(sr::UncertaintyRegion, nsplits=2)
    subregionvec = fill(sr, nsplits^3)
    trl_subregions!(subregionvec, sr, nsplits)
    return subregionvec
end

# Initialize UncertaintyRegion for aligning two PointSets
UncertaintyRegion(x::Union{AbstractPointSet, AbstractGMM}, y::Union{AbstractPointSet, AbstractGMM}, R::RotationVec = RotationVec(0.0,0.0,0.0), T::SVector{3} = SVector{3}(0.0,0.0,0.0)) = UncertaintyRegion(translation_limit(x, y))
TranslationRegion(x::Union{AbstractPointSet, AbstractGMM}, y::Union{AbstractPointSet, AbstractGMM}, R::RotationVec = RotationVec(0.0,0.0,0.0), T::SVector{3} = SVector{3}(0.0,0.0,0.0)) = TranslationRegion(R, zero(SVector{3}), translation_limit(x, y))
RotationRegion(x:: Union{AbstractPointSet, AbstractGMM}, y::Union{AbstractPointSet, AbstractGMM},   R::RotationVec = RotationVec(0.0,0.0,0.0), T::SVector{3} = SVector{3}(0.0,0.0,0.0)) = RotationRegion(RotationVec(0.0,0.0,0.0), T, π)
    