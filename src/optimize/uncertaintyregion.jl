## splitting up searchcubes 

function cuberanges(center, widths)
    t = eltype(center)
    return  NTuple{length(center),Tuple{t,t}}([(center[i]-widths[i], center[i]+widths[i]) for i=1:length(center)])
end

cuberanges(R::RotationVec, T::SVector{3}, σᵣ, σₜ) = cuberanges((R.sx,R.sy,R.sz,T[1],T[2],T[3]), (σᵣ,σᵣ,σᵣ,σₜ,σₜ,σₜ))
cuberanges(R::RotationVec, σᵣ) = cuberanges((R.sx,R.sy,R.sz), (σᵣ,σᵣ,σᵣ))
cuberanges(T::SVector{3}, σₜ) = cuberanges((T[1],T[2],T[3]), (σₜ,σₜ,σₜ))


"""
    sbrngs = subranges(ranges, nsplits)

Takes `ranges`, a nested tuple describing intervals for each dimension in rigid-transformation space
defining a hypercube, and splits the hypercube into `nsplits` even components along each dimension.
If the cube is N-dimensional, the number of returned sub-cubes will be `nsplits^N`.
"""
function subranges(ranges, nsplits::Int)
    t = eltype(eltype(ranges))

    # calculate even splititng points for each dimension
    splitvals = [range(r[1], stop=r[2], length=nsplits+1) |> collect for r in ranges]
    splits = [[(splitvals[i][j], splitvals[i][j+1]) for j=1:nsplits] for i=1:6]
    f(x) = splits[x[1]][x[2]]
    children = fill(ranges, nsplits^6)
    for (i,I) in enumerate(CartesianIndices(NTuple{6,UnitRange{Int}}(fill(1:nsplits, 6))))
        children[i] = NTuple{6,Tuple{t,t}}(map(x->f(x), enumerate(Tuple(I))))
    end
    return children
end

## supertype for search regions
abstract type SearchRegion{T} end
CoordinateTransformations.AffineMap(sr::SearchRegion) = AffineMap(sr.R, sr.T)

## rigid transformation

"""
Describes an transformation uncertainty region centered at rotation R and translation T, with rotation and translation half-widths of σᵣ and σₜ respectively
"""
struct UncertaintyRegion{T<:Real} <: SearchRegion{T}
    R::RotationVec{T}
    T::SVector{3,T}
    σᵣ::T
    σₜ::T
    ranges::NTuple{6,Tuple{T,T}}
end
UncertaintyRegion(R,T,σᵣ,σₜ) = UncertaintyRegion(R, T, σᵣ, σₜ, cuberanges(R, T, σᵣ, σₜ))
UncertaintyRegion(σᵣ, σₜ)    = UncertaintyRegion(one(RotationVec), zero(SVector{3}), σᵣ, σₜ)
UncertaintyRegion(σₜ)        = UncertaintyRegion(one(RotationVec), zero(SVector{3}), π, σₜ)
UncertaintyRegion()         = UncertaintyRegion(one(RotationVec), zero(SVector{3}), π, Inf)

# for speeding up hashing and performance of the priority queue in the branch and bound procedure
const hash_UncertaintyRegion_seed = UInt === UInt64 ? 0x4de49213ae1a23bf : 0xef78ce68
function hash(B::UncertaintyRegion, h::UInt)
    h += hash_UncertaintyRegion_seed
    h = hash(B.ranges, h)
    return h
end

## only rotation
struct RotationRegion{T<:Real} <: SearchRegion{T}
    R::RotationVec{T}
    T::SVector{3,T}
    σᵣ::T
    ranges::NTuple{3,Tuple{T,T}}
end
RotationRegion(R,T,σᵣ)   = RotationRegion(R, T, σᵣ, cuberanges(R, σᵣ))
RotationRegion(σᵣ)       = RotationRegion(one(RotationVec), zero(SVector{3}), σᵣ)
RotationRegion()         = RotationRegion(one(RotationVec), zero(SVector{3}), π)

UncertaintyRegion(rr::RotationRegion) = UncertaintyRegion(rr.R, rr.T, rr.σᵣ, 0)
RotationRegion(ur::UncertaintyRegion) = RotationRegion(ur.R, ur.T, ur.σᵣ)

# for speeding up hashing and performance of the priority queue in the branch and bound procedure
const hash_RotationRegion_seed = UInt === UInt64 ? 0xee63e114344da2b9 : 0xe6cb1eb7
function hash(B::RotationRegion, h::UInt)
    h += hash_RotationRegion_seed
    h = hash(B.ranges, h)
    return h
end


## only translation
struct TranslationRegion{T<:Real} <: SearchRegion{T}
    R::RotationVec{T}
    T::SVector{3,T}
    σₜ::T
    ranges::NTuple{3,Tuple{T,T}}
end
TranslationRegion(R,T,σₜ)   = TranslationRegion(R, T, σₜ, cuberanges(T, σₜ))
TranslationRegion(σₜ)       = TranslationRegion(one(RotationVec), zero(SVector{3}), σₜ)
TranslationRegion()        = TranslationRegion(one(RotationVec), zero(SVector{3}), Inf)

UncertaintyRegion(tr::TranslationRegion) = UncertaintyRegion(tr.R, tr.T, 0, tr.σₜ)
TranslationRegion(ur::UncertaintyRegion) = TranslationRegion(ur.R, ur.T, ur.σₜ)

# for speeding up hashing and performance of the priority queue in the branch and bound procedure
const hash_TranslationRegion_seed = UInt === UInt64 ? 0x24f59aedb6bf903f : 0x76f5f734
function hash(B::TranslationRegion, h::UInt)
    h += hash_TranslationRegion_seed
    h = hash(B.ranges, h)
    return h
end
