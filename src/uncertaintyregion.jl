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
    ranges::NTuple{6,Tuple{N,N}}
end

function UncertaintyRegion(R::RotationVec,T::SVector{3},σᵣ::Number,σₜ::Number)
    t = promote_type(eltype(R), eltype(T), typeof(σᵣ), typeof(σₜ))
    return UncertaintyRegion{t}(RotationVec{t}(R), SVector{3,t}(T), t(σᵣ), t(σₜ), NTuple{6,Tuple{t,t}}(cuberanges(R,T,σᵣ,σₜ)))
end
UncertaintyRegion(σᵣ::Number, σₜ::Number) = UncertaintyRegion(one(RotationVec), zero(SVector{3}), σᵣ, σₜ)    
UncertaintyRegion(σₜ::Number) = UncertaintyRegion(one(RotationVec), zero(SVector{3}), 2π, σₜ)
UncertaintyRegion() = UncertaintyRegion(one(RotationVec), zero(SVector{3}), 2π, 1.0)
UncertaintyRegion(block::UncertaintyRegion) = block;

center(ur::UncertaintyRegion) = (ur.R.sx, ur.R.sy, ur.R.sz, ur.T...);

# for speeding up hashing and performance of the priority queue in the branch and bound procedure
const hash_UncertaintyRegion_seed = UInt === UInt64 ? 0x4de49213ae1a23bf : 0xef78ce68
function hash(B::UncertaintyRegion, h::UInt)
    h += hash_UncertaintyRegion_seed
    h = hash(B.ranges, h)
    return h
end

## only rotation
struct RotationRegion{N<:Real} <: SearchRegion{N}
    R::RotationVec{N}
    T::SVector{3,N}
    σᵣ::N
    ranges::NTuple{3,Tuple{N,N}}
end
function RotationRegion(R::RotationVec,T::SVector{3},σᵣ::Number)
    t = promote_type(eltype(R), eltype(T), typeof(σᵣ))
    return RotationRegion{t}(RotationVec{t}(R), SVector{3,t}(T), t(σᵣ), NTuple{3,Tuple{t,t}}(cuberanges(R,σᵣ)))
end
RotationRegion(R,T,σᵣ::Number) = RotationRegion(R, T, σᵣ, cuberanges(R, σᵣ))
RotationRegion(σᵣ::Number) = RotationRegion(one(RotationVec), zero(SVector{3}), σᵣ)
RotationRegion() = RotationRegion(Float64(2π))

center(rr::RotationRegion) = (rr.R.sx, rr.R.sy, rr.R.sz);

UncertaintyRegion(rr::RotationRegion{T}) where T = UncertaintyRegion(rr.R, rr.T, rr.σᵣ, zero(T))
RotationRegion(ur::UncertaintyRegion) = RotationRegion(ur.R, ur.T, ur.σᵣ)

# for speeding up hashing and performance of the priority queue in the branch and bound procedure
const hash_RotationRegion_seed = UInt === UInt64 ? 0xee63e114344da2b9 : 0xe6cb1eb7
function hash(B::RotationRegion, h::UInt)
    h += hash_RotationRegion_seed
    h = hash(B.ranges, h)
    return h
end


## only translation
struct TranslationRegion{N<:Real} <: SearchRegion{N}
    R::RotationVec{N}
    T::SVector{3,N}
    σₜ::N
    ranges::NTuple{3,Tuple{N,N}}
end
function TranslationRegion(R::RotationVec,T::SVector{3},σₜ::Number)
    t = promote_type(eltype(R), eltype(T), typeof(σₜ))
    return TranslationRegion{t}(RotationVec{t}(R), SVector{3,t}(T), t(σₜ), NTuple{3,Tuple{t,t}}(cuberanges(T,σₜ)))
end
TranslationRegion(R,T,σₜ)   = TranslationRegion(R, T, σₜ, cuberanges(T, σₜ))
TranslationRegion(σₜ)       = TranslationRegion(one(RotationVec{typeof(σₜ)}), zero(SVector{3, typeof(σₜ)}), σₜ)
TranslationRegion()         = TranslationRegion(1.0)

center(tr::TranslationRegion) = (tr.T...,);

UncertaintyRegion(tr::TranslationRegion{T}) where T = UncertaintyRegion(tr.R, tr.T, zero(T), tr.σₜ)
TranslationRegion(ur::UncertaintyRegion) = TranslationRegion(ur.R, ur.T, ur.σₜ)

# for speeding up hashing and performance of the priority queue in the branch and bound procedure
const hash_TranslationRegion_seed = UInt === UInt64 ? 0x24f59aedb6bf903f : 0x76f5f734
function hash(B::TranslationRegion, h::UInt)
    h += hash_TranslationRegion_seed
    h = hash(B.ranges, h)
    return h
end

# Split SearchRegion
function subregions!(subregionvec::Vector{<:UncertaintyRegion}, ur::UncertaintyRegion, nsplits=2)
    sranges = subranges(ur.ranges, nsplits)
    σᵣ = ur.σᵣ / nsplits
    σₜ = ur.σₜ / nsplits
    for (i,sr) in enumerate(sranges)
        c = center(sr)
        R = RotationVec(c[1:3]...)
        T = SVector{3}(c[4:6]...)
        subregionvec[i] = UncertaintyRegion(R,T,σᵣ,σₜ,sr)
    end
end
function subregions!(subregionvec::Vector{<:RotationRegion}, rr::RotationRegion, nsplits=2)
    sranges = subranges(rr.ranges, nsplits)
    σᵣ = rr.σᵣ / nsplits
    for (i,sr) in enumerate(sranges)
        R = RotationVec(center(sr)...)
        subregionvec[i] = RotationRegion(R,rr.T,σᵣ,sr)
    end
end
function subregions!(subregionvec::Vector{<:TranslationRegion}, tr::TranslationRegion, nsplits=2)
    sranges = subranges(tr.ranges, nsplits)
    σₜ = tr.σₜ / nsplits
    for (i,sr) in enumerate(sranges)
        T = SVector{3}(center(sr))
        subregionvec[i] = TranslationRegion(tr.R,T,σₜ,sr)
    end
end

function subregions(sr::SearchRegion, nsplits=2)
    subregionvec = fill(sr, nsplits^length(sr.ranges))
    subregions!(subregionvec, sr, nsplits)
    return subregionvec
end

# Initialize UncertaintyRegion for aligning two PointSets

"""
    lim = translation_limit(gmmx, gmmy)

Computes the largest translation needed to ensure that the searchspace contains the best alignment transformation.
"""
function translation_limit(gmmx::AbstractSingleGMM, gmmy::AbstractSingleGMM)
    trlim = typemin(promote_type(numbertype(gmmx),numbertype(gmmy)))
    for gaussians in (gmmx.gaussians, gmmy.gaussians)
        if !isempty(gaussians)
            trlim = max(trlim, maximum(gaussians) do gauss
                    maximum(abs, gauss.μ) end)
        end
    end
    return trlim
end

function translation_limit(mgmmx::AbstractMultiGMM, mgmmy::AbstractMultiGMM)
    trlim = typemin(promote_type(numbertype(mgmmx),numbertype(mgmmy)))
    for key in keys(mgmmx.gmms) ∩ keys(mgmmy.gmms)
        trlim = max(trlim, translation_limit(mgmmx.gmms[key], mgmmy.gmms[key]))
    end
    return trlim
end

translation_limit(x::AbstractSinglePointSet, y::AbstractSinglePointSet) = max(maximum(abs.(x.coords)), maximum(abs.(y.coords)))

function translation_limit(x::AbstractMultiPointSet, y::AbstractMultiPointSet)
    trlim = typemin(promote_type(numbertype(x),numbertype(y)))
    for key in keys(x.pointsets) ∩ keys(y.pointsets)
        trlim = max(trlim, translation_limit(x.pointsets[key], y.pointsets[key]))
    end
    return trlim
end

UncertaintyRegion(x::Union{AbstractPointSet, AbstractGMM}, y::Union{AbstractPointSet, AbstractGMM}, R::RotationVec = RotationVec(0.0,0.0,0.0), T::SVector{3} = SVector{3}(0.0,0.0,0.0)) = UncertaintyRegion(translation_limit(x, y))
TranslationRegion(x::Union{AbstractPointSet, AbstractGMM}, y::Union{AbstractPointSet, AbstractGMM}, R::RotationVec = RotationVec(0.0,0.0,0.0), T::SVector{3} = SVector{3}(0.0,0.0,0.0)) = TranslationRegion(R, zero(SVector{3}), translation_limit(x, y))
RotationRegion(x:: Union{AbstractPointSet, AbstractGMM}, y::Union{AbstractPointSet, AbstractGMM},   R::RotationVec = RotationVec(0.0,0.0,0.0), T::SVector{3} = SVector{3}(0.0,0.0,0.0)) = RotationRegion(RotationVec(0.0,0.0,0.0), T, π)
    