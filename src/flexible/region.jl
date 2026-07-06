## Search region for articulated alignment: the six rigid parameters of an
## `UncertaintyRegion` plus one angular interval per joint.

"""
    FlexibleRegion(rigid::UncertaintyRegion, φ, σφ)
    FlexibleRegion(rigid::UncertaintyRegion, K::Integer)

A search region over `(R, T, φ₁…φ_K)`: an `UncertaintyRegion` `rigid` for the rigid
rotation/translation, plus per-joint angle centers `φ` and half-widths `σφ` (length `K`).

The `K`-argument form covers the full angular range of every joint (`φ = 0`, `σφ = π`). With
`K = 0` the region carries no joints and projects to `rigid` unchanged.
"""
struct FlexibleRegion{T, K} <: SearchRegion{T}
    rigid::UncertaintyRegion{T}
    φ::SVector{K, T}
    σφ::SVector{K, T}
    function FlexibleRegion{T, K}(rigid, φ, σφ) where {T, K}
        return new{T, K}(rigid, φ, σφ)
    end
end

function FlexibleRegion(rigid::UncertaintyRegion{S}, φ::AbstractVector, σφ::AbstractVector) where {S}
    length(φ) == length(σφ) || throw(DimensionMismatch("φ and σφ must share length; got $(length(φ)) and $(length(σφ))"))
    K = length(φ)
    t = promote_type(S, eltype(φ), eltype(σφ))
    rigidt = UncertaintyRegion(rigid.R, rigid.T, t(rigid.σᵣ), t(rigid.σₜ))
    return FlexibleRegion{t, K}(rigidt, SVector{K, t}(φ), SVector{K, t}(σφ))
end

function FlexibleRegion(rigid::UncertaintyRegion{S}, K::Integer) where {S}
    return FlexibleRegion(rigid, zero(SVector{K, S}), SVector{K, S}(ntuple(_ -> S(π), K)))
end

center(fr::FlexibleRegion) = (center(fr.rigid)..., fr.φ...)

UncertaintyRegion(fr::FlexibleRegion) = fr.rigid

njoints(fr::FlexibleRegion{T, K}) where {T, K} = K

# for the priority queue in the branch-and-bound procedure
const hash_FlexibleRegion_seed = UInt === UInt64 ? 0x7b1f4a2c9d3e6058 : 0x5c2e8f13
function Base.hash(B::FlexibleRegion, h::UInt)
    h += hash_FlexibleRegion_seed
    h = Base.hash(center(B), h)
    return h
end

"""
    subregions(fr::FlexibleRegion, nsplits=2; rotscale=1, trlscale=1, jointscales=(1,…))

Subdivide `fr` along whichever coordinate group — rotation, translation, or a single joint —
has the largest scaled uncertainty. Splitting one group at a time keeps the branching factor
bounded (`nsplits^3` when rotation or translation wins, `nsplits` for a joint) instead of the
`nsplits^(6+K)` of an all-axis split.

The scale factors convert the groups' half-widths to a common footing before they are
compared: an angular half-width contributes a displacement of order `halfwidth * radius`, so
`rotscale` and each `jointscales[k]` should be the relevant rotation radius, while `trlscale`
is a translation displacement and defaults to unity. The center transform's optimum is
unaffected by the choice; it only orders the search.
"""
function subregions(
        fr::FlexibleRegion{T, K}, nsplits::Int = 2;
        rotscale = one(T), trlscale = one(T), jointscales::NTuple{K} = ntuple(_ -> one(T), K)
    ) where {T, K}
    rotw = fr.rigid.σᵣ * rotscale
    trlw = fr.rigid.σₜ * trlscale
    # group 0 = rotation, group -1 = translation, group k>0 = joint k
    group = rotw >= trlw ? 0 : -1
    widest = max(rotw, trlw)
    for k in 1:K
        jw = fr.σφ[k] * jointscales[k]
        if jw > widest
            widest = jw
            group = k
        end
    end

    if group == 0
        return [FlexibleRegion(u, fr.φ, fr.σφ) for u in rot_subregions(fr.rigid, nsplits)]
    elseif group == -1
        return [FlexibleRegion(u, fr.φ, fr.σφ) for u in trl_subregions(fr.rigid, nsplits)]
    end

    # split joint `group`'s angular interval into `nsplits` even pieces
    σ = fr.σφ[group]
    σ2 = σ / nsplits
    lower = fr.φ[group] - σ + σ2
    children = FlexibleRegion{T, K}[]
    for i in 0:(nsplits - 1)
        c = lower + 2 * i * σ2
        φnew = Base.setindex(fr.φ, c, group)
        σφnew = Base.setindex(fr.σφ, σ2, group)
        push!(children, FlexibleRegion{T, K}(fr.rigid, φnew, σφnew))
    end
    return children
end
