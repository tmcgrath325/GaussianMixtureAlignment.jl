loose_distance_bounds(x::AbstractPoint, y::AbstractPoint, sr::Number, st::Number) = loose_distance_bounds(x.coords, y.coords, sr, st)
tight_distance_bounds(x::AbstractPoint, y::AbstractPoint, sr::Number, st::Number) = tight_distance_bounds(x.coords, y.coords, sr, st)

function squared_dist_bounds(x::AbstractSinglePointSet, y::AbstractSinglePointSet, σᵣ::Number, σₜ::Number; 
    distance_bound_fun::Union{typeof(tight_distance_bounds),typeof(loose_distance_bounds)} = loose_distance_bounds, 
    correspondence = hungarian_assignment,
    lohifun = lohi_interval) 

    matches = correspondence(x.coords, y.coords)
    
    bnds = lohifun(0.0, 0.0)
    for (i,j) in matches
        (matchlb, matchub) = distance_bound_fun(x[i], y[j], σᵣ, σₜ) 
        bnds = bnds + lohifun(matchlb^2, matchub^2)
    end
    return bnds
end

function squared_dist_bounds(x::AbstractMultiPointSet{N,T,K}, y::AbstractMultiPointSet{N,S,L}, σᵣ, σₜ; lohifun=lohi_interval, kwargs...) where {N,T,K,S,L}
    bnds = lohifun(0.0, 0.0)
    matches = Dict{promote_type(K,L), Vector{Tuple{Int,Int}}}()
    for key in keys(x.pointsets) ∩ keys(y.pointsets)
        keylb, keyub, keymatches = squared_dist_bounds(x.pointsets[key], y.pointsets[key], σᵣ, σₜ; kwargs...)
        bnds = bnds + lohifun(keylb^2, keyub^2)
        push!(matches, keymatches)
    end
    return bnds
end

squared_dist_bounds(x::AbstractPointSet, y::AbstractPointSet, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number; kwargs...
    ) = squared_dist_bounds(R*x, y-T, σᵣ, σₜ; kwargs...)

squared_dist_bounds(x::AbstractPointSet, y::AbstractPointSet, ur::UncertaintyRegion; kwargs...
    ) = squared_dist_bounds(x, y, ur.R, ur.T, ur.σᵣ, ur.σₜ; kwargs...)

squared_dist_bounds(x::AbstractPointSet, y::AbstractPointSet, sr::SearchRegion; kwargs...
    ) = squared_dist_bounds(x, y, UncertaintyRegion(sr); kwargs...)
