loose_distance_bounds(x::AbstractPoint, y::AbstractPoint, sr::Number, st::Number) = loose_distance_bounds(x.coords, y.coords, sr, st)
tight_distance_bounds(x::AbstractPoint, y::AbstractPoint, sr::Number, st::Number) = tight_distance_bounds(x.coords, y.coords, sr, st)

function squared_dist_bounds(x::AbstractSinglePointSet, y::AbstractSinglePointSet, σᵣ::Number, σₜ::Number; 
    distance_bound_fun::Union{typeof(tight_distance_bounds),typeof(loose_distance_bounds)} = loose_distance_bounds, 
    correspondence = hungarian_assignment) 

    matches = correspondence(x.coords, y.coords)
    
    # sum bounds for each pair of points
    lb = 0.
    ub = 0.
    for (i,j) in matches
        matchbounds = distance_bound_fun(x[i], y[j], σᵣ, σₜ) 
        lb += matchbounds.lowerbound^2
        ub += matchbounds.upperbound^2
    end
    return SearchRegionBounds(lb, ub)
end

function squared_dist_bounds(x::AbstractMultiPointSet, y::AbstractMultiPointSet, σᵣ, σₜ; kwargs...)
    # sum bounds for each matched pair of pointsets
    lb = 0.
    ub = 0.
    for key in keys(x.pointsets) ∩ keys(y.pointsets)
        pointsetbounds = squared_dist_bounds(x.pointsets[key], y.pointsets[key], σᵣ, σₜ; kwargs...)
        lb += pointsetbounds.lowerbound
        ub += pointsetbounds.upperbound
    end
    return SearchRegionBounds(lb, ub)
end

squared_dist_bounds(x::AbstractPointSet, y::AbstractPointSet, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number; kwargs...
    ) = squared_dist_bounds(R*x, y-T, σᵣ, σₜ; kwargs...)

squared_dist_bounds(x::AbstractPointSet, y::AbstractPointSet, ur::UncertaintyRegion; kwargs...
    ) = squared_dist_bounds(x, y, ur.R, ur.T, ur.σᵣ, ur.σₜ; kwargs...)

squared_dist_bounds(x::AbstractPointSet, y::AbstractPointSet, sr::SearchRegion; kwargs...
    ) = squared_dist_bounds(x, y, UncertaintyRegion(sr); kwargs...)
