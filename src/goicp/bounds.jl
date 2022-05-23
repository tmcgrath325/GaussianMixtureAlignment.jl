loose_distance_bounds(x::AbstractPoint, y::AbstractPoint, args...) = loose_distance_bounds(x.coords, y.coords, args...)
tight_distance_bounds(x::AbstractPoint, y::AbstractPoint, args...) = tight_distance_bounds(x.coords, y.coords, args...)

function squared_dist_bounds(x::AbstractSinglePointSet, y::AbstractSinglePointSet, σᵣ::Number, σₜ::Number; distance_bound_fun = tight_distance_bounds, correspondence = hungarian_assignment) 
    matches = correspondence(x.coords, y.coords)
    
    # sum bounds for each pair of points
    lb = 0.
    ub = 0.
    for (i,j) in matches
        (matchlb, matchub) = distance_bound_fun(x[i], y[j], σᵣ, σₜ).^2  
        lb += matchlb
        ub += matchub 
    end
    return lb, ub
end

function squared_dist_bounds(x::AbstractMultiPointSet, y::AbstractMultiPointSet, σᵣ, σₜ; kwargs...)
    # sum bounds for each matched pair of pointsets
    lb = 0.
    ub = 0.
    for key in keys(x.pointsets) ∩ keys(y.pointsets)
        lb, ub = (lb, ub) .+ squared_dist_bounds(mgmmx.gmms[key], mgmmy.gmms[key], σᵣ, σₜ; kwargs...)
    end
    return lb, ub
end

squared_dist_bounds(x::AbstractPointSet, y::AbstractPointSet, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number; kwargs...
    ) = squared_dist_bounds(R*x, y-T, σᵣ, σₜ; kwargs...)

squared_dist_bounds(x::AbstractPointSet, y::AbstractPointSet, ur::UncertaintyRegion; kwargs...
    ) = squared_dist_bounds(x, y, ur.R, ur.T, ur.σᵣ, ur.σₜ; kwargs...)

squared_dist_bounds(x::AbstractPointSet, y::AbstractPointSet, sr::SearchRegion; kwargs...
    ) = squared_dist_bounds(x, y, UncertaintyRegion(sr); kwargs...)
