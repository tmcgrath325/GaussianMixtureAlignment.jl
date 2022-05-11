loose_distance_bounds(x::AbstractPoint, y::AbstractPoint, args...) = loose_distance_bounds(x.coords, y.coords, args...)
tight_distance_bounds(x::AbstractPoint, y::AbstractPoint, args...) = tight_distance_bounds(x.coords, y.coords, args...)

function squared_dist_bounds(x::AbstractSinglePointSet, y::AbstractSinglePointSet, σᵣ::Number, σₜ::Number; distance_bound_fun = tight_distance_bounds) 
    # sum bounds for each pair of points
    lb = 0.
    ub = 0.
    for xpt in x
        for ypt in y
            lb, ub = (lb, ub) .+ distance_bound_fun(xpt, ypt, σᵣ, σₜ).^2  
        end
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
    ) = sum(abs2, [R.sx, R.sy, R.sz]) ? infbounds(x,y) : squared_dist_bounds(R*x, y-T, σᵣ, σₜ; kwargs...)
