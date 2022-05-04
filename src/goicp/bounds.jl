loose_distance_bounds(x::AbstractPoint, y::AbstractPoint, args...) = loose_distance_bounds(x.coords, y.coords, args...)
tight_distance_bounds(x::AbstractPoint, y::AbstractPoint, args...) = tight_distance_bounds(x.coords, y.coords, args...)

function (distance_bound_fun::Union{loose_distance_bounds, tight_distance_bounds})(x::AbstractSinglePointSet, y::AbstractSinglePointSet, σᵣ<:Number, σₜ<:Number) 
    # sum bounds for each pair of points
    lb = 0.
    ub = 0.
    for (i,xpt) in enumerate(x) 
        for (j,ypt) in enumerate(y)
            lb, ub = (lb, ub) .+ distance_bound_fun(xpt, ypt, σᵣ, σₜ)  
        end
    end
    return lb, ub
end

function (distance_bound_fun::Union{loose_distance_bounds, tight_distance_bounds})(x::AbstractMultiPointSet, y::AbstractMultiPointSet, σᵣ, σₜ)
    # sum bounds for each matched pair of pointsets
    lb = 0.
    ub = 0.
    for key in keys(x.pointsets) ∩ keys(y.pointsets)
        lb, ub = (lb, ub) .+ distance_bound_fun(mgmmx.gmms[key], mgmmy.gmms[key], σᵣ, σₜ)
    end
    return lb, ub
end

(distance_bound_fun::Union{loose_distance_bounds, tight_distance_bounds})(x::AbstractPointSet, y::AbstractPointSet, R::RotationVec, T::SVector{3}, σᵣ<:Number, σₜ<:Number
    ) = sum(abs2, [R.sx, R.sy, R.sz]) ? infbounds(x,y) : distance_bound_fun(R*x, y-T, σᵣ, σₜ)
