const sqrt3 = √(3)
const sqrt2pi = √(2π)
const pisq = Float64(π^2)

function infbounds(x,y) 
    typeinf = typemax(promote_type(numbertype(x), numbertype(y)))
    return (typeinf, typeinf) 
end

function loose_distance_bounds(x::SVector{3,<:Number}, y::SVector{3,<:Number}, σᵣ::Number, σₜ::Number)
    ubdist = norm(x - y)
    γₜ = sqrt3 * σₜ 
    γᵣ = 2 * sin(min(sqrt3 * σᵣ, π) / 2)  
    return (max(ubdist - γₜ - γᵣ, 0), ubdist)
end
loose_distance_bounds(x::SVector{3}, y::SVector{3}, R::RotationVec, T::SVector{3}, σᵣ, σₜ
    ) = sum(abs2, [R.sx, R.sy, R.sz]) ? infbounds(x,y) : loose_distance_bounds(R*x, y-T, σᵣ, σₜ)
loose_distance_bounds(x::SVector{3}, y::SVector{3}, block::UncertaintyRegion) = loose_distance_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ)
loose_distance_bounds(x::SVector{3}, y::SVector{3}, block::SearchRegion) = loose_distance_bounds(x, y, UncertaintyRegion(block))


"""
    lb, ub = tight_distance_bounds(x::SVector{3,<:Number}, y::SVector{3,<:Number}, σᵣ::Number, σₜ::Number)
    lb, ub = tight_distance_bounds(x::SVector{3,<:Number}, y::SVector{3,<:Number}, R::RotationVec, T<:SVector{3}, σᵣ::Number, σₜ::Number)

Within an uncertainty region, find the bounds on distance between two points x and y.

See [Campbell & Peterson, 2016](https://arxiv.org/abs/1603.00150)
"""
function tight_distance_bounds(x::SVector{3,<:Number}, y::SVector{3,<:Number}, σᵣ::Number, σₜ::Number)
    # prepare positions and angles
    xnorm, ynorm = norm(x), norm(y)
    if xnorm*ynorm == 0
        cosα = one(promote_type(eltype(x),eltype(y)))
    else
        cosα = dot(x, y)/(xnorm*ynorm) 
    end
    cosβ = cos(min(sqrt3*σᵣ, π))

    # upper bound distance at hypercube center
    ubdist = norm(x - y)
    
    # lower bound distance from the nearest point on the "spherical cap"
    if cosα >= cosβ
        lbdist = max(abs(xnorm-ynorm) - sqrt3*σₜ, 0)
    else
        lbdist = try max(√(xnorm^2 + ynorm^2 - 2*xnorm*ynorm*(cosα*cosβ+√((1-cosα^2)*(1-cosβ^2)))) - sqrt3*σₜ, 0)  # law of cosines
        catch e     # when the argument for the square root is negative (within machine precision of 0, usually)
            0
        end
    end

    # evaluate objective function at each distance to get upper and lower bounds
    return (lbdist, ubdist)
end

tight_distance_bounds(x::SVector{3,<:Number}, y::SVector{3,<:Number}, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number
    ) = sum(abs2, [R.sx, R.sy, R.sz]) ? infbounds(x,y) : tight_distance_bounds(R*x, y-T, σᵣ, σₜ)
tight_distance_bounds(x::SVector{3}, y::SVector{3}, block::UncertaintyRegion) = tight_distance_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ)
tight_distance_bounds(x::SVector{3}, y::SVector{3}, block::Union{RotationRegion, TranslationRegion}) = tight_distance_bounds(x, y, UncertaintyRegion(block))
