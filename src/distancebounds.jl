const sqrt3 = √(3)
const sqrt2pi = √(2π)
const pisq = Float64(π^2)

function infbounds(x,y) 
    typeinf = typemax(promote_type(numbertype(x), numbertype(y)))
    return (typeinf, typeinf) 
end

function loose_distance_bounds(x::SVector{3,<:Number}, y::SVector{3,<:Number}, σᵣ::Number, σₜ::Number, maximize::Bool = false)
    ubdist = norm(x - y)
    γₜ = sqrt3 * σₜ 
    γᵣ = 2 * sin(min(sqrt3 * σᵣ, π) / 2) * norm(x)
    lb, ub = maximize ? (max(ubdist - γₜ - γᵣ, 0), ubdist) : (ubddist + γₜ + γᵣ, ubdist)
    numtype = promote_type(typeof(lb), typeof(ub))
    return numtype(lb), numtype(ub)
end
loose_distance_bounds(x::SVector{3}, y::SVector{3}, R::RotationVec, T::SVector{3}, σᵣ, σₜ, maximize::Bool = false,
    ) = (R.sx^2 + R.sy^2 + R.sz^2) > pisq ? infbounds(x,y) : loose_distance_bounds(R*x, y-T, σᵣ, σₜ, maximize) # loose_distance_bounds(R*x, y-T, σᵣ, σₜ)
loose_distance_bounds(x::SVector{3}, y::SVector{3}, block::UncertaintyRegion, maximize::Bool = false) = loose_distance_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ, maximize)
loose_distance_bounds(x::SVector{3}, y::SVector{3}, block::SearchRegion, maximize::Bool = false) = loose_distance_bounds(x, y, UncertaintyRegion(block), maximize)


"""
    lb, ub = tight_distance_bounds(x::SVector{3,<:Number}, y::SVector{3,<:Number}, σᵣ::Number, σₜ::Number)
    lb, ub = tight_distance_bounds(x::SVector{3,<:Number}, y::SVector{3,<:Number}, R::RotationVec, T<:SVector{3}, σᵣ::Number, σₜ::Number)

Within an uncertainty region, find the bounds on distance between two points x and y. 

See [Campbell & Peterson, 2016](https://arxiv.org/abs/1603.00150)
"""
function tight_distance_bounds(x::SVector{3,<:Number}, y::SVector{3,<:Number}, σᵣ::Number, σₜ::Number, maximize::Bool = false)
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
    
    if maximize
        # this case is intended for situations where the objective function scales negatively with distance\
        # lbdist, which will be the further point on the spherical cap, will be larger than ubdist
        if cosα + cosβ >= π
            lbdist = xnorm + ynorm + sqrt3*σₜ
        else
            lbdist = √(xnorm^2 + ynorm^2 - 2*xnorm*ynorm*(cosα*cosβ-√((1-cosα^2)*(1-cosβ^2)))) + sqrt3*σₜ
        end
    else
        # lower bound distance from the nearest point on the "spherical cap"
        if cosα >= cosβ
            lbdist = max(abs(xnorm-ynorm) - sqrt3*σₜ, 0)
        else
            lbdist = try max(√(xnorm^2 + ynorm^2 - 2*xnorm*ynorm*(cosα*cosβ+√((1-cosα^2)*(1-cosβ^2)))) - sqrt3*σₜ, 0)  # law of cosines
            catch e     # when the argument for the square root is negative (within machine precision of 0, usually)
                0
            end
        end
    end

    # evaluate objective function at each distance to get upper and lower bounds
    numtype = promote_type(typeof(lbdist), typeof(ubdist))
    return (numtype(lbdist), numtype(ubdist))
end

tight_distance_bounds(x::SVector{3,<:Number}, y::SVector{3,<:Number}, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number, maximize::Bool = false,
    ) = (R.sx^2 + R.sy^2 + R.sz^2) > pisq ? infbounds(x,y) : tight_distance_bounds(R*x, y-T, σᵣ, σₜ, maximize) # tight_distance_bounds(R*x, y-T, σᵣ, σₜ) 
tight_distance_bounds(x::SVector{3}, y::SVector{3}, block::UncertaintyRegion, maximize::Bool = false) = tight_distance_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ, maximize)
tight_distance_bounds(x::SVector{3}, y::SVector{3}, block::Union{RotationRegion, TranslationRegion}, maximize::Bool = false) = tight_distance_bounds(x, y, UncertaintyRegion(block), maximize)
