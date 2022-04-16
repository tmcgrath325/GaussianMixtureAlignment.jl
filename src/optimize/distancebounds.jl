loose_distance_bounds(x<:SVector{3,Number}, y<:SVector{3,Number}, γᵣ<:Number, γₜ<:Number) = (norm(x - y), max(ubdist - γᵣ - γₜ))
loose_distance_bounds(x<:SVector{3}, y<:SVector{3}, R::RotationVec, T<:SVector{3}, γᵣ, γₜ) = distance_bounds(R*x .+ T, y, γᵣ, γₜ)

function tight_distance_bounds(x<:SVector{3,Number}, y<:SVector{3,Number}, R::RotationVec, T<:SVector{3}, γᵣ<:Number, γₜ<:Number)
    # return Inf for bounds if the rotation lies outside the π-sphere
    if γᵣ > π
        inf = typemax(promote_type(numbertype(x),numbertype(y)))
        return inf, inf
    end

    # prepare positions and angles
    x0 = R*x
    xnorm, ynorm = norm(x), norm(y-T)
    if xnorm*ynorm == 0
        cosα = one(promote_type(eltype(x),eltype(y)))
    else
        cosα = dot(x0, y-T)/(xnorm*ynorm)
    end
    cosβ = cos(min(sqrt3*γᵣ/2, π))

    # upper bound distance at hypercube center
    ubdist = norm(x0 + t0 - y.μ)
    
    # lower bound distance from the nearest point on the "spherical cap"
    if cosα >= cosβ
        lbdist = max(abs(xnorm-ynorm) - sqrt3*twidth/2, 0)
    else
        lbdist = try max(√(xnorm^2 + ynorm^2 - 2*xnorm*ynorm*(cosα*cosβ+√((1-cosα^2)*(1-cosβ^2)))) - sqrt3*twidth/2, 0)  # law of cosines
        catch e     # when the argument for the square root is negative (within machine precision of 0, usually)
            0
        end
    end

    # upperbound of dot product between directional constraints (minimizes objective)
    if length(x.dirs) == 0 || length(y.dirs) == 0
        cosγ = 1.
    else
        # NOTE: Avoid list comprehension (slow), but perform more matrix multiplications
        cosγ = -1
        for xdir in x.dirs
            for ydir in y.dirs
                cosγ = max(cosγ, dot(R0*xdir, ydir))
            end
        end
    end

    if cosγ >= cosβ
        lbdot = 1.
    else
        lbdot = cosγ*cosβ + √(1-cosγ^2)*√(1-cosβ^2)
        # lbdot = cosγ*cosβ + √(1 - cosγ^2 - cosβ^2 + cosγ^2*cosβ^2)
    end

    # evaluate objective function at each distance to get upper and lower bounds
    return -overlap(lbdist^2, s, w, lbdot), -overlap(ubdist^2, s, w, cosγ)

end