const sqrt3 = √(3)

"""
    val = objectivefun(dist, σx, σy, ϕx, ϕy)

Calculates the unnormalized overlap between two Gaussian distributions with means
`σx` and `σy`, weights `ϕx` and `ϕy`, and means separated by distance `dist`.
"""
function objectivefun(dist, σx, σy, ϕx, ϕy)
    return -ϕx*ϕy * exp(-dist^2 / (2*(σx^2 + σy^2))) # / sqrt(2π*(σx^2 + σy^2))^dim 
end

""" 
    xrot = roderigues(x, rx, ry, rz)

Uses Roderigues' rotation formula to returns the vector `x` rotated by the axis defined 
by `rx`, `ry`, and `rz`, and an angle equal to the norm of the axis. 
"""
function rodrigues(x, rx, ry, rz)
    angle = √(rx^2 + ry^2 + rz^2) 
    if angle == 0
        axis = SVector(1,0,0)
    else
        axis = SVector(rx,ry,rz)/angle
    end
    return x*cos(angle) + cross(axis, x)*sin(angle) + axis*dot(axis, x)*(1-cos(angle))
end

"""
    lowerbound, upperbound = get_bounds(x, y, rwidth, twidth, X)

Finds the bounds for overlap between two isotropic Gaussian distributions for a particular
region in 6-dimensional rigid rotation space, defined by `rwidth`, `twidth`, and feature
vector `X`.

`X` is a feature vector containing rotation axis components (`rx`, `ry`, and `rz`), and
translation components (`tx`, `ty`, and `tz`).

`rwidth` and `twidth` represent the sizes of the rotation and translatoin cubes, respectively,
around the point defined by `X`.

See [Campbell & Peterson, 2016](https://arxiv.org/abs/1603.00150)
"""
function get_bounds(x::IsotropicGaussian, y::IsotropicGaussian, rwidth, twidth, X)
    rx, ry, rz, tx, ty, tz = X
    
    # return Inf for bounds if the rotation lies outside the π-sphere
    if rx^2 + ry^2 + rz^2 > π^2
        inf = typemax(promote_type(eltype(x),eltype(y)))
        return inf, inf
    end

    # prepare positions and angles
    t0 = SVector(tx,ty,tz)
    x0 = rodrigues(x.μ,rx,ry,rz)
    xnorm, ynorm = norm(x.μ), norm(y.μ-t0)
    if xnorm*ynorm == 0
        cosα = one(promote_type(eltype(x),eltype(y)))
    else
        # α = try acos(dot(x0, y.μ-t0)/(xnorm*ynorm))
        # catch e
        #     if isa(e, DomainError)
        #         acos(copysign(1., dot(x0, y.μ-t0)))
        #     end
        # end
        cosα = dot(x0, y.μ-t0)/(xnorm*ynorm)
    end
    cosβ = cos(min(sqrt3*rwidth/2, π))
    # @show cosα, cosβ

    # upper bound distance at hypercube center
    ubdist = norm(x0 + t0 - y.μ)
    
    # lower bound distance from the nearest point on the "spherical cap"
    if cosα >= cosβ
        lbdist = max(abs(xnorm-ynorm) - sqrt3*twidth/2, 0)
    else
        lbdist = max(√(xnorm^2 + ynorm^2 - 2*xnorm*ynorm*(cosα*cosβ+√((1-cosα^2)*(1-cosβ^2)))) - sqrt3*twidth/2, 0)  # law of cosines
    end

    # evaluate objective function at each distance to get upper and lower bounds
    return objectivefun(lbdist, x.σ, y.σ, x.ϕ, y.ϕ), objectivefun(ubdist, x.σ, y.σ, x.ϕ, y.ϕ)
end

# function get_bounds(gmmx::MolGMM, gmmy::MolGMM, rwidth, twidth, rx, ry, rz, tx, ty, tz)
#     lb = 0.
#     ub = 0.
#     for atomx in gmmx.atoms
#         for atomy in gmmy.atoms
#             l, u = get_bounds(atomx, atomy, rwidth, twidth, rx, ry, rz, tx, ty, tz)  
#             lb, ub = lb+l, ub+u
#         end
#     end
#     return lb, ub
# end

function get_bounds(gmmx::IsotropicGMM, gmmy::IsotropicGMM, rwidth, twidth, X)
    lb = 0.
    ub = 0.
    for x in gmmx.gaussians 
        for y in gmmy.gaussians
            l, u = get_bounds(x, y, rwidth, twidth, X)  
            lb, ub = lb+l, ub+u
        end
    end
    return lb, ub
end