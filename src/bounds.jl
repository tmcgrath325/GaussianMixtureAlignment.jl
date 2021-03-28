const sqrt3 = √(3)
const sqrt2pi = √(2π)

function pairwise_consts(gmmx::IsotropicGMM, gmmy::IsotropicGMM)
    t = promote_type(eltype(gmmx),eltype(gmmy))
    pσ, pϕ = zeros(t, length(gmmx), length(gmmy)), zeros(t, length(gmmx), length(gmmy))
    for (i,gaussx) in enumerate(gmmx.gaussians)
        for (j,gaussy) in enumerate(gmmy.gaussians)
            pσ[i,j] = √(gaussx.σ^2 + gaussy.σ^2)
            pϕ[i,j] = gaussx.ϕ * gaussy.ϕ
        end
    end
    return pσ, pϕ
end

"""
    val = objectivefun(dist, s, w, ndims)

Calculates the unnormalized overlap between two Gaussian distributions with width `s`, 
weight `w', and squared distance `distsq`, in `ndims`-dimensional space.
"""
function objectivefun(distsq, s, w, ndims)
    return -w * exp(-distsq / (2*s^2)) / (sqrt2pi * s)^ndims
end

"""
    val = objectivefun(dist, σx, σy, ϕx, ϕy, ndims)

Calculates the unnormalized overlap between two Gaussian distributions with variances
`σx` and `σy`, weights `ϕx` and `ϕy`, and means separated by distance `dist`, in 
`ndims`-dimensional space.
"""
function objectivefun(dist, σx, σy, ϕx, ϕy, ndims)
    return objectivefun(dist^2, sqrt(σx^2 + σy^2), ϕx*ϕy, ndims)
end

"""
    rotmatrix = rot(rx, ry, rz)

Takes a rotation defined by an axis specified by `rx`, `ry`, and `rz`, and an angle equal
to the norm of the axis, and returns a rotation matrix by converting the angle-axis
parameterization to a [unit quaternion](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)
parametrization.
"""
function rot(rx, ry, rz)
    # Angle-Axis parameterization
    angle = √(rx^2 + ry^2 + rz^2) 
    if angle==0
        return SMatrix{3,3,typeof(rx)}(I)
    end
    # Convert to quaternion
    qr = cos(angle/2)
    nrm = angle/sin(angle/2)
    qx = rx/nrm
    qy = ry/nrm
    qz = rz/nrm
    R = @SMatrix([1 - 2*(qy^2 + qz^2)  2*(qx*qy - qz*qr)    2*(qx*qz + qy*qr);
                  2*(qx*qy + qz*qr)    1 - 2*(qx^2 + qz^2)  2*(qy*qz - qx*qr);
                  2*(qx*qz - qy*qr)    2*(qy*qz + qx*qr)    1 - 2*(qx^2 + qy^2)])
    return R
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
function get_bounds(x::IsotropicGaussian, y::IsotropicGaussian, rwidth, twidth, X, R0=rot(X[1:3]...), t0=SVector(X[4:6]...), s=√(x.σ^2 + y.σ^2), w=x.ϕ*y.ϕ)
    rx, ry, rz, tx, ty, tz = X
    
    # return Inf for bounds if the rotation lies outside the π-sphere
    if rx^2 + ry^2 + rz^2 > π^2
        inf = typemax(promote_type(eltype(x),eltype(y)))
        return inf, inf
    end

    # prepare positions and angles
    x0 = R0*x.μ
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
    return objectivefun(lbdist^2, s, w, 3), objectivefun(ubdist^2, s, w, 3)
end

function get_bounds(gmmx::IsotropicGMM, gmmy::IsotropicGMM, rwidth, twidth, X, pσ=nothing, pϕ=nothing)
    # prepare pairwise widths and weights
    if isnothing(pσ) || isnothing(pϕ)
        pσ, pϕ = pairwise_consts(gmmx, gmmy)
    end

    # sum bounds for each pair of points
    rx, ry, rz, tx, ty, tz = X
    lb = 0.
    ub = 0.
    R0 = rot(rx, ry, rz)
    t0 = SVector(tx, ty, tz)
    for (i,x) in enumerate(gmmx.gaussians) 
        for (j,y) in enumerate(gmmy.gaussians)
            l, u = get_bounds(x, y, rwidth, twidth, X, R0, t0, pσ[i,j], pϕ[i,j])  
            lb, ub = lb+l, ub+u
        end
    end
    return lb, ub
end