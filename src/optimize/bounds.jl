const sqrt3 = √(3)
const sqrt2pi = √(2π)

# prepare pairwise values for `σx^2 + σy^2` and `ϕx * ϕy` for all gaussians in `gmmx` and `gmmy`
function pairwise_consts(gmmx::AbstractIsotropicGMM, gmmy::AbstractIsotropicGMM)
    t = promote_type(numbertype(gmmx),numbertype(gmmy))
    pσ, pϕ = zeros(t, length(gmmx), length(gmmy)), zeros(t, length(gmmx), length(gmmy))
    for (i,gaussx) in enumerate(gmmx.gaussians)
        for (j,gaussy) in enumerate(gmmy.gaussians)
            pσ[i,j] = gaussx.σ^2 + gaussy.σ^2
            pϕ[i,j] = gaussx.ϕ * gaussy.ϕ
        end
    end
    return pσ, pϕ
end

function pairwise_consts(mgmmx::AbstractMultiGMM{N,T,K}, mgmmy::AbstractMultiGMM{N,S,K}) where {N,T,S,K}
    t = promote_type(numbertype(mgmmx),numbertype(mgmmy))
    mpσ, mpϕ = Dict{K, Matrix{t}}(), Dict{K, Matrix{t}}()
    for key in keys(mgmmx.gmms) ∩ keys(mgmmy.gmms)
        pσ, pϕ = pairwise_consts(mgmmx.gmms[key], mgmmy.gmms[key])
        push!(mpσ, Pair(key, pσ))
        push!(mpϕ, Pair(key, pϕ))
    end
    return mpσ, mpϕ
end


"""
    lowerbound, upperbound = get_bounds(x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian, rwidth, twidth, X)
    lowerbound, upperbound = get_bounds(gmmx::AbstractSingleGMM, gmmy::AbstractSingleGMM, rwidth, twidth, X)
    lowerbound, upperbound = get_bounds(mgmmx::AbstractMultiGMM, gmmy::AbstractMultiGMM, rwidth, twidth, X)

Finds the bounds for overlap between two isotropic Gaussian distributions, two isotropic GMMs, or `two sets of 
labeled isotropic GMMs for a particular region in 6-dimensional rigid rotation space, defined by `rwidth`, `twidth`, 
and feature vector `X`.

`X` is a feature vector containing rotation axis components (`rx`, `ry`, and `rz`), and
translation components (`tx`, `ty`, and `tz`).

`rwidth` and `twidth` represent the sizes of the rotation and translatoin cubes, respectively,
around the point defined by `X`.

See [Campbell & Peterson, 2016](https://arxiv.org/abs/1603.00150)
"""
function get_bounds(x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian, rwidth, twidth, θ::Real, R0, t0, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ)
    # return Inf for bounds if the rotation lies outside the π-sphere
    if θ > π
        inf = typemax(promote_type(numbertype(x),numbertype(y)))
        return inf, inf
    end

    # prepare positions and angles
    x0 = R0*x.μ
    xnorm, ynorm = norm(x.μ), norm(y.μ-t0)
    if xnorm*ynorm == 0
        cosα = one(promote_type(numbertype(x),numbertype(y)))
    else
        cosα = dot(x0, y.μ-t0)/(xnorm*ynorm)
    end
    cosβ = cos(min(sqrt3*rwidth/2, π))

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

get_bounds(x::AbstractGaussian, y::AbstractGaussian, rwidth, twidth, tform::AffineMap, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ
    ) = get_bounds(x, y, rwidth, twidth, tform.linear.theta, RotMatrix(tform.linear), tform.translation, s, w)

get_bounds(x::AbstractGaussian, y::AbstractGaussian, rwidth, twidth, X, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ
    ) = get_bounds(x, y, rwidth, twidth, AffineMap(X...), s, w)

function get_bounds(gmmx::AbstractSingleGMM, gmmy::AbstractSingleGMM, rwidth, twidth, θ::Real, R0, t0, pσ=nothing, pϕ=nothing)
    # prepare pairwise widths and weights, if not provided
    if isnothing(pσ) || isnothing(pϕ)
        pσ, pϕ = pairwise_consts(gmmx, gmmy)
    end

    # sum bounds for each pair of points
    lb = 0.
    ub = 0.
    for (i,x) in enumerate(gmmx.gaussians) 
        for (j,y) in enumerate(gmmy.gaussians)
            lb, ub = (lb, ub) .+ get_bounds(x, y, rwidth, twidth, θ, R0, t0, pσ[i,j], pϕ[i,j])  
        end
    end
    return lb, ub
end

function get_bounds(mgmmx::AbstractMultiGMM, mgmmy::AbstractMultiGMM, rwidth, twidth, θ::Real, R0, t0, mpσ=nothing, mpϕ=nothing)
    # prepare pairwise widths and weights, if not provided
    if isnothing(mpσ) || isnothing(mpϕ)
        mpσ, mpϕ = pairwise_consts(mgmmx, mgmmy)
    end

    # sum bounds for each pair of points
    lb = 0.
    ub = 0.
    for key in keys(mgmmx.gmms) ∩ keys(mgmmy.gmms)
        lb, ub = (lb, ub) .+ get_bounds(mgmmx.gmms[key], mgmmy.gmms[key], rwidth, twidth, θ, R0, t0, mpσ[key], mpϕ[key])
    end
    return lb, ub
end

get_bounds(gmmx::AbstractGMM, gmmy::AbstractGMM, rwidth, twidth, tform::AffineMap, pσ=nothing, pϕ=nothing
    ) = get_bounds(gmmx, gmmy, rwidth, twidth, tform.linear.theta, RotMatrix(tform.linear), tform.translation, pσ, pϕ)

get_bounds(gmmx::AbstractGMM, gmmy::AbstractGMM, rwidth, twidth, X, pσ=nothing, pϕ=nothing
    ) = get_bounds(gmmx, gmmy, rwidth, twidth, AffineMap(X...), pσ, pϕ)