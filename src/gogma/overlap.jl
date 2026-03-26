"""
    ovlp = overlap(distsq, s, w)

Calculates the unnormalized overlap between two Gaussian distributions with width `s`,
weight `w', and squared distance `distsq`.
"""
function overlap(distsq::Real, s::Real, w::Real)
    return w * exp(-distsq / (2*s)) # / (sqrt2pi * sqrt(s))^ndims
    # Note, the normalization term for the Gaussians is left out, since it is not required that the total "volume" of each Gaussian
    # is equal to 1 (e.g. satisfying the requirements for a probability distribution)
end

"""
    ovlp = overlap(dist, σx, σy, ϕx, ϕy)

Calculates the unnormalized overlap between two Gaussian distributions with variances
`σx` and `σy`, weights `ϕx` and `ϕy`, and means separated by distance `dist`.
"""
function overlap(dist::Real, σx::Real, σy::Real, ϕx::Real, ϕy::Real)
    return overlap(dist^2, σx^2 + σy^2, ϕx*ϕy)
end

"""
    ovlp = overlap(x::IsotropicGaussian, y::IsotropicGaussian)

Calculates the unnormalized overlap between two `IsotropicGaussian` objects.
"""
function overlap(x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian, s=x.σ^2+y.σ^2, w=x.ϕ*y.ϕ)
    return overlap(sum(abs2, x.μ.-y.μ), s, w)
end

"""
    ovlp = overlap(x::AbstractSingleGMM, y::AbstractSingleGMM)

Calculates the unnormalized overlap between two `AbstractSingleGMM` objects.
"""
function overlap(x::AbstractSingleGMM, y::AbstractSingleGMM, pσ=nothing, pϕ=nothing)
    # prepare pairwise widths and weights, if not provided
    if isnothing(pσ) && isnothing(pϕ)
        pσ, pϕ = pairwise_consts(x, y)
    end

    # sum overlaps for all pairwise combinations of Gaussians between x and y
    ovlp = zero(promote_type(numbertype(x),numbertype(y)))
    for (i,gx) in enumerate(x.gaussians)
        for (j,gy) in enumerate(y.gaussians)
            ovlp += overlap(gx, gy, pσ[i,j], pϕ[i,j])
        end
    end
    return ovlp
end

"""
    ovlp = overlap(x::AbstractMultiGMM, y::AbstractMultiGMM)

Calculates the unnormalized overlap between two `AbstractMultiGMM` objects.
"""
function overlap(x::AbstractMultiGMM, y::AbstractMultiGMM, mpσ=nothing, mpϕ=nothing, interactions=nothing)
    # prepare pairwise widths and weights, if not provided
    if isnothing(mpσ) && isnothing(mpϕ)
        mpσ, mpϕ = pairwise_consts(x, y, interactions)
    end

    # sum overlaps from each keyed pairs of GMM
    ovlp = zero(promote_type(numbertype(x),numbertype(y)))
    for k1 in keys(mpσ)
        for k2 in keys(mpσ[k1])
            ovlp += overlap(x.gmms[k1], y.gmms[k2], mpσ[k1][k2], mpϕ[k1][k2])
        end
    end
    return ovlp
end

"""
    l2dist = distance(x, y)

Calculates the L2 distance between two GMMs made up of spherical Gaussian distributions.
"""
function distance(x::AbstractGMM, y::AbstractGMM)
    return overlap(x,x) + overlap(y,y) - 2*overlap(x,y)
end

"""
    tani = tanimoto(x, y)

Calculates the tanimoto distance based on Gaussian overlap between two GMMs.
"""
function tanimoto(x::AbstractGMM, y::AbstractGMM)
    o = overlap(x,y)
    return o / (overlap(x,x) + overlap(y,y) - o)
end

## Forces

"""
    force!(f, x, y)

Add to `f` the force on model `x` due to its overlap with model `y`, i.e. the gradient of
`overlap(x, y)` with respect to the mean positions of the Gaussians in `x`. `f` must be a
mutable vector of length equal to the spatial dimension.

Supports `AbstractIsotropicGaussian`, `AbstractIsotropicGMM`, and `AbstractMultiGMM` inputs.
"""
function force!(f::AbstractVector, x::AbstractVector, y::AbstractVector, s::Real, w::Real)
    Δ = y - x
    f .+= Δ / s * overlap(sum(abs2, Δ), s, w)
end

function force!(f::AbstractVector, x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian,
                s=x.σ^2+y.σ^2, w=x.ϕ*y.ϕ; coef=1)
    return force!(f, x.μ, y.μ, s, coef*w)
end

function force!(f::AbstractVector, x::AbstractIsotropicGaussian, y::AbstractIsotropicGMM, pσ=nothing, pϕ=nothing; kwargs...)
    if isnothing(pσ) && isnothing(pϕ)
        xσsq = x.σ^2
        pσ = [xσsq + gy.σ^2 for gy in y.gaussians]
        pϕ = [x.ϕ * gy.ϕ for gy in y.gaussians]
    end
    for (gy, s, w) in zip(y.gaussians, pσ, pϕ)
        force!(f, x, gy, s, w; kwargs...)
    end
end

function force!(f::AbstractVector, x::AbstractIsotropicGMM, y::AbstractIsotropicGMM, pσ=nothing, pϕ=nothing; kwargs...)
    if isnothing(pσ) && isnothing(pϕ)
        pσ, pϕ = pairwise_consts(x, y)
    end
    for (i,gx) in enumerate(x.gaussians)
        force!(f, gx, y, pσ[i,:], pϕ[i,:]; kwargs...)
    end
end

function force!(f::AbstractVector, x::AbstractMultiGMM, y::AbstractMultiGMM; interactions=nothing)
    mpσ, mpϕ = pairwise_consts(x, y, interactions)
    for k1 in keys(mpσ)
        for k2 in keys(mpσ[k1])
            # don't pass coef as a keyword argument, since the interaction coefficient is baked into mpϕ
            force!(f, x.gmms[k1], y.gmms[k2], mpσ[k1][k2], mpϕ[k1][k2])
        end
    end
end