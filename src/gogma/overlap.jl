"""
    ovlp = overlap(distsq, s, w)

Calculate the unnormalized overlap between two Gaussian distributions with width `s`,
weight `w`, and squared distance `distsq`.
"""
function overlap(distsq::Real, s::Real, w::Real)
    return w * exp(-distsq / (2 * s)) # / (sqrt2pi * sqrt(s))^ndims
    # Note, the normalization term for the Gaussians is left out, since it is not required that the total "volume" of each Gaussian
    # is equal to 1 (e.g. satisfying the requirements for a probability distribution)
end

"""
    ovlp = overlap(dist, σx, σy, ϕx, ϕy)

Calculate the unnormalized overlap between two Gaussian distributions with standard deviations
`σx` and `σy`, weights `ϕx` and `ϕy`, and means separated by distance `dist`.
"""
function overlap(dist::Real, σx::Real, σy::Real, ϕx::Real, ϕy::Real)
    return overlap(dist^2, σx^2 + σy^2, ϕx * ϕy)
end

"""
    ovlp = overlap(x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian, s=x.σ^2+y.σ^2, w=x.ϕ*y.ϕ)

Calculate the unnormalized overlap between two `AbstractIsotropicGaussian` objects.
`s` and `w` are the combined width and weight; supply precomputed values to avoid
redundant calculation when calling in a loop.
"""
function overlap(x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian, s = x.σ^2 + y.σ^2, w = x.ϕ * y.ϕ)
    return overlap(sum(abs2, x.μ .- y.μ), s, w)
end

"""
    ovlp = overlap(x::AbstractSingleGMM, y::AbstractSingleGMM)

Calculate the unnormalized overlap between two `AbstractSingleGMM` objects.
"""
function overlap(x::AbstractSingleGMM, y::AbstractSingleGMM, pσ = nothing, pϕ = nothing)
    # prepare pairwise widths and weights, if not provided
    if isnothing(pσ) && isnothing(pϕ)
        pσ, pϕ = pairwise_consts(x, y)
    end

    # sum overlaps for all pairwise combinations of Gaussians between x and y
    ovlp = zero(promote_type(numbertype(x), numbertype(y)))
    for (i, gx) in enumerate(x.gaussians)
        for (j, gy) in enumerate(y.gaussians)
            ovlp += overlap(gx, gy, pσ[i, j], pϕ[i, j])
        end
    end
    return ovlp
end

"""
    ovlp = overlap(x::AbstractLabeledIsotropicGMM, y::AbstractLabeledIsotropicGMM; interactions=nothing)

Calculate the unnormalized overlap between two `AbstractLabeledIsotropicGMM` objects. The
optional keyword argument `interactions` is a dictionary mapping `(label1, label2)` pairs to
coefficient values; see `pairwise_consts` for the expected format. When omitted, only Gaussians
with equal labels contribute, each with coefficient 1.
"""
function overlap(x::AbstractLabeledIsotropicGMM, y::AbstractLabeledIsotropicGMM; interactions = nothing)
    pσ, pϕ = pairwise_consts(x, y, interactions)
    return overlap(x, y, pσ, pϕ)
end

"""
    ovlp = overlap(x::AbstractMultiGMM, y::AbstractMultiGMM; interactions=nothing)

Calculate the unnormalized overlap between two `AbstractMultiGMM` objects. The optional
keyword argument `interactions` is a dictionary mapping `(key1, key2)` pairs to coefficient
values; see `pairwise_consts` for the expected format.
"""
function overlap(x::AbstractMultiGMM, y::AbstractMultiGMM, mpσ = nothing, mpϕ = nothing; interactions = nothing)
    # prepare pairwise widths and weights, if not provided
    if isnothing(mpσ) && isnothing(mpϕ)
        mpσ, mpϕ = pairwise_consts(x, y, interactions)
    end

    # sum overlaps from each keyed pairs of GMM
    ovlp = zero(promote_type(numbertype(x), numbertype(y)))
    for k1 in keys(mpσ)
        for k2 in keys(mpσ[k1])
            ovlp += overlap(x.gmms[k1], y.gmms[k2], mpσ[k1][k2], mpϕ[k1][k2])
        end
    end
    return ovlp
end

function overlap(x::AbstractMultiGMM, y::AbstractMultiGMM, mpσ, mpϕ, interactions)
    Base.depwarn(
        "Passing `interactions` as the 5th positional argument to `overlap` is deprecated; " *
            "use `overlap(x, y, mpσ, mpϕ; interactions=interactions)` instead.",
        :overlap
    )
    return overlap(x, y, mpσ, mpϕ; interactions)
end

"""
    l2dist = distance(x, y)

Calculates the L2 distance between two GMMs made up of spherical Gaussian distributions.
"""
function distance(x::AbstractGMM, y::AbstractGMM)
    return overlap(x, x) + overlap(y, y) - 2 * overlap(x, y)
end

"""
    tani = tanimoto(x, y)

Calculates the tanimoto distance based on Gaussian overlap between two GMMs.
"""
function tanimoto(x::AbstractGMM, y::AbstractGMM)
    o = overlap(x, y)
    return o / (overlap(x, x) + overlap(y, y) - o)
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
    return f .+= Δ / s * overlap(sum(abs2, Δ), s, w)
end

function force!(
        f::AbstractVector, x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian,
        s = x.σ^2 + y.σ^2, w = x.ϕ * y.ϕ; coef = 1
    )
    return force!(f, x.μ, y.μ, s, coef * w)
end

function force!(f::AbstractVector, x::AbstractIsotropicGaussian, y::AbstractIsotropicGMM, pσ = nothing, pϕ = nothing; kwargs...)
    if isnothing(pσ) && isnothing(pϕ)
        xσsq = x.σ^2
        pσ = [xσsq + gy.σ^2 for gy in y.gaussians]
        pϕ = [x.ϕ * gy.ϕ for gy in y.gaussians]
    end
    for (gy, s, w) in zip(y.gaussians, pσ, pϕ)
        force!(f, x, gy, s, w; kwargs...)
    end
    return
end

function force!(f::AbstractVector, x::AbstractIsotropicGMM, y::AbstractIsotropicGMM, pσ = nothing, pϕ = nothing; kwargs...)
    if isnothing(pσ) && isnothing(pϕ)
        pσ, pϕ = pairwise_consts(x, y)
    end
    for (i, gx) in enumerate(x.gaussians)
        force!(f, gx, y, pσ[i, :], pϕ[i, :]; kwargs...)
    end
    return
end

function force!(f::AbstractVector, x::AbstractLabeledIsotropicGMM, y::AbstractLabeledIsotropicGMM, pσ = nothing, pϕ = nothing; interactions = nothing, kwargs...)
    if isnothing(pσ) && isnothing(pϕ)
        pσ, pϕ = pairwise_consts(x, y, interactions)
    end
    for (i, gx) in enumerate(x.gaussians)
        force!(f, gx, y, pσ[i, :], pϕ[i, :]; kwargs...)
    end
    return
end

function force!(f::AbstractVector, x::AbstractMultiGMM, y::AbstractMultiGMM; interactions = nothing)
    mpσ, mpϕ = pairwise_consts(x, y, interactions)
    for k1 in keys(mpσ)
        for k2 in keys(mpσ[k1])
            # don't pass coef as a keyword argument, since the interaction coefficient is baked into mpϕ
            force!(f, x.gmms[k1], y.gmms[k2], mpσ[k1][k2], mpϕ[k1][k2])
        end
    end
    return
end

"""
    f = force(x, y)

Return the force on model `x` due to its overlap with model `y`, i.e. the gradient of
`overlap(x, y)` with respect to the mean positions of the Gaussians in `x`, as a newly
allocated vector. See [`force!`](@ref) for the mutating form.

Supports `AbstractIsotropicGaussian`, `AbstractIsotropicGMM`, and `AbstractMultiGMM` inputs.
"""
function force(
        x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian,
        s = x.σ^2 + y.σ^2, w = x.ϕ * y.ϕ; coef = 1
    )
    f = zeros(promote_type(eltype(x.μ), eltype(y.μ)), length(x.μ))
    force!(f, x, y, s, w; coef)
    return f
end

function force(x::AbstractIsotropicGMM, y::AbstractIsotropicGMM, pσ = nothing, pϕ = nothing; kwargs...)
    f = zeros(promote_type(numbertype(x), numbertype(y)), dims(x))
    force!(f, x, y, pσ, pϕ; kwargs...)
    return f
end

function force(x::AbstractMultiGMM, y::AbstractMultiGMM; interactions = nothing)
    f = zeros(promote_type(numbertype(x), numbertype(y)), dims(x))
    force!(f, x, y; interactions)
    return f
end
