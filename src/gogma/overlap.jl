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
    ovlp = overlap(x::IsotropicGaussian, y::IsotropicGaussian, xtform=identity)

Calculates the unnormalized overlap between two `IsotropicGaussian` objects.
"""
function overlap(x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian, s=nothing, w=nothing)
    if isnothing(s)
        s = x.σ^2 + y.σ^2
    end
    if isnothing(w)
        w = x.ϕ*y.ϕ
    end
    return overlap(sum(abs2, x.μ.-y.μ), s, w) 
end

"""
    ovlp = overlap(x::AbstractSingleGMM, y::AbstractSingleGMM, xtform=identity)

Calculates the unnormalized overlap between two `AbstractSingleGMM` objects.
"""
function overlap(x::AbstractSingleGMM, y::AbstractSingleGMM, pσ=nothing, pϕ=nothing)
    # prepare pairwise widths and weights, if not provided
    if isnothing(pσ) || isnothing(pϕ)
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
    ovlp = overlap(x::AbstractMultiGMM, y::AbstractMultiGMM, xtform=identity)

Calculates the unnormalized overlap between two `AbstractMultiGMM` objects.
"""
function overlap(x::AbstractMultiGMM, y::AbstractMultiGMM, mpσ=nothing, mpϕ=nothing)
    # prepare pairwise widths and weights, if not provided
    if isnothing(mpσ) || isnothing(mpϕ)
        mpσ, mpϕ = pairwise_consts(x, y)
    end
    
    # sum overlaps from each keyed pairs of GMM
    ovlp = zero(promote_type(numbertype(x),numbertype(y)))
    for k in keys(x.gmms) ∩ keys(y.gmms)
        ovlp += overlap(x.gmms[k], y.gmms[k], mpσ[k], mpϕ[k])
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