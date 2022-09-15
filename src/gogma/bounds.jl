loose_distance_bounds(x::AbstractGaussian, y::AbstractGaussian, args...) = loose_distance_bounds(x.μ, y.μ, args...)
tight_distance_bounds(x::AbstractGaussian, y::AbstractGaussian, args...) = tight_distance_bounds(x.μ, y.μ, args...)

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
    lowerbound, upperbound = gauss_l2_bounds(x::Union{IsotropicGaussian, AbstractGMM}, y::Union{IsotropicGaussian, AbstractGMM}, σᵣ, σₜ)
    lowerbound, upperbound = gauss_l2_bounds(x, y, R::RotationVec, T::SVector{3}, σᵣ, σₜ)

Finds the bounds for overlap between two isotropic Gaussian distributions, two isotropic GMMs, or `two sets of 
labeled isotropic GMMs for a particular region in 6-dimensional rigid rotation space, defined by `R`, `T`, `σᵣ` and `σₜ`.

`R` and `T` represent the rotation and translation, respectively, that are at the center of the uncertainty region. If they are not provided, 
the uncertainty region is assumed to be centered at the origin (i.e. x has already been transformed).

`σᵣ` and `σₜ` represent the sizes of the rotation and translation uncertainty regions.

See [Campbell & Peterson, 2016](https://arxiv.org/abs/1603.00150)
"""
function gauss_l2_bounds(x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian, R::RotationVec, T::SVector{3}, σᵣ, σₜ, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; distance_bound_fun = tight_distance_bounds)
    (lbdist, ubdist) = distance_bound_fun(R*x.μ, y.μ-T, σᵣ, σₜ)

    if length(x.dirs) == 0 || length(y.dirs) == 0
        lbdot = 1.
        cosγ = 1.
    else
        cosβ = cos(min(sqrt3*σᵣ, π))
        cosγ = -1.
        for xdir in x.dirs
            for ydir in y.dirs
                cosγ = max(cosγ, dot(xdir, ydir))
            end
        end
        if cosγ >= cosβ
            lbdot = 1.
        else
            lbdot = cosγ*cosβ + √(1-cosγ^2)*√(1-cosβ^2)
            # lbdot = cosγ*cosβ + √(1 - cosγ^2 - cosβ^2 + cosγ^2*cosβ^2)
        end
    end

    # evaluate objective function at each distance to get upper and lower bounds
    return -overlap(lbdist^2, s, w, lbdot), -overlap(ubdist^2, s, w, cosγ)
end

# gauss_l2_bounds(x::AbstractGaussian, y::AbstractGaussian, R::RotationVec, T::SVector{3}, σᵣ, σₜ, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; kwargs...
#     ) = gauss_l2_bounds(R*x, y-T, σᵣ, σₜ, tform.translation, s, w; kwargs...)

gauss_l2_bounds(x::AbstractGaussian, y::AbstractGaussian, block::UncertaintyRegion, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; kwargs...
    ) = gauss_l2_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ, s, w; kwargs...)

gauss_l2_bounds(x::AbstractGaussian, y::AbstractGaussian, block::SearchRegion, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; kwargs...
    ) = gauss_l2_bounds(x, y, UncertaintyRegion(block), s, w; kwargs...)



function gauss_l2_bounds(gmmx::AbstractSingleGMM, gmmy::AbstractSingleGMM, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number, pσ=nothing, pϕ=nothing; kwargs...)
    # prepare pairwise widths and weights, if not provided
    if isnothing(pσ) || isnothing(pϕ)
        pσ, pϕ = pairwise_consts(gmmx, gmmy)
    end

    # sum bounds for each pair of points
    lb = 0.
    ub = 0.
    for (i,x) in enumerate(gmmx.gaussians) 
        for (j,y) in enumerate(gmmy.gaussians)
            lb, ub = (lb, ub) .+ gauss_l2_bounds(x, y, R, T, σᵣ, σₜ, pσ[i,j], pϕ[i,j]; kwargs...)  
        end
    end
    return lb, ub
end

function gauss_l2_bounds(mgmmx::AbstractMultiGMM, mgmmy::AbstractMultiGMM, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number, mpσ=nothing, mpϕ=nothing)
    # prepare pairwise widths and weights, if not provided
    if isnothing(mpσ) || isnothing(mpϕ)
        mpσ, mpϕ = pairwise_consts(mgmmx, mgmmy)
    end

    # sum bounds for each pair of points
    lb = 0.
    ub = 0.
    for key in keys(mgmmx.gmms) ∩ keys(mgmmy.gmms)
        lb, ub = (lb, ub) .+ gauss_l2_bounds(mgmmx.gmms[key], mgmmy.gmms[key], R, T, σᵣ, σₜ, mpσ[key], mpϕ[key])
    end
    return lb, ub
end

# gauss_l2_bounds(x::AbstractGMM, y::AbstractGMM, R::RotationVec, T::SVector{3}, args...; kwargs...
#     ) = gauss_l2_bounds(R*x, y-T, args...; kwargs...)

gauss_l2_bounds(x::AbstractGMM, y::AbstractGMM, block::UncertaintyRegion, args...; kwargs...
    ) = gauss_l2_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ, args...; kwargs...)

gauss_l2_bounds(x::AbstractGMM, y::AbstractGMM, block::SearchRegion, args...; kwargs...
    ) = gauss_l2_bounds(x, y, UncertaintyRegion(block), args...; kwargs...)