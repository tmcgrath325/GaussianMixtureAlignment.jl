loose_distance_bounds(x::AbstractGaussian, y::AbstractGaussian, args...) = loose_distance_bounds(x.μ, y.μ, args...)
tight_distance_bounds(x::AbstractGaussian, y::AbstractGaussian, args...) = tight_distance_bounds(x.μ, y.μ, args...)

function validate_interactions(interactions::Dict{Tuple{K,K},V}) where {K,V<:Number}
    for (k1,k2) in keys(interactions)
        if k1 != k2
            if haskey(interactions, (k2,k1))
                return false
            end
        end
    end
    return true
end

# prepare pairwise values for `σx^2 + σy^2` and `ϕx * ϕy` for all gaussians in `gmmx` and `gmmy`
function pairwise_consts(gmmx::AbstractIsotropicGMM, gmmy::AbstractIsotropicGMM, interactions=nothing)
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

function pairwise_consts(mgmmx::AbstractMultiGMM{N,T,K}, mgmmy::AbstractMultiGMM{N,S,K}, interactions::Union{Nothing,Dict{Tuple{K,K},V}}=nothing) where {N,T,S,K,V <: Number}
    t = promote_type(numbertype(mgmmx),numbertype(mgmmy), isnothing(interactions) ? numbertype(mgmmx) : V)
    xkeys = keys(mgmmx.gmms)
    ykeys = keys(mgmmy.gmms)
    if isnothing(interactions)
        interactions = Dict{Tuple{K,K},t}()
        for key in xkeys ∩ ykeys
            interactions[(key,key)] = one(t)
        end
    else
        @assert validate_interactions(interactions) "Interactions must not include redundant key pairs (i.e. (k1,k2) and (k2,k1))"
    end
    mpσ, mpϕ = Dict{K, Dict{K, Matrix{t}}}(), Dict{K, Dict{K,Matrix{t}}}()
    ukeys = unique(Iterators.flatten(keys(interactions)))
    for key1 in ukeys
        if key1 ∈ xkeys 
            push!(mpσ, key1 => Dict{K, Matrix{t}}())
            push!(mpϕ, key1 => Dict{K, Matrix{t}}())
            for key2 in ukeys
                keypair = (key1,key2)
                keypair = haskey(interactions, keypair) ? keypair : (key2,key1)
                if key2 ∈ ykeys && haskey(interactions, keypair)
                    pσ, pϕ = pairwise_consts(mgmmx.gmms[key1], mgmmy.gmms[key2])
                    push!(mpσ[key1], key2 => pσ)
                    push!(mpϕ[key1], key2 => interactions[keypair] .* pϕ)
                end
            end
            if isempty(mpσ[key1])
                delete!(mpσ, key1)
                delete!(mpϕ, key1)
            end
        end
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
    (lbdist, ubdist) = distance_bound_fun(R*x.μ, y.μ-T, σᵣ, σₜ, w < 0)

    # evaluate objective function at each distance to get upper and lower bounds
    return -overlap(lbdist^2, s, w), -overlap(ubdist^2, s, w)
end

# gauss_l2_bounds(x::AbstractGaussian, y::AbstractGaussian, R::RotationVec, T::SVector{3}, σᵣ, σₜ, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; kwargs...
#     ) = gauss_l2_bounds(R*x, y-T, σᵣ, σₜ, tform.translation, s, w; kwargs...)

gauss_l2_bounds(x::AbstractGaussian, y::AbstractGaussian, block::UncertaintyRegion, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; kwargs...
    ) = gauss_l2_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ, s, w; kwargs...)

gauss_l2_bounds(x::AbstractGaussian, y::AbstractGaussian, block::SearchRegion, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; kwargs...
    ) = gauss_l2_bounds(x, y, UncertaintyRegion(block), s, w; kwargs...)



function gauss_l2_bounds(gmmx::AbstractSingleGMM, gmmy::AbstractSingleGMM, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number, pσ=nothing, pϕ=nothing, interactions=nothing; kwargs...)
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

function gauss_l2_bounds(mgmmx::AbstractMultiGMM, mgmmy::AbstractMultiGMM, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number, mpσ=nothing, mpϕ=nothing, interactions=nothing)
    # prepare pairwise widths and weights, if not provided
    if isnothing(mpσ) || isnothing(mpϕ)
        mpσ, mpϕ = pairwise_consts(mgmmx, mgmmy, interactions)
    end

    # sum bounds for each pair of points
    lb = 0.
    ub = 0.
    for (key1, intrs) in mpσ
        for (key2, pσ) in intrs
            lb, ub = (lb, ub) .+ gauss_l2_bounds(mgmmx.gmms[key1], mgmmy.gmms[key2], R, T, σᵣ, σₜ, pσ, mpϕ[key1][key2])
        end
    end
    return lb, ub
end

# gauss_l2_bounds(x::AbstractGMM, y::AbstractGMM, R::RotationVec, T::SVector{3}, args...; kwargs...
#     ) = gauss_l2_bounds(R*x, y-T, args...; kwargs...)

gauss_l2_bounds(x::AbstractGMM, y::AbstractGMM, block::UncertaintyRegion, args...; kwargs...
    ) = gauss_l2_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ, args...; kwargs...)

gauss_l2_bounds(x::AbstractGMM, y::AbstractGMM, block::SearchRegion, args...; kwargs...
    ) = gauss_l2_bounds(x, y, UncertaintyRegion(block), args...; kwargs...)