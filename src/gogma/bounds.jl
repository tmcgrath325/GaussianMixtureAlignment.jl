loose_distance_bounds(x::AbstractGaussian, y::AbstractGaussian, args...) = loose_distance_bounds(x.μ, y.μ, args...)
tight_distance_bounds(x::AbstractGaussian, y::AbstractGaussian, args...) = tight_distance_bounds(x.μ, y.μ, args...)

function validate_interactions(interactions::Dict{Tuple{K, K}, V}) where {K, V <: Number}
    for (k1, k2) in keys(interactions)
        if k1 != k2
            if haskey(interactions, (k2, k1))
                return false
            end
        end
    end
    return true
end

"""
    pσ, pϕ = pairwise_consts(gmmx, gmmy, interactions=nothing)

Precompute, for every pair of Gaussians drawn from `gmmx` and `gmmy`, the combined variance
`σx^2 + σy^2` (`pσ`) and the amplitude product `ϕx * ϕy` (`pϕ`). For `AbstractIsotropicGMM`
inputs the results are dense matrices indexed by Gaussian; for `AbstractMultiGMM` inputs they
are nested dictionaries keyed by component label, with `interactions` weighting the
cross-label terms.

These constants depend only on the Gaussians' widths and amplitudes, not on the relative
transformation, so they are invariant under the rigid search. They are the per-pair inputs to
`gauss_l2_bounds` and `local_align`. The `*_align` entry points call `pairwise_consts` once and
capture the result in the bounds and local-refinement closures passed to `branchbound`; computing
them once amortizes the `O(length(gmmx) * length(gmmy))` work across every node of the search.
"""
function pairwise_consts(gmmx::AbstractIsotropicGMM, gmmy::AbstractIsotropicGMM, interactions::Nothing = nothing)
    t = promote_type(numbertype(gmmx), numbertype(gmmy))
    pσ, pϕ = zeros(t, length(gmmx), length(gmmy)), zeros(t, length(gmmx), length(gmmy))
    for (i, gaussx) in enumerate(gmmx.gaussians)
        for (j, gaussy) in enumerate(gmmy.gaussians)
            pσ[i, j] = gaussx.σ^2 + gaussy.σ^2
            pϕ[i, j] = gaussx.ϕ * gaussy.ϕ
        end
    end
    return pσ, pϕ
end

function pairwise_consts(gmmx::AbstractLabeledIsotropicGMM{N, T, K}, gmmy::AbstractLabeledIsotropicGMM{N, S, K}, interactions::Nothing = nothing) where {N, T, S, K}
    t = promote_type(T, S)
    self_interactions = Dict{Tuple{K, K}, t}()
    for label in unique(gmmx.labels) ∩ unique(gmmy.labels)
        self_interactions[(label, label)] = one(t)
    end
    return pairwise_consts(gmmx, gmmy, self_interactions)
end

function pairwise_consts(gmmx::AbstractLabeledIsotropicGMM{N, T, K}, gmmy::AbstractLabeledIsotropicGMM{N, S, K}, interactions::Dict{Tuple{K, K}, V}) where {N, T, S, K, V <: Number}
    validate_interactions(interactions) || throw(ArgumentError("Interactions must not include redundant key pairs (i.e. (k1,k2) and (k2,k1))"))
    t = promote_type(T, S, V)
    pσ, pϕ = zeros(t, length(gmmx), length(gmmy)), zeros(t, length(gmmx), length(gmmy))
    for (i, gaussx) in enumerate(gmmx.gaussians)
        for (j, gaussy) in enumerate(gmmy.gaussians)
            keypair = (gmmx.labels[i], gmmy.labels[j])
            keypair = haskey(interactions, keypair) ? keypair : (keypair[2], keypair[1])
            pσ[i, j] = gaussx.σ^2 + gaussy.σ^2
            pϕ[i, j] = (haskey(interactions, keypair) ? interactions[keypair] : zero(t)) * gaussx.ϕ * gaussy.ϕ
        end
    end
    return pσ, pϕ
end

function pairwise_consts(mgmmx::AbstractMultiGMM{N, T, K}, mgmmy::AbstractMultiGMM{N, S, K}, interactions::Nothing = nothing) where {N, T, S, K}
    t = promote_type(T, S)
    self_interactions = Dict{Tuple{K, K}, t}()
    for key in keys(mgmmx.gmms) ∩ keys(mgmmy.gmms)
        self_interactions[(key, key)] = one(t)
    end
    return pairwise_consts(mgmmx, mgmmy, self_interactions)
end

function pairwise_consts(mgmmx::AbstractMultiGMM{N, T, K}, mgmmy::AbstractMultiGMM{N, S, K}, interactions::Dict{Tuple{K, K}, V}) where {N, T, S, K, V <: Number}
    t = promote_type(T, S, V)
    xkeys = keys(mgmmx.gmms)
    ykeys = keys(mgmmy.gmms)
    validate_interactions(interactions) || throw(ArgumentError("Interactions must not include redundant key pairs (i.e. (k1,k2) and (k2,k1))"))
    mpσ, mpϕ = Dict{K, Dict{K, Matrix{t}}}(), Dict{K, Dict{K, Matrix{t}}}()
    ukeys = unique(Iterators.flatten(keys(interactions)))
    for key1 in ukeys
        if key1 ∈ xkeys
            push!(mpσ, key1 => Dict{K, Matrix{t}}())
            push!(mpϕ, key1 => Dict{K, Matrix{t}}())
            for key2 in ukeys
                keypair = (key1, key2)
                keypair = haskey(interactions, keypair) ? keypair : (key2, key1)
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
function gauss_l2_bounds(x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian, R::RotationVec, T::SVector{3}, σᵣ, σₜ, s = x.σ^2 + y.σ^2, w = x.ϕ * y.ϕ; distance_bound_fun = tight_distance_bounds)
    (lbdist, ubdist) = distance_bound_fun(R * x.μ, y.μ - T, σᵣ, σₜ, w < 0)

    # evaluate objective function at each distance to get upper and lower bounds
    return -overlap(lbdist^2, s, w), -overlap(ubdist^2, s, w)
end

# gauss_l2_bounds(x::AbstractGaussian, y::AbstractGaussian, R::RotationVec, T::SVector{3}, σᵣ, σₜ, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; kwargs...
#     ) = gauss_l2_bounds(R*x, y-T, σᵣ, σₜ, tform.translation, s, w; kwargs...)

gauss_l2_bounds(
    x::AbstractGaussian, y::AbstractGaussian, block::UncertaintyRegion, s = x.σ^2 + y.σ^2, w = x.ϕ * y.ϕ; kwargs...
) = gauss_l2_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ, s, w; kwargs...)

gauss_l2_bounds(
    x::AbstractGaussian, y::AbstractGaussian, block::SearchRegion, s = x.σ^2 + y.σ^2, w = x.ϕ * y.ϕ; kwargs...
) = gauss_l2_bounds(x, y, UncertaintyRegion(block), s, w; kwargs...)


function gauss_l2_bounds(gmmx::AbstractSingleGMM, gmmy::AbstractSingleGMM, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number, pσ = nothing, pϕ = nothing, interactions = nothing; kwargs...)
    # prepare pairwise widths and weights, if not provided
    if isnothing(pσ) || isnothing(pϕ)
        pσ, pϕ = pairwise_consts(gmmx, gmmy)
    end

    # sum bounds for each pair of points
    lb = 0.0
    ub = 0.0
    for (i, x) in enumerate(gmmx.gaussians)
        for (j, y) in enumerate(gmmy.gaussians)
            lb, ub = (lb, ub) .+ gauss_l2_bounds(x, y, R, T, σᵣ, σₜ, pσ[i, j], pϕ[i, j]; kwargs...)
        end
    end
    return lb, ub
end

function gauss_l2_bounds(mgmmx::AbstractMultiGMM, mgmmy::AbstractMultiGMM, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number, mpσ = nothing, mpϕ = nothing, interactions = nothing)
    # prepare pairwise widths and weights, if not provided
    if isnothing(mpσ) || isnothing(mpϕ)
        mpσ, mpϕ = pairwise_consts(mgmmx, mgmmy, interactions)
    end

    # sum bounds for each pair of points
    lb = 0.0
    ub = 0.0
    for (key1, intrs) in mpσ
        for (key2, pσ) in intrs
            lb, ub = (lb, ub) .+ gauss_l2_bounds(mgmmx.gmms[key1], mgmmy.gmms[key2], R, T, σᵣ, σₜ, pσ, mpϕ[key1][key2])
        end
    end
    return lb, ub
end

# gauss_l2_bounds(x::AbstractGMM, y::AbstractGMM, R::RotationVec, T::SVector{3}, args...; kwargs...
#     ) = gauss_l2_bounds(R*x, y-T, args...; kwargs...)

gauss_l2_bounds(
    x::AbstractGMM, y::AbstractGMM, block::UncertaintyRegion, args...; kwargs...
) = gauss_l2_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ, args...; kwargs...)

gauss_l2_bounds(
    x::AbstractGMM, y::AbstractGMM, block::SearchRegion, args...; kwargs...
) = gauss_l2_bounds(x, y, UncertaintyRegion(block), args...; kwargs...)
