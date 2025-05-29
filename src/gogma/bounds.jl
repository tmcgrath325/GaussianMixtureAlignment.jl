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

function pairwise_consts(gmmx::AbstractLabeledIsotropicGMM{N,T,K}, gmmy::AbstractLabeledIsotropicGMM{N,S,K}, interactions::Nothing=nothing) where {N,T,S,K}
    t = promote_type(T, S)
    self_interactions = Dict{Tuple{K,K},t}()
    labels = unique(vcat(collect(g.label for g in x.gaussians), collect(g.label for g in y.gaussians)))
    for l in labels
        self_interactions[(l,l)] = one(t)
    end 
    return pairwise_consts(gmmx, gmmy, self_interactions)
end

function pairwise_consts(gmmx::AbstractLabeledIsotropicGMM{N,T,K}, gmmy::AbstractLabeledIsotropicGMM{N,S,K}, interactions::Dict{Tuple{K,K},V}) where {N,T,S,K,V<:Number}
    @assert validate_interactions(interactions) "Interactions must not include redundant key pairs (i.e. (k1,k2) and (k2,k1))"
    t = promote_type(T, S, V)
    pσ, pϕ = zeros(t, length(gmmx), length(gmmy)), zeros(t, length(gmmx), length(gmmy))
    for (i,gaussx) in enumerate(gmmx.gaussians)
        for (j,gaussy) in enumerate(gmmy.gaussians)
            keypair = (gaussx.label, gaussy.label)
            keypair = haskey(interactions, keypair) ? keypair : (keypair[2], keypair[1])
            pσ[i,j] = gaussx.σ^2 + gaussy.σ^2
            pϕ[i,j] = ( haskey(interactions, keypair) ? interactions[keypair] : zero(t) ) * gaussx.ϕ * gaussy.ϕ
        end
    end
    return pσ, pϕ
end

function pairwise_consts(mgmmx::AbstractMultiGMM{N,T,K}, mgmmy::AbstractMultiGMM{N,S,K}, interactions::Nothing=nothing) where {N,T,S,K}
    t = promote_type(T, S)
    self_interactions = Dict{Tuple{K,K},t}()
    for key in keys(mgmmx.gmms) ∩ keys(mgmmy.gmms)
        self_interactions[(key,key)] = one(t)
    end 
    pairwise_consts(mgmmx, mgmmy, self_interactions)
end

function pairwise_consts(mgmmx::AbstractMultiGMM{N,T,K}, mgmmy::AbstractMultiGMM{N,S,K}, interactions::Dict{Tuple{K,K},V}) where {N,T,S,K,V <: Number}
    t = promote_type(T, S, V)
    xkeys = keys(mgmmx.gmms)
    ykeys = keys(mgmmy.gmms)
    @assert validate_interactions(interactions) "Interactions must not include redundant key pairs (i.e. (k1,k2) and (k2,k1))"

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

function pairwise_consts(y::AbstractGMM, xs::AbstractVector{<:AbstractGMM}, interactions=nothing)
    t = promote_type(numbertype(y), numbertype.(xs)...)
    ms = [y, xs...]
    pσ = Matrix{Matrix{t}}(undef, length(ms), length(ms))
    pϕ = Matrix{Matrix{t}}(undef, length(ms), length(ms))
    for (i,mi) in enumerate(ms)
        for (j,mj) in enumerate(ms)
            pσ[i,j], pϕ[i,j] = pairwise_consts(mi,mj,interactions)
        end
    end
    return pσ, pϕ
end


"""
    interval = gauss_l2_bounds(x::Union{IsotropicGaussian, AbstractGMM}, y::Union{IsotropicGaussian, AbstractGMM}, σᵣ, σₜ)
    interval = gauss_l2_bounds(x, y, R::RotationVec, T::SVector{3}, σᵣ, σₜ)

Finds the bounds for overlap between two isotropic Gaussian distributions, two isotropic GMMs, or `two sets of 
labeled isotropic GMMs for a particular region in 6-dimensional rigid rotation space, defined by `R`, `T`, `σᵣ` and `σₜ`.

`R` and `T` represent the rotation and translation, respectively, that are at the center of the uncertainty region. If they are not provided, 
the uncertainty region is assumed to be centered at the origin (i.e. x has already been transformed).

`σᵣ` and `σₜ` represent the sizes of the rotation and translation uncertainty regions.

See [Campbell & Peterson, 2016](https://arxiv.org/abs/1603.00150)
"""
function gauss_l2_bounds(μx::SVector{3}, μy::SVector{3}, σᵣ, σₜ, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; distance_bound_fun = tight_distance_bounds, lohifun = lohi_interval)
    (lbdist, ubdist) = distance_bound_fun(μx, μy, σᵣ, σₜ, w < 0)

    return lohifun(-overlap(lbdist^2, s, w), -overlap(ubdist^2, s, w))
end

gauss_l2_bounds(x::AbstractIsotropicGaussian, y::AbstractIsotropicGaussian, R::RotationVec, T::SVector{3}, σᵣ, σₜ, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; distance_bound_fun = tight_distance_bounds, lohifun = lohi_interval) = gauss_l2_bounds(R*x.μ, y.μ-T, σᵣ, σₜ, s, w; distance_bound_fun = distance_bound_fun, lohifun = lohifun)

# gauss_l2_bounds(x::AbstractGaussian, y::AbstractGaussian, R::RotationVec, T::SVector{3}, σᵣ, σₜ, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; kwargs...
#     ) = gauss_l2_bounds(R*x, y-T, σᵣ, σₜ, tform.translation, s, w; kwargs...)

gauss_l2_bounds(x::AbstractGaussian, y::AbstractGaussian, block::UncertaintyRegion, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; kwargs...
    ) = gauss_l2_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ, s, w; kwargs...)

gauss_l2_bounds(x::AbstractGaussian, y::AbstractGaussian, block::SearchRegion, s=x.σ^2 + y.σ^2, w=x.ϕ*y.ϕ; kwargs...
    ) = gauss_l2_bounds(x, y, UncertaintyRegion(block), s, w; kwargs...)



function gauss_l2_bounds(gmmx::AbstractSingleGMM, gmmy::AbstractSingleGMM, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number, pσ=nothing, pϕ=nothing, interactions=nothing; lohifun = lohi_interval, kwargs...)
    if isnothing(pσ) || isnothing(pϕ)
        pσ, pϕ = pairwise_consts(gmmx, gmmy)
    end
    trackub = lohifun !== lohi_interval # replace?

    bnds = lohifun(0.0, 0.0)
    ub = 0.0
    for (i,x) in enumerate(gmmx.gaussians) 
        for (j,y) in enumerate(gmmy.gaussians)
            pbnds = gauss_l2_bounds(x, y, R, T, σᵣ, σₜ, pσ[i,j], pϕ[i,j]; lohifun=lohifun, kwargs...)  
            bnds = bnds + pbnds
            if trackub
                ub = ub + hival(pbnds)
            end
        end
    end
    if trackub
        return lohifun(loval(bnds), ub)
    else
        return bnds
    end
end

function gauss_l2_bounds(gmmx::AbstractSingleGMM, gmmy::AbstractSingleGMM, σᵣ::Number, σₜ::Number, pσ=nothing, pϕ=nothing, interactions=nothing; lohifun=lohi_interval, kwargs...)
    if isnothing(pσ) || isnothing(pϕ)
        pσ, pϕ = pairwise_consts(gmmx, gmmy)
    end
    trackub = lohifun !== lohi_interval

    bnds = lohifun(0.0, 0.0)
    ub = 0.0
    for (i,x) in enumerate(gmmx.gaussians) 
        for (j,y) in enumerate(gmmy.gaussians)
            pbnds = gauss_l2_bounds(x.μ, y.μ, σᵣ, σₜ, pσ[i,j], pϕ[i,j]; kwargs...)  
            bnds = bnds + pbnds 
            if trackub
                ub = ub + hival(pbnds)
            end
        end
    end
    if trackub
        return lohifun(loval(bnds), ub)
    else
        return bnds
    end
end

function gauss_l2_bounds(mgmmx::AbstractMultiGMM, mgmmy::AbstractMultiGMM, R::RotationVec, T::SVector{3}, σᵣ::Number, σₜ::Number, mpσ=nothing, mpϕ=nothing, interactions=nothing; lohifun=lohi_interval, kwargs...)
    if isnothing(mpσ) || isnothing(mpϕ)
        mpσ, mpϕ = pairwise_consts(mgmmx, mgmmy, interactions)
    end
    trackub = lohifun !== lohi_interval

    bnds = lohifun(0.0, 0.0)
    ub = 0.0
    for (key1, intrs) in mpσ
        for (key2, pσ) in intrs
            pbnds = gauss_l2_bounds(mgmmx.gmms[key1], mgmmy.gmms[key2], R, T, σᵣ, σₜ, pσ, mpϕ[key1][key2]; lohifun=lohifun, kwargs...)
            bnds = bnds + pbnds
            if trackub
                ub = ub + hival(pbnds)
            end
        end
    end
    if trackub
        return lohifun(loval(bnds), ub)
    else
        return bnds
    end
end

function gauss_l2_bounds(mgmmx::AbstractMultiGMM, mgmmy::AbstractMultiGMM, σᵣ::Number, σₜ::Number, mpσ=nothing, mpϕ=nothing, interactions=nothing; lohifun=lohi_interval, kwargs...)
    if isnothing(mpσ) || isnothing(mpϕ)
        mpσ, mpϕ = pairwise_consts(mgmmx, mgmmy, interactions)
    end
    trackub = lohifun !== lohi_interval

    bnds = lohifun(0.0, 0.0)
    ub = 0.0
    for (key1, intrs) in mpσ
        for (key2, pσ) in intrs
            pbnds = gauss_l2_bounds(mgmmx.gmms[key1], mgmmy.gmms[key2], σᵣ, σₜ, pσ, mpϕ[key1][key2]; lohifun=lohifun, kwargs...)
            bnds = bnds + pbnds
            if trackub
                ub = ub + hival(pbnds)
            end
        end
    end
    if trackub
        return lohifun(loval(bnds), ub)
    else
        return bnds
    end
end


# gauss_l2_bounds(x::AbstractGMM, y::AbstractGMM, R::RotationVec, T::SVector{3}, args...; kwargs...
#     ) = gauss_l2_bounds(R*x, y-T, args...; kwargs...)

gauss_l2_bounds(x::AbstractGMM, y::AbstractGMM, block::UncertaintyRegion, args...; kwargs...
    ) = gauss_l2_bounds(x, y, block.R, block.T, block.σᵣ, block.σₜ, args...; kwargs...)

gauss_l2_bounds(x::AbstractGMM, y::AbstractGMM, block::SearchRegion, args...; kwargs...
    ) = gauss_l2_bounds(x, y, UncertaintyRegion(block), args...; kwargs...)


function gauss_l2_bounds!(tformedxs, y::AbstractGMM, xs::AbstractVector{<:AbstractGMM}, blocks::AbstractVector{<:SearchRegion}, pσs=nothing, pϕs=nothing, interactions=nothing; lohifun=lohi_interval, kwargs...)
    if isnothing(pσs) || isnothing(pϕs)
        pσs, pϕs = pairwise_consts(y, xs, interactions)
    end
    trackub = lohifun !== lohi_interval

    # tformedxs = [b.R*x+b.T for (x,b) in zip(xs,blocks)]
    for (i,(x,b)) in enumerate(zip(xs,blocks))
        tformedxs[i] = b.R*x + b.T
    end

    bnds = lohifun(0.0, 0.0)
    ub = 0.0
    for (i,x) in enumerate(tformedxs) # all of the pairwise overlaps with y
        pbnds = gauss_l2_bounds(x, y, blocks[i].σᵣ, blocks[i].σₜ, pσs[i+1,1], pϕs[i+1,1]; lohifun=lohifun, kwargs...)
        bnds = bnds + pbnds
        if trackub
            ub = ub + hival(pbnds)
        end
    end
    for (i,(xi,bi)) in enumerate(zip(tformedxs, blocks)) # all other pairwise overlaps
        for j in i+1:length(xs)
            pbnds = gauss_l2_bounds(xi, tformedxs[j], bi.σᵣ + blocks[j].σᵣ, bi.σₜ + blocks[j].σₜ, pσs[i+1,j+1], pϕs[i+1,j+1]; lohifun=lohifun, kwargs...)
            bnds = bnds + pbnds
            if trackub
                ub = ub + hival(pbnds)
            end
        end
    end
    if trackub
        return lohifun(loval(bnds), ub)
    else
        return bnds
    end
end