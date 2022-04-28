function alignment_objective(X, gmmx::AbstractGMM, gmmy::AbstractGMM, R=nothing, T=nothing, pσ=nothing, pϕ=nothing)
    return -overlap(AffineMap(X...)(gmmx), gmmy, pσ, pϕ)
end

# alignment objective for rigid Ration (i.e. the first stage of TIV-GOGMA)
function R_alignment_objective(X, gmmx::AbstractGMM, gmmy::AbstractGMM, R=nothing, T=nothing, pσ=nothing, pϕ=nothing)
    if isnothing(T)
        zro = zero(promote_type(numbertype(gmmx), numbertype(gmmy)))
        T = (zro, zro, zro)
    end 
    return alignment_objective((X..., T...), gmmx, gmmy, pσ, pϕ)
end

# alignment objective for translation (i.e. the second stage of TIV-GOGMA)
function T_alignment_objective(X, gmmx::AbstractGMM, gmmy::AbstractGMM, R=nothing, T=nothing, pσ=nothing, pϕ=nothing)
    if isnothing(R)
        zro = zero(promote_type(numbertype(gmmx), numbertype(gmmy)))
        R = (zro, zro, zro)
    end
    return alignment_objective((R..., X...), gmmx, gmmy, pσ, pϕ)
end

"""
    obj, pos = local_align(gmmx, gmmy, block, pσ=nothing, pϕ=nothing; objfun=alignment_objective, R=nothing, T=nothing, rtol=1e-9, maxevals=100)

Performs local alignment within the specified `block` using L-BFGS to minimize objective function `objfun` for the provided GMMs, `gmmx` and `gmmy`.
"""
function local_align(gmmx::AbstractGMM, gmmy::AbstractGMM, block::SearchRegion=UncertaintyRegion(gmmx,gmmy), pσ=nothing, pϕ=nothing; 
                     objfun=alignment_objective, R=nothing, T=nothing, maxevals=100)
    # prepare pairwise widths and weights
    if isnothing(pσ) || isnothing(pϕ)
        pσ, pϕ = pairwise_consts(gmmx, gmmy)
    end

    # set optimization bounds
    lower = [r[1] for r in block.ranges]
    upper = [r[2] for r in block.ranges]

    # set initial guess at the center of the block
    initial_X = center(block)

    # local optimization within the block
    f(X) = objfun(X, gmmx, gmmy, block.R, block.T, pσ, pϕ)
    res = optimize(f, lower, upper, initial_X, Fminbox(LBFGS()), Optim.Options(f_calls_limit=maxevals); autodiff = :forward)
    return Optim.minimum(res), tuple(Optim.minimizer(res)...)
end

local_align(gmmx, gmmy, block::RotationRegion,    args...; kwargs...) = local_align(gmmx, gmmy, block, args...; objfun=R_alignment_objective, kwargs...)
local_align(gmmx, gmmy, block::TranslationRegion, args...; kwargs...) = local_align(gmmx, gmmy, block, args...; objfun=T_alignment_objective, kwargs...)

local_align(gmmx::AbstractGMM, gmmys::Union{AbstractVector{AbstractSingleGMM},AbstractVector{AbstractMultiGMM}}, block,  pσ=nothing, pϕ=nothing; kwargs...) = 
    local_align(gmmx, combine(gmmys), block, pσ, pϕ; kwargs...)