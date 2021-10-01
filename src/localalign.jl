function alignment_objective(X, gmmx::GMM, gmmy::GMM, rot=nothing, trl=nothing, pσ=nothing, pϕ=nothing)
    return overlap(AffineMap(X...)(gmmx), gmmy, pσ, pϕ)
end

# alignment objective for rigid rotation (i.e. the first stage of TIV-GOGMA)
function rot_alignment_objective(X, gmmx::GMM, gmmy::GMM, rot=nothing, trl=nothing, pσ=nothing, pϕ=nothing)
    if isnothing(trl)
        zro = zero(promote_type(eltype(gmmx), eltype(gmmy)))
        trl = (zro, zro, zro)
    end
    return alignment_objective((X..., trl...), gmmx, gmmy, pσ, pϕ)
end

# alignment objective for translation (i.e. the second stage of TIV-GOGMA)
function trl_alignment_objective(X, gmmx::GMM, gmmy::GMM, rot=nothing, trl=nothing, pσ=nothing, pϕ=nothing)
    if isnothing(rot)
        zro = zero(promote_type(eltype(gmmx), eltype(gmmy)))
        rot = (zro, zro, zro)
    end
    return alignment_objective((rot..., X...), gmmx, gmmy, pσ, pϕ)
end

"""
    obj, pos = local_align(gmmx, gmmy, block, pσ=nothing, pϕ=nothing; objfun=alignment_objective, rot=nothing, trl=nothing, rtol=1e-9, maxevals=100)

Performs local alignment within the specified `block` using L-BFGS to minimize objective function `objfun` for the provided GMMs, `gmmx` and `gmmy`.
"""
function local_align(gmmx::GMM, gmmy::GMM, block, pσ=nothing, pϕ=nothing; objfun=alignment_objective, rot=nothing, trl=nothing, rtol=1e-9, maxevals=100)
    # prepare pairwise widths and weights
    if isnothing(pσ) || isnothing(pϕ)
        pσ, pϕ = pairwise_consts(gmmx, gmmy)
    end

    # set optimization bounds
    lower = [r[1] for r in block.ranges]
    upper = [r[2] for r in block.ranges]

    # set initial guess at the center of the block
    initial_X = [x for x in block.center]

    # local optimization within the block
    f(X) = objfun(X, gmmx, gmmy, rot, trl, pσ, pϕ)
    res = optimize(f, lower, upper, initial_X, Fminbox(LBFGS()), Optim.Options(f_calls_limit=maxevals); autodiff = :forward)
    return Optim.minimum(res), tuple(Optim.minimizer(res)...)
end

local_align(gmmx::GMM, gmmys::Union{AbstractVector{IsotropicGMM},AbstractVector{MultiGMM}}, block,  pσ=nothing, pϕ=nothing; kwargs...) = 
    local_align(gmmx, combine(gmmys), block, pσ, pϕ; kwargs...)