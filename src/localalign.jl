tformwithparams(X::NTuple{6},x) = RotationVec(X[1:3]...)*x + SVector{3}(X[4:6]...)
distobj(X,x,y,args...) = rmsd(tformwithparams(X,x),y)
overlapobj(X,x,y,args...) = -overlap(tformwithparams(X,x), y, args...)

function alignment_objective(X::NTuple{6}, x::AbstractModel, y::AbstractModel, args...; objfun=distobj)
    return objfun(X,x,y,args...)
end

alignment_objective(X::NTuple{6}, x::AbstractGMM, y::AbstractGMM, args...) = alignment_objective(X, x, y; objfun=overlapobj)

# alignment objective for rigid Ration (i.e. the first stage of TIV-GOGMA)
function alignment_objective(X::NTuple{3}, gmmx::AbstractModel, gmmy::AbstractModel, block::RotationRegion, args...)
    return alignment_objective((X..., block.T...), gmmx, gmmy, args...)
end

# alignment objective for translation (i.e. the second stage of TIV-GOGMA)
function alignment_objective(X::NTuple{3}, gmmx::AbstractModel, gmmy::AbstractModel, block::TranslationRegion, args...)
    return overlap_alignment_objective((block.R..., X...), gmmx, gmmy, args...)
end


"""
    obj, pos = local_align(x, y, block, pσ=nothing, pϕ=nothing; R=nothing, T=nothing, rtol=1e-9, maxevals=100)

Performs local alignment within the specified `block` using L-BFGS to minimize objective function `objfun` for the provided GMMs, `x` and `y`.
"""
function local_align(x::AbstractModel, y::AbstractModel, block::SearchRegion=UncertaintyRegion(x,y), args...; 
                     maxevals=100)

    # set initial guess at the center of the block
    initial_X = center(block)
    ndims = length(initial_X)

    # local optimization within the block
    f(X) = alignment_objective(NTuple{ndims}(X), x, y, block, args...)
    # res = optimize(f, lower, upper, initial_X, Fminbox(LBFGS()), Optim.Options(f_calls_limit=maxevals); autodiff = :forward)
    res = optimize(f, [initial_X...], LBFGS(), Optim.Options(f_calls_limit=maxevals); autodiff = :forward)
    return Optim.minimum(res), tuple(Optim.minimizer(res)...)
end

local_align(x, y, block::RotationRegion,    args...; kwargs...) = local_align(x, y, block, args...; objfun=R_alignment_objective, kwargs...)
local_align(x, y, block::TranslationRegion, args...; kwargs...) = local_align(x, y, block, args...; objfun=T_alignment_objective, kwargs...)
local_align(x, y; kwargs...) = local_align(x, y, UncertaintyRegion(x,y); kwargs...)