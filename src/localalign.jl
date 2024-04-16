tformwithparams(X,x) = RotationVec(X[1:3]...)*x + SVector{3}(X[4:6]...)
# function tformwithparams(X,x) 
#     if sum(abs2, X[1:3]) == 0 # handled for autodiff around 0
#         T = eltype(X)
#         θ = norm(X[1:3])
#         a = θ > 0 ? X[1] / θ : one(T)
#         b = θ > 0 ? X[2] / θ : zero(T)
#         c = θ > 0 ? X[3] / θ : zero(T)
#         R = AngleAxis(θ, a, b, c)
#         @show R
#     else
#         R = RotationVec(X[1:3]...)
#     end
#     t = SVector{3}(X[4:6]...)
#     @show (R*x)[1]
#     return R*x + t
# end

overlapobj(X,x,y,args...) = -overlap(tformwithparams(X,x), y, args...)

function distanceobj(X, x, y; correspondence = hungarian_assignment)
    tformedx = tformwithparams(X,x)
    return squared_deviation(tformedx, y, correspondence(tformedx,y))
end

function alignment_objective(X, x::AbstractModel, y::AbstractModel, args...; objfun=overlapobj)
    return objfun(X,x,y,args...)
end

# alignment objective for a rigid transformation
alignment_objective(X, x::AbstractModel, y::AbstractModel, block::UncertaintyRegion, args...; kwargs...) = alignment_objective(X,x,y,args...; kwargs...)

# alignment objective for rigid rotation (i.e. the first stage of TIV-GOGMA)
function alignment_objective(X, gmmx::AbstractModel, gmmy::AbstractModel, block::RotationRegion, args...; kwargs...)
    return alignment_objective((X..., block.T...), gmmx, gmmy, args...; kwargs...)
end

# alignment objective for translation (i.e. the second stage of TIV-GOGMA)
function alignment_objective(X, gmmx::AbstractModel, gmmy::AbstractModel, block::TranslationRegion, args...; kwargs...)
    return alignment_objective((block.R.sx, block.R.sy, block.R.sz, X...), gmmx, gmmy, args...; kwargs...)
end


"""
    obj, pos = local_align(x, y, block, pσ=nothing, pϕ=nothing; R=nothing, T=nothing, maxevals=100)

Performs local alignment within the specified `block` using L-BFGS to minimize objective function `objfun` for the provided GMMs, `x` and `y`.
"""
function local_align(x::AbstractModel, y::AbstractModel, block::SearchRegion, args...; 
                     maxevals=100, kwargs...)

    # set initial guess at the center of the block
    initial_X = center(block)
    # if (typeof(block) <: UncertaintyRegion && sum(abs2, initial_X[1:Int(end/2)]) == 0) || (typeof(block) <: RotationRegion && sum(abs2, initial_X) == 0)
    #     T = eltype(initial_X)
    #     initial_X = initial_X .+ [eps(T), zeros(T, length(initial_X)-1)...]
    # end

    # local optimization within the block
    f(X) = alignment_objective(X, x, y, block, args...; kwargs...)
    # res = optimize(f, lower, upper, initial_X, Fminbox(LBFGS()), Optim.Options(f_calls_limit=maxevals); autodiff = :forward)
    res = optimize(f, [initial_X...], LBFGS(), Optim.Options(f_calls_limit=maxevals); autodiff = :forward)
    return Optim.minimum(res), tuple(Optim.minimizer(res)...)
end

local_align(x::AbstractModel, y::AbstractModel; kwargs...) = local_align(x, y, UncertaintyRegion(x,y); kwargs...)