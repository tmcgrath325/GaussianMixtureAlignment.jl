function alignment_objective(X, gmmx::IsotropicGMM, gmmy::IsotropicGMM, rot=nothing, trl=nothing, pσ=nothing, pϕ=nothing)
    t = promote_type(eltype(gmmx), eltype(gmmy))
    rx, ry, rz, tx, ty, tz = X
    # prepare transformation
    R = rotmat(rx, ry, rz)
    tr = SVector(tx, ty, tz)

    # prepare pairwise widths and weights
    if isnothing(pσ) || isnothing(pϕ)
        pσ, pϕ = pairwise_consts(gmmx, gmmy)
    end

    # sum pairwise overlap values
    objval = zero(t)
    for (i,x) in enumerate(gmmx.gaussians)
        for (j,y) in enumerate(gmmy.gaussians)
            distsq = sum(abs2, R*x.μ+tr - y.μ)
            if length(x.dirs)==0 || length(y.dirs)==0
                dirdot = one(t)
            else
                # NOTE: Avoid list comprehension (slow), but perform more matrix multiplications
                # xdirs = [R*xdir for xdir in x.dirs]
                dirdot = -one(t)
                for xdir in x.dirs
                    for ydir in y.dirs
                        dirdot = max(dirdot, dot(R*xdir,ydir))
                    end
                end
                # dirdot = maximum([dot(xdir, ydir) for xdir in xdirs for ydir in y.dirs])
            end
            objval += objectivefun(distsq, pσ[i,j], pϕ[i,j], dirdot) # , 3)
        end
    end
    return objval
end

function alignment_objective(X, mgmmx::MultiGMM, mgmmy::MultiGMM, rot=nothing, trl=nothing, mpσ=nothing, mpϕ=nothing)
    # prepare pairwise widths and weights
    if isnothing(mpσ) || isnothing(mpϕ)
        mpσ, mpϕ = pairwise_consts(mgmmx, mgmmy)
    end 
    
    # sum overlap values for gmms of each feature
     objval = zero(promote_type(eltype(mgmmx), eltype(mgmmy)))
     for key in keys(mgmmx.gmms) ∩ keys(mgmmy.gmms)
        objval += alignment_objective(X, mgmmx.gmms[key], mgmmy.gmms[key], rot, trl, mpσ[key], mpϕ[key])
     end
     return objval
end

function rot_alignment_objective(X, gmmx::Union{IsotropicGMM,MultiGMM}, gmmy::Union{IsotropicGMM,MultiGMM}, rot=nothing, trl=nothing, pσ=nothing, pϕ=nothing)
    if isnothing(trl)
        zro = zero(promote_type(eltype(gmmx), eltype(gmmy)))
        trl = (zro, zro, zro)
    end
    return alignment_objective((X..., trl...), gmmx, gmmy, pσ, pϕ)
end

function trl_alignment_objective(X, gmmx::Union{IsotropicGMM,MultiGMM}, gmmy::Union{IsotropicGMM,MultiGMM}, rot=nothing, trl=nothing, pσ=nothing, pϕ=nothing)
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
function local_align(gmmx::Union{IsotropicGMM,MultiGMM}, gmmy::Union{IsotropicGMM,MultiGMM}, block, pσ=nothing, pϕ=nothing; objfun=alignment_objective, rot=nothing, trl=nothing, rtol=1e-9, maxevals=100)
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

local_align(gmmx::Union{IsotropicGMM,MultiGMM}, gmmys::Union{AbstractVector{IsotropicGMM},AbstractVector{MultiGMM}}, block,  pσ=nothing, pϕ=nothing; kwargs...) = 
    local_align(gmmx, combine(gmmys), block, pσ, pϕ; kwargs...)