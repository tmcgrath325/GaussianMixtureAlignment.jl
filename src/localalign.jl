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
            objval += objectivefun(distsq, pσ[i,j], pϕ[i,j], 3)
        end
    end
    return objval
end

function rot_alignment_objective(X, gmmx::IsotropicGMM, gmmy::IsotropicGMM, rot=nothing, trl=nothing, pσ=nothing, pϕ=nothing)
    if isnothing(trl)
        zro = zero(promote_type(eltype(gmmx), eltype(gmmy)))
        trl = (zro, zro, zro)
    end
    return alignment_objective((X..., trl...), gmmx, gmmy, pσ, pϕ)
end

function trl_alignment_objective(X, gmmx::IsotropicGMM, gmmy::IsotropicGMM, rot=nothing, trl=nothing, pσ=nothing, pϕ=nothing)
    if isnothing(rot)
        zro = zero(promote_type(eltype(gmmx), eltype(gmmy)))
        rot = (zro, zro, zro)
    end
    return alignment_objective((rot..., X...), gmmx, gmmy, pσ, pϕ)
end

function local_align(gmmx::IsotropicGMM, gmmy::IsotropicGMM, block, pσ=nothing, pϕ=nothing; objfun=alignment_objective, rot=nothing, trl=nothing, rtol=1e-9, maxevals=100)
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