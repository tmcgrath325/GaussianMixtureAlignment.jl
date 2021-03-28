function alignment_objective(X, gmmx::IsotropicGMM, gmmy::IsotropicGMM, pσ=nothing, pϕ=nothing)
    t = promote_type(eltype(gmmx), eltype(gmmy))
    rx, ry, rz, tx, ty, tz = X
    # prepare transformation
    R = rot(rx, ry, rz)
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

function local_align(gmmx::IsotropicGMM, gmmy::IsotropicGMM, block::Block=Block(gmmx,gmmy), pσ=nothing, pϕ=nothing; rtol=1e-9, maxevals=250)
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
    f(X) = alignment_objective(X, gmmx, gmmy, pσ, pϕ)
    res = optimize(f, lower, upper, initial_X, Fminbox(LBFGS()), Optim.Options(f_calls_limit=maxevals))
    return Optim.minimum(res), Optim.minimizer(res)
end