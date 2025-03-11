function align_local_points(P, Q; maxevals=1000, tformfun=AffineMap)
    # set initial guess at the center of the block
    initial_X = zeros(tformfun === AffineMap ? 6 : 3)
    trlim = tformfun !== LinearMap ? translation_limit(P, Q) : 0;
    lower =     tformfun === AffineMap ? [-π, -π, -π, -trlim, -trlim, -trlim] :
                tformfun === LinearMap ? [-π, -π, -π] :
                [-trlim, -trlim, -trlim]
    upper =     tformfun === AffineMap ? [π, π, π, trlim, trlim, trlim] :
                tformfun === LinearMap ? [π, π, π] :
                [trlim, trlim, trlim]

    # local optimization within the block
    function f(X)
        tform = build_tform(tformfun, X)
        score = squared_deviation(tform(P), Q)
        return score
    end
    # res = optimize(f, initial_X, LBFGS(), Optim.Options(f_calls_limit=maxevals)) # ; autodiff = :forward)
    res = optimize(f, lower, upper, initial_X, Fminbox(LBFGS()), Optim.Options(f_calls_limit=maxevals)) #; autodiff = :forward)
    # @show res.iterations
    return Optim.minimum(res), tuple(Optim.minimizer(res)...)
end

function iterate_local_alignment(P, Q; correspondence = hungarian_assignment, iterations=100, tformfun=AffineMap, kwargs...)
    matches = correspondence(P, Q)
    prevmatches = matches
    score = Inf
    tformparams = zeros(tformfun === AffineMap ? 6 : 3);
    it = 0
    while (it < iterations)
        it += 1
        matchedP, matchedQ = matched_points(P,Q,matches)
        score, tformparams = align_local_points(matchedP, matchedQ; tformfun=tformfun, kwargs...)
        tform = build_tform(tformfun, tformparams)
        prevmatches = matches
        matches = correspondence(tform(P), Q)
        if matches == prevmatches
            break
        end
    end
    return score, tformparams
end

function iterate_local_alignment(P, Q, block; tformfun=AffineMap, kwargs...)
    block_tform = AffineMap(block.R, block.T)
    tformedP = block_tform(P)
    score, opt_tformparams = iterate_local_alignment(tformedP, Q; tformfun=tformfun, kwargs...)
    opt_tform = build_tform(tformfun, opt_tformparams)
    tform = opt_tform ∘ block_tform
    tform = AffineMap(RotationVec(tform.linear), tform.translation)
    tformparams = tformfun === LinearMap ? (tform.linear.sx, tform.linear.sy, tform.linear.sz) :
                  tformfun === Translation ? (tform.translation...,) :
                  (tform.linear.sx, tform.linear.sy, tform.linear.sz, tform.translation...)
    return score, tformparams
end