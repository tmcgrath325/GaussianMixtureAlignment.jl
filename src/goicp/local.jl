function align_local_points(P, Q; maxevals=1000, tformfun=AffineMap)
    # set initial guess at the center of the block
    initial_X = zeros(tformfun === AffineMap ? 6 : 3);

    # local optimization within the block
    function f(X)
        tform = tformfun((X...,))
        score = squared_deviation(tform(P), Q)
        @show X
        @show score
        return score
    end
    res = optimize(f, initial_X, LBFGS(), Optim.Options(f_calls_limit=maxevals); autodiff = :forward)
    @show Optim.minimizer(res)
    return Optim.minimum(res), tuple(Optim.minimizer(res)...)
end

function iterate_local_alignment(P, Q; correspondence = hungarian_assignment, iterations=1000, tformfun=AffineMap, kwargs...)
    matches = correspondence(P, Q)
    prevmatches = matches
    score = Inf
    tformparams = zeros(tformfun === AffineMap ? 6 : 3);
    it = 0
    while (it < iterations)
        it += 1
        matchedP, matchedQ = matched_points(P,Q,matches)
        score, tformparams = align_local_points(matchedP, matchedQ; tformfun=tformfun, kwargs...)
        @show score, tformparams
        tform = tformfun(tformparams)
        prevmatches = matches
        matches = correspondence(tform(P), Q)
        if matches === prevmatches
            break
        end
    end
    @show it
    return score, tformparams
end

function iterate_local_alignment(P, Q, block; tformfun=AffineMap, kwargs...)
    blocktform = AffineMap(block.R, block.T)
    tformedP = blocktform(P)
    score, opt_tformparams = iterate_local_alignment(tformedP, Q; kwargs...)
    @show opt_tformparams
    opt_tform = tformfun(opt_tformparams)
    tform = opt_tform ∘ opt_tform
    tformparams = tformfun === LinearMap ? (tform.linear.sx, tform.linear.sy, tform.linear.sz) :
                  tformfun === Translation ? (tform.translation...,) :
                  (tform.linear.sx, tform.linear.sy, tform.linear.sz, tform.translation...)
    return score, tformparams
end