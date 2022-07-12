# Speed up rotation search of TIV-GOGMA by using the ROCS alignment algorithm to produce a best rotation
function rocs_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM; kwargs...)
    t = promote_type(numbertype(gmmx),numbertype(gmmy))
    p = t(π)

    # rotation alignment
    rocs_res = rocs_align(gmmx, gmmy)
    rotpos = rot_to_axis(rocs_res.tform.linear)

    # translation alignment
    trl_res = trl_gogma_align(gmmx, gmmy; rot=rotpos, kwargs...)

    # local alignment in the full transformation space
    pos = NTuple{6, t}([rotpos..., trl_res.tform_params...])
    trlim = translation_limit(gmmx, gmmy)
    localblock = Block(((-p,p), (-p,p), (-p,p), (-trlim,trlim), (-trlim,trlim), (-trlim,trlim)), pos, trl_res.upperbound, -Inf)
    min, bestpos = local_align(gmmx, gmmy, localblock)

    return GMAlignmentResult(gmmx, gmmy, min, trl_res.lowerbound, AffineMap(bestpos...), bestpos, trl_res.obj_calls, trl_res.num_splits, trl_res.num_blocks, trl_res.stagnant_splits)

end 