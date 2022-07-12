"""
    lim = translation_limit(gmmx, gmmy)

Computes the largest translation needed to ensure that the searchspace contains the best alignment transformation.
"""
function translation_limit(gmmx::AbstractSingleGMM, gmmy::AbstractSingleGMM)
    trlim = typemin(promote_type(numbertype(gmmx),numbertype(gmmy)))
    for gaussians in (gmmx.gaussians, gmmy.gaussians)
        if !isempty(gaussians)
            trlim = max(trlim, maximum(gaussians) do gauss
                    maximum(abs, gauss.μ) end)
        end
    end
    return trlim
end

function translation_limit(mgmmx::AbstractMultiGMM, mgmmy::AbstractMultiGMM)
    trlim = typemin(promote_type(numbertype(mgmmx),numbertype(mgmmy)))
    for key in keys(mgmmx.gmms) ∩ keys(mgmmy.gmms)
        trlim = max(trlim, translation_limit(mgmmx.gmms[key], mgmmy.gmms[key]))
    end
    return trlim
end