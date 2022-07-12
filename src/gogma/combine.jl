"""
    gmm = combine(gmmx::IsotropicGMM, gmmy::IsotropicGMM)
    mgmm = combine(mgmmx::MultiGMM, mgmmy::MultiGMM)
    gmm = combine(gmms::Union{AbstractVector{<:IsotropicGMM},AbstractVector{<:MultiGMM}})

Creates a new `IsotropicGMM` or `MultiGMM` by concatenating the vectors of `IsotroicGaussian`s in
the input GMMs. 
"""
function combine(gmmx::AbstractSingleGMM, gmmy::AbstractSingleGMM)
    if dims(gmmx) != dims(gmmy)
        throw(ArgumentError("GMMs must have the same dimensionality"))
    end
    t = promote_type(typeof(gmmx), typeof(gmmy))
    return t(vcat(gmmx.gaussians, gmmy.gaussians))
end

function combine(mgmmx::IsotropicMultiGMM, mgmmy::IsotropicMultiGMM)
    if dims(mgmmx) != dims(mgmmy)
        throw(ArgumentError("GMMs must have the same dimensionality"))
    end
    t = IsotropicGMM{dims(mgmmx),promote_type(numbertype(mgmmx), numbertype(mgmmy))}
    d = promote_type(typeof(mgmmx.gmms), typeof(mgmmy.gmms))
    gmms = d()
    xkeys, ykeys = keys(mgmmx.gmms), keys(mgmmy.gmms)
    for key in xkeys ∪ ykeys
        if key ∈ xkeys && key ∈ykeys
            push!(gmms, Pair(key, convert(t, combine(mgmmx.gmms[key], mgmmy.gmms[key]))))
        elseif key ∈ xkeys
            push!(gmms, Pair(key, convert(t, mgmmx.gmms[key])))
        else
            @show convert(t,mgmmy.gmms[key])
            push!(gmms, Pair(key, convert(t, mgmmy.gmms[key])))
        end
    end
    return promote_type(typeof(mgmmx), typeof(mgmmy))(gmms)
end

function combine(gmms::Union{AbstractVector{<:AbstractSingleGMM},AbstractVector{<:AbstractMultiGMM}})
    if length(gmms) > 1
        return combine([combine(gmms[1],gmms[2]), gmms[3:end]...])
    elseif length(gmms) == 1
        return gmms[1]
    else
        throw(ArgumentError("provided no GMMs to combine"))
    end
end