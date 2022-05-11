# There is some concern about the inferability of the functions below. Using Test.@inferred did not throw any errors

# function (tform::Union{LinearMap,Translation,AffineMap})(x::AbstractIsotropicGaussian)
#     T = typeof(x)
#     otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][5:end] # first 4 fields must be `μ`, `σ`, `ϕ`, and `dirs`
#     return T.name.wrapper(tform(x.μ), x.σ, x.ϕ, [tform.linear*dir for dir in x.dirs], otherfields...)
# end

# function (tform::Union{LinearMap,Translation,AffineMap})(x::AbstractIsotropicGMM)
#     T = typeof(x)
#     otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first field must be `gaussians`
#     return T.name.wrapper([tform(g) for g in x.gaussians], otherfields...)
# end

# function (tform::Union{LinearMap,Translation,AffineMap})(x::AbstractIsotropicMultiGMM)
#     T = typeof(x)
#     otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first field must be `gmms`
#     rotgmms = [tform(x.gmms[key]) for key in keys(x.gmms)]
#     gmmdict = Dict{eltype(keys(x.gmms)),eltype(rotgmms)}()
#     for (i,key) in enumerate(keys(x.gmms))
#         push!(gmmdict, key=>rotgmms[i])
#     end
#     return T.name.wrapper(gmmdict, otherfields...)
# end

function Base.:*(R::AbstractMatrix, x::AbstractIsotropicGaussian)
    ty = typeof(x)
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][5:end] # first 4 fields must be `μ`, `σ`, `ϕ`, and `dirs`
    return ty.name.wrapper(R*x.μ, x.σ, x.ϕ, [R*dir for dir in x.dirs], otherfields...)
end

function Base.:+(x::AbstractIsotropicGaussian, T::AbstractVector,)
    ty = typeof(x)
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][5:end] # first 4 fields must be `μ`, `σ`, `ϕ`, and `dirs`
    return ty.name.wrapper(x.μ+T, x.σ, x.ϕ, x.dirs, otherfields...)
end

function Base.:-(x::AbstractIsotropicGaussian, T::AbstractVector,)
    ty = typeof(x)
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][5:end] # first 4 fields must be `μ`, `σ`, `ϕ`, and `dirs`
    return ty.name.wrapper(x.μ-T, x.σ, x.ϕ, x.dirs, otherfields...)
end

function Base.:*(R::AbstractMatrix, x::AbstractIsotropicGMM)
    ty = typeof(x)
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first fields must be `gaussians`
    return ty.name.wrapper([R*g for g in x.gaussians], otherfields...)
end

function Base.:+(x::AbstractIsotropicGMM, T::AbstractVector,)
    ty = typeof(x)
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first fields must be `gaussians`
    return ty.name.wrapper([g+T for g in x.gaussians], otherfields...)
end

function Base.:-(x::AbstractIsotropicGMM, T::AbstractVector,)
    ty = typeof(x)
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first fields must be `gaussians`
    return ty.name.wrapper([g-T for g in x.gaussians], otherfields...)
end

function Base.:*(R::AbstractMatrix, x::AbstractIsotropicMultiGMM)
    ty = typeof(x)
    gmmkeys = keys(x.gmms)
    gmmdict = Dict(first(gmmkeys)=>R*x.gmms[first(gmmkeys)])
    for (i,key) in enumerate(gmmkeys)
        i === 1 ? continue : push!(gmmdict, key=>R*x.gmms[key])
    end
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first field must be `gmms`
    return ty.name.wrapper(gmmdict, otherfields...)
end

function  Base.:+(x::AbstractIsotropicMultiGMM, T::AbstractVector)
    ty = typeof(x)
    gmmkeys = keys(x.gmms)
    gmmdict = Dict(first(gmmkeys)=>x.gmms[first(gmmkeys)]+T)
    for (i,key) in enumerate(gmmkeys)
        i === 1 ? continue : push!(gmmdict, key=>x.gmms[key]+T)
    end
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first field must be `gmms`
    return ty.name.wrapper(gmmdict, otherfields...)
end

function  Base.:-(x::AbstractIsotropicMultiGMM, T::AbstractVector)
    ty = typeof(x)
    gmmkeys = keys(x.gmms)
    gmmdict = Dict(first(gmmkeys)=>x.gmms[first(gmmkeys)]-T)
    for (i,key) in enumerate(gmmkeys)
        i === 1 ? continue : push!(gmmdict, key=>x.gmms[key]-T)
    end
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first field must be `gmms`
    return ty.name.wrapper(gmmdict, otherfields...)
end

