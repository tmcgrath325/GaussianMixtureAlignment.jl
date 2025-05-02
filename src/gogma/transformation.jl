function Base.:*(R::AbstractMatrix{W}, x::IsotropicGaussian{N,V}) where {N,V,W}
    numtype = promote_type(V, W)
    return IsotropicGaussian{N,numtype}(R*x.μ, x.σ, x.ϕ)
end

function Base.:+(x::IsotropicGaussian{N,V}, T::AbstractVector{W}) where {N,V,W}
    numtype = promote_type(V, W)
    return IsotropicGaussian{N,numtype}(x.μ.+T, x.σ, x.ϕ)
end

Base.:-(x::IsotropicGaussian, T::AbstractVector,) = x + (-T)

function Base.:*(R::AbstractMatrix{W}, x::LabeledIsotropicGaussian{N,V,K}) where {N,V,K,W}
    numtype = promote_type(V, W)
    return LabeledIsotropicGaussian{N,numtype,K}(R*x.μ, x.σ, x.ϕ, x.label)
end

function Base.:+(x::LabeledIsotropicGaussian{N,V,K}, T::AbstractVector{W}) where {N,V,K,W}
    numtype = promote_type(V, W)
    return LabeledIsotropicGaussian{N,numtype,K}(x.μ.+T, x.σ, x.ϕ, x.label)
end

Base.:-(x::LabeledIsotropicGaussian, T::AbstractVector,) = x + (-T)

function Base.:*(R::AbstractMatrix{W}, x::IsotropicGMM{N,V}) where {N,V,W}
    numtype = promote_type(V, W)
    return IsotropicGMM{N,numtype}([R*g for g in x.gaussians])
end

function Base.:+(x::IsotropicGMM{N,V}, T::AbstractVector{W}) where {N,V,W}
    numtype = promote_type(V, W)
    return IsotropicGMM{N,numtype}([g+T for g in x.gaussians])
end

Base.:-(x::IsotropicGMM, T::AbstractVector,) = x + (-T)

function Base.:*(R::AbstractMatrix{W}, x::LabeledIsotropicGMM{N,V,K}) where {N,V,W,K}
    numtype = promote_type(V, W)
    return LabeledIsotropicGMM{N,numtype,K}([R*g for g in x.gaussians])
end

function Base.:+(x::LabeledIsotropicGMM{N,V,K}, T::AbstractVector{W}) where {N,V,W,K}
    numtype = promote_type(V, W)
    return LabeledIsotropicGMM{N,numtype,K}([g+T for g in x.gaussians])
end

Base.:-(x::LabeledIsotropicGMM, T::AbstractVector,) = x + (-T)

function  Base.:*(R::AbstractMatrix{W}, x::IsotropicMultiGMM{N,V,K}) where {N,V,K,W}
    numtype = promote_type(V, W)
    gmmdict = Dict{K, IsotropicGMM{N,numtype}}()
    for (key, gmm) in x.gmms
        push!(gmmdict, key=>R*gmm)
    end
    return IsotropicMultiGMM(gmmdict)
end

function  Base.:+(x::IsotropicMultiGMM{N,V,K}, T::AbstractVector{W}) where {N,V,K,W}
    numtype = promote_type(V, W)
    gmmdict = Dict{K, IsotropicGMM{N,numtype}}()
    for  (key, gmm) in x.gmms
        push!(gmmdict, key=>gmm+T)
    end
    return IsotropicMultiGMM(gmmdict)
end

Base.:-(x::IsotropicMultiGMM, T::AbstractVector) = x + (-T)




# There is some concern about the inferability of the functions below. Using Test.@inferred did not throw any errors

# function Base.:*(R::AbstractMatrix, x::AbstractIsotropicGaussian)
#     ty = typeof(x)
#     otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][5:end] # first 4 fields must be `μ`, `σ`, `ϕ`, and `dirs`
#     return ty.name.wrapper(R*x.μ, x.σ, x.ϕ, [R*dir for dir in x.dirs], otherfields...)
# end

# function Base.:+(x::AbstractIsotropicGaussian, T::AbstractVector,)
#     ty = typeof(x)
#     otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][5:end] # first 4 fields must be `μ`, `σ`, `ϕ`, and `dirs`
#     return ty.name.wrapper(x.μ.+T, x.σ, x.ϕ, x.dirs, otherfields...)
# end

# Base.:-(x::AbstractIsotropicGaussian, T::AbstractVector,) = x + (-T)

# function Base.:*(R::AbstractMatrix, x::AbstractIsotropicGMM)
#     ty = typeof(x)
#     otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first fields must be `gaussians`
#     return ty.name.wrapper([R*g for g in x.gaussians], otherfields...)
# end

# function Base.:+(x::AbstractIsotropicGMM, T::AbstractVector,)
#     ty = typeof(x)
#     otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first fields must be `gaussians`
#     return ty.name.wrapper([g+T for g in x.gaussians], otherfields...)
# end

# Base.:-(x::AbstractIsotropicGMM, T::AbstractVector,) = x + (-T)


# function Base.:*(R::AbstractMatrix, x::AbstractIsotropicMultiGMM)
#     ty = typeof(x)
#     gmmkeys = keys(x.gmms)
#     gmmdict = Dict(first(gmmkeys)=>R*x.gmms[first(gmmkeys)])
#     for (i,key) in enumerate(gmmkeys)
#         i === 1 ? continue : push!(gmmdict, key=>R*x.gmms[key])
#     end
#     otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first field must be `gmms`
#     return ty.name.wrapper(gmmdict, otherfields...)
# end

# function  Base.:+(x::AbstractIsotropicMultiGMM, T::AbstractVector)
#     ty = typeof(x)
#     gmmkeys = keys(x.gmms)
#     gmmdict = Dict(first(gmmkeys)=>x.gmms[first(gmmkeys)]+T)
#     for (i,key) in enumerate(gmmkeys)
#         i === 1 ? continue : push!(gmmdict, key=>x.gmms[key]+T)
#     end
#     otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first field must be `gmms`
#     return ty.name.wrapper(gmmdict, otherfields...)
# end

# Base.:-(x::AbstractIsotropicMultiGMM, T::AbstractVector) = x + (-T)


