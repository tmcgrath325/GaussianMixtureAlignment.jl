# Extend rotation and transformation methods from Rotations and CoordinateTransformations

function Rotations.AngleAxis(rx, ry, rz) 
    # Note: promoting with Float64 here in order to avoid strange behavior when passing AngleAxis{Int} to AffineMap()
    #       I don't think this is a big deal, since RotMatrix{Int} is almost never valid (identity, reflections, etc...)
    t = promote_type(typeof(rx), typeof(ry), typeof(rz), Float64)
    theta = √(rx^2+ry^2+rz^2)
    return theta == 0 ? AngleAxis(zero(t),one(t),zero(t),zero(t)) : AngleAxis(theta, rx, ry, rz)
end

function rot_to_axis(R::RotationVec)
    aa = AngleAxis(R)
    return aa.theta.*(aa.axis_x, aa.axis_y, aa.axis_z)
end

function affinemap_to_params(tform::AffineMap)
    return (rot_to_axis(tform.linear)..., tform.translation)
end

CoordinateTransformations.LinearMap(rx,ry,rz) = CoordinateTransformations.LinearMap(AngleAxis(rx,ry,rz))
CoordinateTransformations.AffineMap(rx,ry,rz,tx,ty,tz) = CoordinateTransformations.AffineMap(AngleAxis(rx,ry,rz), SVector(tx,ty,tz))

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
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first field must be `gmms`
    gmmdict = Dict{eltype(keys(x.gmms)),eltype(x)}()
    for (i,key) in enumerate(keys(x.gmms))
        push!(gmmdict, key=>R*x.gmms[key])
    end
    return ty.name.wrapper(gmmdict, otherfields...)
end

function  Base.:+(x::AbstractIsotropicMultiGMM, T::AbstractVector)
    ty = typeof(x)
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first field must be `gmms`
    gmmdict = Dict{eltype(keys(x.gmms)),eltype(x)}()
    for (i,key) in enumerate(keys(x.gmms))
        push!(gmmdict, key=>x.gmms[key]+T)
    end
    return ty.name.wrapper(gmmdict, otherfields...)
end

function  Base.:-(x::AbstractIsotropicMultiGMM, T::AbstractVector)
    ty = typeof(x)
    otherfields = [getfield(x,fname) for fname in fieldnames(typeof(x))][2:end] # first field must be `gmms`
    gmmdict = Dict{eltype(keys(x.gmms)),eltype(x)}()
    for (i,key) in enumerate(keys(x.gmms))
        push!(gmmdict, key=>x.gmms[key]-T)
    end
    return ty.name.wrapper(gmmdict, otherfields...)
end

