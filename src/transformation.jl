# Extend rotation and transformation methods from Rotations and CoordinateTransformations

function Rotations.AngleAxis(rx, ry, rz) 
    # Note: promoting with Float64 here in order to avoid strange behavior when passing AngleAxis{Int} to AffineMap()
    #       I don't think this is a big deal, since the only time a RotMatrix{Int} will be valid is for the identity matrix
    t = promote_type(typeof(rx), typeof(ry), typeof(rz), Float64)
    theta = √(rx^2+ry^2+rz^2)
    return theta == 0 ? AngleAxis(zero(t),one(t),zero(t),zero(t)) : AngleAxis(theta, rx, ry, rz)
end

function rot_to_axis(R)
    aa = AngleAxis(R)
    return aa.theta.*(aa.axis_x, aa.axis_y, aa.axis_z)
end

CoordinateTransformations.AffineMap(rx,ry,rz,tx,ty,tz) = AffineMap(AngleAxis(rx,ry,rz), SVector(tx,ty,tz))

function (tform::AffineMap)(x::IsotropicGaussian)
    return IsotropicGaussian(tform(x.μ), x.σ, x.ϕ, [tform.linear*dir for dir in x.dirs])
end

function (tform::AffineMap)(x::IsotropicGMM)
    return IsotropicGMM([tform(g) for g in x.gaussians])
end

function (tform::AffineMap)(x::MultiGMM)
    tformgmms = [tform(x.gmms[key]) for key in keys(x.gmms)]
    gmmdict = Dict{eltype(keys(x.gmms)),eltype(tformgmms)}()
    for (i,key) in enumerate(keys(x.gmms))
        push!(gmmdict, key=>tformgmms[i])
    end
    return MultiGMM(gmmdict)
end