function objectivefun_gpu(dist, σx, σy, ϕx, ϕy)
    return -ϕx*ϕy * CUDA.exp(-dist^2 / (2*(σx^2 + σy^2))) # / sqrt(2π*(σx^2 + σy^2))^dim 
end

function rot_gpu(rx, ry, rz)
    # Angle-Axis parameterization
    angle = CUDA.sqrt((rx^2 + ry^2 + rz^2))
    if angle==0
        return SMatrix{3,3,typeof(rx)}(I)
    end
    # Convert to quaternion
    qr = CUDA.cos(angle/2)
    nrm = angle/CUDA.sin(angle/2)
    qx = rx/nrm
    qy = ry/nrm
    qz = rz/nrm
    R = @SMatrix([1 - 2*(qy^2 + qz^2)  2*(qx*qy - qz*qr)    2*(qx*qz + qy*qr);
                  2*(qx*qy + qz*qr)    1 - 2*(qx^2 + qz^2)  2*(qy*qz - qx*qr);
                  2*(qx*qz - qy*qr)    2*(qy*qz + qx*qr)    1 - 2*(qx^2 + qy^2)])
    return R
end

function get_bounds_gpu(x::IsotropicGaussian, y::IsotropicGaussian, rwidth, twidth, R0, t0)
    # prepare positions and angles
    x0 = R0*x.μ
    xnorm, ynorm = CUDA.norm(x.μ), CUDA.norm(y.μ-t0)
    if xnorm*ynorm == 0
        cosα = 1.
    else
        cosα = CUDA.dot(x0, y.μ-t0)/(xnorm*ynorm)
    end
    cosβ = CUDA.cos(min(sqrt3*rwidth/2, π))

    # upper bound distance at hypercube center
    ubdist = CUDA.norm(x0 + t0 - y.μ)
    
    # lower bound distance from the nearest point on the "spherical cap"
    if cosα >= cosβ
        lbdist = max(CUDA.abs(xnorm-ynorm) - sqrt3*twidth/2, 0)
    else
        lbdist = max(√(xnorm^2 + ynorm^2 - 2*xnorm*ynorm*(cosα*cosβ+√((1-cosα^2)*(1-cosβ^2)))) - sqrt3*twidth/2, 0)  # law of cosines
    end

    # evaluate objective function at each distance to get upper and lower bounds
    return objectivefun_gpu(lbdist, x.σ, y.σ, x.ϕ, y.ϕ), objectivefun_gpu(ubdist, x.σ, y.σ, x.ϕ, y.ϕ)
end

function get_bounds_gpu(xgausses, ygausses, rwidth, twidth, X)
    rx, ry, rz, tx, ty, tz = X
    if rx^2 + ry^2 + rz^2 > π^2
        return Inf, Inf
    end
    lb = 0.
    ub = 0.
    R0 = rot_gpu(rx, ry, rz)
    t0 = SVector(tx, ty, tz)
    for x in xgausses
        for y in ygausses
            l, u = get_bounds_gpu(x, y, rwidth, twidth, R0, t0)  
            lb, ub = lb+l, ub+u
        end
    end
    return lb, ub
end

function bounds_kernel!(b, x, y, rw, tw, X)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(b)
        @inbounds b[i] = get_bounds_gpu(x, y, rw, tw, X[i])
    end
    return
end