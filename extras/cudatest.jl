using CUDA
# using GaussianMixtureAlignment
# using MolecularGraph
# import Adapt
using BenchmarkTools

# mol1 = removehydrogens(sdftomol("./data/E1050_3d.sdf"))
# mol2 = removehydrogens(sdftomol("./data/E1103_3d.sdf"))
# mol1pts = [a.coords for a in mol1.nodeattrs]
# mol2pts = [a.coords for a in mol2.nodeattrs]

# xpts = [rand(3) * 1000 for i=1:1000]
# ypts = [rand(3) * 1000 for i=1:1234]

# gmmx = IsotropicGMM([IsotropicGaussian(x, 1.0, 1.0) for x in xpts])
# gmmy = IsotropicGMM([IsotropicGaussian(y, 1.0, 1.0) for y in ypts])

# x = CuArray(gmmx.gaussians);
# y = CuArray(gmmy.gaussians);

# o = CuArray(fill(0.0, length(x), length(y)))

# Adapt.@adapt_structure IsotropicGMM

# function gpu_distsq!(x, y, dsq)
#     xindex = threadIdx().x
#     yindex = threadIdx().y
#     xstride = blockDim().x
#     ystride = blockDim().y
#     for i = xindex:xstride:length(x)
#         for j = yindex:ystride:length(y)
#             @inbounds dsq[i,j] = 0.0
#             for k = 1:1:3
#                 @inbounds dsq[i,j] += (x[i].μ[k] - y[j].μ[k])^2
#             end
#         end
#     end
#     return
# end

# function gpu_overlap!(o, x, y)
#     idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#     strx = blockDim().x * gridDim().x
#     stry = blockDim().y * gridDim().y
#     Nx, Ny = size(o)
#     for j = idy:stry:Ny
#         for i = idx:strx:Nx
#             @inbounds o[i,j] = overlap(x[i], y[j], nothing, nothing)
#         end
#     end
#     return
# end

# function bench_overlap!(o, x, y)
#     kernel = @cuda launch=false gpu_overlap!(o, x, y)
#     config = launch_configuration(kernel.fun)
#     threads = min(length(o), config.threads)
#     blocks = cld(length(o), threads)

#     CUDA.@sync begin
#         kernel(o, x, y; threads, blocks)
#     end
# end
 
# @show @btime bench_overlap!($o, $x, $y)
# @show @btime overlap(gmmx, gmmy)

function cpu_table!(t,a,b)
    for i = 1:length(a)
        for j = 1:length(b)
            @inbounds t[i, j] = exp(a[i] + b[j])
        end
    end
end

function gpu_table!(t, a, b)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    strx = blockDim().x * gridDim().x
    stry = blockDim().y * gridDim().y
    Nx, Ny = size(t)
    for j = idy:stry:Ny
        for i = idx:strx:Nx
            @inbounds t[i, j] = exp(a[i] + b[j])
        end
    end
    return
end

function bench_gpu_table!(t,a, b)
    kernel = @cuda launch=false gpu_table!(t, a, b)
    config = launch_configuration(kernel.fun)
    threads = min(length(t), config.threads)
    blocks = cld(length(t), threads)

    CUDA.@sync begin
        kernel(t,a,b; threads, blocks)
    end
end

a = Vector{Float32}(1:1000)
b = Vector{Float32}(1:1000)
t = fill(zero(Float32), length(a), length(b))
a_d = CuArray(a)
b_d = CuArray(b)
t_d = CuArray(t)

# CUDA.@profile gpu_table!(t_d, a_d, b_d)
