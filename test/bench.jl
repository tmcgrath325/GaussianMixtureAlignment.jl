using GOGMA
using CUDA
using BenchmarkTools
using StaticArrays

xpts = [10*rand(3).-6 for i in 1:50]
ypts = [10*rand(3).-4 for i in 1:50]
gmmx = IsotropicGMM([IsotropicGaussian(x, 1, 1) for x in xpts])
gmmy = IsotropicGMM([IsotropicGaussian(y, 1, 1) for y in ypts])
t = promote_type(eltype(gmmx), eltype(gmmy))
xgausses = CuArray(gmmx.gaussians)
ygausses = CuArray(gmmy.gaussians)
bl = Block(gmmx, gmmy)
subcntrs = GOGMA.subcenters(bl.center, bl.rwidth, bl.twidth, 4)
gpusubcntrs = CuArray(subcntrs)
bounds = CuArray(fill((zero(t),zero(t)), length(subcntrs)))
rw = rand()
tw = rand()

function bench_bounds_gpu!(b, x, y, rw, tw, X)
    numblocks = ceil(Int, length(y)/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks GOGMA.bounds_kernel!(b, x, y, rw, tw, X)
    end
end

function bench_bounds_cpu(gmmx, gmmy, rw, tw, X)
    bounds = fill((0.,0.), length(X))
    for (i,center) in enumerate(X)
        bounds[i] = get_bounds(gmmx, gmmy, rw, tw, center)
    end
    return bounds
end

function bench_bounds_threads(gmmx, gmmy, rw, tw, X)
    bounds = fill((0.,0.), length(X))
    @Threads.threads for i=1:length(X)
        bounds[i] = get_bounds(gmmx, gmmy, rw, tw, X[i])
    end
    return bounds
end

@btime bounds = bench_bounds_cpu(gmmx, gmmy, rw, tw, subcntrs);

@btime bench_bounds_threads(gmmx, gmmy, rw, tw, subcntrs)

@btime bench_bounds_gpu!($bounds, $xgausses, $ygausses, $bl.rwidth, $bl.twidth, $gpusubcntrs)

