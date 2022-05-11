using GaussianMixtureAlignment
using StaticArrays
using CoordinateTransformations
using Rotations
# Note: ProfileView and BenchmarkTools are not added by this package
using ProfileView
using BenchmarkTools

using GaussianMixtureAlignment: gogma_align, tiv_gogma_align, overlap

# small problem
xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]
ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
gmmx = IsotropicGMM([IsotropicGaussian(x, 1, 1) for x in xpts])
gmmy = IsotropicGMM([IsotropicGaussian(y, 1, 1) for y in ypts])

@btime gogma_align(gmmx, gmmy; maxsplits=1e3)
@btime tiv_gogma_align(gmmx, gmmy; maxsplits=1e3)

@ProfileView.profview gogma_align(gmmx, gmmy; maxsplits=1e5)
@ProfileView.profview tiv_gogma_align(gmmx, gmmy)

# steroid-sized problem
randpts = 10*rand(3,50)
randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(10*rand(3)...))
gmmx = IsotropicGMM([IsotropicGaussian(randpts[:,i],1,1) for i=1:size(randpts,2)])
gmmy = randtform(gmmx)
min_overlap_score = overlap(gmmx,gmmx)
@btime res = gogma_align(gmmx,gmmy; maxsplits=1E3)
@btime tiv_res = tiv_gogma_align(gmmx,gmmy,0.5,0.5; maxsplits=1E3)

@ProfileView.profview res = gogma_align(gmmx, gmmy; maxsplits=1e4)
@ProfileView.profview tiv_res = tiv_gogma_align(gmmx, gmmy, 0.5, 0.5; maxsplits=1e4)