using GaussianMixtureAlignment
using CoordinateTransformations
# Note: ProfileView and BenchmarkTools are not added by this package
using ProfileView
using BenchmarkTools

# small problem
xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]
ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
gmmx = IsotropicGMM([IsotropicGaussian(x, 1, 1) for x in xpts])
gmmy = IsotropicGMM([IsotropicGaussian(y, 1, 1) for y in ypts])

@btime gogma_align(gmmx, gmmy; maxblocks=1e3)
@btime tiv_gogma_align(gmmx, gmmy; maxblocks=1e3)

@ProfileView.profview gogma_align(gmmx, gmmy; maxblocks=1e5)
@ProfileView.profview tiv_gogma_align(gmmx, gmmy)

# steroid-sized problem
randpts = 10*rand(3,50)
randtform = AffineMap(10*rand(6)...)
gmmx = IsotropicGMM([IsotropicGaussian(randpts[:,i],1,1) for i=1:size(randpts,2)])
gmmy = randtform(gmmx)
min_overlap_score = overlap(gmmx,gmmx)
@btime res = gogma_align(gmmx,gmmy; maxblocks=1E3)
@btime tiv_res = tiv_gogma_align(gmmx,gmmy,0.5,0.5; maxblocks=1E3)

@ProfileView.profview res = gogma_align(gmmx, gmmy; maxblocks=1e4)
@ProfileView.profview tiv_res = tiv_gogma_align(gmmx, gmmy, 0.5, 0.5; maxblocks=1e4)