using GaussianMixtureAlignment
using ProfileView

# two sets of points, each forming a 3-4-5 triangle
xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]] 
ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
σ = ϕ = 1.
gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

# make sure this runs without an error
@time res1 = gogma_align(gmmx, gmmy; maxblocks=1E5)
# @ProfileView.profview gogma_align(gmmx, gmmy)
# res2 = tiv_gogma_align(gmmx, gmmy; maxblocks=1E5)