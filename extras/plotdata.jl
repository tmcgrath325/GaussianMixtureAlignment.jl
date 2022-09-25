using GaussianMixtureAlignment
using MolecularGraph

using GaussianMixtureAlignment: UncertaintyRegion, gauss_l2_bounds

GMA = GaussianMixtureAlignment

### Generate data

# two sets of points, each forming a 3-4-5 triangle
xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]] 
ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]

# two steroid molecules with H removed
mol1 = removehydrogens(sdftomol("./data/E1050_3d.sdf"))
mol2 = removehydrogens(sdftomol("./data/E1103_3d.sdf"))
mol1pts = [a.coords for a in mol1.nodeattrs]
mol2pts = [a.coords for a in mol2.nodeattrs]

σ = ϕ = 1.
gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

gmm1 = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in mol1pts])
gmm2 = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in mol2pts])

println("start GOGMA:")
println("\tSimple problem, lowest lb block")
res1 = gogma_align(gmmx, gmmy; maxsplits=1000, nextblockfun=GMA.lowestlbblock)
fname1 = "simple_lowestlb_gogma_short.mp4"

println("\tSimple problem, random block")
res2 = gogma_align(gmmx, gmmy; maxsplits=1000, nextblockfun=GMA.randomblock)
fname2 = "simple_rand_gogma_short.mp4"

println("\tSteroids, lowest lb block")
res3 = gogma_align(gmm1, gmm2; maxsplits=1000, nextblockfun=GMA.lowestlbblock)
fname3 = "steroids_lowestlb_gogma_short.mp4"

println("\tSteroids, random block")
res4 = gogma_align(gmm1, gmm2; maxsplits=1000, nextblockfun=GMA.randomblock)
fname4 = "steroids_rand_gogma_short.mp4"

results = [res1, res2, res3, res4]
fnames = [fname1, fname2, fname3, fname4]
searchregions = [UncertaintyRegion(gmmx, gmmy), UncertaintyRegion(gmmx, gmmy), UncertaintyRegion(gmm1, gmm2), UncertaintyRegion(gmm1, gmm2)]
lims = [gauss_l2_bounds(gmmx, gmmy, searchregions[1]), gauss_l2_bounds(gmmx, gmmy, searchregions[2]), gauss_l2_bounds(gmm1, gmm2, searchregions[3]), gauss_l2_bounds(gmm1, gmm2, searchregions[4])]

# results = [res4]
# fnames = [fname4]
# searchregions = [UncertaintyRegion(gmm1, gmm2)]
# lims = [gauss_l2_bounds(gmm1, gmm2, searchregions[1])]
