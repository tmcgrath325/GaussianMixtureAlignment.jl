using GaussianMixtureAlignment
using ProfileView
using MolecularGraph

GMA = GaussianMixtureAlignment

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

@ProfileView.profview gogma_align(gmmx, gmmy; maxsplits=10000)
@ProfileView.profview gogma_align(gmm1, gmm2; maxsplits=10000)
@ProfileView.profview gogma_align(gmmx, gmmy; maxsplits=10000, nextblockfun=GMA.randomblock)
@ProfileView.profview gogma_align(gmm1, gmm2; maxsplits=10000, nextblockfun=GMA.randomblock)