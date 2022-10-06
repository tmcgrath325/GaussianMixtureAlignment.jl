using GaussianMixtureAlignment
using MolecularGraph
using MutableConvexHulls
using PairedLinkedLists
using StaticArrays
using Rotations
using CairoMakie

using GaussianMixtureAlignment: UncertaintyRegion, gauss_l2_bounds
GMA = GaussianMixtureAlignment

# function for generating/saving animated plots of the lb-rv space
include("makeplot.jl")

### Generate data
# two sets of points, each forming a 3-4-5 triangle
xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]] 
ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]

# two steroid molecules with H removed
mol1 = removehydrogens(sdftomol("./data/A7864.sdf"))
mol2 = removehydrogens(sdftomol("./data/E0588.sdf"))
mol1pts = [a.coords for a in mol1.nodeattrs]
mol2pts = [a.coords for a in mol2.nodeattrs]

# σ = 1.0
# ϕ = 1.0
# gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
# gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

# gmm1 = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in mol1pts])
# gmm2 = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in mol2pts])

psx = PointSet(xpts)
psy = PointSet(ypts)
ps1 = PointSet(mol1pts)
ps2 = PointSet(mol2pts)

# Stanford Bunny, downsampled
include("bunny.jl")
bungmm1 = IsotropicGMM([IsotropicGaussian(x, 0.005, 1.0) for x in downpts])
R = RotationVec(π/4, π/2, π/4)
bungmm2 = R * bungmm1

psbun1 = PointSet(downpts)
psbun2 = R * psbun1

ppsbun1 = R*PointSet(leftbunpts)
ppsbun2 = PointSet(rightbunpts)
pgmmbun1 = R*(IsotropicGMM([IsotropicGaussian(x, 0.005, 1.0) for x in leftbunpts]) - GMA.centroid(hcat(leftbunpts...)))
pgmmbun2 = IsotropicGMM([IsotropicGaussian(x, 0.005, 1.0) for x in rightbunpts]) - GMA.centroid(hcat(rightbunpts...))

boundsfun = (x, y, bl) -> GMA.gauss_l2_bounds(x, y, bl)
# boundsfun = (x, y, bl) -> GMA.squared_dist_bounds(x, y, bl; correspondence = GMA.hungarian_assignment, distance_bound_fun = GMA.tight_distance_bounds)
# boundsfun = (x, y, bl) -> GMA.squared_dist_bounds(x, y, bl; correspondence = GMA.closest_points, distance_bound_fun = GMA.tight_distance_bounds)

### Generate plots 
# Plot 1
# println("\tSimple problem, lowest lb block")
# res = goicp_align(gmmx, gmmy; maxsplits=1000, nextblockfun=GMA.lowestlbblock, atol=0.001)
# searchregion = UncertaintyRegion(gmmx, gmmy)
# fname = "simple_lowestlb_goicp.mp4"
# lim = gauss_l2_bounds(gmmx, gmmy, searchregion)
# firstpoint = boundsfun(gmmx, gmmy, searchregion)
# makeplot(res, searchregion, fname, lim, firstpoint)
# # attempt to free memory
# res = nothing
# GC.gc()

# # Plot 2
# println("\tSimple problem, random block")
# res = goicp_align(gmmx, gmmy; maxsplits=1000, nextblockfun=GMA.randomblock, atol=0.002)
# searchregion = UncertaintyRegion(gmmx, gmmy)
# fname = "simple_rand_goicp.mp4"
# lim = gauss_l2_bounds(gmmx, gmmy, searchregion)
# firstpoint = boundsfun(gmmx, gmmy, searchregion)
# makeplot(res, searchregion, fname, lim, firstpoint)
# # attempt to free memory
# res = nothing
# GC.gc()

# # Plot 3
# println("\tSteroids, lowest lb block")
# res = goicp_align(ps1, ps2; maxsplits=200, nextblockfun=GMA.lowestlbblock, atol=0.001)
# searchregion = UncertaintyRegion(ps1, ps2)
# fname = "steroids_lowestlb_goicp.mp4"
# lim = (0., 400.) # GMA.squared_dist_bounds(ps1, ps
# firstpoint = boundsfun(ps1, ps2, searchregion)
# makeplot(res, searchregion, fname, lim, firstpoint)
# # attempt to free memory
# res = nothing
# GC.gc()

# # Plot 4
# println("\tSteroids, random block")
# res = goicp_align(ps1, ps2; maxsplits=1000, nextblockfun=GMA.randomblock, atol=0.001)
# searchregion = UncertaintyRegion(ps1, ps2)
# fname = "steroids_rand_goicp.mp4"
# lim = lim = (0., 400.) # GMA.squared_dist_bounds(p
# firstpoint = boundsfun(ps1, ps2, searchregion)
# makeplot(res, searchregion, fname, lim, firstpoint)
# # attempt to free memoryres = nothing
# GC.gc()

# # Plot 5
# println("\tBunny, lowest lb block")
# res = goicp_align(psbun1, psbun2; maxsplits=1000, nextblockfun=GMA.lowestlbblock, atol=0.001)
# searchregion = UncertaintyRegion(psbun1, psbun2)
# fname = "bun_lowestlb_goicp.mp4"
# lim = boundsfun(psbun1, psbun2, searchregion) # (-2000., 0.)
# firstpoint = boundsfun(psbun1, psbun2, searchregion)
# makeplot(res, searchregion, fname, lim, firstpoint)
# # attempt to free memory
# res = nothing
# GC.gc()

# # Plot 6
# println("\tBunny, random block")
# res = goicp_align(psbun1, psbun2; maxsplits=1000, nextblockfun=GMA.randomblock, atol=0.001)
# searchregion = UncertaintyRegion(psbun1, psbun2)
# fname = "bun_rand_goicp.mp4"
# lim = boundsfun(psbun1, psbun2, searchregion)
# firstpoint = boundsfun(psbun1, psbun2, searchregion)
# makeplot(res, searchregion, fname, lim, firstpoint)
# # attempt to free memoryres = nothing
# GC.gc()

# Plot 7
println("\tPartial Bun, lowest lb block")
res = gogma_align(pgmmbun1, pgmmbun2; maxsplits=1000, nextblockfun=GMA.lowestlbblock, atol=0.001)
searchregion = UncertaintyRegion(pgmmbun1, pgmmbun2)
fname = "partialbun_lowestlb_gogma.mp4"
lim = (-4000., -100.) # boundsfun(pgmmbun1, pgmmbun2, searchregion) # 
firstpoint = boundsfun(pgmmbun1, pgmmbun2, searchregion)
makeplot(res, searchregion, fname, lim, firstpoint)
# attempt to free memory
res = nothing
GC.gc()

# Plot 8
println("\tPartial Bun, random block")
res = gogma_align(pgmmbun1, pgmmbun2; maxsplits=1000, nextblockfun=GMA.randomblock, atol=0.001)
searchregion = UncertaintyRegion(pgmmbun1, pgmmbun2)
fname = "partialbun_rand_gogma.mp4"
lim = (-4000., -100.) #  boundsfun(pgmmbun1, pgmmbun2, searchregion)
firstpoint = boundsfun(pgmmbun1, pgmmbun2, searchregion)
makeplot(res, searchregion, fname, lim, firstpoint)
