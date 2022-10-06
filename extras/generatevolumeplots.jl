using GaussianMixtureAlignment
using LinearAlgebra
using StaticArrays
using Rotations
using CoordinateTransformations
using MolecularGraph
using PlyIO
using MutableConvexHulls
using PairedLinkedLists
using CairoMakie

using GaussianMixtureAlignment: TranslationRegion, RotationRegion, UncertaintyRegion, gauss_l2_bounds
GMA = GaussianMixtureAlignment

# function for generating/saving animated plots of the lb-rv space
include("volumeplot.jl")

# boundsfun = (x, y, bl) -> GMA.squared_dist_bounds(x, y, bl; correspondence = GMA.closest_points, distance_bound_fun = GMA.tight_distance_bounds)
boundsfun = (x, y, bl) -> GMA.gauss_l2_bounds(x, y, bl)

### Generate data
# a grid of points (there will be several local minima)
xpts = [Float64[i,j,k] for i=1:3 for j=1:3 for k=1:3]
ypts = [[i,j,k] .+ 0.5 for i=1:2 for j=1:2 for k=1:2]

mol1 = removehydrogens(sdftomol("./data/A7864.sdf"))
mol2 = removehydrogens(sdftomol("./data/Q1570.sdf"))
mol1pts = [a.coords for a in mol1.nodeattrs]
mol2pts = [a.coords for a in mol2.nodeattrs]

σ = 1.0
ϕ = 1.0
gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

gmm1 = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in mol1pts])
# R = RotationVec(π,0,0)
T = SVector{3}(8.0, 3.0, 3.0)
gmm2 = gmm1 + T # IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in mol2pts])

psx = PointSet(xpts)
psy = PointSet(ypts)

ps1 = PointSet(mol1pts)
ps2 = PointSet(mol2pts)

# Stanford Bunny, downsampled
include("bunny.jl")
bungmm1 = IsotropicGMM([IsotropicGaussian(x, 0.02, ϕ) for x in downpts])
R = RotationVec(π/4, π/4, π/4) 
bungmm2 = R * bungmm1

# println("Steroid, lowest lb block")
# # res = GMA.trl_goicp_align(ps1, ps2; atol=0.0001, maxsplits=1000, nextblockfun=GMA.lowestlbblock);
# res = gogma_align(gmm1, gmm2; atol=0.0000001, maxsplits=600, nextblockfun=GMA.lowestlbblock, blockfun=GMA.TranslationRegion, tformfun=Translation)
# searchregion = TranslationRegion(gmm1, gmm2)
# fname = "steroid_translation_volume_lowestlb.mp4"
# lim = (-300, 0) # boundsfun(gmm1, gmm2, searchregion)
# firstpoint = boundsfun(gmm1, gmm2, searchregion)
# volumeplot(res, searchregion, fname, lim, firstpoint)
# # attempt to free memory
# res = nothing
# GC.gc()

# println("Steroid, rand lb block")
# # res = GMA.trl_goicp_align(ps1, ps2; atol=0.0001, maxsplits=1000, nextblockfun=GMA.lowestlbblock);
# res = gogma_align(gmm1, gmm2; atol=0.0000001, maxsplits=600, nextblockfun=GMA.randomblock, blockfun=GMA.TranslationRegion, tformfun=Translation)
# searchregion = TranslationRegion(gmm1, gmm2)
# fname = "steroid_translation_volume_rand.mp4"
# lim = (-300, 0) # boundsfun(gmm1, gmm2, searchregion)
# firstpoint = boundsfun(gmm1, gmm2, searchregion)
# volumeplot(res, searchregion, fname, lim, firstpoint)
# # attempt to free memory
# res = nothing
# GC.gc()

println("Bunny, lowest lb block")
# res = GMA.trl_goicp_align(ps1, ps2; atol=0.0001, maxsplits=1000, nextblockfun=GMA.lowestlbblock);
res = gogma_align(bungmm1, bungmm2; atol=0.0000001, maxsplits=600, nextblockfun=GMA.lowestlbblock, blockfun=GMA.TranslationRegion, tformfun=Translation)
searchregion = TranslationRegion(bungmm1, bungmm2)
fname = "steroid_translation_volume_lowestlb.mp4"
lim = (-300, 0) # boundsfun(bungmm1, bungmm2, searchregion)
firstpoint = boundsfun(bungmm1, bungmm2, searchregion)
volumeplot(res, searchregion, fname, lim, firstpoint)
# attempt to free memory
res = nothing
GC.gc()

println("Bunny, rand lb block")
# res = GMA.trl_goicp_align(ps1, ps2; atol=0.0001, maxsplits=1000, nextblockfun=GMA.lowestlbblock);
res = gogma_align(bungmm1, bungmm2; atol=0.0000001, maxsplits=600, nextblockfun=GMA.randomblock, blockfun=GMA.TranslationRegion, tformfun=Translation)
searchregion = TranslationRegion(bungmm1, bungmm2)
fname = "steroid_translation_volume_rand.mp4"
lim = (-300, 0) # boundsfun(bungmm1, bungmm2, searchregion)
firstpoint = boundsfun(bungmm1, bungmm2, searchregion)
volumeplot(res, searchregion, fname, lim, firstpoint)
# attempt to free memory
res = nothing
GC.gc()

