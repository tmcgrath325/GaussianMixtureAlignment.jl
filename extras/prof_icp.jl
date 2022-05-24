using GaussianMixtureAlignment
using StaticArrays
using CoordinateTransformations
using Rotations
# Note: ProfileView and BenchmarkTools are not added by this package
using ProfileView
using BenchmarkTools

using GaussianMixtureAlignment: goicp_align

# small problem
xcoords = hcat([[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]...)
ycoords = hcat([[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]...)
weights = [1.,1.,1.]
xset = PointSet(xcoords, weights)
yset = PointSet(ycoords, weights)

goicp_align(xset, yset; maxsplits=1e3)
goih_align(xset, yset; maxsplits=100)

# larger problem
randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(5*rand(3)...))
ycoords = rand(3, 50) * 10 .- 20
xcoords = randtform(ycoords)
weights = ones(Float64, 50)
xset = PointSet(xcoords, weights)
yset = PointSet(ycoords, weights)

@ProfileView.profview goicp_align(xset, yset; maxsplits=1e3)
@ProfileView.profview goih_align(xset, yset; maxsplits=100)
