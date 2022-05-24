using GaussianMixtureAlignment
using StaticArrays
using CoordinateTransformations
using Rotations

using GaussianMixtureAlignment: goicp_align, goih_align

randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(5*rand(3)...))
ycoords = rand(3, 20) * 10 .- 20
xcoords = randtform(ycoords)
weights = ones(Float64, 20)
xset = PointSet(xcoords, weights)
yset = PointSet(ycoords, weights)

print("GO-ICP:")   
@show res_icp.num_splits
@show res_icp.upperbound, res_icp.lowerbound

print("GO-IH:")
@time res_ih  = goih_align(xset, yset, maxsplits=100000)
@show res_ih.num_splits
@show res_ih.upperbound, res_ih.lowerbound
