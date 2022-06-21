using GaussianMixtureAlignment
using StaticArrays
using CoordinateTransformations
using Rotations
# Note: ProfileView and BenchmarkTools are not added by this package
using ProfileView
using BenchmarkTools

using GaussianMixtureAlignment: goicp_align

# small problem
# xcoords = hcat([[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]...);
# ycoords = hcat([[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]...);
# weights = [1.,1.,1.];
# xset = PointSet(xcoords, weights);
# yset = PointSet(ycoords, weights);

# @btime goicp_align(xset, yset; maxsplits=100);
# @btime tiv_goicp_align(xset, yset; maxsplits=100);
# @ProfileView.profview goicp_align(xset, yset; maxsplits=1000);
# @ProfileView.profview tiv_goicp_align(xset, yset; maxsplits=1000);

# larger problem
numpts = 10;
size = 50;
randpts = size / 2 * rand(3,numpts) .- size;
randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(size/4*rand(3)...));
weights = ones(Float64, numpts);
xset = PointSet(randpts, weights);
yset = PointSet(randtform(randpts), weights);

# @time res = goicp_align(xset, yset)
@time res = tiv_goicp_align(xset, yset, 2, 2)
@show inv(res.tform) ∘ randtform 
@show res.num_splits
# @ProfileView.profview goicp_align(xset, yset; maxsplits=1000);
# @ProfileView.profview tiv_goicp_align(xset, yset, 1, 1; maxsplits=1000);
