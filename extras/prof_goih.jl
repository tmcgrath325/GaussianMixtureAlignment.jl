using GaussianMixtureAlignment
using StaticArrays
using CoordinateTransformations
using Rotations
# Note: ProfileView and BenchmarkTools are not added by this package
using ProfileView
using BenchmarkTools

## Perfectly matched points

# small problem
# xcoords = hcat([[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]...);
# ycoords = hcat([[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]...);
# weights = [1.,1.,1.];
# xset = PointSet(xcoords, weights);
# yset = PointSet(ycoords, weights);

# @btime goih_align(xset, yset; maxsplits=100);
# @btime tiv_goih_align(xset, yset; maxsplits=100);
# @ProfileView.profview goih_align(xset, yset; maxsplits=1000);
# @ProfileView.profview tiv_goih_align(xset, yset; maxsplits=1000);

# larger problem
numpts = 50;
size = 50;
randpts = size / 2 * rand(3,numpts) .- size;
randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(size/4*rand(3)...));
weights = ones(Float64, numpts);
xset = PointSet(randpts, weights);
yset = PointSet(randtform(randpts), weights);

# @time goih_align(xset, yset)
@time res = tiv_goih_align(xset, yset, 1, 1)
@show inv(res.tform) ∘ randtform 
@show res.num_splits
# @ProfileView.profview goih_align(xset, yset; maxsplits=100);
@ProfileView.profview tiv_goih_align(xset, yset, 1, 1);


## with missing points
# numpts = 50;
# size = 50;
# num_missing = 10;
# randpts = size / 2 * rand(3,numpts) .- size;
# randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(size/4*rand(3)...));
# weights = ones(Float64, numpts);
# xset = PointSet(randpts, weights);
# yset = PointSet(randtform(randpts[:, 1:end-num_missing]), weights[1:end-num_missing]);

# @time res = tiv_goih_align(xset, yset, 1, 1)
# @show inv(res.tform) ∘ randtform 
# @show res.num_splits
