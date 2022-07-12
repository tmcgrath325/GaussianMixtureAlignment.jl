using GaussianMixtureAlignment
using StaticArrays
using CoordinateTransformations
using Rotations
# Note: ProfileView and BenchmarkTools are not added by this package
using ProfileView
using BenchmarkTools

using PlotlyJS
using GaussianMixtureAlignment: plotdrawing, drawPointSet
const GMA = GaussianMixtureAlignment

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
# numpts = 52;
# size = 50;
# randpts = size / 2 * rand(3,numpts) .- size;
# randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(size/4*rand(3)...));
# weights = ones(Float64, numpts);
# xset = PointSet(randpts, weights);
# yset = PointSet(randtform(randpts), weights);

# @time goih_align(xset, yset)
# @time res = tiv_goih_align(xset, yset, 1, 1)
# @show inv(res.tform) ∘ randtform 
# @show res.num_splits
# @ProfileView.profview goih_align(xset, yset; maxsplits=100);
# @ProfileView.profview tiv_goih_align(xset, yset, 1, 1);


## with missing points
# numpts = 10;
# size = 10;
# num_missing = 2;
# num_extra = 2;
# tiv_factor = 2;
# randpts = size / 2 * rand(3,numpts) .- size;
# extrapts = size / 2 * rand(3,num_extra) .- size;
# randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(size/4*rand(3)...));
# weights = ones(Float64, numpts);
# xset = PointSet(hcat(randpts[:, 1:end-num_missing], extrapts), vcat(weights[1:end-num_missing], ones(num_extra)));
# yset = PointSet(randtform(randpts), weights);

# @time res = tiv_goih_align(xset, yset, tiv_factor, tiv_factor; atol=0.5, centerinputs=false)
@time res = goih_align(xset, yset)
@show inv(res.tform) ∘ randtform 
@show res.num_splits

# plotdrawing([drawPointSet(res.rotation_result.tform(res.rotation_result.x); opacity=0.5), drawPointSet(res.rotation_result.y; color=GMA.default_colors[2], opacity=0.5)])
plotdrawing([drawPointSet(res.tform(xset); opacity=0.5), drawPointSet(yset; color=GMA.default_colors[2], opacity=0.5)])

@ProfileView.profview goih_align(xset, yset)