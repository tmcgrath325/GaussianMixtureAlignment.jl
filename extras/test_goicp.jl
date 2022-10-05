using GaussianMixtureAlignment
using StaticArrays
using CoordinateTransformations
using Rotations
# Note: ProfileView and BenchmarkTools are not added by this package
using ProfileView
using BenchmarkTools

# using PlotlyJS
# using GaussianMixtureAlignment: plotdrawing, drawPointSet, default_colors
const GMA = GaussianMixtureAlignment


# small problem
xcoords = hcat([[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]...);
ycoords = hcat([[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]...);
weights = [1.,1.,1.];
xset = PointSet(xcoords, weights);
yset = PointSet(ycoords, weights);

# @btime goicp_align(xset, yset; maxsplits=100);
# @btime tiv_goicp_align(xset, yset; maxsplits=100);
@ProfileView.profview goicp_align(xset, yset; maxsplits=100000);
# @ProfileView.profview tiv_goicp_align(xset, yset; maxsplits=1000);

# larger problem
# numpts = 50;
# size = 50;
# num_missing = 10;
# num_extra = 10;
# tiv_factor = 2;
# randpts = size / 2 * rand(3,numpts) .- size;
# extrapts = size / 2 * rand(3,num_extra) .- size;
# randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(size/4*rand(3)...));
# weights = ones(Float64, numpts);
# xset = PointSet(hcat(randpts[:, 1:end-num_missing], extrapts), vcat(weights[1:end-num_missing], ones(num_extra)));
# yset = PointSet(randtform(randpts), weights);

# @time res = goicp_align(xset, yset; centerinputs=true)
# @time res = tiv_goicp_align(xset, yset, tiv_factor, tiv_factor; centerinputs=false)
# @show res.num_splits
# @ProfileView.profview tiv_goicp_align(xset, yset; maxsplits=1000);
# @ProfileView.profview tiv_goicp_align(xset, yset, 1, 1; maxsplits=1000);

# tiv_plot = [drawPointSet(res.rotation_result.x; opacity=0.5), drawPointSet(res.rotation_result.y; color=GMA.default_colors[2], opacity=0.5)];
# tiv_rot_plot = [drawPointSet(res.rotation_result.tform(res.rotation_result.x); opacity=0.5), drawPointSet(res.rotation_result.y; color=GMA.default_colors[2], opacity=0.5)];
# rot_plot = [drawPointSet(res.rotation_result.tform(res.x); opacity=0.5), drawPointSet(res.y; color=GMA.default_colors[2], opacity=0.5)];
# original_plot = [drawPointSet(xset; opacity=0.5), drawPointSet(yset; color=default_colors[2], opacity=0.5)];
# alignment_plot = [drawPointSet(res.tform(xset); opacity=0.5), drawPointSet(yset; color=default_colors[2], opacity=0.5)];
# plotdrawing(alignment_plot)