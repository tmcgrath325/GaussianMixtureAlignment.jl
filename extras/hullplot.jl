using GaussianMixtureAlignment
using MutableConvexHulls
using PairedLinkedLists
using GLMakie
using GaussianMixtureAlignment: UncertaintyRegion, gauss_l2_bounds, lowestlbnode

f = Figure()
axis = Axis(f[1, 1])
diag = [Point2f(-9,-9), Point2f(0,0)]
points1 = Observable(Point2f[])
points2 = Observable(Point2f[])
colors = Observable(Int[])

lines!(axis, diag, color=:black)
scatterlines!(axis, points1, color=:pink, marker=:circle, markersize=20, overdraw=true)
scatter!(axis, points2, color=:blue, marker=:circle, markersize=3)

hull = MutableLowerConvexHull{Tuple{Float64,Float64}}()
initblock = UncertaintyRegion(gmmx, gmmy)
lb, ub = gauss_l2_bounds(gmmx, gmmy, initblock)
addpoint!(hull, (lb, ub))

interval = 1000
nframes = Int(floor(length(res1.removedpoints)/interval))
@show length(res1.removedpoints)
@show interval
@show nframes
record(f, "simple_lowestlb_gogma.mp4", 1:nframes) do frame
    idx = interval * frame
    previdx = idx - interval
    @show idx
    for i = previdx+1:idx
        rmnode = MutableConvexHulls.getfirst(x -> x.data == res1.removedpoints[i], ListNodeIterator(hull.points))
        removepoint!(hull, rmnode)
        mergepoints!(hull, res1.addedpoints[i])
    end

    empty!(points1.val)
    push!(points1.val, collect(hull.hull)...)
    empty!(points2.val)
    push!(points2.val, collect(hull.points)...)
    notify(points1)
    notify(points2)

    lowestnode = lowestlbnode(hull)
    limits!(axis, lowestnode.data[1]*1.1, 0, lowestnode.data[2]*1.1, 0)
end
