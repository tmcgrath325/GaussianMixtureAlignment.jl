using GaussianMixtureAlignment
using MutableConvexHulls
using PairedLinkedLists
using CairoMakie
using GaussianMixtureAlignment: UncertaintyRegion, gauss_l2_bounds, lowestlbnode

f = Figure()
axis = Axis(f[1, 1])
diag = [Point2f(-9,-9), Point2f(0,0)]
ubline = Observable([Point2f(res1.progress[1][2], -9), Point2f(res1.progress[1][2], 0)])
points1 = Observable(Point2f[])
points2 = Observable(Point2f[])
colors = Observable(Int[])

lines!(axis, diag, color=:black)
lines!(axis, ubline, color=:red)
scatterlines!(axis, points1, color=:pink, marker=:circle, markersize=20, overdraw=true)
scatter!(axis, points2, color=:blue, marker=:circle, markersize=1)

hull = MutableLowerConvexHull{Tuple{Float64,Float64}}(CCW, true, x -> (x[1], -x[2]))
initblock = UncertaintyRegion(gmmx, gmmy)
lb, ub = gauss_l2_bounds(gmmx, gmmy, initblock)
addpoint!(hull, (lb, ub))

interval = 100
nframes = Int(floor(length(res1.removedpoints)/interval)+2)
niters = length(res1.addedpoints)
@show niters
@show interval
@show nframes
record(f, "simple_lowestlb_gogma.mp4", 1:nframes) do frame
    if frame == 1
        removepoint!(hull, head(hull.hull))
        mergepoints!(hull, res1.addedpoints[1])
        lowestnode = lowestlbnode(hull)
        frame == 1 && limits!(axis, lowestnode.data[1]*1.1, 0, lowestnode.data[1]*1.1, 0)

        currentub = res1.progress[1][1]
        empty!(ubline.val)
        push!(ubline.val, (currentub, -9), (currentub, 9))
    else
        idx = interval * (frame - 1)
        previdx = frame == 2 ? 1 : idx - interval
        idx = frame == nframes ? niters : idx
        for i = previdx+1:idx
            rmnode = MutableConvexHulls.getfirst(x -> x.data == res1.removedpoints[i], ListNodeIterator(hull.points))
            removepoint!(hull, rmnode)
            mergepoints!(hull, res1.addedpoints[i])
        end

        currentub = findfirst(x -> x[1] >= idx, res1.progress)
        currentub = isnothing(currentub) ? res1.progress[end][2] : currentub[2]
        empty!(ubline.val)
        push!(ubline.val, (currentub, -9), (currentub, 0))
    end 

    empty!(ubline.val)
    push!(ubline.val, (currentub, -9), (currentub, 9))
    empty!(points1.val)
    push!(points1.val, collect(hull.hull)...)
    empty!(points2.val)
    push!(points2.val, collect(hull.points)...)
    notify(ubline)
    notify(points1)
    notify(points2)
end
