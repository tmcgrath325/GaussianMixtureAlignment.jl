using MutableConvexHulls
using PairedLinkedLists
using CairoMakie

### Plot Data
for (res, searchregion, fname, lim) in zip(results, searchregions, fnames, lims)
    println("Starting $fname:")
    hull = ChanLowerConvexHull{eltype(res.addedpoints[1])}(CCW, true, x -> (x[1], -x[2]))
    σₒ = searchregion.σₜ
    lb, ub = gauss_l2_bounds(gmmx, gmmy, searchregion)
    llim = min(lim[1], 0)
    ulim = max(lim[2], 0)
    addpoint!(hull, (lb, ub, searchregion))

    # initial figure and traces
    f = Figure()
    axis = Axis(f[1, 1], xlabel = "Lower Bound", ylabel = "Realized Value")
    diag = [Point2f(llim,llim), Point2f(ulim,ulim)]
    ubline = Observable([Point2f(res.progress[1][2], llim), Point2f(res.progress[1][2], ulim)])
    points1 = Observable(Point2f[])
    points2 = Observable(Point2f[])
    colors = Observable(Int[])
    ncolors = 8

    lines!(axis, diag, color=:black)
    lines!(axis, ubline, color=:red)
    scatterlines!(axis, points1, color=:black, marker=:circle, markersize=10, overdraw=true)
    scatter!(axis, points2, color=colors, colormap=:Accent_8, colorrange=(1, 8), marker=:circle, markersize=1)
    limits!(axis, llim, ulim, llim, ulim)

    # plot a frame after a specified interval
    interval = 1
    nframes = length(res.removedpoints) # Int(floor(length(res.removedpoints)/interval)+2)
    niters = length(res.addedpoints)
    @show niters
    @show interval
    @show nframes
    record(f, fname, 1:nframes) do frame
        if frame % 10 == 0
            @show frame
        end
        if frame == 1
            removepoint!(hull, head(hull.hull))
            mergepoints!(hull, res.addedpoints[1])

            currentub = res.progress[1][1]
            empty!(ubline.val)
            push!(ubline.val, (currentub, llim), (currentub, ulim))
        else
            idx = interval * (frame - 1)
            previdx = frame == 2 ? 1 : idx - interval
            idx = frame == nframes ? niters : idx
            for i = previdx+1:idx
                rmnode = MutableConvexHulls.getfirst(x -> x.data == res.removedpoints[i], ListNodeIterator(hull.hull))
                subhull = getfirst(x -> x.points===rmnode.target.list, hull.subhulls)
                removepoint!(subhull, rmnode.target)
                deletenode!(rmnode)
                mergepoints!(hull, res.addedpoints[i])
            end

            ubidx = findfirst(x -> x[1] >= idx, res.progress)
            currentub = isnothing(ubidx) ? res.progress[end][2] : res.progress[ubidx][2]
            empty!(ubline.val)
            push!(ubline.val, (currentub, llim), (currentub, ulim))
        end 

        empty!(ubline.val)
        push!(ubline.val, (currentub, llim), (currentub, ulim))
        notify(ubline)

        hullpts = [[x[1], x[2]] for x in hull.hull]

        empty!(points1.val)
        push!(points1.val, hullpts...)
        notify(points1)

        allpts = vcat([collect(x.points) for x in hull.subhulls]...)

        empty!(points2.val)
        push!(points2.val, [(pt[1], pt[2]) for pt in allpts]...)
        notify(points2)

        empty!(colors.val)
        push!(colors.val, [Int(round(log2(σₒ / pt[3].σₜ) - 1)) % 8 + 1 for pt in allpts]...)
        notify(colors)
    end
end
