### Plot Data
# for (res, searchregion, fname, lim) in zip(results, searchregions, fnames, lims)
function makeplot(res, searchregion, fname, lim, firstpoint)
    println("Starting $fname:")
    hull = ChanLowerConvexHull{eltype(res.addedpoints[1])}(CCW, true, x -> (x[1], -x[2]))
    σₒ = searchregion.σₜ
    (lb, ub) = firstpoint
    xllim = min(lim[1], 0)
    xulim = max(lim[2], 0)
    yllim = xllim # min(res.upperbound, 0) * 1.1
    yulim = xulim # max(res.upperbound, 0) * 1.1
    addpoint!(hull, (lb, ub, searchregion))

    # initial figure and traces
    f = Figure()
    axis = Axis(f[1, 1], xlabel = "Lower Bound", ylabel = "Realized Value", aspect=DataAspect())
    diag = [Point2f(xllim,xllim), Point2f(xulim,xulim)]
    ubline = Observable([Point2f(res.progress[1][2], yllim), Point2f(res.progress[1][2], yulim)])
    centerubline = Observable([Point2f(xllim, res.progress[1][3]), Point2f(xulim, res.progress[1][3])])
    points1 = Observable(Point2f[])
    points2 = Observable(Point2f[])
    colors = Observable(Int[])
    ncolors = 8

    lines!(axis, diag, color=:black, linewidth=3)
    lines!(axis, ubline, color=:red, linewidth=3)
    lines!(axis, centerubline, color=:blue, linewidth=3)
    scatterlines!(axis, points1, color=:grey, marker=:circle, markersize=12, linewidth=2, overdraw=true)
    scatter!(axis, points2, color=colors, colormap=:Dark2_8, colorrange=(1, ncolors), marker=:circle, markersize=5)
    limits!(axis, xllim, xulim, yllim, yulim)

    # plot a frame after a specified interval
    interval = 1
    nframes = Int(floor(length(res.removedpoints)/interval)+1) # length(res.removedpoints) # Int(floor(length(res.removedpoints)/interval)+2)
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

            currentub = res.progress[1][2]
            empty!(ubline.val)
            push!(ubline.val, (currentub, yllim), (currentub, yulim))
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
            currentcenterub = isnothing(ubidx) ? res.progress[end][3] : res.progress[ubidx][3]
            empty!(ubline.val)
            push!(ubline.val, (currentub, yllim), (currentub, yulim))
            empty!(centerubline.val)
            push!(centerubline.val, (xllim, currentcenterub), (xulim, currentcenterub))
        end 
        notify(ubline)
        notify(centerubline)

        hullpts = [[x[1], x[2]] for x in hull.hull]

        empty!(points1.val)
        push!(points1.val, hullpts...)
        notify(points1)

        allpts = vcat([collect(x.points) for x in hull.subhulls]...)

        empty!(points2.val)
        push!(points2.val, [(pt[1], pt[2]) for pt in allpts]...)
        notify(points2)

        empty!(colors.val)
        push!(colors.val, [Int(round(log2(σₒ / pt[3].σₜ) - 1)) % ncolors + 1 for pt in allpts]...)
        notify(colors)
    end
end
