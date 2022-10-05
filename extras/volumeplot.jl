const Dark2_8_colors = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666"
]

function searchregionaxis(fig, sr::TranslationRegion; kwargs...)
    llc = GMA.center(sr) .- sr.σₜ
    urc = GMA.center(sr) .+ sr.σₜ
    Axis3(fig[1,2]; limits=(llc[1], urc[1], llc[2], urc[2], llc[3], urc[3]), kwargs...)
end

function searchregionaxis(fig, sr::RotationRegion; kwargs...)
    llc = GMA.center(sr) .- sr.σᵣ
    urc = GMA.center(sr) .+ sr.σᵣ
    Axis3(fig[1,2]; limits=(llc..., urc...), kwargs...)
end

function searchregioncube!(ax, sr::TranslationRegion; kwargs...)
    llc = GMA.center(sr) .- sr.σₜ
    width = 2 * sr.σₜ
    diag = (width, width, width)
    return wireframe!(ax, Makie.FRect3D(Point3f(llc), Point3f(diag)); kwargs...)
end

function searchregioncube!(ax, sr::RotationRegion; kwargs...)
    llc = GMA.center(sr) .- sr.σᵣ
    width = 2 * sr.σᵣ
    diag = (width, width, width)
    return wireframe!(ax, Makie.FRect3D(Point3f(llc), Point3f(diag)); kwargs...)
end

function subcube!(ax, sr::TranslationRegion; kwargs...)
    llc = GMA.center(sr) .- sr.σₜ
    width = 2 * sr.σₜ
    diag = (width, width, width)
    return poly!(ax, Makie.FRect3D(Point3f(llc), Point3f(diag)); kwargs...)
end

function subcube!(ax, sr::RotationRegion)
    llc = GMA.center(sr) .- sr.σᵣ
    width = 2 * sr.σᵣ
    diag = (width, width, width)
    return poly!(ax, Makie.FRect3D(Point3f(llc), Point3f(diag)); kwargs...)
end

### Plot Data
# for (res, searchregion, fname, lim) in zip(results, searchregions, fnames, lims)
function volumeplot(res, searchregion, fname, lim, firstpoint)
    println("Starting $fname:")

    hull = ChanLowerConvexHull{eltype(res.addedpoints[1])}(CCW, true, x -> (x[1], -x[2]))
    (lb, ub) = firstpoint
    addpoint!(hull, (lb, ub, searchregion))

    σₒ = searchregion.σₜ
    xllim = min(lim[1], 0)
    xulim = max(lim[2], 0)
    yllim = xllim # min(res.upperbound, 0) * 1.1
    yulim = xulim # max(res.upperbound, 0) * 1.1

    # initial figure and traces
    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel = "Lower Bound", ylabel = "Realized Value", aspect=DataAspect())
    ax2 = Axis3(fig[1, 2]; aspect=:data, viewmode=:fit)
    diag = [Point2f(xllim,xllim), Point2f(xulim,xulim)]
    ubline = Observable([Point2f(res.progress[1][2], yllim), Point2f(res.progress[1][2], yulim)])
    centerubline = Observable([Point2f(xllim, res.progress[1][3]), Point2f(xulim, res.progress[1][3])])
    points1 = Observable(Point2f[])
    points2 = Observable(Point2f[])
    colors = Observable(Int[])
    ncolors = 8

    # searchregioncube!(ax2, searchregion; color=:black, strokewidth=1, shading=true)
    # lines!(ax1, diag, color=:black, linewidth=3)
    # lines!(ax1, ubline, color=:red, linewidth=3)
    # lines!(ax1, centerubline, color=:blue, linewidth=3)
    # scatterlines!(ax1, points1, color=:grey, marker=:circle, markersize=12, linewidth=2, overdraw=true)
    # scatter!(ax1, points2, color=colors, colormap=:Dark2_8, colorrange=(1, ncolors), marker=:circle, markersize=5)
    # limits!(ax1, xllim, xulim, yllim, yulim)

    # plot a frame after a specified interval
    interval = 1
    nframes = length(res.removedpoints) # Int(floor(length(res.removedpoints)/interval)+2) # Int(floor(length(res.removedpoints)/interval)+2)
    niters = length(res.addedpoints)
    @show niters
    @show interval
    @show nframes
    currentub = res.progress[1][2]
    record(fig, fname, 1:nframes) do frame
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
        # notify(ubline)
        # notify(centerubline)

        empty!(fig) 
        # 2d plot
        ax1 = Axis(fig[1, 1], xlabel = "Lower Bound", ylabel = "Realized Value", aspect=DataAspect())
        hullpts = [[x[1], x[2]] for x in hull.hull]

        empty!(points1.val)
        push!(points1.val, hullpts...)
        # notify(points1)

        allpts = vcat([collect(x.points) for x in hull.subhulls]...)

        empty!(points2.val)
        push!(points2.val, [(pt[1], pt[2]) for pt in allpts]...)
        # notify(points2)

        lines!(ax1, diag, color=:black, linewidth=3)
        lines!(ax1, ubline, color=:red, linewidth=3)
        lines!(ax1, centerubline, color=:blue, linewidth=3)
        scatterlines!(ax1, points1, color=:grey, marker=:circle, markersize=12, linewidth=2, overdraw=true)
        scatter!(ax1, points2, color=colors, colormap=:Dark2_8, colorrange=(1, ncolors), marker=:circle, markersize=5)
        limits!(ax1, xllim, xulim, yllim, yulim)
        
        # 3d plot
        # empty!(ax2.scene)
        ax2 = Axis3(fig[1, 2]; aspect=:data, viewmode=:fit)
        # searchregioncube!(ax2, searchregion; color=:black, strokewidth=1, shading=true)

        empty!(colors.val)
        push!(colors.val, [Int(round(log2(σₒ / pt[3].σₜ) - 1)) % ncolors + 1 for pt in allpts]...)
        # notify(colors)
        for (i,pt) in enumerate(allpts)
            (pt[1] < currentub) && subcube!(ax2, pt[3]; color=(Dark2_8_colors[colors.val[i]],0.2), strokewidth=0, shading=true, transparency=true)
        end

        reset_limits!(ax2)
        ax2.azimuth[] = 1.7pi + 2pi * frame / (nframes / ceil(nframes/200)) 
        fig
    end
end
