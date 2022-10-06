using PlyIO

function inside(pt, space::GMA.TranslationRegion)
    center = space.T
    halfwidth = space.σₜ
    return (center[1] - halfwidth <= pt[1] < center[1] + halfwidth) && (center[2] - halfwidth <= pt[2] < center[2] + halfwidth) && (center[3] - halfwidth <= pt[3] < center[3] + halfwidth)
end

function downsample(pts, space)
    
end

# Stanford Bunny, downsampled
bunny = load_ply("./data/bun_zipper_res4.ply")
props = bunny["vertex"].properties
bunpts = [[props[1][i], props[3][i], props[2][i]] for i in 1:length(props[1])]

# voxel downsampling
xmin, xmax = minimum(props[1]), maximum(props[1])
xw, xc = xmax - xmin, 0.5 * (xmin + xmax)
ymin, ymax = minimum(props[2]), maximum(props[2])
yw, yc = ymax - ymin, 0.5 * (ymin + ymax)
zmin, zmax = minimum(props[3]), maximum(props[3])
zw, zc = zmax - zmin, 0.5 * (zmin + zmax)
space = GMA.TranslationRegion(RotationVec(0.,0.,0.,), SVector{3}(xc, yc, zc), max(xw, yw, zw))
targetnpts = 200

subspaces = [space]
downpts = eltype(bunpts)[]
ppvox = Int[]
while length(downpts) < targetnpts
    newpts = eltype(bunpts)[]
    newppvox = Int[]
    for ssp in subspaces
        ptsinside = filter(x -> inside(x, ssp), bunpts)
        len = length(ptsinside)
        if len > 0
            push!(newppvox, len)
            push!(newpts, sum(ptsinside)./len)
        end
    end

    length(newpts) > targetnpts && break

    empty!(downpts)
    empty!(ppvox)
    append!(downpts, newpts)
    append!(ppvox, newppvox)

    newspaces = eltype(subspaces)[]
    for ssp in subspaces
        append!(newspaces, GMA.subregions(ssp))
    end
    empty!(subspaces)
    append!(subspaces, newspaces)
end

# partial point sets
leftbunpts = filter(x -> x[1] <= xc + xw / 4, downpts)
rightbunpts = filter(x -> x[1] >= xc - xw / 4, downpts)

# # create PointSet from downsampled points
# psbun = PointSet(downpts)

# # viz
# fig, ax, l = scatter([pt[1] for pt in downpts], [pt[2] for pt in downpts], [pt[3] for pt in downpts]; markersize=25, axis=(; type=Axis3, aspect=:data, viewmode=:fit))
# fig2, ax2, l = scatter([pt[1] for pt in bunpts], [pt[2] for pt in bunpts], [pt[3] for pt in bunpts]; axis=(; type=Axis3, aspect=:data, viewmode=:fit))

# nframes = 400
# record(fig, "downsampled_bun.mp4", 1:nframes) do frame
#     ax.azimuth[] = 1.7pi + 2pi * frame / (nframes / ceil(nframes/200)) 
# end
# record(fig2, "full_bun.mp4", 1:nframes) do frame
#     ax2.azimuth[] = 1.7pi + 2pi * frame / (nframes / ceil(nframes/200)) 
# end


