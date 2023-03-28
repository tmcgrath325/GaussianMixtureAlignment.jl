using MakieCore
using Makie.Colors

const makie_default_colors = Makie.wong_colors()

const halfway_radius = sqrt(3) / 2

θs = range(0, 2π, length=32)
const cosθs = [cos(θ) for θ in θs]
const sinθs = [sin(θ) for θ in θs]

function flat_circle!(pos, r, dim::Int; kwargs...)
    if dim == 3
        xs = [r * cosθ + pos[1] for cosθ in cosθs]
        ys = [r * sinθ + pos[2] for sinθ in sinθs]
        zs = fill(pos[3], 32)
    elseif dim == 2
        xs = [r * cosθ + pos[1] for cosθ in cosθs]
        ys = fill(pos[2], 32)
        zs = [r * sinθ + pos[3] for sinθ in sinθs]
    elseif dim == 1
        xs = fill(pos[1], 32)
        ys = [r * cosθ + pos[2] for cosθ in cosθs]
        zs = [r * sinθ + pos[3] for sinθ in sinθs]
    end
    lines!(xs,ys,zs; kwargs...)
end

function wire_sphere!(pos, r; kwargs...)
    for dim in 1:3
        flat_circle!(pos, r, dim; kwargs...)
        halfwaypos = Float32[0,0,0]
        halfwaypos[dim] = r / 2;
        flat_circle!(pos .- halfwaypos, r * halfway_radius, dim; kwargs...)
        flat_circle!(pos .+ halfwaypos, r * halfway_radius, dim; kwargs...)
    end
end

function solid_sphere(pos, r; kwargs...)
    mesh!(Sphere(Point{3}(pos), r); kwargs...)
end

