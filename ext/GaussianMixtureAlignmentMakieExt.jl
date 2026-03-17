module GaussianMixtureAlignmentMakieExt

using GaussianMixtureAlignment
import GaussianMixtureAlignment: gaussiandisplay, gaussiandisplay!, gmmdisplay, gmmdisplay!, multigmmdisplay, multigmmdisplay!
using Makie
using GeometryBasics: Sphere
using Colors: RGB

import Makie: plot!
using Makie: @recipe, lines!, mesh!, Theme

const HALFWAY_RADIUS = sqrt(3) / 2
const EQUAL_VOL_CONST = 3*√π/4

const DEFAULT_COLORS = [            # CUD colors: https://jfly.uni-koeln.de/color/#assign
    RGB(0/255,   114/255, 178/255), # blue
    RGB(230/255, 159/255, 0/255  ), # orange
    RGB(0/255,   158/255, 115/255), # green
    RGB(204/255, 121/255, 167/255), # reddish purple
    RGB(86/255,  180/255, 233/255), # sky blue
    RGB(213/255, 94/255,  0/255  ), # vermillion
    RGB(240/255, 228/255, 66/255 ), # yellow
]

# use cached calculations for positions of points on a circle
θs = range(0, 2π, length=32)        # use 32 points (arbitrary)
const cosθs = [cos(θ) for θ in θs]
const sinθs = [sin(θ) for θ in θs]

equal_volume_radius(σ, ϕ) = (EQUAL_VOL_CONST*abs(ϕ))^(1/3) * σ

function flat_circle!(f, pos, r, dim::Int; kwargs...)
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
    lines!(f, xs,ys,zs; kwargs...)
end

function wire_sphere!(f, pos, r; kwargs...)
    for dim in 1:3
        flat_circle!(f, pos, r, dim; kwargs...)
        halfwaypos = Float32[0,0,0]
        halfwaypos[dim] = r / 2;
        flat_circle!(f, pos .- halfwaypos, r * HALFWAY_RADIUS, dim; kwargs...)
        flat_circle!(f, pos .+ halfwaypos, r * HALFWAY_RADIUS, dim; kwargs...)
    end
end

function solid_sphere!(f, pos, r; kwargs...)
    mesh!(f, Sphere(GeometryBasics.Point{3}(pos...), r); kwargs...)
end

@recipe(GaussianDisplay, g) do scene
    Theme(
        display = :wire,
        color = DEFAULT_COLORS[1],
        label = "",
    )
end

function plot!(gd::GaussianDisplay{<:Tuple{<:GaussianMixtureAlignment.AbstractIsotropicGaussian}})
    g = gd[:g][]
    disp = gd[:display][]
    color = gd[:color][]
    label = gd[:label][]
    plotfun = disp == :wire ? wire_sphere! : ( disp == :solid ? solid_sphere! : throw(ArgumentError("Unrecognized display option: `$disp`")))
    plotfun(gd, g.μ, g.σ; color=color, label)
    return gd
end

@recipe(GMMDisplay, g) do scene
    Theme(
        display = :wire,
        palette = DEFAULT_COLORS,
        color = nothing,
        label = "",
    )
end

function plot!(gd::GMMDisplay{<:Tuple{<:GaussianMixtureAlignment.AbstractIsotropicGMM}})
    gmm = gd[:g][]
    disp = gd[:display][]
    color = gd[:color][]
    palette = gd[:palette][]
    label = gd[:label][]
    for (i, gauss) in enumerate(gmm)
        col = isnothing(color) ? palette[(i-1) % length(palette) + 1] : color
        gaussiandisplay!(gd, gauss; display=disp, color=col, label)
    end
    return gd
end

function plot!(gd::GMMDisplay{<:Tuple{<:GaussianMixtureAlignment.AbstractIsotropicMultiGMM{N,T,K}}}) where {N,T,K}
    mgmm = gd[:g][]
    disp = gd[:display][]
    color = gd[:color][]
    palette = gd[:palette][]
    allkeys = collect(keys(mgmm))
    len = length(allkeys)
    for (i, k) in enumerate(allkeys)
        col = isnothing(color) ? palette[(i-1) % len + 1] : color
        haskey(mgmm, k) && gmmdisplay!(gd, mgmm[k]; display=disp, color=col, palette=palette, label=string(k))
    end
    return gd
end

# Needed to get legends working, see https://github.com/MakieOrg/Makie.jl/issues/1148
Makie.get_plots(p::GMMDisplay) = p.plots

@recipe(MultiGMMDisplay, g) do scene
    Theme(
        display = :wire,
        palette = DEFAULT_COLORS,
        color = nothing,
        label = "",
    )
end

function plot!(gd::MultiGMMDisplay{<:Tuple{<:GaussianMixtureAlignment.AbstractIsotropicMultiGMM{N,T,K}}}) where {N,T,K}
    mgmm = gd[:g][]
    disp = gd[:display][]
    color = gd[:color][]
    palette = gd[:palette][]
    allkeys = collect(keys(mgmm))
    len = length(allkeys)
    for (i, k) in enumerate(allkeys)
        col = isnothing(color) ? palette[(i-1) % len + 1] : color
        haskey(mgmm, k) && gmmdisplay!(gd, mgmm[k]; display=disp, color=col, palette=palette, label=string(k))
    end
    return gd
end

end
