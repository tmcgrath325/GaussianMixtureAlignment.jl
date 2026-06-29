module GaussianMixtureAlignmentMakieExt

using GaussianMixtureAlignment
import GaussianMixtureAlignment: gaussiandisplay, gaussiandisplay!, gmmdisplay, gmmdisplay!
using Makie
using GeometryBasics: GeometryBasics, Sphere
using Colors: RGB

import Makie: plot!
using Makie: @recipe, lines!, mesh!, Attributes

const HALFWAY_RADIUS = sqrt(3) / 2
const EQUAL_VOL_CONST = 3 * √π / 4

const DEFAULT_COLORS = [            # CUD colors: https://jfly.uni-koeln.de/color/#assign
    RGB(0 / 255, 114 / 255, 178 / 255), # blue
    RGB(230 / 255, 159 / 255, 0 / 255), # orange
    RGB(0 / 255, 158 / 255, 115 / 255), # green
    RGB(204 / 255, 121 / 255, 167 / 255), # reddish purple
    RGB(86 / 255, 180 / 255, 233 / 255), # sky blue
    RGB(213 / 255, 94 / 255, 0 / 255), # vermillion
    RGB(240 / 255, 228 / 255, 66 / 255), # yellow
]

# use cached calculations for positions of points on a circle
θs = range(0, 2π, length = 32)        # use 32 points (arbitrary)
const cosθs = [cos(θ) for θ in θs]
const sinθs = [sin(θ) for θ in θs]

equal_volume_radius(σ, ϕ) = (EQUAL_VOL_CONST * abs(ϕ))^(1 / 3) * σ

function flat_circle!(f, pos, r, dim::Int, attrs::Attributes)
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
    return lines!(f, attrs, xs, ys, zs)
end

function wire_sphere!(f, pos, r, attrs::Attributes)
    for dim in 1:3
        flat_circle!(f, pos, r, dim, attrs)
        halfwaypos = Float32[0, 0, 0]
        halfwaypos[dim] = r / 2
        flat_circle!(f, pos .- halfwaypos, r * HALFWAY_RADIUS, dim, attrs)
        flat_circle!(f, pos .+ halfwaypos, r * HALFWAY_RADIUS, dim, attrs)
    end
    return
end

function solid_sphere!(f, pos, r, attrs::Attributes)
    return mesh!(f, attrs, Sphere(GeometryBasics.Point{3}(pos...), r))
end

"""
    gaussiandisplay([fig_or_ax,] g; display=:wire, color=DEFAULT_COLORS[1], label="", alpha=1, transparency=false)

Visualize an `AbstractIsotropicGaussian` as a sphere centered at `g.μ` with radius `g.σ`.

# Arguments
- `g`: the Gaussian to display

# Keyword arguments
- `display`: `:wire` (wireframe, default) or `:solid` (filled mesh)
- `color`: color of the sphere
- `label`: legend label
- `alpha`: transparency level (0 = fully transparent, 1 = fully opaque)
- `transparency`: if true, Makie uses Order Independent Transparency (default `false`)

Any additional attributes valid for `Lines` (wire) or `Mesh` (solid) are forwarded to Makie.

See also `gaussiandisplay!`.
"""
@recipe GaussianDisplay (g,) begin
    display = :wire
    color = @inherit color DEFAULT_COLORS[1]
    label = ""
    alpha = 1.0f0
    transparency = false
end

function plot!(gd::GaussianDisplay{<:Tuple{<:GaussianMixtureAlignment.AbstractIsotropicGaussian}})
    g = gd[:g][]
    disp = gd[:display][]
    if disp == :wire
        wire_sphere!(gd, g.μ, g.σ, Makie.shared_attributes(gd, Lines))
    elseif disp == :solid
        solid_sphere!(gd, g.μ, g.σ, Makie.shared_attributes(gd, Mesh))
    else
        throw(ArgumentError("Unrecognized display option: `$disp`"))
    end
    return gd
end

"""
    gmmdisplay([fig_or_ax,] g; display=:wire, palette=DEFAULT_COLORS, color=nothing, label="", alpha=1, transparency=false)

Visualize an `AbstractIsotropicGMM` (or `AbstractIsotropicMultiGMM`) as a collection of spheres,
one per Gaussian component.

# Arguments
- `g`: the GMM to display

# Keyword arguments
- `display`: `:wire` (wireframe, default) or `:solid` (filled mesh)
- `palette`: color palette cycled across components (ignored when `color` is set)
- `color`: if set, all components are drawn in this color
- `label`: legend label
- `alpha`: transparency level (0 = fully transparent, 1 = fully opaque)
- `transparency`: if true, Makie uses Order Independent Transparency (default `false`)

See also `gmmdisplay!`.
"""
@recipe GMMDisplay (g,) begin
    display = :wire
    palette = DEFAULT_COLORS
    color = nothing
    label = ""
    alpha = 1.0f0
    transparency = false
end

function plot!(gd::GMMDisplay{<:Tuple{<:GaussianMixtureAlignment.AbstractIsotropicGMM}})
    gmm = gd[:g][]
    disp = gd[:display][]
    color = gd[:color][]
    palette = gd[:palette][]
    label = gd[:label][]
    base_attrs = Makie.shared_attributes(gd, GaussianDisplay; drop = [:color, :label, :palette])
    for (i, gauss) in enumerate(gmm)
        col = isnothing(color) ? palette[(i - 1) % length(palette) + 1] : color
        gaussiandisplay!(gd, base_attrs, gauss; color = col, label)
    end
    return gd
end

function plot!(gd::GMMDisplay{<:Tuple{<:GaussianMixtureAlignment.AbstractIsotropicMultiGMM{N, T, K}}}) where {N, T, K}
    mgmm = gd[:g][]
    color = gd[:color][]
    palette = gd[:palette][]
    base_attrs = Makie.shared_attributes(gd, GMMDisplay; drop = [:color, :label])
    allkeys = collect(keys(mgmm))
    len = length(allkeys)
    for (i, k) in enumerate(allkeys)
        col = isnothing(color) ? palette[(i - 1) % len + 1] : color
        haskey(mgmm, k) && gmmdisplay!(gd, base_attrs, mgmm[k]; color = col, label = string(k))
    end
    return gd
end

# Needed to get legends working, see https://github.com/MakieOrg/Makie.jl/issues/1148
Makie.get_plots(p::GMMDisplay) = p.plots

end
