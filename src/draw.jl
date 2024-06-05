import MakieCore: plot!
using MakieCore: @recipe, lines!, mesh!, Theme
using GeometryBasics: Sphere
using Colors: RGB

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
    )
end

function plot!(gd::GaussianDisplay{<:NTuple{<:Any, <:AbstractIsotropicGaussian}})
    gauss = [gd[i][] for i=1:length(gd)]
    disp = gd[:display][]
    color = gd[:color][]
    label = gd[:label][]
    plotfun = disp == :wire ? wire_sphere! : ( disp == :solid ? solid_sphere! : throw(ArgumentError("Unrecognized display option: `$disp`")))
    for g in gauss
        plotfun(gd, g.μ, g.σ; color=color, label)
    end
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

function plot!(gd::GMMDisplay{<:NTuple{<:Any,<:AbstractIsotropicGMM}})
    gmms = [gd[i][] for i=1:length(gd)]
    len = length(gmms)
    disp = gd[:display][]
    color = gd[:color][]
    palette = gd[:palette][]
    label = gd[:label][]
    for (i,gmm) in enumerate(gmms)
        col = isnothing(color) ? palette[(i-1) % len + 1] : color
        gaussiandisplay!(gd, gmm...; display=disp, color=col, label)
    end
    return gd
end

function plot!(gd::GMMDisplay{<:NTuple{<:Any,<:AbstractIsotropicMultiGMM{N,T,K}}}) where {N,T,K}
    mgmms = [gd[i][] for i=1:length(gd)]
    disp = gd[:display][]
    color = gd[:color][]
    palette = gd[:palette][]
    allkeys = Set{K}()
    for mgmm in mgmms
        allkeys = allkeys ∪ keys(mgmm)
    end
    len = length(allkeys)
    for (i,k) in enumerate(allkeys)
        col = isnothing(color) ? palette[(i-1) % len + 1] : color
        for mgmm in mgmms
            haskey(mgmm, k) && gmmdisplay!(gd, mgmm[k]; display=disp, color=col, palette=palette, label=string(k))
        end
    end
    return gd
end
