const HALFWAY_RADIUS = sqrt(3) / 2
const EQUAL_VOL_CONST = 3*√π/4

θs = range(0, 2π, length=32)
const cosθs = [cos(θ) for θ in θs]
const sinθs = [sin(θ) for θ in θs]

equal_volume_radius(σ, ϕ) = (EQUAL_VOL_CONST*ϕ)^(1/3) * σ

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
    Makie.lines!(xs,ys,zs; kwargs...)
end

function wire_sphere!(pos, r; kwargs...)
    for dim in 1:3
        flat_circle!(pos, r, dim; kwargs...)
        halfwaypos = Float32[0,0,0]
        halfwaypos[dim] = r / 2;
        flat_circle!(pos .- halfwaypos, r * HALFWAY_RADIUS, dim; kwargs...)
        flat_circle!(pos .+ halfwaypos, r * HALFWAY_RADIUS, dim; kwargs...)
    end
end

function solid_sphere!(pos, r; kwargs...)
    Makie.mesh!(Makie.Sphere(Makie.Point{3}(pos...), r); kwargs...)
end

function plot!(m::AbstractIsotropicGMM; solid = false, color=Makie.wong_colors()[1], kwargs...)
    plotfun = solid ? solid_sphere! : wire_sphere!
    for g in m
        plotfun(g.μ, equal_volume_radius(g.σ, g.ϕ); color=color, kwargs...)
    end
end

function plot!(m::AbstractIsotropicMultiGMM; colors=Makie.wong_colors(), kwargs...)
    for (i,(k,gmm)) in enumerate(m.gmms)
        idx = (i-1) % length(colors) + 1
        plot!(gmm; color=colors[idx], kwargs...)
    end
end

function plot!(ms::AbstractVector{<:AbstractIsotropicGMM}; colors=Makie.wong_colors(), kwargs...)
    for (i,m) in enumerate(ms)
        idx = (i-1) % length(colors) + 1
        plot!(m; color=colors[idx], kwargs...)
    end
end

function plot!(ms::AbstractVector{<:AbstractIsotropicMultiGMM{N,T,K}}; colors=Makie.wong_colors(), kwargs...) where {N,T,K}
    colormap = Dict{K,Int}()
    cidx = 1
    for m in ms
        for (k,gmm) in m.gmms
            if !haskey(colormap, k)
                push!(colormap, k => cidx)
                cidx = cidx == length(colors) ? 1 : cidx + 1
            end
        end
    end
    for (i,m) in enumerate(ms)
        mcolors = [colors[colormap[k]] for (k,gmm) in m.gmms]
        plot!(m; colors=mcolors, kwargs...)
    end
end

plot!(args...; kwargs...) = plot!([args...]; kwargs...)

function plot(args...; kwargs...)
    f = Makie.Figure()
    ax = Makie.Axis3(f[1,1]; aspect=:data)
    plot!(args...; kwargs...)
    return f
end
