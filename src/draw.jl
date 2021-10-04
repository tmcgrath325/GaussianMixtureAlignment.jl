const default_colors =     
   ["#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf"]  # blue-teal

function plotdrawing(traces::AbstractVector{AbstractTrace})
    layout = Layout(autosize=false, width=800, height=600,
                    margin=attr(l=0, r=0, b=0, t=65),
                    scene=attr(),
                        # aspectmode="cube",)
                        # xaxis=attr(visible=false,), # range=[-size/2, size/2]),
                        # yaxis=attr(visible=false,), # range=[-size/2, size/2]),
                        # zaxis=attr(visible=false,), # range=[-size/2, size/2]))
                    )
    plt = plot(traces, layout)
    return plt
end

function plotdrawing(traces::AbstractVector{<:AbstractVector{<:AbstractTrace}}; size=100)
    tracesvec = AbstractTrace[]
    for trs in traces
        push!(tracesvec, trs...)
    end
    layout = Layout(autosize=false, width=800, height=600,
                    margin=attr(l=0, r=0, b=0, t=65),
                    scene=attr(
                        aspectmode="cube",
                        xaxis=attr(visible=false, range=[-size/2, size/2]),
                        yaxis=attr(visible=false, range=[-size/2, size/2]),
                        zaxis=attr(visible=false, range=[-size/2, size/2]))
                    )
    plt = plot(tracesvec, layout)
    return plt
end

function draw_wireframe_sphere(pos, r, w; npts=11, color=default_colors[1], name=nothing, opacity=1., sizecoef=1.)
    ϕs = range(0, stop=2π, length=npts)
    θs = range(-π/2, stop=π/2, length=npts)
    # vertical
    v_xs = [[r * sizecoef * cos(θ) * sin(ϕ) + pos[1] for θ in θs] for ϕ in ϕs[1:end-1]]
    v_ys = [[r * sizecoef * cos(θ) * cos(ϕ) + pos[2] for θ in θs] for ϕ in ϕs[1:end-1]]
    v_zs = [[r * sizecoef * sin(θ) + pos[3] for θ in θs] for ϕ in ϕs[1:end-1]]
    # horizontal
    h_xs = [[r * sizecoef * cos(θ) * sin(ϕ) + pos[1] for ϕ in ϕs] for θ in θs]
    h_ys = [[r * sizecoef * cos(θ) * cos(ϕ) + pos[2] for ϕ in ϕs] for θ in θs]
    h_zs = [[r * sizecoef * sin(θ) + pos[3] for ϕ in ϕs] for θ in θs] 

    hover = isnothing(name) ? "skip" : nothing

    xs, ys, zs = Float64[], Float64[], Float64[]
    for i=1:length(v_xs)
        append!(xs, v_xs[i])
        push!(xs, NaN)
        append!(ys, v_ys[i])
        push!(ys, NaN)
        append!(zs, v_zs[i])
        push!(zs, NaN)
    end
    for j=1:length(h_xs)
        append!(xs, h_xs[j])
        push!(xs, NaN)
        append!(ys, h_ys[j])
        push!(ys, NaN)
        append!(zs, h_zs[j])
        push!(zs, NaN)
    end

    return scatter3d(;x=xs, y=ys, z=zs, 
                      mode="lines",
                      line=attr(color=color),
                      opacity=opacity,
                      showlegend=false, showscale=false, name=name,
                      hovertemplate="μ = " * string(pos) * "<br>σ = " * string(r) * "<br>ϕ = " *string(w), 
                      hoverinfo=hover,
            )
end

function drawGaussian(gauss::AbstractIsotropicGaussian; sizecoef=1., opacity=1., color=default_colors[1], kwargs...)
    # gaussian centered at μ
    pos = gauss.μ
    r = gauss.σ * sizecoef
    gtrace = draw_wireframe_sphere(pos, gauss.σ, gauss.ϕ; color=color, kwargs...)
    if length(gauss.dirs) < 1
        return (gtrace,)
    end
    # cones to represent geometric constraints
    cntrs = [pos + 1.25*r*dir/norm(dir) for dir in gauss.dirs]
    dtrace = cone(;x=[c[1] for c in cntrs], y=[c[2] for c in cntrs], z=[c[3] for c in cntrs], 
                   u=[d[1] for d in gauss.dirs], v=[d[2] for d in gauss.dirs], w=[d[3] for d in gauss.dirs],
                   colorscale=[[0,color],[1,color]], opacity=opacity,
                   sizemode="absolute", sizeref=0.25,
                   showlegend=false, showscale=false, name="", hoverinfo="skip")
    return (gtrace, dtrace)
end

function drawIsotropicGMM(gmm::AbstractIsotropicGMM; kwargs...)
    # set opacities with weight values
    # weights = [gauss.ϕ for gauss in gmm.gaussians]
    # opacities = weights/maximum(weights) * 0.25

    # add a trace for each gaussian
    traces = AbstractTrace[]
    for i=1:length(gmm)
        push!(traces, drawGaussian(gmm.gaussians[i]; kwargs...)...)
    end
    return traces
end

function drawIsotropicGMMs(gmms::AbstractVector{<:AbstractIsotropicGMM};
                           colors=default_colors, kwargs...)
    traces = AbstractTrace[]
    for (i,gmm) in enumerate(gmms)
        # add traces for each GMM
        color = colors[mod(i-1, length(colors))+1]
        push!(traces, drawIsotropicGMM(gmm; color=color, kwargs...)...)
    end
    return traces
end

function drawMultiGMM(mgmm::AbstractMultiGMM; colordict=Dict{Symbol, String}(), colors=default_colors, kwargs...)
    # add traces from each GMM
    i = 1
    traces = AbstractTrace[]
    for key in keys(mgmm.gmms)
        # assign a color if the Dict doesn't include the key for a feature
        if key ∉ keys(colordict)
            push!(colordict, Pair(key, colors[mod(i-1, length(colors))+1]))
            i += i
        end
        push!(traces, drawIsotropicGMM(mgmm.gmms[key]; color=colordict[key], kwargs...)...)
    end
    return traces
end

function drawMultiGMMs(mgmms::AbstractVector{<:AbstractMultiGMM};
                       colordict=Dict{Symbol, String}(), colors=default_colors, kwargs...)
    # get all keys across the feature GMMs
    allkeys = Set{Symbol}()
    for fgmm in mgmms
        allkeys = allkeys ∪ keys(fgmm.gmms)
    end

    # assign a color to each feature type 
    i = 1
    for key in allkeys
        if key ∉ keys(colordict)
            push!(colordict, Pair(key, colors[mod(i-1, length(colors))+1]))
            i += i
        end
    end

    # add traces from each MultiGMM
    traces = AbstractTrace[]
    for (i,mgmm) in enumerate(mgmms)
        push!(traces, drawMultiGMM(mgmm; colordict, colors=colors, kwargs...)...)
    end
    return traces
end