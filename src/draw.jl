using PlotlyJS

export plotdrawing, drawGaussian, drawIsotropicGMM, drawIsotropicGMMs, drawMultiGMM, drawMultiGMMs

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
        append!(tracesvec, trs)
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

function wireframe_sphere(pos, r, npts=11)
    ϕs = range(0, stop=2π, length=npts)
    θs = range(-π/2, stop=π/2, length=npts)
    # vertical
    v_xs = [[r * cos(θ) * sin(ϕ) + pos[1] for θ in θs] for ϕ in ϕs[1:end-1]]
    v_ys = [[r * cos(θ) * cos(ϕ) + pos[2] for θ in θs] for ϕ in ϕs[1:end-1]]
    v_zs = [[r * sin(θ) + pos[3] for θ in θs] for ϕ in ϕs[1:end-1]]
    # horizontal
    h_xs = [[r * cos(θ) * sin(ϕ) + pos[1] for ϕ in ϕs] for θ in θs]
    h_ys = [[r * cos(θ) * cos(ϕ) + pos[2] for ϕ in ϕs] for θ in θs]
    h_zs = [[r * sin(θ) + pos[3] for ϕ in ϕs] for θ in θs] 

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

    return xs, ys, zs
end

function draw_wireframe_spheres(positions, rs, ws; npts=11, color=default_colors[1], name=nothing, opacity=1.)
    xs, ys, zs = Float64[], Float64[], Float64[]
    for i = 1:length(rs)
        x,y,z = wireframe_sphere(positions[i], rs[i], npts)
        append!(xs,x)
        append!(ys,y)
        append!(zs,z)
    end

    len = Int(length(xs)/length(rs))
    cdata = [[positions[Int(floor((i-1)/len + 1))][1],
              positions[Int(floor((i-1)/len + 1))][2],
              positions[Int(floor((i-1)/len + 1))][3],
              rs[Int(floor((i-1)/len + 1))],
              ws[Int(floor((i-1)/len + 1))]] 
            for i=1:length(xs)]
    
    hovertemp = join(["μ = [%{customdata[0]:.3e}, %{customdata[1]:.3e}, %{customdata[2]:.3e}]<br>",
                      "σ = %{customdata[3]:.3e}<br>",
                      "ϕ = %{customdata[4]:.3e}<br>",])

    
    return scatter3d(;x=xs, y=ys, z=zs, 
                      customdata=cdata,
                      mode="lines",
                      line=attr(color=color),
                      opacity=opacity,
                      showlegend=!isnothing(name), name=isnothing(name) ? "" : name,
                      hovertemplate=hovertemp, 
                      hoverinfo=isnothing(name) ? "skip" : nothing,
            )
end

function arrows(gauss::AbstractIsotropicGaussian, sizecoef=1)
    pos = gauss.μ
    r = gauss.σ * sizecoef
    cntrs = [pos + 1.25*r*dir/norm(dir) for dir in gauss.dirs]

    x=[c[1] for c in cntrs]
    y=[c[2] for c in cntrs]
    z=[c[3] for c in cntrs]

    u=[d[1] for d in gauss.dirs]
    v=[d[2] for d in gauss.dirs]
    w=[d[3] for d in gauss.dirs]

    return x,y,z,u,v,w
end

function draw_arrows(gaussians::AbstractVector{<:AbstractIsotropicGaussian}; sizecoef=1., opacity=1., color=default_colors[1], kwargs...)
    xs, ys, zs, us, vs, ws = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    for gauss in gaussians
        x,y,z,u,v,w = arrows(gauss, sizecoef)
        append!(xs,x)
        append!(ys,y)
        append!(zs,z)
        append!(us,u)
        append!(vs,v)
        append!(ws,w)
    end
    return cone(;x=xs, y=ys, z=zs, u=us, v=vs, w=ws,
                 colorscale=[[0,color],[1,color]], opacity=opacity,
                 sizemode="absolute", sizeref=0.25,
                 showlegend=false, showscale=false, name="", hoverinfo="skip")
end

function drawGaussians(gaussians::AbstractVector{<:AbstractIsotropicGaussian}; sizecoef=1., kwargs...)
    # spheres to represent Gaussian distributions
    positions = [gauss.μ for gauss in gaussians]
    rs = [gauss.σ * sizecoef for gauss in gaussians]
    ws = [gauss.ϕ for gauss in gaussians]
    gtrace = draw_wireframe_spheres(positions, rs, ws; kwargs...)

    # cones to represent geometric constraints
    dtrace = draw_arrows(gaussians; sizecoef=sizecoef, kwargs...)
    
    return [gtrace, dtrace]
end

function drawIsotropicGMM(gmm::AbstractIsotropicGMM; kwargs...)
    return drawGaussians(gmm.gaussians; kwargs...)
end

function drawIsotropicGMMs(gmms::AbstractVector{<:AbstractIsotropicGMM};
                           colors=default_colors, kwargs...)
    traces = AbstractTrace[]
    for (i,gmm) in enumerate(gmms)
        # add traces for each GMM
        color = colors[mod(i-1, length(colors))+1]
        append!(traces, drawIsotropicGMM(gmm; color=color, kwargs...))
    end
    return traces
end

function drawMultiGMM(mgmm::AbstractMultiGMM; colordict=Dict{Symbol, String}(), colors=default_colors, names=keys(mgmm), kwargs...)
    # add traces from each GMM
    i = 1
    traces = AbstractTrace[]
    for (i,key) in enumerate(keys(mgmm))
        # assign a color if the Dict doesn't include the key for a feature
        if key ∉ keys(colordict)
            push!(colordict, Pair(key, colors[mod(i-1, length(colors))+1]))
            i += i
        end
        push!(traces, drawIsotropicGMM(mgmm[key]; color=colordict[key], name=names[i], kwargs...)...)
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
        append!(traces, drawMultiGMM(mgmm; colordict, colors=colors, kwargs...))
    end
    return traces
end