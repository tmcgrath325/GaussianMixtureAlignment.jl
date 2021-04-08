const colors =     
   ["#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf"]   # blue-teal


function draw3d(gmms, tformvecs=fill(zeros(6),length(gmms)); sizecoef=1)
    traces = AbstractTrace[]
    for i=1:length(gmms)
        color = colors[mod(i-1, length(colors))+1]
        gmm = gmms[i]
        rx, ry, rz, tx, ty, tz = tformvecs[i]
        R = rotmat(rx, ry, rz)
        T = @SVector [tx, ty, tz]
        weights = [gauss.ϕ for gauss in gmm.gaussians]
        opacities = weights/maximum(weights) * 0.5
        for i=1:length(gmm)
            push!(traces, drawgaussian(gmm.gaussians[i], R, T, opacities[i], color; sizecoef=sizecoef))
        end
    end
    layout = Layout(autosize=false, width=800, height=600,
                    scene_aspectmode="data",
                    margin=attr(l=0, r=0, b=0, t=65))
    plt = plot(traces, layout)
    return plt
end

function drawgaussian(gauss, R, T, op=0.5, color=nothing; sizecoef=1, npts=10)
    pos = R * gauss.μ + T
    r = gauss.σ * sizecoef
    ϕs = range(0, stop=2π, length=npts)
    θs = range(-π/2, stop=π/2, length=npts)
    xs = [[r * cos(θ) * sin(ϕ) + pos[1] for θ in θs, ϕ in ϕs]...]
    ys = [[r * cos(θ) * cos(ϕ) + pos[2] for θ in θs, ϕ in ϕs]...]
    zs = [[r * sin(θ) + pos[3] for θ in θs, ϕ in ϕs]...]
    return mesh3d(;x=xs, y=ys, z=zs, 
                  color=color, opacity=op, alphahull = 0, 
                  showlegend=false, name="", # hoverinfo="skip",
                  hovertemplate="μ = " * string(gauss.μ) * "<br>σ = " * string(gauss.σ) * "<br>ϕ = " *string(gauss.ϕ)) 
end