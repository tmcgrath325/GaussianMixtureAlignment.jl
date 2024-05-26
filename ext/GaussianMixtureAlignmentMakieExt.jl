module GaussianMixtureAlignmentMakieExt

using GaussianMixtureAlignment
using Makie

# Needed to get legends working, see https://github.com/MakieOrg/Makie.jl/issues/1148
Makie.get_plots(p::GaussianMixtureAlignment.GMMDisplay) = p.plots

end
