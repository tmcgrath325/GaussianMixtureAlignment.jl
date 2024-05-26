# Smoke tests for drawing functions

using Makie   # load GLMakie or CairoMakie externally
using GaussianMixtureAlignment
using Test

@testset "drawing" begin
    tetrahedral = [
        [0.,0.,1.],
        [sqrt(8/9), 0., -1/3],
        [-sqrt(2/9),sqrt(2/3),-1/3],
        [-sqrt(2/9),-sqrt(2/3),-1/3]
    ]
    gmm = IsotropicGMM([IsotropicGaussian(x, 1.2, 1) for x in tetrahedral])
    gmmdisplay(gmm)

    ch_g = IsotropicGaussian(tetrahedral[1], 1.0, 1.0)
    s_gs = [IsotropicGaussian(x, 0.5, 1.0) for (i,x) in enumerate(tetrahedral)]
    mgmmx = IsotropicMultiGMM(Dict(
        :positive => IsotropicGMM([ch_g]),
        :steric => IsotropicGMM(s_gs)
    ))
    gmmdisplay(mgmmx)
end
