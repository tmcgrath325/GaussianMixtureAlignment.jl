# GOGMA.jl

A Julia implementation of the Globally-Optimal Gaussian Mixture Alignment (GOGMA) algorithm [(Campbell, 2016)](https://arxiv.org/abs/1603.00150), with modifications inspired by
[Li et. al. (2018)](https://arxiv.org/abs/1812.11307). 

The GOGMA algorithm uses a branch-and-bound procedure to return a globally optimal alignment of point sets via rigid transformation. In order to improve speed for small point sets, the alignment problem can be split to separately optimize rotational and translational alignments, while still guaranteeing global optimality, through the using of translation invariant vectors (TIVs).

Becaues the runtime of the GOGMA algorithm is O(n^2), and that of the TIV-GOGMA algorithm is O(n^4), they may be unsuitable for use with large point sets without downsampling. 

## Construct Isotropic Gaussian Mixture Models (GMMs) for alignment

```julia
julia> # These are very simple point sets that can be perfectly aligned

julia> xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]];

julia> ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]];

julia> σ = ϕ = 1.;

julia> gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
IsotropicGMM with mean 3 Gaussian distributions.


julia> gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])
IsotropicGMM with mean 3 Gaussian distributions.
```

## Align Isotropic GMMs with TIV-GOGMA

```julia
julia> ub, lb, tform, niters = tiv_gogma_align(gmmx, gmmy);

julia> # upper and lower bounds of alignmnet objective function at search termination

julia> @show ub, lb;
(ub, lb) = (-2.3474784988273276, -2.447055987303346)

julia> # rotation component of the best transformation

julia> rot = GOGMA.rotmat(tform[1:3]...)
3×3 SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
 -1.63136e-10  -1.90925e-10  1.0
  1.0          -2.24997e-10  1.63136e-10
  2.24997e-10   1.0          1.90925e-10

julia> # translation component of the best transformation

julia> trl = SVector(tform[4:6])
3-element SVector{3, Float64} with indices SOneTo(3):
  1.0000000009364591
 -1.3376624620107662
 -2.9326814654410134

julia> # repeat alignment with stricter tolerance

julia> ub, lb, tform, niters = tiv_gogma_align(gmmx, gmmy; atol=0.001, rtol=0.);

julia> @show ub, lb;
(ub, lb) = (-2.3474784988273276, -2.3484784945005486)
```
