```@meta
CurrentModule = GaussianMixtureAlignment
```

# GaussianMixtureAlignment.jl

GaussianMixtureAlignment.jl finds globally-optimal rigid alignments between point sets by
representing them as [Gaussian mixture models (GMMs)](https://en.wikipedia.org/wiki/Mixture_model)
and using a branch-and-bound search to maximize their spatial overlap.

## Background

A GMM represents a point set as a weighted sum of Gaussian distributions, one per point.
The **overlap** between two GMMs is a smooth, orientation-sensitive measure of their similarity:
it is high when corresponding Gaussians are close, and falls off as the models separate. The
alignment problem is to find the rigid transformation (rotation + translation) that maximizes
overlap, which is equivalent to minimizing a sum of squared distances between the two point
sets.

GaussianMixtureAlignment.jl implements the
[GOGMA algorithm (Campbell, 2016)](https://arxiv.org/abs/1603.00150), with modifications
inspired by [Li et al. (2018)](https://arxiv.org/abs/1812.11307). The algorithm uses a
**branch-and-bound** procedure that is guaranteed to return the globally-optimal alignment.
Two search strategies are available:

- **`gogma_align`** — searches rotation and translation jointly in SE(3).
  Runtime is O(n²) in the number of Gaussians.
- **`tiv_gogma_align`** — decomposes the search by first solving for rotation using
  translation-invariant vectors (TIVs), then solving for translation.
  Runtime is O(n⁴) but often faster in practice for small point sets.

When multiple feature types (e.g., different chemical properties) are involved,
`IsotropicMultiGMM` holds one `IsotropicGMM` per feature label and a user-supplied
interaction matrix controls which feature pairs attract or repel.

For aligning raw point sets (no Gaussian smoothing), see `goicp_align` and `goih_align`.

## Quick start

Build a pair of GMMs and compute their overlap:

```jldoctest
julia> using GaussianMixtureAlignment

julia> xpts = [[0., 0., 0.], [3., 0., 0.], [0., 4., 0.]];

julia> ypts = [[1., 1., 1.], [1., -2., 1.], [1., 1., -3.]];

julia> gmmx = IsotropicGMM([IsotropicGaussian(p, 1.0, 1.0) for p in xpts])
IsotropicGMM{3, Float64} with 3 IsotropicGaussian{3, Float64} distributions.

julia> gmmy = IsotropicGMM([IsotropicGaussian(p, 1.0, 1.0) for p in ypts])
IsotropicGMM{3, Float64} with 3 IsotropicGaussian{3, Float64} distributions.

julia> overlap(gmmx, gmmy)
1.1908057504684806
```

Align the GMMs with GOGMA and apply the result:

```julia
julia> using ForwardDiff  # required for gradient-based local refinement

julia> res = gogma_align(gmmx, gmmy; maxsplits=10_000);

julia> # overlap after alignment — equals -res.upperbound
julia> overlap(res.tform(gmmx), gmmy)
```

The branch-and-bound search returns an `AlignmentResults` with:
- `res.tform` — the best rigid transformation found (a `CoordinateTransformations.AffineMap`)
- `res.upperbound` / `res.lowerbound` — primal and dual bounds (the true optimum lies between them)

Tighten the bounds by increasing `maxsplits` or switching to `atol`/`rtol` stopping criteria.

## See also

- [API Reference](@ref) — complete listing of all exported types and functions
- [GOGMA paper](https://arxiv.org/abs/1603.00150) — Campbell & Liu (2016)
