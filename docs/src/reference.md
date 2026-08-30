```@meta
CurrentModule = GaussianMixtureAlignment
```

# API Reference

```@docs
GaussianMixtureAlignment.GaussianMixtureAlignment
```

## Types

### Abstract types

```@docs
AbstractGaussian
AbstractGMM
```

### Concrete GMM types

```@docs
IsotropicGaussian
IsotropicGMM
IsotropicMultiGMM
LabeledIsotropicGMM
StackedLabeledGaussian
StackedLabeledIsotropicGMM
stackedgmm
```

### Point set types

```@docs
PointSet
MultiPointSet
```

## GMM alignment

### Objective function

```@docs
overlap
force
force!
```

### Branch-and-bound aligners

```@docs
gogma_align
rot_gogma_align
trl_gogma_align
tiv_gogma_align
rocs_align
```

### Flexible (articulated) alignment

Align a model whose features are connected by rotatable joints — for example a molecule's
rotatable bonds — optimizing one rotation angle per joint alongside the rigid transform. The
target may be articulated as well, in which case its joint angles are optimized jointly with
the moving model's. With no joints on either model this reduces to [`gogma_align`](@ref).

A flexible model can raise its overlap by folding its own features onto one another; the
`selfoverlap` keyword adds a [`SelfOverlap`](@ref) penalty that charges that self-overlap
within the same branch-and-bound guarantee.

```@docs
flex_gogma_align
FlexibleAlignmentResult
joint_angles
target_joint_angles
aligned
aligned_target
SelfOverlap
penalty
penalty_bounds
```

An articulated model supplies its kinematic tree through the interface below.
[`ArticulatedGMM`](@ref) is the in-package implementation; external models, such as
MolecularGaussians' `PharmacophoreGMM`, provide their own methods.

```@docs
ArticulatedGMM
ArticulatedStackedGMM
Joint
njoints
joint_axis
joint_origin
joint_features
joint_children
flex
FlexibleRegion
```

## Point set alignment

```@docs
kabsch
goicp_align
goih_align
tiv_goicp_align
tiv_goih_align
```

## Visualization

These functions are provided by the `GaussianMixtureAlignmentMakieExt` extension, which is
loaded automatically when Colors, GeometryBasics, and Makie are available.

```@docs
gmmdisplay
gmmdisplay!
gaussiandisplay
gaussiandisplay!
```
