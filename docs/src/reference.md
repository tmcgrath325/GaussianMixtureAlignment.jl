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
