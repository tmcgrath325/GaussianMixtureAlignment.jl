"""
    result = gogma_align(gmmx, gmmy; interactions=nothing, autodiff=AutoForwardDiff(), kwargs...)

Find the globally optimal rigid transformation (rotation + translation) aligning GMM `gmmx`
to GMM `gmmy` using the [GOGMA algorithm](https://arxiv.org/abs/1603.00150).

Returns a `GlobalAlignmentResult`. Additional keyword arguments are forwarded to `branchbound`;
see `?branchbound` for the full list (tolerances, iteration limits, etc.).

# Keyword arguments
- `interactions`: optional interaction-coefficient matrix between labeled GMM components
  (only used when both inputs are `AbstractMultiGMM`)
- `autodiff`: autodiff backend for local optimization (default: `AutoForwardDiff()`)
"""
function gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM; interactions=nothing, autodiff=AutoForwardDiff(), kwargs...)
    pσ, pϕ = pairwise_consts(gmmx,gmmy,interactions)
    boundsfun(x,y,block) = gauss_l2_bounds(x,y,block,pσ,pϕ)
    localfun(x,y,block) = local_align(x,y,block,pσ,pϕ; autodiff)
    return branchbound(gmmx, gmmy; boundsfun=boundsfun, localfun=localfun, kwargs...)
end
"""
    result = rot_gogma_align(gmmx, gmmy; kwargs...)

Find the globally optimal *rotation* aligning GMM `gmmx` to GMM `gmmy` with translation
fixed to zero, using the GOGMA algorithm. Returns a `GlobalAlignmentResult`.

Accepts the same keyword arguments as `gogma_align`.
"""
function rot_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM; interactions=nothing, autodiff=AutoForwardDiff(), kwargs...)
    pσ, pϕ = pairwise_consts(gmmx,gmmy,interactions)
    boundsfun(x,y,block) = gauss_l2_bounds(x,y,block,pσ,pϕ)
    localfun(x,y,block) = local_align(x,y,block,pσ,pϕ; autodiff)
    rot_branchbound(gmmx, gmmy; boundsfun=boundsfun, localfun=localfun, kwargs...)
end
"""
    result = trl_gogma_align(gmmx, gmmy; kwargs...)

Find the globally optimal *translation* aligning GMM `gmmx` to GMM `gmmy` with rotation
fixed to the identity, using the GOGMA algorithm. Returns a `GlobalAlignmentResult`.

Accepts the same keyword arguments as `gogma_align`.
"""
function trl_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM; interactions=nothing, autodiff=AutoForwardDiff(), kwargs...)
    pσ, pϕ = pairwise_consts(gmmx,gmmy,interactions)
    boundsfun(x,y,block) = gauss_l2_bounds(x,y,block,pσ,pϕ)
    localfun(x,y,block) = local_align(x,y,block,pσ,pϕ; autodiff)
    trl_branchbound(gmmx, gmmy; boundsfun=boundsfun, localfun=localfun, kwargs...)
end
"""
    result = tiv_gogma_align(gmmx, gmmy; cutoff_x=Inf, cutoff_y=Inf, autodiff=AutoForwardDiff(), kwargs...)

Find the globally optimal rigid transformation aligning GMM `gmmx` to GMM `gmmy` using the
Translation-Invariant Vector (TIV) decomposition: first a rotation-only search on TIV GMMs,
then a translation-only search on the original GMMs.

`cutoff_x` and `cutoff_y` are radius cutoffs used when constructing the TIV representations of
`gmmx` and `gmmy` (default `Inf`, i.e. all pairwise differences). Smaller values reduce the TIV
GMM size at the cost of some accuracy in the rotation phase.

Returns a `TIVAlignmentResult` whose `.rotation_result` and `.translation_result` fields
hold the individual `GlobalAlignmentResult`s. Accepts the same additional keyword arguments
as `gogma_align`.
"""
function tiv_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM; cutoff_x=Inf, cutoff_y=Inf, autodiff=AutoForwardDiff(), kwargs...)
    tivgmmx, tivgmmy = tivgmm(gmmx, cutoff_x), tivgmm(gmmy, cutoff_y)
    pσ, pϕ = pairwise_consts(gmmx,gmmy)
    tivpσ, tivpϕ = pairwise_consts(tivgmmx,tivgmmy)
    boundsfun(x,y,block) = gauss_l2_bounds(x,y,block,pσ,pϕ)
    rot_boundsfun(x,y,block) = gauss_l2_bounds(x,y,block,tivpσ,tivpϕ)
    localfun(x,y,block) = local_align(x,y,block,pσ,pϕ; autodiff)
    rot_localfun(x,y,block) = local_align(x,y,block,tivpσ,tivpϕ; autodiff)
    tiv_branchbound(gmmx, gmmy, tivgmmx, tivgmmy; boundsfun=boundsfun, rot_boundsfun=rot_boundsfun, localfun=localfun, rot_localfun=rot_localfun, kwargs...)
end

function tiv_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM, cutoff_x, cutoff_y=Inf; kwargs...)
    Base.depwarn("passing TIV radius cutoffs positionally to `tiv_gogma_align` is deprecated; pass them as the `cutoff_x` and `cutoff_y` keyword arguments instead.", :tiv_gogma_align)
    return tiv_gogma_align(gmmx, gmmy; cutoff_x, cutoff_y, kwargs...)
end