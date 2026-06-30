module GaussianMixtureAlignmentThickExt

# ThickGlobalOptimization-based global search. This extension provides `thick_gogma_align` and
# its low-level driver `globalalign`, which align one fixed GMM against one or more mobile GMMs
# simultaneously by branch-and-bound minimization of the negated L2 overlap. The search runs on
# ThickGlobalOptimization's `thicksearch`; per-node bounds reuse the core `gauss_l2_bounds` /
# `overlap` / `pairwise_consts` machinery, wrapped into ThickNumbers intervals.

using GaussianMixtureAlignment: AbstractModel, AbstractGMM, SearchRegion, RotationRegion,
    TranslationRegion, GlobalAlignmentResult, numbertype, dims, translation_limit,
    gauss_l2_bounds, build_tform
import GaussianMixtureAlignment: globalalign, thick_gogma_align, pairwise_consts, overlap,
    UncertaintyRegion

using StaticArrays: SVector
using Rotations: RotationVec
using CoordinateTransformations: AffineMap
using ThickGlobalOptimization: thicksearch, SearchBox, bisect
using ThickNumbers: lohi, mid, wid
using IntervalFastMath: Interval
import Optim
using ADTypes: AutoForwardDiff

# An already-transformed mobile is bounded around its current position; the identity rotation
# and zero translation reproduce the core `gauss_l2_bounds(x, y, R, T, σᵣ, σₜ, …)` distance.
const IDENTITY_ROTATION = RotationVec(0.0, 0.0, 0.0)
const ZERO_TRANSLATION = SVector{3}(0.0, 0.0, 0.0)

lohi_interval(lo, hi) = lohi(Interval, lo, hi)

## SearchRegion <-> SearchBox bridge

# A `SearchBox` interval per rigid-transformation coordinate: the three rotation-vector
# components (half-width `σᵣ`) followed by the three translation components (half-width `σₜ`).
function searchbox(sr::UncertaintyRegion)
    R, T, σᵣ, σₜ = sr.R, sr.T, sr.σᵣ, sr.σₜ
    return SearchBox(
        lohi(Interval, R.sx - σᵣ, R.sx + σᵣ),
        lohi(Interval, R.sy - σᵣ, R.sy + σᵣ),
        lohi(Interval, R.sz - σᵣ, R.sz + σᵣ),
        lohi(Interval, T[1] - σₜ, T[1] + σₜ),
        lohi(Interval, T[2] - σₜ, T[2] + σₜ),
        lohi(Interval, T[3] - σₜ, T[3] + σₜ),
    )
end

function searchbox(srs::AbstractVector{<:UncertaintyRegion})
    intervals = reduce(vcat, [collect(searchbox(sr).box) for sr in srs])
    return SearchBox(intervals)
end

# Reconstruct the `i`-th (zero-based) mobile's region from a `6N`-dimensional box.
function blockregion(box::SearchBox, i::Integer = 0)
    o = 6i
    return UncertaintyRegion(
        RotationVec(mid(box[o + 1]), mid(box[o + 2]), mid(box[o + 3])),
        SVector{3}(mid(box[o + 4]), mid(box[o + 5]), mid(box[o + 6])),
        wid(box[o + 1]) / 2, wid(box[o + 4]) / 2,
    )
end

UncertaintyRegion(box::SearchBox) = blockregion(box, 0)

# Split the rigid block (six coordinates) with the largest rotation-by-translation extent.
function bisect_largest_rigid(box::SearchBox{N}) where {N}
    nblocks = N ÷ 6
    maxblock = argmax(i -> wid(box[6 * (i - 1) + 1]) * wid(box[6 * (i - 1) + 4]), 1:nblocks)
    o = 6 * (maxblock - 1)
    return bisect(box, (o + 1):(o + 6))
end

## Multiple-GMM overlap and pairwise constants (over `[y, xs...]`)

function pairwise_consts(y::AbstractGMM, xs::AbstractVector{<:AbstractGMM}, interactions = nothing)
    t = promote_type(numbertype(y), numbertype.(xs)...)
    ms = [y, xs...]
    n = length(ms)
    pσ = Matrix{Matrix{t}}(undef, n, n)
    pϕ = Matrix{Matrix{t}}(undef, n, n)
    for i in 1:n, j in 1:n
        pσ[i, j], pϕ[i, j] = pairwise_consts(ms[i], ms[j], interactions)
    end
    return pσ, pϕ
end

# Total overlap when all mobiles are aligned to the fixed `y` and to one another.
function overlap(y::AbstractGMM, xs::AbstractVector{<:AbstractGMM}, pσ = nothing, pϕ = nothing, interactions = nothing)
    if isnothing(pσ) && isnothing(pϕ)
        pσ, pϕ = pairwise_consts(y, xs, interactions)
    end
    ovlp = zero(promote_type(numbertype(y), numbertype.(xs)...))
    for (i, xi) in enumerate(xs)
        ovlp += overlap(y, xi, pσ[1, i + 1], pϕ[1, i + 1])
        for j in (i + 1):length(xs)
            ovlp += overlap(xi, xs[j], pσ[i + 1, j + 1], pϕ[i + 1, j + 1])
        end
    end
    return ovlp
end

## Bounds

# Interval bound on the negated overlap for an already-transformed pair with residual rotation
# uncertainty `σᵣ` and translation uncertainty `σₜ`.
interval_bounds(x, y, σᵣ, σₜ, pσ, pϕ; lohifun = lohi_interval) =
    lohifun(gauss_l2_bounds(x, y, IDENTITY_ROTATION, ZERO_TRANSLATION, σᵣ, σₜ, pσ, pϕ)...)

function thick_l2_bounds!(tformedxs, y::AbstractGMM, xs::AbstractVector{<:AbstractGMM}, blocks::AbstractVector{<:SearchRegion}, pσ, pϕ; lohifun = lohi_interval)
    for (i, (x, b)) in enumerate(zip(xs, blocks))
        tformedxs[i] = b.R * x + b.T
    end
    bnds = lohifun(0.0, 0.0)
    for (i, xi) in enumerate(tformedxs)
        bnds += interval_bounds(xi, y, blocks[i].σᵣ, blocks[i].σₜ, pσ[i + 1, 1], pϕ[i + 1, 1]; lohifun)
        for j in (i + 1):length(xs)
            bnds += interval_bounds(xi, tformedxs[j], blocks[i].σᵣ + blocks[j].σᵣ, blocks[i].σₜ + blocks[j].σₜ, pσ[i + 1, j + 1], pϕ[i + 1, j + 1]; lohifun)
        end
    end
    return bnds
end

## Local search

# `thicksearch`'s default `localfun` drives Optim with the Optim-1 `autodiff = :forward` symbol,
# which Optim 2 rejects. This local search uses the Optim 2 / ADTypes backend, and lets
# `thick_gogma_align` control the autodiff backend the way `gogma_align` does. Returns the
# `(minimum, minimizer, f_calls)` triple `thicksearch` expects from a `localfun`.
#
# `f_calls_limit = maxevals` bounds each refinement: the `RotationVec` exponential map is singular
# at the identity, so a block centered at zero rotation yields a `NaN` gradient and an unbounded
# line search would stall there. The cap turns that into a harmless no-op while subdivided blocks
# (nonzero rotation centers) still refine.
function thick_localsearch(func, box; autodiff = AutoForwardDiff(), maxevals = 100, inner_optimizer = Optim.LBFGS(), kwargs...)
    initial_x = collect(mid(box))
    options = Optim.Options(; f_calls_limit = maxevals, iterations = maxevals, kwargs...)
    results = Optim.optimize(func, initial_x, inner_optimizer, options; autodiff)
    return Optim.minimum(results), Optim.minimizer(results), results.f_calls
end

## Result translation

# `thicksearch` reports no termination reason, so infer one of the strings
# `GaussianMixtureAlignment.converged` recognizes: an empty scheduler means the queue was
# exhausted; otherwise a closed bounds gap means the tolerance was met, and anything else is a
# limit. (An authoritative reason would come from a termination field on `ThickSearchResult`.)
function thick_terminated_by(res, atol, rtol)
    isempty(res.scheduler) && return "priority queue empty"
    gap = abs(res.globalrv - res.globallb)
    if (atol > 0 && gap < atol) || (rtol > 0 && gap / max(abs(res.globalrv), abs(res.globallb)) < rtol)
        return "optimum within tolerance"
    end
    return "terminated early"
end

# Translate a `ThickSearchResult` into a `GlobalAlignmentResult` for the mobile occupying the
# `params` coordinates of the flat minimizer. `GlobalAlignmentResult` cannot embed the
# `ThickSearchResult` — ThickGlobalOptimization is a weak dependency. `stagnant_splits` is `0`:
# `thicksearch` exposes no since-last-improvement count.
# Unexplored regions left in the scheduler. `thicksearch`'s default `ConvexHullScheduler` keeps
# them in a `.structure` hull; other schedulers expose no generic count, so fall back to 0.
remaining_blocks(scheduler) = hasproperty(scheduler, :structure) ? length(scheduler.structure) : 0

function alignment_result(res, x::AbstractModel, y::AbstractModel, params::UnitRange, terminated_by)
    p = Tuple(res.bestminimizer[params])
    progress = [(c, v, Tuple(m[params])) for (c, m, v) in res.progress]
    return GlobalAlignmentResult(
        x, y, res.globalrv, res.globallb, build_tform(AffineMap, p), p,
        res.fevals, res.boxsplits, remaining_blocks(res.scheduler), 0, progress, terminated_by,
    )
end

## Drivers

function globalalign(fixed::AbstractModel, mobiles::AbstractVector{<:AbstractModel};
        searchspace = nothing, blockfun = UncertaintyRegion, boxsplitter = bisect_largest_rigid,
        objfun, boundsfun, kwargs...
    )
    all(x -> dims(x) == dims(fixed), mobiles) || throw(ArgumentError("Dimensionality of the GMMs must be equal"))
    if isnothing(searchspace)
        tlimit = translation_limit(fixed, first(mobiles))
        searchspace = [blockfun(tlimit) for _ in mobiles]
    end
    box = searchbox(searchspace)
    return thicksearch(objfun, box; boxsplitter, boundsfun, kwargs...)
end

function thick_gogma_align(y::AbstractGMM, xs::AbstractVector{<:AbstractGMM}; interactions = nothing, lohifun = lohi_interval, autodiff = AutoForwardDiff(), atol = 0.01, rtol = 0.01, kwargs...)
    pσ, pϕ = pairwise_consts(y, xs, interactions)
    tformedxs = collect(xs)
    nmobiles = length(xs)

    boundsfun = function (box)
        blocks = [blockregion(box, i) for i in 0:(nmobiles - 1)]
        return thick_l2_bounds!(tformedxs, y, xs, blocks, pσ, pϕ; lohifun)
    end
    objfun = function (X)
        local_tformedxs = [RotationVec(X[6i + 1], X[6i + 2], X[6i + 3]) * xs[i + 1] + SVector{3}(X[6i + 4], X[6i + 5], X[6i + 6]) for i in 0:(nmobiles - 1)]
        return -overlap(y, local_tformedxs, pσ, pϕ, interactions)
    end
    localfun(func, box; kw...) = thick_localsearch(func, box; autodiff, kw...)

    res = globalalign(y, xs; boxsplitter = bisect_largest_rigid, boundsfun, objfun, localfun, atol, rtol, kwargs...)
    terminated_by = thick_terminated_by(res, atol, rtol)
    return [alignment_result(res, xs[i + 1], y, (6i + 1):(6i + 6), terminated_by) for i in 0:(nmobiles - 1)]
end

thick_gogma_align(fixed::AbstractGMM, mobile::AbstractGMM; kwargs...) =
    only(thick_gogma_align(fixed, [mobile]; kwargs...))

end
