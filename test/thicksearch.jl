# Tests for the GaussianMixtureAlignmentThickExt global-search extension. Reached only when the
# (currently unregistered) weak dependencies resolve in the test environment; see the guard in
# runtests.jl.

using Random
using StaticArrays
using Rotations
using ThickGlobalOptimization
using ThickNumbers
using IntervalFastMath

const THICK_EXT = Base.get_extension(GaussianMixtureAlignment, :GaussianMixtureAlignmentThickExt)

randgmm(rng, n; σ = 1.0, ϕ = 1.0) =
    IsotropicGMM([IsotropicGaussian(SVector{3}(randn(rng, 3)...), σ, ϕ) for _ in 1:n])

@testset "thick_gogma_align matches gogma_align" begin
    # The extension bounds reuse core `gauss_l2_bounds`, so for a single pair the certified
    # optimum must equal `gogma_align`'s. `thick_gogma_align(fixed, mobile)` aligns `mobile`
    # onto `fixed`, mirroring `gogma_align(moving, fixed)`.
    for seed in 1:5
        rng = MersenneTwister(seed)
        x = randgmm(rng, 4)
        y = randgmm(rng, 4)
        gres = gogma_align(x, y; maxsplits = 1.0e3)
        tres = thick_gogma_align(y, x; maxsplits = 1000)
        @test tres isa GMA.GlobalAlignmentResult
        @test tres.upperbound ≈ gres.upperbound atol = 1.0e-8
    end
end

@testset "searchbox / UncertaintyRegion round-trip" begin
    sr = UncertaintyRegion(RotationVec(0.3, -0.2, 0.5), SVector{3}(1.0, -2.0, 3.0), 0.4, 1.5)
    box = THICK_EXT.searchbox(sr)

    # A `SearchBox` carries six intervals: three rotation-vector components then three
    # translation components, each centered on the region with the matching half-width.
    @test wid(box[1]) / 2 ≈ sr.σᵣ
    @test wid(box[4]) / 2 ≈ sr.σₜ

    sr2 = UncertaintyRegion(box)
    @test SVector(sr2.R.sx, sr2.R.sy, sr2.R.sz) ≈ SVector(sr.R.sx, sr.R.sy, sr.R.sz)
    @test sr2.T ≈ sr.T
    @test sr2.σᵣ ≈ sr.σᵣ
    @test sr2.σₜ ≈ sr.σₜ

    # A vector of regions stacks into one 6N-dimensional box; `blockregion(box, i)` recovers
    # the i-th (zero-based) region.
    srs = [sr, UncertaintyRegion(RotationVec(0.1, 0.0, -0.1), SVector{3}(0.0, 1.0, 0.0), 0.2, 0.9)]
    vbox = THICK_EXT.searchbox(srs)
    for (i, s) in enumerate(srs)
        b = THICK_EXT.blockregion(vbox, i - 1)
        @test b.σᵣ ≈ s.σᵣ
        @test b.σₜ ≈ s.σₜ
        @test b.T ≈ s.T
    end
end

@testset "thick_gogma_align vector path" begin
    rng = MersenneTwister(11)
    x = randgmm(rng, 4)
    y = randgmm(rng, 4)

    # A single-element vector reproduces the scalar method exactly.
    scalar = thick_gogma_align(y, x; maxsplits = 1000)
    vec = thick_gogma_align(y, [x]; maxsplits = 1000)
    @test vec isa Vector{<:GMA.GlobalAlignmentResult}
    @test length(vec) == 1
    @test only(vec).upperbound ≈ scalar.upperbound

    # Two mobiles, each a known rigid transform of the fixed GMM, align back onto it: the
    # recovered overlap approaches the self-overlap ideal.
    x1 = RotationVec(0.5, 0.2, -0.3) * y + SVector{3}(1.0, 0.0, 2.0)
    x2 = RotationVec(-0.4, 0.1, 0.6) * y + SVector{3}(-1.0, 1.0, 0.0)
    res = thick_gogma_align(y, [x1, x2]; maxsplits = 4000)
    @test length(res) == 2
    ideal = -overlap(y, y)
    for (xi, r) in zip((x1, x2), res)
        @test r isa GMA.GlobalAlignmentResult
        aligned = r.tform(xi)
        @test -overlap(aligned, y) ≤ 0.9 * ideal      # captured most of the available overlap
    end
end

@testset "AlignmentResults accessors on thick results" begin
    rng = MersenneTwister(3)
    x = randgmm(rng, 3)
    y = randgmm(rng, 3)
    r = thick_gogma_align(y, x; maxsplits = 1000)
    @test GMA.tform(r) === r.tform
    @test GMA.upperbound(r) ≤ 0
    @test GMA.lowerbound(r) ≤ GMA.upperbound(r)
    @test GMA.num_splits(r) ≥ 0
    @test GMA.num_blocks(r) ≥ 0
    @test GMA.obj_calls(r) ≥ 0
    @test sprint(show, r) isa String
end
