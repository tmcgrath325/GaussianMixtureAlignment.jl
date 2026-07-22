using GaussianMixtureAlignment
using Test
using IntervalSets
using LinearAlgebra
using StaticArrays
using Rotations
using CoordinateTransformations
using ForwardDiff
using FiniteDifferences
using ADTypes
using Aqua
using ExplicitImports
using OffsetArrays
using JLD2

using GaussianMixtureAlignment: UncertaintyRegion, RotationRegion, TranslationRegion
using GaussianMixtureAlignment: tight_distance_bounds, loose_distance_bounds, squared_dist_bounds, gauss_l2_bounds, subranges, sqrt3, UncertaintyRegion, subregions, branchbound, rocs_align, overlap, gogma_align, tiv_gogma_align, tiv_goih_align, overlapobj
const GMA = GaussianMixtureAlignment

@testset "Aqua" begin
    # Only run Aqua tests on CI (to avoid slowing down local development)
    get(ENV, "CI", "false") == "true" && Aqua.test_all(GaussianMixtureAlignment)
end

@testset "ExplicitImports" begin
    test_explicit_imports(
        GaussianMixtureAlignment;
        all_explicit_imports_are_public = VERSION >= v"1.11",
        all_qualified_accesses_are_public = VERSION >= v"1.11",
        ignore = (:kabsch_centered, :Options)
    )
end

@testset "public API surface" begin
    # `public` (and `Base.ispublic`) exist only on 1.11+; on the LTS the names are still
    # reachable, just unmarked.
    if VERSION >= v"1.11"
        for name in (
                :branchbound, :UncertaintyRegion, :RotationRegion, :TranslationRegion,
                :icp, :iterative_hungarian,
                :converged, :tform, :upperbound, :lowerbound, :obj_calls,
                :num_splits, :num_blocks, :stagnant_splits, :progress,
            )
            @test Base.ispublic(GaussianMixtureAlignment, name)
            @test !Base.isexported(GaussianMixtureAlignment, name)  # public, not exported
        end
    end
end

@testset "search space bounds" begin
    μx = SVector(3, 0, 0)
    μy = SVector(-4, 0, 0)
    σ = ϕ = 1
    ndims = 3
    sqrt2 = √2

    x, y = IsotropicGaussian(μx, σ, ϕ), IsotropicGaussian(μy, σ, ϕ)

    ### tight_distance_bounds
    # anti-aligned (no rotation) and aligned (180 degree rotation)
    lbdist, ubdist = tight_distance_bounds(x, y, π, 0)
    @test ubdist ≈ 7
    @test lbdist ≈ 1
    # region with closest alignment at 90 degree rotation
    lbdist, ubdist = tight_distance_bounds(x, y, π / 2 / sqrt3, 0)
    @test lbdist ≈ 5
    # translation region centered at origin
    lbdist, ubdist = tight_distance_bounds(x, y, 0, 1 / √3)
    @test lbdist ≈ 6
    @test ubdist ≈ 7
    # centered at x = 1
    lbdist, ubdist = tight_distance_bounds(x + SVector(1, 0, 0), y, 0, 1 / sqrt3)
    @test lbdist ≈ 7
    @test ubdist ≈ 8
    # nearly (anti)parallel points: floating-point rounding pushes
    # dot(x,y)/(norm(x)*norm(y)) just past ±1, where clamping cosα keeps 1-cosα²
    # nonnegative under the √ instead of throwing a DomainError. xpar/1.3xpar give
    # cosα = 1.0000000000000002; the negated pair gives -1.0000000000000002.
    xpar = SVector{3, Float64}(1, 1, 1)
    lbp, ubp = tight_distance_bounds(xpar, 1.3 * xpar, 0.1, 0.0, true)   # parallel, maximize
    @test isfinite(lbp) && isfinite(ubp)
    @test lbp ≥ ubp
    lbn, ubn = tight_distance_bounds(xpar, -1.3 * xpar, 0.1, 0.0)        # anti-parallel, minimize
    @test isfinite(lbn) && isfinite(ubn)
    @test 0 ≤ lbn ≤ ubn
    # nearly-coincident points: the outer law-of-cosines radicand
    # xnorm²+ynorm²-2·xnorm·ynorm·cos(α∓β) is a squared distance (≥ 0) that rounds
    # slightly negative when the points nearly coincide (equal norms, α ≈ β),
    # which threw a DomainError under the √ before the radicand was clamped.
    xthrow = SVector(2.7731554579023157, 0.0, 0.0)                       # threw DomainError pre-fix
    ythrow = SVector(1.9505050369704133, 1.9712740290736779, 0.0)
    lbt, ubt = tight_distance_bounds(xthrow, ythrow, 0.4565073488921946, 0.0)
    @test isfinite(lbt) && isfinite(ubt)
    @test 0 ≤ lbt ≤ ubt
    # dense sweep of near-coincident configurations (equal-ish norms, angle ≈ β):
    # every case must yield finite, nonnegative bounds rather than throwing.
    for r in (0.5, 2.0, 7.5), σᵣ in (0.1, 0.35, 0.45, 0.9), dθ in (-2.0e-9, 0.0, 3.0e-9), dr in (-1.0e-9, 0.0, 2.0e-9)
        β = sqrt3 * σᵣ
        xc = SVector(r, 0.0, 0.0)
        yc = (r + dr) * SVector(cos(β + dθ), sin(β + dθ), 0.0)
        for maxflag in (false, true)
            lb, ub = tight_distance_bounds(xc, yc, σᵣ, 0.0, maxflag)
            @test isfinite(lb) && isfinite(ub)
            @test lb ≥ 0 && ub ≥ 0
        end
    end

    ### loose_distance_bounds
    # full rotation: the nearest point on the circle, matching the tight bound
    lbdist, ubdist = loose_distance_bounds(μx, μy, π, 0)
    @test ubdist ≈ 7
    @test lbdist ≈ 1
    # pure translation: γᵣ = 0, so the loose bound equals the tight one
    lbdist, ubdist = loose_distance_bounds(μx, μy, 0, 1 / √3)
    @test lbdist ≈ 6
    @test ubdist ≈ 7
    # intermediate rotation: the loose lower bound is strictly looser than the tight one
    loose_lb, loose_ub = loose_distance_bounds(μx, μy, π / 2 / sqrt3, 0)
    tight_lb, tight_ub = tight_distance_bounds(μx, μy, π / 2 / sqrt3, 0)
    @test loose_lb < tight_lb
    @test loose_ub ≈ tight_ub ≈ 7
    # the loose bounds must enclose the tight ones across a range of uncertainties: a wider
    # nearest point when minimizing, a wider farthest point when maximizing. Both share the
    # region-center distance as their second element.
    for σᵣ in (0.0, 0.3, 0.8, 1.5), σₜ in (0.0, 0.5, 2.0)
        near_l, ctr_l = loose_distance_bounds(μx, μy, σᵣ, σₜ)
        near_t, ctr_t = tight_distance_bounds(μx, μy, σᵣ, σₜ)
        @test near_l ≤ near_t ≤ ctr_t
        @test ctr_l ≈ ctr_t
        far_l, fctr_l = loose_distance_bounds(μx, μy, σᵣ, σₜ, true)
        far_t, _ = tight_distance_bounds(μx, μy, σᵣ, σₜ, true)
        @test far_l ≥ far_t
        @test fctr_l ≈ ctr_t
    end
    # maximize: μx and μy are anti-aligned, so any rotation cap includes the anti-alignment
    # direction; the farthest distance must equal norm(μx) + norm(μy) + sqrt3*σₜ
    for σᵣ in (0.3, 0.8, 1.5), σₜ in (0.0, 0.5)
        far_t, _ = tight_distance_bounds(μx, μy, σᵣ, σₜ, true)
        @test far_t ≈ norm(μx) + norm(μy) + sqrt3 * σₜ
    end

    ### Gaussian L2 bounds
    # rotation distances, no translation
    # anti-aligned (no rotation) and aligned (180 degree rotation)
    lb, ub = gauss_l2_bounds(x, y, RotationRegion(0.0))
    @test lb ≈ -GMA.overlap(7^2, 2 * σ^2, ϕ * ϕ) atol = 1.0e-16
    @test ub ≈ -GMA.overlap(7^2, 2 * σ^2, ϕ * ϕ)
    lb, ub = gauss_l2_bounds(x, y, RotationRegion(RotationVec(0.0, 0.0, π), SVector{3}(0.0, 0.0, 0.0), 0.0))
    @test lb ≈ ub ≈ -GMA.overlap(1, 2 * σ^2, ϕ * ϕ)
    # region with closest alignment at 90 degree rotation
    lb = gauss_l2_bounds(x, y, RotationRegion(π / 2 / sqrt3))[1]
    @test lb ≈ -GMA.overlap(5^2, 2 * σ^2, ϕ * ϕ)
    lb = gauss_l2_bounds(x, y, RotationRegion(RotationVec(0, 0, π / 4), SVector{3}(0.0, 0.0, 0.0), π / 4 / (sqrt3)))[1]
    @test lb ≈ -GMA.overlap(5^2, 2 * σ^2, ϕ * ϕ)

    # translation distance, no rotation
    # translation region centered at origin
    lb, ub = gauss_l2_bounds(x, y, TranslationRegion(1 / sqrt3))
    @test lb ≈ -GMA.overlap(6^2, 2 * σ^2, ϕ * ϕ)
    @test ub ≈ -GMA.overlap(7^2, 2 * σ^2, ϕ * ϕ)
    # centered with translation of 1 in +x
    lb, ub = gauss_l2_bounds(x + SVector(1, 0, 0), y, TranslationRegion(1 / sqrt3))
    @test lb ≈ -GMA.overlap(7^2, 2 * σ^2, ϕ * ϕ)
    @test ub ≈ -GMA.overlap(8^2, 2 * σ^2, ϕ * ϕ)
    # centered with translation of 3 in +y
    lb, ub = gauss_l2_bounds(x + SVector(0, 3, 0), y, TranslationRegion(1 / sqrt3))
    @test lb ≈ -GMA.overlap((√(58) - 1)^2, 2 * σ^2, ϕ * ϕ)
    @test ub ≈ -GMA.overlap(58, 2 * σ^2, ϕ * ϕ)

end

@testset "divide a searchspace" begin
    blk1 = NTuple{6, Tuple{Float64, Float64}}(((-π, π), (-π, π), (-π, π), (-1, 1), (-1, 1), (-1, 1)))
    subblks1 = subranges(blk1, 2)
    @test length(subblks1) == 2^6
    for i in 1:6
        intv = OpenInterval(0, 0)
        for sblk in subblks1
            rng = sblk[i]
            intv = union(intv, ClosedInterval(rng[1], rng[2]))
        end
        @test intv == ClosedInterval(blk1[i][1], blk1[i][2])
    end
    # @show subblks1[length(subblks1)]

    blk2 = NTuple{6, Tuple{Float64, Float64}}(((0, π), (0, π), (0, π), (0, 1), (0, 1), (0, 1)))
    subblks2 = subranges(blk2, 4)
    @test length(subblks2) == 4^6
    for i in 1:6
        intv = OpenInterval(0, 0)
        for sblk in subblks2
            rng = sblk[i]
            intv = union(intv, ClosedInterval(rng[1], rng[2]))
        end
        @test intv == ClosedInterval(blk2[i][1], blk2[i][2])
    end
end

@testset "GMM interface" begin
    tetrahedral = [
        [0.0, 0.0, 1.0],
        [sqrt(8 / 9), 0.0, -1 / 3],
        [-sqrt(2 / 9), sqrt(2 / 3), -1 / 3],
        [-sqrt(2 / 9), -sqrt(2 / 3), -1 / 3],
    ]
    ch_g = IsotropicGaussian(tetrahedral[1], 1.0, 1.0)
    s_gs = [IsotropicGaussian(x, 0.5, 1.0) for (i, x) in enumerate(tetrahedral)]
    gmm = IsotropicGMM(s_gs)
    @test length(gmm) == 4
    @test gmm[2] == s_gs[2]
    @test collect(gmm) == s_gs       # tests iterate
    @test eltype(gmm) === eltype(typeof(gmm)) === IsotropicGaussian{3, Float64}
    @test convert(IsotropicGMM{3, Float32}, gmm) isa IsotropicGMM{3, Float32}
    @test_throws DimensionMismatch convert(IsotropicGMM{2, Float64}, gmm)
    @test !isa(GMA.coords(gmm), StaticArray)
    @test !isa(GMA.weights(gmm), StaticArray)
    @test !isa(GMA.widths(gmm), StaticArray)
    @test GMA.weights(gmm) == [g.ϕ for g in s_gs] == fill(1.0, 4)
    @test GMA.widths(gmm) == [g.σ for g in s_gs] == fill(0.5, 4)
    mgmmx = IsotropicMultiGMM(
        Dict(
            :positive => IsotropicGMM([ch_g]),
            :steric => gmm
        )
    )
    @test keys(mgmmx) == Set([:positive, :steric])
    @test keytype(mgmmx) === keytype(typeof(mgmmx)) === Symbol
    @test valtype(mgmmx) === valtype(typeof(mgmmx)) === IsotropicGMM{3, Float64}
    @test eltype(mgmmx) === eltype(typeof(mgmmx)) === Pair{Symbol, IsotropicGMM{3, Float64}}
    @test length(mgmmx) == 2
    @test length(mgmmx[:steric]) == 4
    @test mgmmx[:steric][2] == s_gs[2]
    @test collect(mgmmx) == collect(mgmmx.gmms) # tests iterate
    @test get!(valtype(mgmmx), mgmmx, :positive) == mgmmx[:positive]
    gmm = get!(valtype(mgmmx), mgmmx, :acceptor)
    @test isempty(gmm) && gmm isa IsotropicGMM{3, Float64}
    push!(gmm, ch_g)
    @test length(gmm) == 1
    pop!(gmm)
    @test isempty(gmm)
    push!(gmm, ch_g)
    empty!(gmm)
    @test isempty(gmm)
    delete!(mgmmx, :acceptor)
    @test !haskey(mgmmx, :acceptor)
    @test !isa(GMA.coords(mgmmx), StaticArray)
    @test !isa(GMA.weights(mgmmx), StaticArray)
    @test !isa(GMA.widths(mgmmx), StaticArray)
    @test sort(GMA.widths(mgmmx)) == [0.5, 0.5, 0.5, 0.5, 1.0]
    empty!(mgmmx)
    @test isempty(mgmmx)
end

@testset "multi-container show and valtype" begin
    # `show` reports the value type via `valtype`, so it stays correct if the underlying
    # container type changes (no positional `.parameters[2]` access).
    g = IsotropicGaussian([0.0, 0.0, 0.0], 1.0, 1.0)
    mgmm = IsotropicMultiGMM(Dict(:a => IsotropicGMM([g, g]), :b => IsotropicGMM([g])))
    @test sprint(show, mgmm) ==
        "IsotropicMultiGMM{3, Float64, Symbol} with 2 labeled IsotropicGMM{3, Float64} " *
        "models made up of a total of 3 IsotropicGMM{3, Float64} distributions.\n"

    ps = PointSet([0.0 1.0; 0.0 0.0; 0.0 0.0], [1.0, 1.0])
    mps = MultiPointSet(Dict(:a => ps, :b => ps))
    @test valtype(mps) === valtype(mps.pointsets) === PointSet{3, Float64}
    @test sprint(show, mps) ==
        "MultiPointSet{3, Float64, Symbol} with 2 labeled PointSet{3, Float64} sets " *
        "and a total of 4 points.\n"
end

@testset "combine MultiGMMs with differing keys" begin
    g1 = IsotropicGMM([IsotropicGaussian([0.0, 0.0, 0.0], 1.0, 1.0)])
    g2 = IsotropicGMM([IsotropicGaussian([1.0, 0.0, 0.0], 1.0, 1.0)])
    mgmmx = IsotropicMultiGMM(Dict(:a => g1))
    mgmmy = IsotropicMultiGMM(Dict(:b => g2))
    combined_ref = Ref{Any}()
    captured = mktemp() do path, io
        redirect_stdout(io) do
            combined_ref[] = GMA.combine(mgmmx, mgmmy)
        end
        flush(io)
        read(path, String)
    end
    combined = combined_ref[]
    @test isempty(captured)                    # keys present in only one input must not print
    @test keys(combined) == Set([:a, :b])
    @test combined[:a] == g1
    @test combined[:b] == g2
end

@testset "apply transformations via tform(model)" begin
    # An affine map from CoordinateTransformations applies directly to any model as
    # `tform(model)`, composing the model `*`/`+` methods. The README aligns models with
    # `res.tform(model)`; this contract breaks silently if a `*` or `+` method is removed.
    R = RotationVec(0.3, -0.2, 0.5)
    T = SVector(1.0, -2.0, 3.0)
    affine = AffineMap(R, T)
    linear = LinearMap(R)
    trl = Translation(T)

    gauss = IsotropicGaussian([1.0, 0.0, 0.0], 0.5, 1.0)
    gmm = IsotropicGMM([gauss, IsotropicGaussian([0.0, 1.0, 0.0], 0.5, 2.0)])
    mgmm = IsotropicMultiGMM(Dict(:a => gmm, :b => IsotropicGMM([gauss])))
    ps = PointSet([1.0 0.0 -1.0; 0.0 1.0 0.0; 0.0 0.0 2.0], [1.0, 2.0, 3.0])
    mps = MultiPointSet(Dict(:a => ps))

    # Applying a map as a functor equals decompose-and-apply, for each map flavor.
    @test affine(gauss).μ ≈ R * gauss.μ + T
    @test linear(gauss).μ ≈ R * gauss.μ
    @test trl(gauss).μ ≈ gauss.μ + T
    for model in (gmm, ps)
        @test GMA.coords(affine(model)) ≈ GMA.coords(R * model + T)
        @test GMA.coords(linear(model)) ≈ GMA.coords(R * model)
        @test GMA.coords(trl(model)) ≈ GMA.coords(model + T)
    end
    for model in (mgmm, mps)
        for k in keys(model)
            @test GMA.coords(affine(model)[k]) ≈ GMA.coords(R * model[k] + T)
            @test GMA.coords(linear(model)[k]) ≈ GMA.coords(R * model[k])
            @test GMA.coords(trl(model)[k]) ≈ GMA.coords(model[k] + T)
        end
    end

    # For the Gaussian-mixture types the functor is fully type-inferred.
    for model in (gauss, gmm, mgmm)
        @test (@inferred affine(model)) isa typeof(model)
        @test (@inferred linear(model)) isa typeof(model)
        @test (@inferred trl(model)) isa typeof(model)
    end
end

@testset "bounds for shrinking searchspace around an optimum" begin
    # two sets of points, each forming a 3-4-5 triangle
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    ypts = [[1.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -3.0]]
    σ = ϕ = 1.0
    gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
    gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

    # aligning a GMM to itself
    bigblock = UncertaintyRegion(gmmx, gmmx)
    (lb, ub) = gauss_l2_bounds(gmmx, gmmx, bigblock)
    @test lb ≈ -length(gmmx.gaussians)^2 # / √(4π)^3

    blk = UncertaintyRegion(RotationVec{Float64}(π / 2, π / 2, π / 2), SVector{3, Float64}(1.0, 1.0, 1.0), π / 2, 1.0)
    (lb, ub) = gauss_l2_bounds(gmmx, gmmx, blk)
    for i in 1:20
        blk = subregions(blk)[1]
        (newlb, newub) = gauss_l2_bounds(gmmx, gmmx, blk)
        @test newlb >= lb
        @test newub <= ub
        (lb, ub) = (newlb, newub)
    end
end

@testset "squared_dist_bounds for MultiPointSets" begin
    # The MultiPointSet method sums the single-pointset bounds over shared keys.
    psx_a = PointSet([0.0 3.0 0.0; 0.0 0.0 4.0; 0.0 0.0 0.0])
    psy_a = PointSet([1.0 1.0 1.0; 1.0 -2.0 1.0; 1.0 1.0 -3.0])
    psx_b = PointSet([1.0 0.0; 0.0 1.0; 0.0 0.0])
    psy_b = PointSet([0.0 1.0; 1.0 0.0; 0.0 0.0])

    # Key :c appears in only one input and must be ignored by the key intersection.
    mpsx = MultiPointSet(Dict(:a => psx_a, :b => psx_b, :c => psx_a))
    mpsy = MultiPointSet(Dict(:a => psy_a, :b => psy_b))

    σᵣ, σₜ = 1.0, 1.0
    (lb, ub) = squared_dist_bounds(mpsx, mpsy, σᵣ, σₜ)

    # Equal to the sum of the per-shared-key single-pointset bounds.
    (alb, aub) = squared_dist_bounds(psx_a, psy_a, σᵣ, σₜ)
    (blb, bub) = squared_dist_bounds(psx_b, psy_b, σᵣ, σₜ)
    @test lb ≈ alb + blb
    @test ub ≈ aub + bub
    @test lb <= ub
end

@testset "GOGMA runs without errors" begin
    # two sets of points, each forming a 3-4-5 triangle
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    ypts = [[1.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -3.0]]
    σ = ϕ = 1.0
    gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
    gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

    # make sure this runs without an error
    res1 = gogma_align(gmmx, gmmy; maxsplits = 1.0e3)
    res2 = tiv_gogma_align(gmmx, gmmy; maxsplits = 1.0e3)

    # TIV radius cutoffs are keyword arguments; the positional form is deprecated
    res2_kw = tiv_gogma_align(gmmx, gmmy; cutoff_x = 10.0, cutoff_y = 10.0, maxsplits = 1.0e3)
    @test isfinite(res2_kw.upperbound)
    @test_deprecated tiv_gogma_align(gmmx, gmmy, 10.0, 10.0; maxsplits = 1.0e3)

    # branchbound accepts initial rotation/translation hints by keyword (forwarded through gogma_align)
    res_hint = gogma_align(
        gmmx, gmmy; initial_rotation = RotationVec(0.1, 0.0, 0.0),
        initial_translation = SVector{3}(0.0, 0.0, 0.0), maxsplits = 1.0e2
    )
    @test isfinite(res_hint.upperbound)

    mgmmx = IsotropicMultiGMM(Dict(:x => gmmx, :y => gmmx))
    mgmmy = IsotropicMultiGMM(Dict(:x => gmmy, :y => gmmy))
    res3 = gogma_align(mgmmx, mgmmy; maxsplits = 1.0e3)
    res4 = tiv_gogma_align(mgmmx, mgmmy)

    # ROCS alignment should work perfectly for these GMMs
    @test isapprox(rocs_align(gmmx, gmmy).minimum, -overlap(gmmx, gmmx); atol = 1.0e-12)

    # non-uniform σ and ϕ: the inertial starting orientations depend on the widths through
    # the mass matrix, so recovery here requires `widths` to actually return σ
    gx2 = IsotropicGMM([IsotropicGaussian(x, 0.5 + 0.3i, 1.0 + 0.5i) for (i, x) in enumerate(xpts)])
    gy2 = AffineMap(RotationVec(0.4, -0.3, 0.7), SVector(1.0, -2.0, 0.5))(gx2)
    @test isapprox(rocs_align(gx2, gy2).minimum, -overlap(gx2, gx2); atol = 1.0e-6)
end

@testset "AlignmentResults interface" begin
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    ypts = [[1.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -3.0]]
    σ = ϕ = 1.0
    gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
    gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

    gres = gogma_align(gmmx, gmmy)              # generous budget: certifies the optimum
    tres = tiv_gogma_align(gmmx, gmmy)
    rres = rocs_align(gmmx, gmmy)
    early = gogma_align(gmmx, gmmy; maxsplits = 1) # halted by the split limit

    # accessors mirror the underlying fields
    @test GMA.tform(gres) === gres.tform
    @test GMA.tform(tres) === tres.tform
    @test GMA.tform(rres) === rres.tform
    @test GMA.upperbound(gres) === gres.upperbound
    @test GMA.lowerbound(gres) === gres.lowerbound
    @test GMA.obj_calls(gres) === gres.obj_calls
    @test GMA.num_splits(gres) === gres.num_splits
    @test GMA.num_blocks(gres) === gres.num_blocks
    @test GMA.stagnant_splits(gres) === gres.stagnant_splits
    @test GMA.progress(gres) === gres.progress

    # converged reflects the termination cause
    @test GMA.converged(gres)
    @test GMA.converged(tres)
    @test !GMA.converged(early)
    @test early.terminated_by == "terminated early"
    @test !GMA.converged(rres)                  # ROCS carries no global guarantee

    # TIV aggregates the two sub-searches
    @test GMA.converged(tres) ==
        (GMA.converged(tres.rotation_result) && GMA.converged(tres.translation_result))
    @test GMA.stagnant_splits(tres) ==
        tres.rotation_result.stagnant_splits + tres.translation_result.stagnant_splits
    @test eltype(GMA.progress(tres)) == Tuple{Int, Float64, NTuple{6, Float64}}
    @test !isempty(GMA.progress(tres))
    # the trace ends at the reported optimum
    @test last(GMA.progress(tres))[2] == GMA.upperbound(tres)
    @test last(GMA.progress(tres))[3] == tres.tform_params

    # show produces a readable summary for each result type
    for r in (gres, tres, rres)
        str = sprint(show, MIME"text/plain"(), r)
        @test occursin(string(nameof(typeof(r))), str)
        @test occursin("converged", str)
    end
end

@testset "autodiff kwarg" begin
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    ypts = [[1.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -3.0]]
    σ = ϕ = 1.0
    gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
    gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])
    block = UncertaintyRegion(gmmx, gmmy)
    fdm = central_fdm(5, 1)
    ad = AutoFiniteDifferences(; fdm)

    # local_align directly: should return a finite score
    score_fd, _ = GMA.local_align(gmmx, gmmy, block; autodiff = ad)
    @test isfinite(score_fd)

    # gogma_align with autodiff kwarg: should run without error
    res_fd = gogma_align(gmmx, gmmy; autodiff = ad, maxsplits = 10)
    @test isfinite(res_fd.upperbound)

    # branchbound with a custom localfun closure
    pσ, pϕ = GMA.pairwise_consts(gmmx, gmmy, nothing)
    bndsfun = (x, y, bl) -> gauss_l2_bounds(x, y, bl, pσ, pϕ)
    localfun_fd = (x, y, bl) -> GMA.local_align(x, y, bl, pσ, pϕ; autodiff = ad)
    res_bb = branchbound(gmmx, gmmy; boundsfun = bndsfun, localfun = localfun_fd, maxsplits = 10)
    @test isfinite(res_bb.upperbound)
end

@testset "Evaluation at a point" begin
    pts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    σ = ϕ = 1.0
    gmm = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in pts])

    pt = [1, 1, 1]
    gpt = [g(pt) for g in gmm.gaussians]
    @test gpt ≈ [exp(-3 / 2), exp(-6 / 2), exp(-11 / 2)]
    @test gmm(pt) == sum(gpt)
end

@testset "MultiGMMs with interactions" begin
    tetrahedral = [
        [0.0, 0.0, 1.0],
        [sqrt(8 / 9), 0.0, -1 / 3],
        [-sqrt(2 / 9), sqrt(2 / 3), -1 / 3],
        [-sqrt(2 / 9), -sqrt(2 / 3), -1 / 3],
    ]
    ch_g = IsotropicGaussian(tetrahedral[1], 1.0, 1.0)
    s_gs = [IsotropicGaussian(x, 0.5, 1.0) for (i, x) in enumerate(tetrahedral)]
    mgmmx = IsotropicMultiGMM(
        Dict(
            :positive => IsotropicGMM([ch_g]),
            :steric => IsotropicGMM(s_gs)
        )
    )
    mgmmy = IsotropicMultiGMM(
        Dict(
            :negative => IsotropicGMM([ch_g]),
            :steric => IsotropicGMM(s_gs)
        )
    )
    # interaction validation
    interactions = Dict(
        (:positive, :negative) => 1.0,
        (:negative, :positive) => 1.0,
        (:positive, :positive) => -1.0,
        (:negative, :negative) => -1.0,
        (:steric, :steric) => -1.0,
    )
    @test_throws ArgumentError GMA.pairwise_consts(mgmmx, mgmmy, interactions)
    @test_throws "must not include redundant key pairs" GMA.pairwise_consts(mgmmx, mgmmy, interactions)
    interactions = Dict(
        (:positive, :negative) => 1.0,
        (:positive, :positive) => -1.0,
        (:negative, :negative) => -1.0,
        (:steric, :steric) => -1.0,
    )
    randtform = AffineMap(RotationVec(π * 0.1rand(3)...), SVector{3}(0.1 * rand(3)...))
    res = gogma_align(randtform(mgmmx), mgmmy; interactions = interactions, maxsplits = 5.0e3, nextblockfun = GMA.randomblock)

    # `mgmmx` holds :positive and :steric, `mgmmy` holds :negative and :steric

    # a key pair is resolved in whichever order it is stored
    forward = GMA.pairwise_consts(mgmmx, mgmmy, Dict((:positive, :negative) => 1.0, (:steric, :steric) => -1.0))
    reversed = GMA.pairwise_consts(mgmmx, mgmmy, Dict((:negative, :positive) => 1.0, (:steric, :steric) => -1.0))
    @test forward == reversed
    @test sort(collect(keys(forward[1]))) == [:positive, :steric]
    @test forward[1][:positive][:negative] == GMA.pairwise_consts(mgmmx.gmms[:positive], mgmmy.gmms[:negative])[1]
    @test forward[2][:steric][:steric] == -1.0 .* GMA.pairwise_consts(mgmmx.gmms[:steric], mgmmy.gmms[:steric])[2]

    # a coefficient of zero still produces an entry; an absent pair does not
    zeroed = GMA.pairwise_consts(mgmmx, mgmmy, Dict((:positive, :negative) => 0.0, (:steric, :steric) => -1.0))
    @test haskey(zeroed[2][:positive], :negative)
    @test all(iszero, zeroed[2][:positive][:negative])
    @test !haskey(GMA.pairwise_consts(mgmmx, mgmmy, Dict((:steric, :steric) => -1.0))[2], :positive)

    # a key of `mgmmx` whose partners are all absent from `mgmmy` is dropped entirely
    dropped = GMA.pairwise_consts(mgmmx, mgmmy, Dict((:positive, :positive) => -1.0, (:steric, :steric) => 1.0))
    @test collect(keys(dropped[1])) == [:steric]

    # overlap uses those constants, so it agrees with a per-key-pair sum
    interactions = Dict((:positive, :negative) => 1.5, (:steric, :steric) => -1.0)
    manual = 1.5 * overlap(mgmmx.gmms[:positive], mgmmy.gmms[:negative]) -
        overlap(mgmmx.gmms[:steric], mgmmy.gmms[:steric])
    @test overlap(mgmmx, mgmmy; interactions) ≈ manual

    @test GMA.has_interaction(Dict((:a, :b) => 1.0), :a, :b)
    @test GMA.has_interaction(Dict((:a, :b) => 1.0), :b, :a)     # either ordering
    @test GMA.has_interaction(Dict((:a, :b) => 0.0), :a, :b)     # zero is still present
    @test !GMA.has_interaction(Dict((:a, :b) => 1.0), :a, :a)
end

@testset "LabeledIsotropicGMM" begin
    g1 = IsotropicGaussian([0.0, 0.0, 0.0], 1.0, 1.0)
    g2 = IsotropicGaussian([1.0, 0.0, 0.0], 1.0, 1.0)
    x = LabeledIsotropicGMM([g1, g2], [:A, :B])

    # interface and supertypes
    @test x isa GMA.AbstractLabeledIsotropicGMM{3, Float64, Symbol}
    @test x isa GMA.AbstractIsotropicGMM
    @test length(x) == 2
    @test x[2] == g2
    @test collect(x) == [g1, g2]              # iterate
    @test eltype(x) === eltype(typeof(x)) === IsotropicGaussian{3, Float64}

    # constructors
    xcopy = LabeledIsotropicGMM(x)
    @test xcopy.gaussians == x.gaussians && xcopy.labels == x.labels
    empty_gmm = LabeledIsotropicGMM{3, Float64, Symbol}()
    @test isempty(empty_gmm) && empty_gmm isa LabeledIsotropicGMM{3, Float64, Symbol}
    @test_throws DimensionMismatch LabeledIsotropicGMM([g1, g2], [:A])

    # convert and promote
    @test convert(LabeledIsotropicGMM{3, Float32, Symbol}, x) isa LabeledIsotropicGMM{3, Float32, Symbol}
    xf32 = LabeledIsotropicGMM([IsotropicGaussian([0.0f0, 0.0f0, 0.0f0], 1.0f0, 1.0f0)], [:A])
    @test promote_type(typeof(x), typeof(xf32)) === LabeledIsotropicGMM{3, Float64, Symbol}

    # transformations preserve labels and produce a LabeledIsotropicGMM
    R = RotationVec(0.3, 0.1, -0.2)
    @test R * x isa LabeledIsotropicGMM{3, Float64, Symbol}
    @test (R * x).labels == x.labels
    @test (x + SVector(1.0, 2.0, 3.0)).labels == x.labels
    @test [g.μ for g in (x - SVector(1.0, 0.0, 0.0))] == [SVector(-1.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0)]

    # overlap: default uses same-label pairs only, each with coefficient 1
    y = LabeledIsotropicGMM(
        [
            IsotropicGaussian([0.5, 0.2, 0.0], 1.0, 1.0),
            IsotropicGaussian([1.7, 0.0, 0.3], 1.0, 1.0),
        ], [:A, :B]
    )
    same_label = overlap(g1, y.gaussians[1]) + overlap(g2, y.gaussians[2])
    @test overlap(x, y) ≈ same_label
    @test overlap(x, y) ≈ overlap(x, y; interactions = Dict((:A, :A) => 1.0, (:B, :B) => 1.0))
    @test overlap(x, y; interactions = Dict{Tuple{Symbol, Symbol}, Float64}()) == 0.0
    @test !(overlap(x, y; interactions = Dict((:A, :B) => 1.0)) ≈ overlap(x, y))

    # overlap is invariant when the same rigid transform is applied to both
    @test overlap(R * x, R * y) ≈ overlap(x, y)

    # redundant key pairs are rejected
    @test_throws ArgumentError overlap(x, y; interactions = Dict((:A, :B) => 1.0, (:B, :A) => 1.0))
    @test_throws "must not include redundant key pairs" GMA.pairwise_consts(x, y, Dict((:A, :B) => 1.0, (:B, :A) => 1.0))

    # force and force! agree, and interactions change the result
    f = zeros(3)
    force!(f, x, y)
    @test f ≈ force(x, y)
    @test !(force(x, y) ≈ force(x, y; interactions = Dict((:A, :B) => 1.0)))
end

@testset "zero-weight pairs contribute nothing to overlap" begin
    # `overlap` and `gauss_l2_bounds` skip Gaussian pairs whose weight is zero. Every pair
    # of a labeled GMM whose labels do not interact has such a weight, so the skip has to
    # reproduce a summation over all pairs exactly.
    gx = [
        IsotropicGaussian([0.0, 0.0, 0.0], 1.0, 1.0),
        IsotropicGaussian([1.0, 0.0, 0.0], 1.3, 2.0),
        IsotropicGaussian([0.0, 1.0, 0.5], 0.8, 1.5),
    ]
    gy = [
        IsotropicGaussian([0.5, 0.2, 0.0], 1.0, 1.0),
        IsotropicGaussian([1.7, 0.0, 0.3], 1.1, 0.7),
        IsotropicGaussian([0.2, 0.9, 0.1], 0.9, 1.2),
    ]
    x = LabeledIsotropicGMM(gx, [:A, :B, :C])
    y = LabeledIsotropicGMM(gy, [:A, :B, :C])
    # :C interacts with nothing, and (:A,:B) is negative, as a charge penalty would be
    interactions = Dict((:A, :A) => 1.0, (:B, :B) => 2.0, (:A, :B) => -1.5)

    pσ, pϕ = GMA.pairwise_consts(x, y, interactions)
    @test count(iszero, pϕ) > 0                                    # the skip is exercised
    # pσ/pϕ are indexed [x, y]
    @test size(pσ) == size(pϕ) == (length(gx), length(gy))
    coefficient(a, b) = get(interactions, (a, b), get(interactions, (b, a), 0.0))
    @test all(
        pσ[i, j] == gx[i].σ^2 + gy[j].σ^2 &&
            pϕ[i, j] == coefficient(x.labels[i], y.labels[j]) * gx[i].ϕ * gy[j].ϕ
            for i in eachindex(gx), j in eachindex(gy)
    )
    naive = sum(
        pϕ[i, j] * exp(-sum(abs2, gx[i].μ - gy[j].μ) / (2 * pσ[i, j]))
            for i in eachindex(gx), j in eachindex(gy)
    )
    @test overlap(x, y; interactions) ≈ naive
    @test overlap(x, y, pσ, pϕ) ≈ naive

    # pσ/pϕ index the Gaussians from 1, so offset inputs are rejected, not mis-indexed
    @test_throws "offset arrays are not supported" overlap(x, y, OffsetArray(pσ, 0:2, 0:2), OffsetArray(pϕ, 0:2, 0:2))

    # an explicitly zero coefficient behaves exactly like an omitted one
    @test overlap(x, y; interactions = Dict((:A, :A) => 0.0)) == 0.0
    @test overlap(x, y; interactions = Dict((:A, :A) => 0.0, (:B, :B) => 2.0)) ≈
        overlap(x, y; interactions = Dict((:B, :B) => 2.0))

    # gauss_l2_bounds skips the same pairs, and must match a sum over every pair
    R, T = RotationVec(0.2, -0.1, 0.35), SVector(0.3, -0.2, 0.1)
    σᵣ, σₜ = 0.4, 0.25
    lb, ub = gauss_l2_bounds(x, y, R, T, σᵣ, σₜ, pσ, pϕ)
    naive_lb, naive_ub = 0.0, 0.0
    for i in eachindex(gx), j in eachindex(gy)
        naive_lb, naive_ub = (naive_lb, naive_ub) .+
            gauss_l2_bounds(gx[i], gy[j], R, T, σᵣ, σₜ, pσ[i, j], pϕ[i, j])
    end
    @test lb ≈ naive_lb
    @test ub ≈ naive_ub
    # the objective at the region's center lies within the bounds it brackets
    @test lb <= -overlap(R * x + T, y, pσ, pϕ) <= ub
end

@testset "pairwise_consts resolves each label pair once" begin
    # `pairwise_consts` indexes a coefficient table rather than hashing every Gaussian
    # pair, so it must agree with a direct per-pair lookup for every label arrangement.
    function reference(gmmx, gmmy, interactions::Dict{Tuple{K, K}, V}) where {K, V}
        t = promote_type(GMA.numbertype(gmmx), GMA.numbertype(gmmy), V)
        n, m = length(gmmx), length(gmmy)
        pσ, pϕ = zeros(t, n, m), zeros(t, n, m)
        for i in 1:n, j in 1:m
            gaussx, gaussy = gmmx.gaussians[i], gmmy.gaussians[j]
            key = (gmmx.labels[i], gmmy.labels[j])
            key = haskey(interactions, key) ? key : reverse(key)
            pσ[i, j] = gaussx.σ^2 + gaussy.σ^2
            pϕ[i, j] = (haskey(interactions, key) ? interactions[key] : zero(t)) * gaussx.ϕ * gaussy.ϕ
        end
        return pσ, pϕ
    end

    gauss(x, σ, ϕ) = IsotropicGaussian([x, 0.0, 0.0], σ, ϕ)
    x = LabeledIsotropicGMM([gauss(0.0, 1.0, 1.0), gauss(1.0, 1.3, 2.0), gauss(2.0, 0.7, 0.5)], [:A, :B, :A])
    y = LabeledIsotropicGMM([gauss(0.5, 1.0, 1.5), gauss(1.5, 0.9, 0.8)], [:B, :C])

    # (:B, :A) is stored in the order opposite to how the labels are encountered, :C
    # interacts with nothing, and :A never meets itself across the two GMMs
    for interactions in (
            Dict((:B, :A) => -1.5, (:B, :B) => 2.0),
            Dict((:A, :B) => -1.5, (:B, :B) => 2.0),
            Dict((:A, :A) => 1.0),
            Dict((:C, :C) => 3.0),
            Dict((:A, :B) => 0.0, (:B, :B) => 2.0),   # an explicit zero coefficient
            Dict{Tuple{Symbol, Symbol}, Float64}(),   # nothing interacts
        )
        @test GMA.pairwise_consts(x, y, interactions) == reference(x, y, interactions)
    end

    # a label present in one GMM but absent from the other
    @test GMA.pairwise_consts(x, y, Dict((:A, :C) => 2.5)) == reference(x, y, Dict((:A, :C) => 2.5))

    # labels need not be Symbols, and repeated labels share a table entry
    xi = LabeledIsotropicGMM([gauss(0.0, 1.0, 1.0), gauss(1.0, 1.0, 1.0)], [7, 7])
    yi = LabeledIsotropicGMM([gauss(0.5, 1.0, 1.0)], [7])
    @test GMA.pairwise_consts(xi, yi, Dict((7, 7) => 2.0)) == reference(xi, yi, Dict((7, 7) => 2.0))

    # empty GMMs produce empty constants rather than erroring on an empty label table
    empty_gmm = LabeledIsotropicGMM{3, Float64, Symbol}()
    epσ, epϕ = GMA.pairwise_consts(empty_gmm, y, Dict((:A, :B) => 1.0))
    @test size(epσ) == size(epϕ) == (0, length(y))
    @test overlap(empty_gmm, y; interactions = Dict((:A, :B) => 1.0)) == 0.0

    # the coefficient helper is symmetric in its labels and zero for absent pairs
    @test GMA.interaction_coefficient(Dict((:A, :B) => 1.5), :A, :B) == 1.5
    @test GMA.interaction_coefficient(Dict((:A, :B) => 1.5), :B, :A) == 1.5
    @test GMA.interaction_coefficient(Dict((:A, :B) => 1.5), :A, :A) == 0.0

    # redundant key pairs are still rejected
    @test_throws "must not include redundant key pairs" GMA.pairwise_consts(x, y, Dict((:A, :B) => 1.0, (:B, :A) => 2.0))
end

@testset "overlap keeps zero-valued weights that carry a derivative" begin
    # pϕ = coefficient * ϕx * ϕy, so differentiating with respect to an amplitude that is
    # itself zero yields a weight with zero value and a nonzero partial. Such a term still
    # contributes to the derivative and must not be skipped: `iszero` on a dual number is
    # false unless the partials vanish too, which is what makes the skip sound.
    @test !iszero(ForwardDiff.Dual(0.0, 1.0))
    @test iszero(ForwardDiff.Dual(0.0, 0.0))

    yg = LabeledIsotropicGMM([IsotropicGaussian([0.5, 0.0, 0.0], 1.0, 2.0)], [:A])
    coefficient = 3.0
    ovlp(ϕ) = overlap(
        LabeledIsotropicGMM([IsotropicGaussian(SVector(0.0, 0.0, 0.0), 1.0, ϕ)], [:A]),
        yg; interactions = Dict((:A, :A) => coefficient)
    )

    @test ovlp(0.0) == 0.0                       # the value really is zero at ϕ = 0
    # d/dϕ [c * ϕ * ϕy * exp(-d²/2s)] = c * ϕy * exp(-d²/2s)
    analytic = coefficient * 2.0 * exp(-0.5^2 / (2 * (1.0^2 + 1.0^2)))
    @test ForwardDiff.derivative(ovlp, 0.0) ≈ analytic
    @test ForwardDiff.derivative(ovlp, 0.0) ≈ central_fdm(5, 1)(ovlp, 0.0)
    @test ForwardDiff.derivative(ovlp, 0.0) != 0.0
end

@testset "Forces" begin
    μx = randn(SVector{3, Float64})
    μy = randn(SVector{3, Float64})
    σx = 1 + rand()
    σy = 1 + rand()
    ϕx = 1 + rand()
    ϕy = 1 + rand()
    x = IsotropicGaussian(μx, σx, ϕx)
    y = IsotropicGaussian(μy, σy, ϕy)
    f = zeros(3)
    force!(f, x, y)
    ovlp(μ) = overlap(IsotropicGaussian(μ, σx, ϕx), y)
    @test f ≈ ForwardDiff.gradient(ovlp, μx)
    @test force(x, y) ≈ f

    tetrahedral = [
        [0.0, 0.0, 1.0],
        [sqrt(8 / 9), 0.0, -1 / 3],
        [-sqrt(2 / 9), sqrt(2 / 3), -1 / 3],
        [-sqrt(2 / 9), -sqrt(2 / 3), -1 / 3],
    ]
    ch_g = IsotropicGaussian(tetrahedral[1], 1.0, 1.0)
    s_gs = [IsotropicGaussian(x, 0.5, 1.0) for (i, x) in enumerate(tetrahedral)]
    mgmmx = IsotropicMultiGMM(
        Dict(
            :positive => IsotropicGMM([ch_g]),
            :steric => IsotropicGMM(s_gs)
        )
    )
    mgmmy = IsotropicMultiGMM(
        Dict(
            :negative => IsotropicGMM([ch_g]),
            :steric => IsotropicGMM(s_gs)
        )
    )
    fliptform = AffineMap(RotationVec(π, 0, 0), [0, 0, 3]) ∘ AffineMap(RotationVec(0, 0, π), [0, 0, 0])
    mgmmy = fliptform(mgmmy)
    interactions = Dict(
        (:positive, :negative) => 1.0,
        (:positive, :positive) => -1.0,
        (:negative, :negative) => -1.0,
        (:steric, :steric) => -1.0,
    )
    f = zeros(3)
    force!(f, mgmmx, mgmmy; interactions = interactions)
    movlp(μ) = overlap(IsotropicMultiGMM(Dict(:positive => IsotropicGMM([ch_g + μ]), :steric => IsotropicGMM([g + μ for g in s_gs]))), mgmmy; interactions)
    @test f ≈ ForwardDiff.gradient(movlp, zeros(3))
    @test force(mgmmx, mgmmy; interactions) ≈ f
    @test_deprecated overlap(mgmmx, mgmmy, nothing, nothing, interactions)
end

@testset "GO-ICP and GO-IH run without errors" begin
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    ypts = [[1.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -3.0]]

    xset = PointSet(xpts)
    yset = PointSet(ypts)

    goicp_res = goicp_align(xset, yset)
    @test GaussianMixtureAlignment.transform_columns(goicp_res.tform, xset.coords) ≈ yset.coords
    goih_res = goih_align(xset, yset)
    @test GaussianMixtureAlignment.transform_columns(goih_res.tform, xset.coords) ≈ yset.coords

    # TIV radius cutoffs are keyword arguments; the positional form is deprecated
    tiv_icp = tiv_goicp_align(xset, yset; cutoff_x = 10.0, cutoff_y = 10.0)
    @test isfinite(tiv_icp.upperbound)
    @test GaussianMixtureAlignment.transform_columns(tiv_icp.tform, xset.coords) ≈ yset.coords
    @test_deprecated tiv_goicp_align(xset, yset, 10.0, 10.0)
    tiv_ih = tiv_goih_align(xset, yset; cutoff_x = 10.0, cutoff_y = 10.0)
    @test isfinite(tiv_ih.upperbound)
    @test GaussianMixtureAlignment.transform_columns(tiv_ih.tform, xset.coords) ≈ yset.coords
    @test_deprecated tiv_goih_align(xset, yset, 10.0, 10.0)

end

@testset "Kabsch" begin
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    ypts = [[1.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -3.0]]

    xset = PointSet(xpts, ones(3))
    yset = PointSet(ypts, ones(3))

    tform = kabsch(xset, yset)

    @test yset.coords ≈ tform(xset).coords
end

@testset "GO-ICP" begin
    ycoords = rand(3, 50) * 5 .- 10
    randtform = AffineMap(RotationVec(π * rand(3)...), SVector{3}(5 * rand(3)...))
    xcoords = GaussianMixtureAlignment.transform_columns(randtform, ycoords)

    xset = PointSet(xcoords, ones(size(xcoords, 2)))
    yset = PointSet(xcoords, ones(size(ycoords, 2)))

    res = goicp_align(yset, xset)
    @test res.lowerbound == 0
    @test res.upperbound < 1.0e-15
end

@testset "globally optimal iterative hungarian" begin
    ycoords = rand(3, 5) * 5 .- 10
    randtform = AffineMap(RotationVec(π * rand(3)...), SVector{3}(5 * rand(3)...))
    xcoords = GaussianMixtureAlignment.transform_columns(randtform, ycoords)

    xset = PointSet(xcoords, ones(size(xcoords, 2)))
    yset = PointSet(xcoords, ones(size(ycoords, 2)))

    res = goih_align(yset, xset)
    @test res.lowerbound == 0
    @test res.upperbound < 1.0e-15
end

@testset "rotation-only and translation-only alignment" begin
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    ypts = [[1.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -3.0]]
    σ = ϕ = 1.0
    gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
    gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

    rot_res = rot_gogma_align(gmmx, gmmy; maxsplits = 200)
    @test isfinite(rot_res.upperbound)
    @test rot_res.tform isa LinearMap

    trl_res = trl_gogma_align(gmmx, gmmy; maxsplits = 200)
    @test isfinite(trl_res.upperbound)
    @test trl_res.tform isa Translation

    xset = PointSet(xpts)
    yset = PointSet(ypts)
    trl_icp = GMA.trl_goicp_align(xset, yset; maxsplits = 1000)
    @test isfinite(trl_icp.upperbound)
    @test trl_icp.tform isa Translation
end

@testset "branchbound options and error paths" begin
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    ypts = [[1.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -3.0]]
    σ = ϕ = 1.0
    gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
    gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

    # centerinputs=true centers each model before searching
    res_center = gogma_align(gmmx, gmmy; centerinputs = true, maxsplits = 200)
    @test isfinite(res_center.upperbound)

    # separatesplit=true splits rotation and translation axes independently
    res_sep = gogma_align(gmmx, gmmy; separatesplit = true, maxsplits = 200)
    @test isfinite(res_sep.upperbound)

    # nsplits must be even
    @test_throws ArgumentError branchbound(gmmx, gmmx; nsplits = 3)
    @test_throws "`nsplits` must be even" branchbound(gmmx, gmmx; nsplits = 3)

    # dimensionality mismatch
    gmm2d = IsotropicGMM([IsotropicGaussian(SVector(0.0, 0.0), σ, ϕ)])
    @test_throws ArgumentError branchbound(gmmx, gmm2d)
    @test_throws "Dimensionality" branchbound(gmmx, gmm2d)
end

@testset "GMM similarity metrics" begin
    g = IsotropicGaussian([0.0, 0.0, 0.0], 1.0, 1.0)
    gmmx = IsotropicGMM([g])

    @test GMA.distance(gmmx, gmmx) ≈ 0 atol = 1.0e-15
    @test GMA.tanimoto(gmmx, gmmx) ≈ 1 atol = 1.0e-15

    far = IsotropicGMM([IsotropicGaussian([100.0, 0.0, 0.0], 1.0, 1.0)])
    @test GMA.distance(gmmx, far) > 0
    @test GMA.tanimoto(gmmx, far) < 1
end

@testset "combine vector and varargs forms" begin
    g1 = IsotropicGMM([IsotropicGaussian([0.0, 0.0, 0.0], 1.0, 1.0)])
    g2 = IsotropicGMM([IsotropicGaussian([1.0, 0.0, 0.0], 1.0, 1.0)])
    g3 = IsotropicGMM([IsotropicGaussian([2.0, 0.0, 0.0], 1.0, 1.0)])

    @test length(GMA.combine([g1, g2])) == 2
    @test GMA.combine([g1]) === g1
    @test length(GMA.combine(g1, g2, g3)) == 3

    @test_throws ArgumentError GMA.combine(IsotropicGMM{3, Float64}[])
    @test_throws "no GMMs to combine" GMA.combine(IsotropicGMM{3, Float64}[])

    g2d = IsotropicGMM([IsotropicGaussian(SVector(0.0, 0.0), 1.0, 1.0)])
    @test_throws ArgumentError GMA.combine(g1, g2d)
    @test_throws "same dimensionality" GMA.combine(g1, g2d)

    # MultiGMM combine error
    mg1 = IsotropicMultiGMM(Dict(:a => g1))
    mg2 = IsotropicMultiGMM(Dict(:a => IsotropicGMM([IsotropicGaussian(SVector(0.0, 0.0), 1.0, 1.0)])))
    @test_throws ArgumentError GMA.combine(mg1, mg2)
    @test_throws "same dimensionality" GMA.combine(mg1, mg2)
end

@testset "overlap 5-arg form and force non-mutating for GMMs" begin
    x = IsotropicGaussian([0.0, 0.0, 0.0], 1.0, 1.0)
    y = IsotropicGaussian([1.0, 0.0, 0.0], 2.0, 0.5)
    @test overlap(1.0, x.σ, y.σ, x.ϕ, y.ϕ) ≈ overlap(x, y) atol = 1.0e-15

    # force on a GMM vs itself at zero displacement is zero (forces cancel by symmetry)
    g0 = IsotropicGaussian([0.0, 0.0, 0.0], 1.0, 1.0)
    gmm = IsotropicGMM([g0])
    @test force(gmm, gmm) ≈ zeros(3) atol = 1.0e-15

    # force!(f, SingleGaussian, SingleGMM) must agree with the per-pair sum
    gmmx = IsotropicGMM([x, y])
    gmmy = IsotropicGMM([y])
    f1 = zeros(3); force!(f1, x, gmmy)
    f2 = zeros(3); force!(f2, x, y)
    @test f1 ≈ f2 atol = 1.0e-15

    # force non-mutating for GMM vs GMM
    tetrahedral = [[0.0, 0.0, 1.0], [sqrt(8 / 9), 0.0, -1 / 3], [-sqrt(2 / 9), sqrt(2 / 3), -1 / 3], [-sqrt(2 / 9), -sqrt(2 / 3), -1 / 3]]
    sym = IsotropicGMM([IsotropicGaussian(p, 0.5, 1.0) for p in tetrahedral])
    @test force(sym, sym) ≈ zeros(3) atol = 1.0e-10
end

@testset "SearchRegion constructors and conversions" begin
    # UncertaintyRegion convenience constructors
    ur0 = UncertaintyRegion()
    @test ur0.σᵣ ≈ π && ur0.σₜ ≈ 1.0

    ur1 = UncertaintyRegion(2.0)       # σₜ only; σᵣ defaults to π
    @test ur1.σₜ ≈ 2.0 && ur1.σᵣ ≈ π

    ur2 = UncertaintyRegion(π / 4, 3.0)  # (σᵣ, σₜ)
    @test ur2.σᵣ ≈ π / 4 && ur2.σₜ ≈ 3.0

    @test UncertaintyRegion(ur2) === ur2  # identity when already UncertaintyRegion

    # RotationRegion
    rr = RotationRegion()
    @test rr.σᵣ ≈ Float64(π)

    rr2 = RotationRegion(ur2)
    @test rr2.σᵣ ≈ ur2.σᵣ

    # RotationRegion ↔ UncertaintyRegion round-trip
    ur3 = UncertaintyRegion(rr2)
    @test ur3.σᵣ ≈ rr2.σᵣ && iszero(ur3.σₜ)
    @test RotationRegion(ur3).σᵣ ≈ ur3.σᵣ

    # TranslationRegion
    tr = TranslationRegion()
    @test tr.σₜ ≈ 1.0

    tr2 = TranslationRegion(ur2)
    @test tr2.σₜ ≈ ur2.σₜ

    ur4 = UncertaintyRegion(tr2)
    @test ur4.σₜ ≈ tr2.σₜ && iszero(ur4.σᵣ)
    @test TranslationRegion(ur4).σₜ ≈ ur4.σₜ

    # AffineMap from SearchRegion
    af = AffineMap(ur2)
    @test af isa AffineMap

    # rot_subregions halves σᵣ, leaves σₜ unchanged
    rsubs = GMA.rot_subregions(ur2)
    @test length(rsubs) == 8
    @test all(s.σᵣ ≈ ur2.σᵣ / 2 for s in rsubs)
    @test all(s.σₜ ≈ ur2.σₜ for s in rsubs)

    # trl_subregions halves σₜ, leaves σᵣ unchanged
    tsubs = GMA.trl_subregions(ur2)
    @test length(tsubs) == 8
    @test all(s.σₜ ≈ ur2.σₜ / 2 for s in tsubs)
    @test all(s.σᵣ ≈ ur2.σᵣ for s in tsubs)
end

@testset "distance bounds: explicit R and block-dispatch forms" begin
    x = SVector(3.0, 0.0, 0.0)
    y = SVector(-4.0, 0.0, 0.0)
    R = RotationVec(0.0, 0.0, 0.0)
    T = SVector(0.0, 0.0, 0.0)

    # R-form equals the σ-form when R is the identity and T is zero
    lb_r, ub_r = GMA.tight_distance_bounds(x, y, R, T, Float64(π), 0.0)
    lb_s, ub_s = GMA.tight_distance_bounds(x, y, Float64(π), 0.0)
    @test lb_r ≈ lb_s && ub_r ≈ ub_s

    lb_lr, ub_lr = GMA.loose_distance_bounds(x, y, R, T, Float64(π), 0.0)
    lb_ls, ub_ls = GMA.loose_distance_bounds(x, y, Float64(π), 0.0)
    @test lb_lr ≈ lb_ls && ub_lr ≈ ub_ls

    # UncertaintyRegion block dispatch
    ur = UncertaintyRegion(R, T, Float64(π), 0.0)
    lb_ur, ub_ur = GMA.tight_distance_bounds(x, y, ur)
    @test lb_ur ≈ lb_s && ub_ur ≈ ub_s

    # Union{RotationRegion,TranslationRegion} dispatch (line 75 in distancebounds.jl)
    rr = RotationRegion(R, T, Float64(π))
    lb_rr, ub_rr = GMA.tight_distance_bounds(x, y, rr)
    @test lb_rr ≈ lb_s

    # gauss_l2_bounds with a non-UncertaintyRegion SearchRegion block
    xg = IsotropicGaussian(x, 1.0, 1.0)
    yg = IsotropicGaussian(y, 1.0, 1.0)
    lb_gur, ub_gur = gauss_l2_bounds(xg, yg, ur)
    lb_grr, ub_grr = gauss_l2_bounds(xg, yg, rr)
    @test lb_gur ≈ lb_grr
end

@testset "model utility functions" begin
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    ps = PointSet(xpts)

    # centroid and center_translation via AbstractModel dispatch
    c = GMA.centroid(ps)
    @test c ≈ [1.0, 4 / 3, 0.0] atol = 1.0e-10   # equal weights → arithmetic mean

    ct = GMA.center_translation(ps)
    @test ct isa Translation
    @test ct.translation ≈ -c atol = 1.0e-10

    # IsotropicGMM(ps) and PointSet(gmm) round-trip
    gmm = IsotropicGMM(ps)
    @test gmm isa IsotropicGMM{3, Float64}
    @test length(gmm) == length(ps)
    @test GMA.coords(gmm) ≈ GMA.coords(ps) atol = 1.0e-15

    ps2 = PointSet(gmm)
    @test ps2 isa PointSet{3, Float64}
    @test ps2.coords ≈ ps.coords atol = 1.0e-15

    # MultiPointSet(mgmm) preserves keys and coordinates
    mgmm = IsotropicMultiGMM(Dict(:a => gmm))
    mps = MultiPointSet(mgmm)
    @test mps isa MultiPointSet{3, Float64, Symbol}
    @test GMA.coords(mps[:a]) ≈ GMA.coords(ps) atol = 1.0e-15

    # translation_limit
    lim = GMA.translation_limit(ps, ps2)
    @test lim ≈ maximum(abs.(GMA.coords(ps))) atol = 1.0e-15

    # affinemap_to_params round-trip with build_tform
    R = RotationVec(0.3, -0.2, 0.5)
    T = SVector(1.0, -2.0, 3.0)
    tform = AffineMap(R, T)
    params = GMA.affinemap_to_params(tform)
    @test length(params) == 6
    rebuilt = GMA.build_tform(AffineMap, params)
    @test rebuilt.translation ≈ T atol = 1.0e-10
    @test Matrix(RotationVec(rebuilt.linear)) ≈ Matrix(R) atol = 1.0e-10
end

@testset "Point and PointSet interface" begin
    # PointSet from a vector of vectors
    vs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ps = PointSet(vs)
    @test ps isa PointSet{3, Float64}
    @test size(ps.coords, 2) == 3

    # iterate(AbstractSinglePointSet)
    pts = collect(ps)
    @test length(pts) == 3
    @test all(p isa GMA.Point{3, Float64} for p in pts)

    # indexed size(model, idx)
    gmm = IsotropicGMM([IsotropicGaussian([0.0, 0.0, 0.0], 1.0, 1.0), IsotropicGaussian([1.0, 0.0, 0.0], 1.0, 1.0)])
    @test size(gmm, 1) == length(gmm)
    @test size(gmm, 2) == 3

    mgmm = IsotropicMultiGMM(Dict(:a => gmm))
    @test size(mgmm, 1) == length(mgmm)
    @test size(mgmm, 2) == 3
    @test eltype(mgmm) === Pair{Symbol, IsotropicGMM{3, Float64}}

    @test size(ps, 1) == 3
    @test size(ps, 2) == length(ps)

    # get(mgmm, k, default) with a missing key
    @test get(mgmm, :missing, gmm) === gmm

    # Point arithmetic
    p = GMA.Point(SVector(1.0, 0.0, 0.0), 2.0)
    R = RotationVec(0.0, 0.0, π / 2)
    Rp = R * p
    @test Rp.coords ≈ SVector(0.0, 1.0, 0.0) atol = 1.0e-10
    @test Rp.weight == p.weight
    mp = p - SVector(1.0, 0.0, 0.0)
    @test mp.coords ≈ SVector(0.0, 0.0, 0.0) atol = 1.0e-10

    # weights(MultiPointSet) returns a Dict
    mps = MultiPointSet(Dict(:a => ps))
    w = GMA.weights(mps)
    @test w isa Dict
    @test haskey(w, :a)
end

@testset "MultiPointSet arithmetic and tivpointset" begin
    ps = PointSet([0.0 3.0; 0.0 0.0; 0.0 0.0])
    mps = MultiPointSet(Dict(:a => ps))
    R = RotationVec(0.0, 0.0, π / 2)
    T = SVector(1.0, 0.0, 0.0)

    rotated = R * mps
    @test GMA.coords(rotated[:a]) ≈ R * GMA.coords(ps) atol = 1.0e-10

    shifted = mps + T
    @test GMA.coords(shifted[:a]) ≈ GMA.coords(ps) .+ T atol = 1.0e-10

    subtracted = mps - T
    @test GMA.coords(subtracted[:a]) ≈ GMA.coords(ps) .- T atol = 1.0e-10

    # GMM subtraction operators
    g = IsotropicGaussian([1.0, 0.0, 0.0], 1.0, 1.0)
    gmm = IsotropicGMM([g])
    mgmm = IsotropicMultiGMM(Dict(:a => gmm))
    tv = [1.0, 0.0, 0.0]
    @test (g - tv).μ ≈ SVector(0.0, 0.0, 0.0) atol = 1.0e-10
    @test GMA.coords(gmm - tv) ≈ GMA.coords(gmm) .- tv atol = 1.0e-10
    @test GMA.coords((mgmm - tv)[:a]) ≈ GMA.coords(gmm) .- tv atol = 1.0e-10

    # tivpointset for MultiPointSet preserves keys
    ps3 = PointSet([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    mps3 = MultiPointSet(Dict(:a => ps3, :b => ps3))
    tivmps = GMA.tivpointset(mps3)
    @test tivmps isa MultiPointSet
    @test Set(keys(tivmps.pointsets)) == Set(keys(mps3.pointsets))
end

@testset "Kabsch, kabsch_matches, and translation_align" begin
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    ypts = [[1.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -3.0]]
    xset = PointSet(xpts)
    yset = PointSet(ypts)

    # kabsch_centered(PointSet, PointSet) returns a LinearMap wrapping a rotation matrix
    Rc = GMA.kabsch_centered(xset, yset)
    @test Rc isa LinearMap

    # kabsch_centered_matches and kabsch_matches
    matches = [(1, 1), (2, 2), (3, 3)]
    Rm = GMA.kabsch_centered_matches(xset, yset, matches)
    @test Rm isa LinearMap
    tform_m = GMA.kabsch_matches(xset, yset, matches)
    @test tform_m isa AffineMap

    # translation_align: pure centroid-matching shift
    offset = [2.0, 0.0, 0.0]
    xshift = PointSet([p .+ offset for p in xpts])
    t = GMA.translation_align(xset, xshift)
    @test t isa Translation
    @test t.translation ≈ offset atol = 1.0e-10
end

@testset "correspondence and squared_deviation for MultiPointSet" begin
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    ypts = [[1.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -3.0]]
    xset = PointSet(xpts)
    yset = PointSet(ypts)

    # hungarian_assignment(MultiPointSet, MultiPointSet) returns a Dict
    mpsx = MultiPointSet(Dict(:a => xset))
    mpsy = MultiPointSet(Dict(:a => yset))
    mdict = GMA.hungarian_assignment(mpsx, mpsy)
    @test mdict isa Dict{Symbol, Vector{Tuple{Int, Int}}}

    # squared_deviation for MultiPointSet matches single-key result
    matches_a = GMA.hungarian_assignment(xset, yset)
    sq_multi = GMA.squared_deviation(mpsx, mpsy, mdict)
    sq_single = GMA.squared_deviation(xset, yset, matches_a)
    @test sq_multi ≈ sq_single atol = 1.0e-10

    # rmsd
    P = xset.coords
    Q = yset.coords
    @test GMA.rmsd(P, Q) ≈ sqrt(GMA.squared_deviation(P, Q) / size(P, 2)) atol = 1.0e-10
end

@testset "GMM copy constructors and show methods" begin
    g = IsotropicGaussian([1.0, 0.0, 0.0], 0.5, 2.0)

    # copy constructors
    g2 = IsotropicGaussian(g)
    @test g2.μ == g.μ && g2.σ == g.σ && g2.ϕ == g.ϕ

    gmm = IsotropicGMM([g])
    gmm2 = IsotropicGMM(gmm)
    @test length(gmm2) == 1 && gmm2[1] == g

    mgmm = IsotropicMultiGMM(Dict(:a => gmm))
    mgmm2 = IsotropicMultiGMM(mgmm)
    @test keys(mgmm2) == keys(mgmm)

    # show methods produce non-empty output containing recognizable substrings
    @test occursin("μ", sprint(show, g))
    @test occursin("IsotropicGMM", sprint(show, gmm))

    p = GMA.Point(SVector(1.0, 0.0, 0.0), 2.0)
    @test occursin("ϕ", sprint(show, p))

    ps = PointSet([1.0 0.0; 0.0 1.0; 0.0 0.0])
    @test occursin("2 points", sprint(show, ps))
end

@testset "inertial_transforms model dispatch and invert" begin
    xpts = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    gmmx = IsotropicGMM([IsotropicGaussian(x, 1.0, 1.0) for x in xpts])

    # model dispatch (line 93 in rocs/rocsalign.jl)
    tforms = GMA.inertial_transforms(gmmx)
    @test length(tforms) == 4

    # invert=true (line 81): each inverted transform undoes the forward one
    itforms = GMA.inertial_transforms(gmmx; invert = true)
    @test length(itforms) == 4
    p = SVector(1.0, 2.0, 3.0)
    for (t, it) in zip(tforms, itforms)
        @test it(t(p)) ≈ p atol = 1.0e-10
    end
end

@testset "generic axes" begin
    coords_plain = [0.0 3.0 0.0; 0.0 0.0 4.0; 0.0 0.0 0.0]

    # PointSet is a 1-based conversion boundary: offset coords are accepted and stripped.
    ps_ref = PointSet(coords_plain)
    ps_off = PointSet(OffsetArray(coords_plain, 0:2, 0:2))
    @test axes(ps_off.coords) == axes(ps_ref.coords)
    @test ps_off.coords == ps_ref.coords

    # transform_columns(Translation, ...) declares require_one_based_indexing.
    trl = Translation(SVector(1.0, 0.0, 0.0))
    @test GMA.transform_columns(trl, coords_plain) isa Matrix
    @test_throws "offset arrays are not supported" GMA.transform_columns(trl, OffsetArray(coords_plain, 0:2, 0:2))

    # icp and iterative_hungarian with raw AbstractMatrix inputs declare require_one_based_indexing;
    # the PointSet overloads (which always hold 1-based coords) continue to work.
    P = [0.0 3.0 0.0; 0.0 0.0 4.0; 0.0 0.0 0.0]
    Q = [1.0 1.0 1.0; 1.0 -2.0 1.0; 1.0 1.0 -3.0]
    @test_throws "offset arrays are not supported" GMA.icp(OffsetArray(P, 0:2, 0:2), OffsetArray(Q, 0:2, 0:2))
    @test_throws "offset arrays are not supported" GMA.iterative_hungarian(OffsetArray(P, 0:2, 0:2), OffsetArray(Q, 0:2, 0:2))
    xset = PointSet(P); yset = PointSet(Q)
    @test GMA.icp(xset, yset) isa Vector{<:Tuple{Int, Int}}
    @test GMA.iterative_hungarian(xset, yset) isa Vector{<:Tuple{Int, Int}}
end

@testset "TIV alignment with interactions" begin
    models = JLD2.load(joinpath(@__DIR__, "..", "assets", "test_models.jld2"))["models"]
    x = models[1]

    # tivgmm on a labeled GMM keeps each TIV's endpoint widths, weights, and labels;
    # zero-length TIVs (i == j) are excluded as rotation-independent
    tiv = GMA.tivgmm(x)
    @test tiv isa GMA.TIVGMM{3, Float64, Symbol}
    @test length(tiv) == length(x)^2 - length(x)
    @test length(GMA.tivgmm(x, 3.0)) == 3 * length(x)                  # radius cutoff limits the count
    # each TIV's mean and endpoint data are consistent with a pair of source features
    @test all(eachindex(tiv.gaussians)) do i
        any(Iterators.product(pairs(x.gaussians), pairs(x.gaussians))) do ((a, ga), (b, gb))
            tiv.gaussians[i].μ == ga.μ - gb.μ &&
                (tiv.headσ[i], tiv.headϕ[i], tiv.headlabels[i]) == (ga.σ, ga.ϕ, x.labels[a]) &&
                (tiv.tailσ[i], tiv.tailϕ[i], tiv.taillabels[i]) == (gb.σ, gb.ϕ, x.labels[b])
        end
    end

    # two-term pairwise constants: hand-computed on a two-feature toy
    ga = GMA.IsotropicGaussian(SVector(0.0, 0.0, 0.0), 0.5, 2.0)
    gb = GMA.IsotropicGaussian(SVector(1.0, 0.0, 0.0), 1.0, 3.0)
    toy = LabeledIsotropicGMM([ga, gb], [:A, :B])
    ttoy = GMA.tivgmm(toy)
    toyints = Dict((:A, :A) => 1.0, (:B, :B) => 1.0, (:A, :B) => 0.25)
    tpσ, tpϕ = GMA.tiv_pairwise_consts(ttoy, ttoy, toyints)
    i_ab = findfirst(==((:A, :B)), collect(zip(ttoy.headlabels, ttoy.taillabels)))
    let i = i_ab, j = i_ab
        s_h = 2 * 0.5^2                              # head-head: both endpoints are feature a
        s_t = 2 * 1.0^2                              # tail-tail: both endpoints are feature b
        s_sum = s_h + s_t
        @test tpσ[i, j] ≈ SVector(s_sum^2 / s_h, s_sum^2 / s_t)
        @test tpϕ[i, j] ≈ SVector(1.0 * 2.0^2, 1.0 * 3.0^2)
    end

    # direction sensitivity: an antiparallel match pairs head with tail, so its terms carry the
    # cross coefficient instead of the self coefficients
    i_ba = findfirst(==((:B, :A)), collect(zip(ttoy.headlabels, ttoy.taillabels)))
    @test tpϕ[i_ab, i_ba] ≈ SVector(0.25 * 2.0 * 3.0, 0.25 * 3.0 * 2.0)

    # with no interactions dictionary, only equal-label endpoint pairs contribute with
    # coefficient 1 (matching the labeled `pairwise_consts` default)
    self_toy = Dict((:A, :A) => 1.0, (:B, :B) => 1.0)
    @test GMA.tiv_pairwise_consts(ttoy, ttoy, nothing) == GMA.tiv_pairwise_consts(ttoy, ttoy, self_toy)
    @test GMA.tiv_pairwise_consts(ttoy, ttoy, nothing)[2][i_ab, i_ba] == SVector(0.0, 0.0)

    # the two-term kernel sums per-term overlaps, and its bounds sandwich the objective
    @test overlap(0.7, SVector(2.0, 4.0), SVector(1.5, -0.5)) ≈
        overlap(0.7, 2.0, 1.5) + overlap(0.7, 4.0, -0.5)
    let block = UncertaintyRegion(RotationVec(0.1, -0.2, 0.3), SVector(0.0, 0.0, 0.0), 0.3, 0.0)
        mixσ, mixϕ = SVector(2.0, 4.0), SVector(1.5, -0.5)
        lb, ub = gauss_l2_bounds(ga, gb, block, mixσ, mixϕ)
        obj_center = -overlap(sum(abs2, block.R * ga.μ - gb.μ), mixσ, mixϕ)
        @test ub ≈ obj_center
        @test lb <= ub
    end

    # repulsion penalizes: a doubly-repulsive TIV pair has both term weights negative,
    # unlike a product of coefficients, for which the signs would cancel
    rep = Dict((:A, :A) => -1.0, (:B, :B) => -0.5)
    reppϕ = GMA.tiv_pairwise_consts(ttoy, ttoy, rep)[2]
    @test all(<(0), reppϕ[i_ab, i_ab])

    # end-to-end: a known rotation is recovered, and interactions are honored in both stages
    Rtrue = RotationVec(0.5, -0.3, 0.8)
    labels = sort(unique(x.labels))
    self_interactions = Dict((l, l) => 1.0 for l in labels)
    res = tiv_gogma_align(Rtrue * x, x; interactions = self_interactions, cutoff_x = 3.0, cutoff_y = 3.0, maxsplits = 2.0e3)
    Rfound = RotationVec(res.tform_params[1:3]...)
    @test rad2deg(rotation_angle(Rtrue * Rfound)) < 1.0e-2             # recovers the rotation
    @test res.upperbound ≈ -overlap(x, x; interactions = self_interactions) rtol = 1.0e-3

    # half-matches steer the rotation search: with a single interacting feature amid inert
    # ones, every TIV has a non-interacting endpoint, so a per-TIV coefficient *product*
    # would zero the entire rotation objective; summed endpoint terms recover the rotation
    # from the interacting-feature halves alone
    hm_pts = [SVector(0.0, 0.0, 0.0), SVector(2.0, 0.0, 0.0), SVector(0.0, 3.0, 0.5), SVector(-1.0, -1.0, 2.0)]
    hm = LabeledIsotropicGMM([IsotropicGaussian(p, 0.4, 1.0) for p in hm_pts], [:A, :B, :B, :B])
    hm_ints = Dict((:A, :A) => 1.0)
    hmpϕ = GMA.tiv_pairwise_consts(GMA.tivgmm(hm), GMA.tivgmm(hm), hm_ints)[2]
    @test all(v -> iszero(v[1]) || iszero(v[2]), hmpϕ)                 # every pair product is zero
    @test any(v -> !iszero(v), hmpϕ)                                   # but half-matches remain
    hm_res = tiv_gogma_align(Rtrue * hm, hm; interactions = hm_ints, maxsplits = 5.0e3)
    @test rad2deg(rotation_angle(Rtrue * RotationVec(hm_res.tform_params[1:3]...))) < 1.0e-2

    # coplanar models: a 180° spin about the plane normal maps every TIV to its negative, so
    # the unlabeled TIV objective is exactly degenerate; endpoint labels break the tie
    pl_pts = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.5, 0.0), SVector(0.3, 1.2, 0.0), SVector(-0.4, -0.9, 0.0)]
    pl = LabeledIsotropicGMM([IsotropicGaussian(p, 0.5, 1.0) for p in pl_pts], [:A, :A, :B, :B])
    pl_tiv = GMA.tivgmm(pl)
    pl_plain = GMA.tivgmm(IsotropicGMM(pl.gaussians))
    spin = RotationVec(0.0, 0.0, π)
    @test overlap(spin * pl_plain, pl_plain) ≈ overlap(pl_plain, pl_plain)
    plpσ, plpϕ = GMA.tiv_pairwise_consts(pl_tiv, pl_tiv, nothing)
    @test overlap(spin * pl_tiv, pl_tiv, plpσ, plpϕ) < overlap(pl_tiv, pl_tiv, plpσ, plpϕ)

    # the interactions keyword is threaded through without breaking the unlabeled path
    @test isfinite(tiv_gogma_align(Rtrue * IsotropicGMM(x.gaussians), IsotropicGMM(x.gaussians); cutoff_x = 3.0, cutoff_y = 3.0, maxsplits = 2.0e3).upperbound)
end

@testset "StackedLabeledGaussian and StackedLabeledIsotropicGMM" begin
    g = StackedLabeledGaussian([0.0, 0.0, 1.0], [0.5, 0.8], [1.0, 2.0], [:a, :b])
    @test g isa StackedLabeledGaussian{3, Float64, 2, Symbol}
    @test g isa GMA.AbstractIsotropicGaussian{3, Float64}
    @test g.μ === SVector(0.0, 0.0, 1.0)
    @test g.σ === SVector(0.5, 0.8) && g.ϕ === SVector(1.0, 2.0) && g.labels === SVector(:a, :b)

    # point evaluation is the sum of the slot Gaussians
    pos = [0.3, -0.2, 0.4]
    @test g(pos) ≈ sum(ϕ * exp(-sum(abs2, pos - g.μ) / (2σ^2)) for (σ, ϕ) in zip(g.σ, g.ϕ))

    # numeric promotion in the outer constructor
    @test StackedLabeledGaussian([0, 0, 1], [1, 1], [1.0f0, 2.0f0], [:a, :b]) isa StackedLabeledGaussian{3, Float32, 2, Symbol}

    # fail-fast validation
    @test_throws DimensionMismatch StackedLabeledGaussian([0.0, 0.0, 1.0], [0.5], [1.0, 2.0], [:a, :b])
    @test_throws "all slot widths must be positive" StackedLabeledGaussian([0.0, 0.0, 1.0], [0.5, 0.0], [1.0, 0.0], [:a, :a])
    @test_throws "all slot widths must be positive" StackedLabeledGaussian([0.0, 0.0, 1.0], [0.5, -1.0], [1.0, 2.0], [:a, :b])

    # single-slot lift from a plain Gaussian
    lifted = StackedLabeledGaussian(IsotropicGaussian([1.0, 0.0, 0.0], 0.7, 1.5), :c)
    @test lifted isa StackedLabeledGaussian{3, Float64, 1, Symbol}
    @test lifted.σ == SVector(0.7) && lifted.labels == SVector(:c)

    # copy, convert, promote
    @test StackedLabeledGaussian(g) == g
    @test convert(StackedLabeledGaussian{3, Float32, 2, Symbol}, g) isa StackedLabeledGaussian{3, Float32, 2, Symbol}
    @test promote_type(typeof(g), StackedLabeledGaussian{3, Float32, 2, Symbol}) === typeof(g)

    g2 = StackedLabeledGaussian([1.0, 1.0, 0.0], [0.6, 1.0], [1.5, 0.0], [:a, :a])
    gmm = StackedLabeledIsotropicGMM([g, g2])
    @test gmm isa StackedLabeledIsotropicGMM{3, Float64, 2, Symbol}
    @test gmm isa GMA.AbstractStackedLabeledIsotropicGMM{3, Float64, 2, Symbol}
    @test gmm isa GMA.AbstractIsotropicGMM{3, Float64}
    @test !(gmm isa GMA.AbstractLabeledIsotropicGMM)
    @test length(gmm) == 2
    @test gmm[2] == g2
    @test collect(gmm) == [g, g2]
    @test eltype(gmm) === eltype(typeof(gmm)) === StackedLabeledGaussian{3, Float64, 2, Symbol}
    @test isempty(StackedLabeledIsotropicGMM{3, Float64, 2, Symbol}())
    @test StackedLabeledIsotropicGMM(gmm).gaussians == gmm.gaussians
    @test convert(StackedLabeledIsotropicGMM{3, Float32, 2, Symbol}, gmm) isa StackedLabeledIsotropicGMM{3, Float32, 2, Symbol}
    @test gmm(pos) ≈ g(pos) + g2(pos)

    # show smoke tests
    @test occursin("labels", sprint(show, g))
    @test occursin("StackedLabeledGaussian", sprint(show, gmm))

    # accessors: weights sum slot amplitudes; widths are amplitude-weighted RMS widths
    @test GMA.coords(gmm) == [g.μ g2.μ]
    @test GMA.weights(gmm) == [3.0, 1.5]
    @test GMA.widths(gmm) ≈ [sqrt((1.0 * 0.5^2 + 2.0 * 0.8^2) / 3.0), 0.6]
    allpad = StackedLabeledIsotropicGMM([StackedLabeledGaussian([0.0, 0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [:a, :a])])
    @test GMA.weights(allpad) == [0.0]
    @test GMA.widths(allpad) == [0.0]

    # per-point construction with automatic padding
    padded = StackedLabeledIsotropicGMM(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.5, 0.8], [0.9]], [[1.0, 2.0], [1.5]], [[:a, :b], [:c]]
    )
    @test padded isa StackedLabeledIsotropicGMM{3, Float64, 2, Symbol}
    @test padded.gaussians[2].σ == SVector(0.9, 1.0)
    @test padded.gaussians[2].ϕ == SVector(1.5, 0.0)
    @test padded.gaussians[2].labels == SVector(:c, :c)
    @test_throws DimensionMismatch StackedLabeledIsotropicGMM([[0.0, 0.0, 0.0]], [[0.5]], [[1.0, 2.0]], [[:a]])
    @test_throws "has no features" StackedLabeledIsotropicGMM([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.5], Float64[]], [[1.0], Float64[]], [[:a], Symbol[]])

    # single-slot lift from a labeled GMM (no grouping)
    labgmm = LabeledIsotropicGMM([IsotropicGaussian([0.0, 0.0, 0.0], 0.5, 1.0), IsotropicGaussian([0.0, 0.0, 0.0], 0.8, 2.0)], [:a, :b])
    lifted_gmm = StackedLabeledIsotropicGMM(labgmm)
    @test lifted_gmm isa StackedLabeledIsotropicGMM{3, Float64, 1, Symbol}
    @test length(lifted_gmm) == 2

    # stackedgmm groups exactly equal means and pads shorter groups
    grouped = stackedgmm(labgmm)
    @test grouped isa StackedLabeledIsotropicGMM{3, Float64, 2, Symbol}
    @test length(grouped) == 1
    @test grouped.gaussians[1].σ == SVector(0.5, 0.8)
    @test grouped.gaussians[1].labels == SVector(:a, :b)
end

@testset "stacked transformations" begin
    g = StackedLabeledGaussian([0.0, 0.0, 1.0], [0.5, 0.8], [1.0, 2.0], [:a, :b])
    gmm = StackedLabeledIsotropicGMM([g, StackedLabeledGaussian([1.0, 1.0, 0.0], [0.6, 1.0], [1.5, 0.0], [:a, :a])])
    R = RotationVec(0.4, -0.3, 0.7)
    T = SVector(1.0, -2.0, 0.5)
    affine = AffineMap(R, T)

    tg = affine(g)
    @test tg isa typeof(g)
    @test tg.μ ≈ R * g.μ + T
    @test tg.σ == g.σ && tg.ϕ == g.ϕ && tg.labels == g.labels
    @test (@inferred affine(g)) isa typeof(g)
    @test ((g + T) - T).μ ≈ g.μ

    tgmm = affine(gmm)
    @test tgmm isa typeof(gmm)
    @test (@inferred affine(gmm)) isa typeof(gmm)
    @test all(tG.σ == G.σ && tG.ϕ == G.ϕ && tG.labels == G.labels for (tG, G) in zip(tgmm.gaussians, gmm.gaussians))
    @test GMA.coords(tgmm) ≈ R * GMA.coords(gmm) .+ T

    # overlap is invariant under a shared rigid transformation
    other = StackedLabeledIsotropicGMM([StackedLabeledGaussian([0.5, 0.2, -0.3], [0.7, 0.9], [1.0, 0.5], [:a, :b])])
    @test overlap(affine(gmm), affine(other)) ≈ overlap(gmm, other)
end

@testset "stacked equals duplicated labeled model" begin
    # a labeled model with duplicate-mean groups of sizes 3, 2, and 1, including a negative
    # amplitude and label pairs with negative or absent interaction coefficients
    pts = [SVector(0.0, 0.0, 0.0), SVector(3.0, 0.0, 0.0), SVector(0.0, 4.0, 0.0)]
    feats = [
        [(0.5, 1.0, :a), (0.8, 2.0, :b), (1.1, -0.5, :c)],
        [(0.6, 1.5, :a), (0.9, 0.7, :c)],
        [(1.0, 1.2, :b)],
    ]
    gaussians = IsotropicGaussian{3, Float64}[]
    labels = Symbol[]
    for (p, fs) in zip(pts, feats), (σ, ϕ, lab) in fs
        push!(gaussians, IsotropicGaussian(p, σ, ϕ))
        push!(labels, lab)
    end
    lab_x = LabeledIsotropicGMM(gaussians, labels)
    stk_x = stackedgmm(lab_x)
    @test stk_x isa StackedLabeledIsotropicGMM{3, Float64, 3, Symbol}
    @test length(stk_x) == 3 && length(lab_x) == 6

    tform = AffineMap(RotationVec(0.3, -0.2, 0.5), SVector(1.0, -1.0, 2.0))
    lab_y, stk_y = tform(lab_x), stackedgmm(tform(lab_x))
    interactions = Dict((:a, :a) => 1.0, (:b, :b) => 0.5, (:a, :c) => -0.3)

    # overlaps match, with and without interactions
    @test overlap(stk_x, stk_y; interactions) ≈ overlap(lab_x, lab_y; interactions) rtol = 1e-12
    @test overlap(stk_x, stk_y) ≈ overlap(lab_x, lab_y) rtol = 1e-12

    # pairwise constants: one SVector{Lx*Ly} entry per mean pair, zero weight for padded
    # and non-interacting slot pairs
    pσ, pϕ = GMA.pairwise_consts(stk_x, stk_y, interactions)
    @test pσ isa Matrix{SVector{9, Float64}} && pϕ isa Matrix{SVector{9, Float64}}
    @test size(pσ) == (3, 3)
    # point 3 has one feature (:b): against point 3 only the (1, 1) slot pair can interact
    @test count(!iszero, pϕ[3, 3]) == 1
    @test pϕ[3, 3][1] ≈ 0.5 * 1.2 * 1.2
    @test pσ[3, 3][1] ≈ 1.0^2 + 1.0^2
    # slot-pair weights carry the interaction coefficients: (:a, :c) → -0.3
    g1, g1y = stk_x.gaussians[1], stk_y.gaussians[2]
    m_ac = findfirst(m -> (fldmod1(m, 3) == (1, 2)), 1:9)  # slot 1 (:a) of x vs slot 2 (:c) of y
    @test pϕ[1, 2][m_ac] ≈ -0.3 * g1.ϕ[1] * g1y.ϕ[2]

    # the mean-pair distance bounds are shared, so bounds match the duplicated model exactly
    plσ, plϕ = GMA.pairwise_consts(lab_x, lab_y, interactions)
    R, T = RotationVec(0.1, 0.2, -0.1), SVector(0.5, -0.3, 0.2)
    for (σᵣ, σₜ) in ((0.5, 0.5), (0.1, 1.0), (1.0, 0.05))
        b_lab = gauss_l2_bounds(lab_x, lab_y, R, T, σᵣ, σₜ, plσ, plϕ)
        b_stk = gauss_l2_bounds(stk_x, stk_y, R, T, σᵣ, σₜ, pσ, pϕ)
        @test all(isapprox.(b_lab, b_stk; atol = 1e-12))
        @test b_stk[1] <= -overlap(AffineMap(R, T)(stk_x), stk_y, pσ, pϕ) <= b_stk[2] + 1e-12
    end

    # pσ/pϕ index the Gaussians from 1
    @test_throws ArgumentError overlap(stk_x, stk_y, OffsetArray(pσ, -1, -1), OffsetArray(pϕ, -1, -1))

    # Gaussian-pair overlap sums the slot cross-terms at the shared distance
    gx, gy = stk_x.gaussians[1], stk_y.gaussians[2]
    distsq = sum(abs2, gx.μ - gy.μ)
    manual = sum(
        GMA.interaction_coefficient(interactions, gx.labels[k], gy.labels[l]) * gx.ϕ[k] * gy.ϕ[l] *
            exp(-distsq / (2 * (gx.σ[k]^2 + gy.σ[l]^2)))
            for k in 1:3, l in 1:3
    )
    @test overlap(gx, gy; interactions) ≈ manual rtol = 1e-12
    @test overlap(gx, gy) ≈ sum(
        (gx.labels[k] == gy.labels[l]) * gx.ϕ[k] * gy.ϕ[l] * exp(-distsq / (2 * (gx.σ[k]^2 + gy.σ[l]^2)))
            for k in 1:3, l in 1:3
    ) rtol = 1e-12

    # forces match the duplicated model
    f_stk, f_lab = zeros(3), zeros(3)
    GMA.force!(f_stk, stk_x, stk_y; interactions)
    GMA.force!(f_lab, lab_x, lab_y; interactions)
    @test f_stk ≈ f_lab rtol = 1e-10

    # mixed stacked × labeled overlaps lift the labeled side
    @test overlap(stk_x, lab_y; interactions) ≈ overlap(lab_x, lab_y; interactions) rtol = 1e-12
    @test overlap(lab_x, stk_y; interactions) ≈ overlap(lab_x, lab_y; interactions) rtol = 1e-12

    # unlabeled models cannot be paired with stacked ones
    iso = IsotropicGMM([IsotropicGaussian(SVector(0.0, 0.0, 0.0), 1.0, 1.0)])
    @test_throws "wrap the IsotropicGMM in a LabeledIsotropicGMM" GMA.pairwise_consts(stk_x, iso)
    @test_throws "wrap the IsotropicGMM in a LabeledIsotropicGMM" GMA.pairwise_consts(iso, stk_x)
end

@testset "stacked TIV" begin
    # nonnegative amplitudes: TIV selection scores take geometric means of amplitudes
    pts = [SVector(0.0, 0.0, 0.0), SVector(3.0, 0.0, 0.0), SVector(0.0, 4.0, 0.0)]
    feats = [
        [(0.5, 1.0, :a), (0.8, 2.0, :b), (1.1, 0.5, :c)],
        [(0.6, 1.5, :a), (0.9, 0.7, :c)],
        [(1.0, 1.2, :b)],
    ]
    gaussians = IsotropicGaussian{3, Float64}[]
    labels = Symbol[]
    for (p, fs) in zip(pts, feats), (σ, ϕ, lab) in fs
        push!(gaussians, IsotropicGaussian(p, σ, ϕ))
        push!(labels, lab)
    end
    lab_x = LabeledIsotropicGMM(gaussians, labels)
    stk_x = stackedgmm(lab_x)
    tform = AffineMap(RotationVec(0.3, -0.2, 0.5), SVector(1.0, -1.0, 2.0))
    lab_y, stk_y = tform(lab_x), stackedgmm(tform(lab_x))
    interactions = Dict((:a, :a) => 1.0, (:b, :b) => 0.5, (:a, :c) => -0.3)

    # structure: one TIV per ordered pair of distinct stacked points, with slot-wise endpoint data
    tivs_x = GMA.tivgmm(stk_x, Inf)
    @test tivs_x isa GMA.StackedTIVGMM{3, Float64, 3, Symbol}
    @test length(tivs_x) == length(stk_x)^2 - length(stk_x)
    @test length(tivs_x.headσ) == length(tivs_x) && length(tivs_x.taillabels) == length(tivs_x)

    # a mean-duplicated model produces the same TIV overlap once its zero-length TIVs
    # (between co-located duplicate features) are removed: those pair overlaps are
    # rotation-independent constants that a stacked model deliberately omits
    tivd_full = GMA.tivgmm(lab_x, Inf)
    keep = [i for i in eachindex(tivd_full.gaussians) if !iszero(norm(tivd_full.gaussians[i].μ))]
    tivd_x = GMA.TIVGMM(
        tivd_full.gaussians[keep], tivd_full.headσ[keep], tivd_full.headϕ[keep], tivd_full.headlabels[keep],
        tivd_full.tailσ[keep], tivd_full.tailϕ[keep], tivd_full.taillabels[keep]
    )
    tivd_yfull = GMA.tivgmm(lab_y, Inf)
    keepy = [i for i in eachindex(tivd_yfull.gaussians) if !iszero(norm(tivd_yfull.gaussians[i].μ))]
    tivd_y = GMA.TIVGMM(
        tivd_yfull.gaussians[keepy], tivd_yfull.headσ[keepy], tivd_yfull.headϕ[keepy], tivd_yfull.headlabels[keepy],
        tivd_yfull.tailσ[keepy], tivd_yfull.tailϕ[keepy], tivd_yfull.taillabels[keepy]
    )
    tivs_y = GMA.tivgmm(stk_y, Inf)

    dpσ, dpϕ = GMA.tiv_pairwise_consts(tivd_x, tivd_y, interactions)
    spσ, spϕ = GMA.tiv_pairwise_consts(tivs_x, tivs_y, interactions)
    @test spσ isa Matrix{SVector{162, Float64}}  # 2 ⋅ 3² ⋅ 3² terms per TIV pair
    for R in (RotationVec(0.0, 0.0, 0.0), RotationVec(0.3, -0.2, 0.5), RotationVec(-1.0, 0.4, 0.9))
        @test overlap(R * tivs_x, tivs_y, spσ, spϕ) ≈ overlap(R * tivd_x, tivd_y, dpσ, dpϕ) rtol = 1e-10
    end
    dpσ0, dpϕ0 = GMA.tiv_pairwise_consts(tivd_x, tivd_y, nothing)
    spσ0, spϕ0 = GMA.tiv_pairwise_consts(tivs_x, tivs_y, nothing)
    R = RotationVec(0.3, -0.2, 0.5)
    @test overlap(R * tivs_x, tivs_y, spσ0, spϕ0) ≈ overlap(R * tivd_x, tivd_y, dpσ0, dpϕ0) rtol = 1e-10

    # full TIV alignment recovers the self-overlap, matching the labeled model
    res_stk = tiv_gogma_align(stk_x, stk_y; interactions, maxsplits = 2e3)
    res_lab = tiv_gogma_align(lab_x, lab_y; interactions, maxsplits = 2e3)
    @test res_stk.upperbound ≈ res_lab.upperbound rtol = 1e-6
    @test res_stk.upperbound ≈ -overlap(stk_x, stk_x; interactions) rtol = 1e-4

    # mixed stacked × labeled TIV alignment lifts the labeled side
    res_mixed = tiv_gogma_align(stk_x, lab_y; interactions, maxsplits = 1e3)
    @test isfinite(res_mixed.upperbound)
end

@testset "stacked GOGMA and local alignment" begin
    pts = [SVector(0.0, 0.0, 0.0), SVector(3.0, 0.0, 0.0), SVector(0.0, 4.0, 0.0)]
    feats = [
        [(0.5, 1.0, :a), (0.8, 2.0, :b), (1.1, 0.5, :c)],
        [(0.6, 1.5, :a), (0.9, 0.7, :c)],
        [(1.0, 1.2, :b)],
    ]
    μs = eltype(pts)[]
    σs, ϕs, labelss = Vector{Float64}[], Vector{Float64}[], Vector{Symbol}[]
    for (p, fs) in zip(pts, feats)
        push!(μs, p)
        push!(σs, [f[1] for f in fs])
        push!(ϕs, [f[2] for f in fs])
        push!(labelss, [f[3] for f in fs])
    end
    stk_x = StackedLabeledIsotropicGMM(μs, σs, ϕs, labelss)
    tform = AffineMap(RotationVec(0.3, -0.2, 0.5), SVector(1.0, -1.0, 2.0))
    stk_y = tform(stk_x)
    interactions = Dict((:a, :a) => 1.0, (:b, :b) => 0.5, (:a, :c) => -0.3)

    res = gogma_align(stk_x, stk_y; interactions, maxsplits = 2e3)
    @test res.upperbound ≈ -overlap(stk_x, stk_x; interactions) rtol = 1e-4

    resr = GMA.rot_gogma_align(stk_x, stk_x; interactions, maxsplits = 500)
    rest = GMA.trl_gogma_align(stk_x, stk_x; interactions, maxsplits = 500)
    @test resr.upperbound ≈ -overlap(stk_x, stk_x; interactions) rtol = 1e-4
    @test rest.upperbound ≈ -overlap(stk_x, stk_x; interactions) rtol = 1e-4

    # rocs_align consumes the stacked weights/widths accessors
    rres = rocs_align(stk_x, stk_y)
    @test rres.minimum ≈ -overlap(stk_x, stk_x) atol = 1e-9

    # autodiff: gradients through padded slots are finite and match the duplicated model
    lab_equiv = LabeledIsotropicGMM(
        [IsotropicGaussian(p, σ, ϕ) for (p, fs) in zip(pts, feats) for (σ, ϕ, _) in fs],
        [lab for fs in feats for (_, _, lab) in fs]
    )
    lab_yequiv = tform(lab_equiv)
    pσ, pϕ = GMA.pairwise_consts(stk_x, stk_y, interactions)
    plσ, plϕ = GMA.pairwise_consts(lab_equiv, lab_yequiv, interactions)
    X = [0.1, -0.2, 0.3, 0.5, -0.1, 0.2]
    grad_stk = ForwardDiff.gradient(X -> overlapobj(X, stk_x, stk_y, pσ, pϕ), X)
    grad_lab = ForwardDiff.gradient(X -> overlapobj(X, lab_equiv, lab_yequiv, plσ, plϕ), X)
    @test all(isfinite, grad_stk)
    @test grad_stk ≈ grad_lab rtol = 1e-8

    res_ad = gogma_align(stk_x, stk_y; interactions, autodiff = AutoForwardDiff(), maxsplits = 100)
    @test isfinite(res_ad.upperbound)
end
