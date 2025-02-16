using GaussianMixtureAlignment
using Test
using IntervalSets
using LinearAlgebra
using StaticArrays
using Rotations
using CoordinateTransformations
using ForwardDiff

using GaussianMixtureAlignment: UncertaintyRegion, RotationRegion, TranslationRegion
using GaussianMixtureAlignment: tight_distance_bounds, loose_distance_bounds, gauss_l2_bounds, subranges, sqrt3, UncertaintyRegion, subregions, branchbound, rocs_align, overlap, gogma_align, tiv_gogma_align, tiv_goih_align, overlapobj
const GMA = GaussianMixtureAlignment

@testset "search space bounds" begin
    μx = SVector(3,0,0)
    μy = SVector(-4,0,0)
    σ = ϕ = 1
    ndims = 3
    sqrt2 = √2

    x, y = IsotropicGaussian(μx, σ, ϕ), IsotropicGaussian(μy, σ, ϕ)

    ### tight_distance_bounds
    # anti-aligned (no rotation) and aligned (180 degree rotation)
    lbdist, ubdist = tight_distance_bounds(x,y,π,0)
    @test ubdist ≈ 7
    @test lbdist ≈ 1
    # region with closest alignment at 90 degree rotation
    lbdist, ubdist = tight_distance_bounds(x,y,π/2/sqrt3,0)
    @test lbdist ≈ 5
    # translation region centered at origin
    lbdist, ubdist = tight_distance_bounds(x,y,0,1/√3)
    @test lbdist ≈ 6
    @test ubdist ≈ 7
    # centered at x = 1
    lbdist, ubdist = tight_distance_bounds(x+SVector(1,0,0),y,0,1/sqrt3)
    @test lbdist ≈ 7
    @test ubdist ≈ 8

    ### loose_distance_bounds

    ### Gaussian L2 bounds
    # rotation distances, no translation
    # anti-aligned (no rotation) and aligned (180 degree rotation)
    lb, ub = gauss_l2_bounds(x,y,RotationRegion(0.))
    @test lb ≈ -GMA.overlap(7^2,2*σ^2,ϕ*ϕ) atol=1e-16
    @test ub ≈ -GMA.overlap(7^2,2*σ^2,ϕ*ϕ)
    lb, ub = gauss_l2_bounds(x,y,RotationRegion(RotationVec(0.,0.,π),SVector{3}(0.,0.,0.),0.))
    @test lb ≈ ub ≈ -GMA.overlap(1,2*σ^2,ϕ*ϕ)
    # region with closest alignment at 90 degree rotation
    lb = gauss_l2_bounds(x,y,RotationRegion(π/2/sqrt3))[1]
    @test lb ≈ -GMA.overlap(5^2,2*σ^2,ϕ*ϕ)
    lb = gauss_l2_bounds(x,y,RotationRegion(RotationVec(0,0,π/4),SVector{3}(0.,0.,0.),π/4/(sqrt3)))[1]
    @test lb ≈ -GMA.overlap(5^2,2*σ^2,ϕ*ϕ)

    # translation distance, no rotation
    # translation region centered at origin
    lb, ub = gauss_l2_bounds(x,y,TranslationRegion(1/sqrt3))
    @test lb ≈ -GMA.overlap(6^2,2*σ^2,ϕ*ϕ)
    @test ub ≈ -GMA.overlap(7^2,2*σ^2,ϕ*ϕ)
    # centered with translation of 1 in +x
    lb, ub = gauss_l2_bounds(x+SVector(1,0,0),y,TranslationRegion(1/sqrt3))
    @test lb ≈ -GMA.overlap(7^2,2*σ^2,ϕ*ϕ)
    @test ub ≈ -GMA.overlap(8^2,2*σ^2,ϕ*ϕ)
    # centered with translation of 3 in +y
    lb, ub = gauss_l2_bounds(x+SVector(0,3,0),y,TranslationRegion(1/sqrt3))
    @test lb ≈ -GMA.overlap((√(58)-1)^2,2*σ^2,ϕ*ϕ)
    @test ub ≈ -GMA.overlap(58,2*σ^2,ϕ*ϕ)

end

@testset "divide a searchspace" begin
    blk1 = NTuple{6,Tuple{Float64,Float64}}(((-π,π), (-π,π), (-π,π), (-1,1), (-1,1), (-1,1)))
    subblks1 = subranges(blk1, 2)
    @test length(subblks1) == 2^6
    for i=1:6
        intv = OpenInterval(0,0)
        for sblk in subblks1
            rng = sblk[i]
            intv = union(intv, ClosedInterval(rng[1], rng[2]))
        end
        @test intv == ClosedInterval(blk1[i][1], blk1[i][2])
    end
    # @show subblks1[length(subblks1)]

    blk2 = NTuple{6,Tuple{Float64,Float64}}(((0,π), (0,π), (0,π), (0,1), (0,1), (0,1)))
    subblks2 = subranges(blk2, 4)
    @test length(subblks2) == 4^6
    for i=1:6
        intv = OpenInterval(0,0)
        for sblk in subblks2
            rng = sblk[i]
            intv = union(intv, ClosedInterval(rng[1], rng[2]))
        end
        @test intv == ClosedInterval(blk2[i][1], blk2[i][2])
    end
end

@testset "GMM interface" begin
    tetrahedral = [
        [0.,0.,1.],
        [sqrt(8/9), 0., -1/3],
        [-sqrt(2/9),sqrt(2/3),-1/3],
        [-sqrt(2/9),-sqrt(2/3),-1/3]
    ]
    ch_g = IsotropicGaussian(tetrahedral[1], 1.0, 1.0)
    s_gs = [IsotropicGaussian(x, 0.5, 1.0) for (i,x) in enumerate(tetrahedral)]
    gmm = IsotropicGMM(s_gs)
    @test length(gmm) == 4
    @test gmm[2] == s_gs[2]
    @test collect(gmm) == s_gs       # tests iterate
    @test eltype(gmm) === eltype(typeof(gmm)) === IsotropicGaussian{3,Float64}
    @test convert(IsotropicGMM{3,Float32}, gmm) isa IsotropicGMM{3,Float32}
    @test_throws DimensionMismatch convert(IsotropicGMM{2,Float64}, gmm)
    mgmmx = IsotropicMultiGMM(Dict(
        :positive => IsotropicGMM([ch_g]),
        :steric => gmm
    ))
    @test keys(mgmmx) == Set([:positive, :steric])
    @test keytype(mgmmx) === keytype(typeof(mgmmx)) === Symbol
    @test valtype(mgmmx) === valtype(typeof(mgmmx)) === IsotropicGMM{3,Float64}
    @test eltype(mgmmx) === eltype(typeof(mgmmx)) === Pair{Symbol, IsotropicGMM{3,Float64}}
    @test length(mgmmx) == 2
    @test length(mgmmx[:steric]) == 4
    @test mgmmx[:steric][2] == s_gs[2]
    @test collect(mgmmx) == collect(mgmmx.gmms) # tests iterate
    @test get!(valtype(mgmmx), mgmmx, :positive) == mgmmx[:positive]
    gmm = get!(valtype(mgmmx), mgmmx, :acceptor)
    @test isempty(gmm) && gmm isa IsotropicGMM{3,Float64}
    push!(gmm, ch_g)
    @test length(gmm) == 1
    pop!(gmm)
    @test isempty(gmm)
    push!(gmm, ch_g)
    empty!(gmm)
    @test isempty(gmm)
    delete!(mgmmx, :acceptor)
    @test !haskey(mgmmx, :acceptor)
    empty!(mgmmx)
    @test isempty(mgmmx)
end

@testset "bounds for shrinking searchspace around an optimum" begin
    # two sets of points, each forming a 3-4-5 triangle
    xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]
    ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
    σ = ϕ = 1.
    gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
    gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

    # aligning a GMM to itself
    bigblock = UncertaintyRegion(gmmx, gmmx)
    (lb,ub) = gauss_l2_bounds(gmmx, gmmx, bigblock)
    @test lb ≈ -length(gmmx.gaussians)^2 # / √(4π)^3

    blk = UncertaintyRegion(RotationVec{Float64}(π/2, π/2, π/2), SVector{3,Float64}(1.0, 1.0, 1.0), π/2, 1.0)
    (lb,ub) = gauss_l2_bounds(gmmx, gmmx, blk)
    for i = 1:20
        blk = subregions(blk)[1]
        (newlb,newub) = gauss_l2_bounds(gmmx, gmmx, blk)
        @test newlb >= lb
        @test newub <= ub
        (lb,ub) = (newlb, newub)
    end
end

@testset "GOGMA runs without errors" begin
    # two sets of points, each forming a 3-4-5 triangle
    xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]
    ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
    σ = ϕ = 1.
    gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
    gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

    # make sure this runs without an error
    res1 = gogma_align(gmmx, gmmy; maxsplits=1E3)
    res2 = tiv_gogma_align(gmmx, gmmy; maxsplits=1E3)

    mgmmx = IsotropicMultiGMM(Dict(:x => gmmx, :y => gmmx))
    mgmmy = IsotropicMultiGMM(Dict(:x => gmmy, :y => gmmy))
    res3 = gogma_align(mgmmx, mgmmy; maxsplits=1E3)
    res4 = tiv_gogma_align(mgmmx, mgmmy)

    # ROCS alignment should work perfectly for these GMMs
    @test isapprox(rocs_align(gmmx, gmmy).minimum, -overlap(gmmx,gmmx); atol=1E-12)
end

@testset "Evaluation at a point" begin
    pts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]
    σ = ϕ = 1.
    gmm = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in pts])

    pt = [1, 1, 1]
    gpt = [g(pt) for g in gmm.gaussians]
    @test gpt ≈ [exp(-3/2), exp(-6/2), exp(-11/2)]
    @test gmm(pt) == sum(gpt)
end

@testset "MultiGMMs with interactions" begin
    tetrahedral = [
        [0.,0.,1.],
        [sqrt(8/9), 0., -1/3],
        [-sqrt(2/9),sqrt(2/3),-1/3],
        [-sqrt(2/9),-sqrt(2/3),-1/3]
    ]
    ch_g = IsotropicGaussian(tetrahedral[1], 1.0, 1.0)
    s_gs = [IsotropicGaussian(x, 0.5, 1.0) for (i,x) in enumerate(tetrahedral)]
    mgmmx = IsotropicMultiGMM(Dict(
        :positive => IsotropicGMM([ch_g]),
        :steric => IsotropicGMM(s_gs)
    ))
    mgmmy = IsotropicMultiGMM(Dict(
        :negative => IsotropicGMM([ch_g]),
        :steric => IsotropicGMM(s_gs)
    ))
    # interaction validation
    interactions = Dict(
        (:positive, :negative) =>  1.0,
        (:negative, :positive) => 1.0,
        (:positive, :positive) => -1.0,
        (:negative, :negative) => -1.0,
        (:steric, :steric) => -1.0,
    )
    @test_throws AssertionError GMA.pairwise_consts(mgmmx, mgmmy, interactions)
    interactions = Dict(
        (:positive, :negative) =>  1.0,
        (:positive, :positive) => -1.0,
        (:negative, :negative) => -1.0,
        (:steric, :steric) => -1.0,
    )
    randtform = AffineMap(RotationVec(π*0.1rand(3)...), SVector{3}(0.1*rand(3)...))
    res = gogma_align(randtform(mgmmx), mgmmy; interactions=interactions, maxsplits=5e3, nextblockfun=GMA.randomblock)
end

@testset "Forces" begin
    μx = randn(SVector{3,Float64})
    μy = randn(SVector{3,Float64})
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

    tetrahedral = [
        [0.,0.,1.],
        [sqrt(8/9), 0., -1/3],
        [-sqrt(2/9),sqrt(2/3),-1/3],
        [-sqrt(2/9),-sqrt(2/3),-1/3]
    ]
    ch_g = IsotropicGaussian(tetrahedral[1], 1.0, 1.0)
    s_gs = [IsotropicGaussian(x, 0.5, 1.0) for (i,x) in enumerate(tetrahedral)]
    mgmmx = IsotropicMultiGMM(Dict(
        :positive => IsotropicGMM([ch_g]),
        :steric => IsotropicGMM(s_gs)
    ))
    mgmmy = IsotropicMultiGMM(Dict(
        :negative => IsotropicGMM([ch_g]),
        :steric => IsotropicGMM(s_gs)
    ))
    fliptform = AffineMap(RotationVec(π,0,0),[0,0,3]) ∘ AffineMap(RotationVec(0,0,π),[0,0,0])
    mgmmy = fliptform(mgmmy)
    interactions = Dict(
        (:positive, :negative) =>  1.0,
        (:positive, :positive) => -1.0,
        (:negative, :negative) => -1.0,
        (:steric, :steric) => -1.0,
    )
    f = zeros(3)
    force!(f, mgmmx, mgmmy; interactions=interactions)
    movlp(μ) = overlap(IsotropicMultiGMM(Dict(:positive => IsotropicGMM([ch_g + μ]),:steric => IsotropicGMM([g + μ for g in s_gs]))), mgmmy, nothing, nothing, interactions)
    @test f ≈ ForwardDiff.gradient(movlp, zeros(3))
end

@testset "GO-ICP and GO-IH run without errors" begin
    xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]
    ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]

    xset = PointSet(xpts);
    yset = PointSet(ypts);

    goicp_res = goicp_align(xset, yset)
    @test GaussianMixtureAlignment.transform_columns(goicp_res.tform, xset.coords) ≈ yset.coords
    goih_res = goih_align(xset, yset)
    @test GaussianMixtureAlignment.transform_columns(goih_res.tform, xset.coords) ≈ yset.coords

end

@testset "Kabsch" begin
    xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]
    ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]

    xset = PointSet(xpts, ones(3))
    yset = PointSet(ypts, ones(3))

    tform = kabsch(xset, yset)

    @test yset.coords ≈ tform(xset).coords
end

@testset "GO-ICP" begin
    ycoords = rand(3,50) * 5 .- 10;
    randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(5*rand(3)...))
    xcoords = GaussianMixtureAlignment.transform_columns(randtform, ycoords)

    xset = PointSet(xcoords, ones(size(xcoords,2)))
    yset = PointSet(xcoords, ones(size(ycoords,2)))

    res = goicp_align(yset, xset)
    @test res.lowerbound == 0
    @test res.upperbound < 1e-15
end

@testset "globally optimal iterative hungarian" begin
    ycoords = rand(3,5) * 5 .- 10;
    randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(5*rand(3)...))
    xcoords = GaussianMixtureAlignment.transform_columns(randtform, ycoords)

    xset = PointSet(xcoords, ones(size(xcoords,2)))
    yset = PointSet(xcoords, ones(size(ycoords,2)))

    res = goih_align(yset, xset)
    @test res.lowerbound == 0
    @test res.upperbound < 1e-15
end
