using GaussianMixtureAlignment
using Test
using IntervalSets
using LinearAlgebra
using StaticArrays
using Rotations
using CoordinateTransformations

using GaussianMixtureAlignment: tight_distance_bounds, loose_distance_bounds, gauss_l2_bounds, subranges, sqrt3, UncertaintyRegion, subregions, branchbound, rocs_align, overlap, gogma_align, tiv_gogma_align, overlapobj
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
    lb, ub = gauss_l2_bounds(x,y,π,0)
    @test lb ≈ -GMA.overlap(1,2*σ^2,ϕ*ϕ, 1.) atol=1e-16
    @test ub ≈ -GMA.overlap(7^2,2*σ^2,ϕ*ϕ, 1.)
    lb, ub = gauss_l2_bounds(RotationVec(0,0,π)*x,y,π,0)
    @test lb ≈ ub ≈ -GMA.overlap(1,2*σ^2,ϕ*ϕ, 1.)
    # region with closest alignment at 90 degree rotation
    lb = gauss_l2_bounds(x,y,π/2/sqrt3,0)[1]
    @test lb ≈ -GMA.overlap(5^2,2*σ^2,ϕ*ϕ, 1.)
    lb = gauss_l2_bounds(RotationVec(0,0,π/4)*x,y,π/4/(sqrt3),0)[1]
    @test lb ≈ -GMA.overlap(5^2,2*σ^2,ϕ*ϕ, 1.) 
    
    # translation distance, no rotation
    # translation region centered at origin
    lb, ub = gauss_l2_bounds(x,y,0,1/sqrt3)
    @test lb ≈ -GMA.overlap(6^2,2*σ^2,ϕ*ϕ, 1.)
    @test ub ≈ -GMA.overlap(7^2,2*σ^2,ϕ*ϕ, 1.)
    # centered with translation of 1 in +x
    lb, ub = gauss_l2_bounds(x+SVector(1,0,0),y,0,1/sqrt3)
    @test lb ≈ -GMA.overlap(7^2,2*σ^2,ϕ*ϕ, 1.)
    @test ub ≈ -GMA.overlap(8^2,2*σ^2,ϕ*ϕ, 1.)
    # centered with translation of 3 in +y 
    lb, ub = gauss_l2_bounds(x+SVector(0,3,0),y,0,1/sqrt3)
    @test lb ≈ -GMA.overlap((√(58)-1)^2,2*σ^2,ϕ*ϕ, 1.)
    @test ub ≈ -GMA.overlap(58,2*σ^2,ϕ*ϕ, 1.)

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

# @testset "ROCS alignment" begin
#     xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]] 
#     ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
#     σ = ϕ = 1.
#     gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
#     gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

    
# end

@testset "GOGMA runs without errors" begin
    # two sets of points, each forming a 3-4-5 triangle
    xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]] 
    ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
    σ = ϕ = 1.
    gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
    gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

    # make sure this runs without an error
    res1 = gogma_align(gmmx, gmmy; maxblocks=1E5)
    res2 = tiv_gogma_align(gmmx, gmmy; maxblocks=1E5)

    mgmmx = IsotropicMultiGMM(Dict(:x => gmmx, :y => gmmy))
    mgmmy = IsotropicMultiGMM(Dict(:y => gmmx, :x => gmmy))
    res3 = gogma_align(mgmmx, mgmmy; maxblocks=1E5)
    res4 = tiv_gogma_align(mgmmx, mgmmy)

    # ROCS alignment should work perfectly for these GMMs
    @test isapprox(rocs_align(gmmx, gmmy; objfun=overlapobj).minimum, -overlap(gmmx,gmmx); atol=1E-12)
end

@testset "GOGMA with directions" begin
    xpt = [1.,0.,0.]
    ypt = [1.,0.,0.]
    xdir = [0.,1.,0.]
    ydir = [1.,0.,0.]
    σ = ϕ = 1.
    x = IsotropicGaussian(xpt, σ, ϕ, [xdir])
    y = IsotropicGaussian(ypt, σ, ϕ, [ydir])
    @test gauss_l2_bounds(x,y,0.,0.) == (-0.5,-0.5)
    @test gauss_l2_bounds(x,y,π/(6*GMA.sqrt3),0.) == (-0.75,-0.5)
    @test gauss_l2_bounds(x,y,π/(2*GMA.sqrt3),0.) == (-1.0,-0.5)

    xpt = [1.,0.,0.]
    ypt = [1.,0.,0.]
    xdir = [-1.,0.,0.]
    ydir = [1.,0.,0.]
    σ = ϕ = 1.
    x = IsotropicGaussian(xpt, σ, ϕ, [xdir])
    y = IsotropicGaussian(ypt, σ, ϕ, [ydir])
    @test gauss_l2_bounds(x,y,0.,0.) == (0.,0.)
    

    xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]] 
    ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
    σ = ϕ = 1.
    xdirs = [[0.,-1.,0], [1.,0.,0.], [0.,1.,0.]]
    ydirs = [[0.,0.,1.], [0.,-1.,0.], [0.,0.,-1]]
    dgmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ, [xdirs[i]]) for (i,x) in enumerate(xpts)])
    dgmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ, [ydirs[i]]) for (i,y) in enumerate(ypts)])
    objminxy = tiv_gogma_align(dgmmx, dgmmy).upperbound
    objminxx = tiv_gogma_align(dgmmx, dgmmx).upperbound
    objminyy = tiv_gogma_align(dgmmy, dgmmy).upperbound
    @test objminxy ≈ objminxx ≈ objminyy

    randdgmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ, [2*rand(3).-1]) for (i,y) in enumerate(ypts)])
    randobjminxy = tiv_gogma_align(dgmmx, randdgmmy).upperbound
    @test randobjminxy > objminxy

    # when σ is very small, only very well aligned Gaussians will contribute to the overlap
    σ = 0.001
    ddgmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ, [xdirs[i], normalize(2*rand(3).-1)]) for (i,x) in enumerate(xpts)])
    ddgmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ, [ydirs[i]]) for (i,y) in enumerate(ypts)])
    ddobjmin = tiv_gogma_align(ddgmmx, ddgmmy).upperbound
    @test ddobjmin ≈ -3.0

    mgmmx = IsotropicMultiGMM(Dict(:one => IsotropicGMM([IsotropicGaussian(xpts[1], σ, ϕ, [xdirs[1]])]),
                                   :two => IsotropicGMM([IsotropicGaussian(xpts[2], σ, ϕ, [xdirs[2]])]),
                                   :three => IsotropicGMM([IsotropicGaussian(xpts[3], σ, ϕ, [xdirs[3]])])))
    mgmmy = IsotropicMultiGMM(Dict(:one => IsotropicGMM([IsotropicGaussian(ypts[1], σ, ϕ, [ydirs[1]])]),
                                   :two => IsotropicGMM([IsotropicGaussian(ypts[2], σ, ϕ, [ydirs[2]])]),
                                   :three => IsotropicGMM([IsotropicGaussian(ypts[3], σ, ϕ, [ydirs[3]])])))
    mobjminxy = tiv_gogma_align(mgmmx, mgmmy).upperbound
    @test mobjminxy ≈ -3.0
end

@testset "Kabsch" begin
    xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]] 
    ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]

    xset = PointSet(xpts, ones(3))
    yset = PointSet(ypts, ones(3))

    tform = kabsch(xset, yset)

    @test yset.coords ≈ tform(xset).coords
end

# @testset "ICP" begin
#     ycoords = rand(3,10) * 5 .- 10;
#     randtform = AffineMap(RotationVec(π/4*rand(3)...), SVector{3}(5*rand(3)...))
#     xcoords = randtform(ycoords)

#     matches = icp(ycoords, xcoords)
# end

@testset "ICP" begin
    for i=1:10
        ycoords = rand(3,10) * 5 .- 10;
        randtform = AffineMap(RotationVec(π/16*rand(3)...), SVector{3}(0*rand(3)...))
        xcoords = randtform(ycoords)

        matches = icp(ycoords, xcoords)

        @test([m[1] for m in matches] == [m[2] for m in matches])
    end
end

@testset "GO-ICP" begin
    ycoords = rand(3,5) * 5 .- 10;
    randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(5*rand(3)...))
    xcoords = randtform(ycoords)

    xset = PointSet(xcoords, ones(size(xcoords,2)))
    yset = PointSet(xcoords, ones(size(ycoords,2)))

    res = goicp_align(yset, xset)
    @test res.lowerbound == 0
    @test res.upperbound < 1e-15
end

@testset "Iterative Hungarian" begin
    for i=1:10
        ycoords = rand(3,10) * 5 .- 10;
        randtform = AffineMap(RotationVec(π/8*rand(3)...), SVector{3}(0*rand(3)...))
        xcoords = randtform(ycoords)

        matches = iterative_hungarian(ycoords, xcoords)

        @test([m[1] for m in matches] == [m[2] for m in matches])
    end
end

@testset "globally optimal iterative hungarian" begin
    ycoords = rand(3,5) * 5 .- 10;
    randtform = AffineMap(RotationVec(π*rand(3)...), SVector{3}(5*rand(3)...))
    xcoords = randtform(ycoords)

    xset = PointSet(xcoords, ones(size(xcoords,2)))
    yset = PointSet(xcoords, ones(size(ycoords,2)))

    res = goih_align(yset, xset)
    @test res.lowerbound == 0
    @test res.upperbound < 1e-15
end

# @testset "TIV-GOGMA (perfect alignment)" begin
#     for i=1:10
#         randpts = 10*rand(3,50)
#         randtform = AffineMap(10*rand(6)...)
#         gmmx = IsotropicGMM([IsotropicGaussian(randpts[:,i],1,1) for i=1:size(randpts,2)])
#         gmmy = randtform(gmmx)
#         min_overlap_score = -overlap(gmmx,gmmx)
#         res = tiv_gogma_align(gmmx,gmmy,0.5,0.5; maxstagnant=1E3)
#         @test isapprox(res.upperbound, min_overlap_score; rtol=0.01)
#         @test isapprox(overlap(res.tform(gmmx), gmmy), -min_overlap_score; rtol=0.01)
#     end

#     for i=1:10
#         randpts = 10*rand(3,50)
#         randtform = AffineMap(10*rand(6)...)
#         mgmmx = IsotropicMultiGMM(Dict([Symbol(j)=>IsotropicGMM([IsotropicGaussian(randpts[:,i+10*(j-1)],1,1) for i=1:Int(size(randpts,2)/5)]) for j=1:5]))
#         mgmmy = randtform(mgmmx)
#         min_overlap_score = -overlap(mgmmx,mgmmx)
#         res = tiv_gogma_align(mgmmx,mgmmy,0.5,0.5; maxstagnant=1E3)
#         @test isapprox(res.upperbound, min_overlap_score; rtol=0.01)
#         @test isapprox(overlap(res.tform(mgmmx), mgmmy), -min_overlap_score; rtol=0.01)
#     end
# end

# @testset "ROCS (perfect alignment)" begin
#     for i=1:10
#         randpts = 10*rand(3,50)
#         randtform = AffineMap(10*rand(6)...)
#         gmmx = IsotropicGMM([IsotropicGaussian(randpts[:,i],1,1) for i=1:size(randpts,2)])
#         gmmy = randtform(gmmx)
#         min_overlap_score = -overlap(gmmx,gmmx)
#         rocs_res = rocs_align(gmmx,gmmy)
#         ovlp, tform  = rocs_res.minimum, rocs_res.tform
#         @test isapprox(ovlp, min_overlap_score; atol=1E-12)
#         @test isapprox(overlap(tform(gmmx), gmmy), -min_overlap_score; atol=1E-12)
#     end

#     for i=1:10
#         randpts = 10*rand(3,50)
#         randtform = AffineMap(10*rand(6)...)
#         mgmmx = IsotropicMultiGMM(Dict([Symbol(j)=>IsotropicGMM([IsotropicGaussian(randpts[:,i+10*(j-1)],1,1) for i=1:Int(size(randpts,2)/5)]) for j=1:5]))
#         mgmmy = randtform(mgmmx)
#         min_overlap_score = -overlap(mgmmx,mgmmx)
#         rocs_res = rocs_align(mgmmx,mgmmy)
#         ovlp, tform  = rocs_res.minimum, rocs_res.tform
#         @test isapprox(ovlp, min_overlap_score; atol=1E-12)
#         @test isapprox(overlap(tform(mgmmx), mgmmy), -min_overlap_score; atol=1E-12)
#     end
# end

# @testset "TIV-GOGMA (missing data alignment)" begin
#     for i=1:10
#         randpts = 10*rand(3,50)
#         randtform = AffineMap(10*rand(6)...)
#         gmmx = IsotropicGMM([IsotropicGaussian(randpts[:,i],1,1) for i=1:size(randpts,2)])
#         gmmy = randtform(IsotropicGMM(gmmx.gaussians[1:40]))
#         min_overlap_score = -overlap(gmmx,inv(randtform)(gmmy))
#         res = tiv_gogma_align(gmmx,gmmy,0.5,0.5; maxstagnant=1E4)
#         @test res.upperbound <= min_overlap_score*0.99
#         @test -overlap(res.tform(gmmx), gmmy) <= min_overlap_score*0.99   
#     end

#     for i=1:10
#         randpts = 10*rand(3,50)
#         randtform = AffineMap(10*rand(6)...)
#         mgmmx = IsotropicMultiGMM(Dict([Symbol(j)=>IsotropicGMM([IsotropicGaussian(randpts[:,i+10*(j-1)],1,1) for i=1:10]) for j=1:5]))
#         mgmmy = randtform(IsotropicMultiGMM(Dict([Symbol(j)=>IsotropicGMM([IsotropicGaussian(randpts[:,i+10*(j-1)],1,1) for i=1:8]) for j=1:5])))
#         min_overlap_score = -overlap(mgmmx,inv(randtform)(mgmmy))
#         res = tiv_gogma_align(mgmmx,mgmmy,0.5,0.5; maxstagnant=1E4)
#         @test res.upperbound <= min_overlap_score*0.99
#         @test -overlap(res.tform(mgmmx), mgmmy) <= min_overlap_score*0.99   
#     end
# end

# @testset "ROCS-GOGMA (missing data alignment)" begin
#     for i=1:10
#         randpts = 10*rand(3,50)
#         randtform = AffineMap(10*rand(6)...)
#         gmmx = IsotropicGMM([IsotropicGaussian(randpts[:,i],1,1) for i=1:size(randpts,2)])
#         gmmy = randtform(IsotropicGMM(gmmx.gaussians[1:40]))
#         min_overlap_score = -overlap(gmmx,inv(randtform)(gmmy))
#         res = rocs_gogma_align(gmmx,gmmy; maxstagnant=1E4)
#         @test res.upperbound <= min_overlap_score*0.99
#         @test -overlap(res.tform(gmmx), gmmy) <= min_overlap_score*0.99
#     end

#     for i=1:10
#         randpts = 10*rand(3,50)
#         randtform = AffineMap(10*rand(6)...)
#         mgmmx = IsotropicMultiGMM(Dict([Symbol(j)=>IsotropicGMM([IsotropicGaussian(randpts[:,i+10*(j-1)],1,1) for i=1:10]) for j=1:5]))
#         mgmmy = randtform(IsotropicMultiGMM(Dict([Symbol(j)=>IsotropicGMM([IsotropicGaussian(randpts[:,i+10*(j-1)],1,1) for i=1:8]) for j=1:5])))
#         min_overlap_score = -overlap(mgmmx,inv(randtform)(mgmmy))
#         res = rocs_gogma_align(mgmmx, mgmmy; maxstagnant=1E4)
#         @test res.upperbound <= min_overlap_score*0.99
#         @test -overlap(res.tform(mgmmx), mgmmy) <= min_overlap_score*0.99
#     end
# end