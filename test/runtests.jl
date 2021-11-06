using GaussianMixtureAlignment
using Test
using IntervalSets
using LinearAlgebra
using CoordinateTransformations

using GaussianMixtureAlignment: get_bounds, subranges, fullBlock

const GMA = GaussianMixtureAlignment
@testset "get bounds" begin
    μx = [3,0,0]
    μy = [-4,0,0]
    σ = ϕ = 1
    ndims = 3
    sqrt2 = √2

    x, y = IsotropicGaussian(μx, σ, ϕ), IsotropicGaussian(μy, σ, ϕ)

    # rotation distances, no translation
    # anti-aligned (no rotation) and aligned (180 degree rotation)
    lb, ub = get_bounds(x,y,2π,0,zeros(6))
    @test lb ≈ -GMA.overlap(1,2*σ^2,ϕ*ϕ, 1.) atol=1e-16
    @test ub ≈ -GMA.overlap(7^2,2*σ^2,ϕ*ϕ, 1.)
    lb, ub = get_bounds(x,y,2π,0,[0,0,π,0,0,0])
    @test lb ≈ ub ≈ -GMA.overlap(1,2*σ^2,ϕ*ϕ, 1.)
    # spheres with closest alignment at 90 degree rotation
    lb = get_bounds(x,y,π/√(3),0,zeros(6))[1]
    @test lb ≈ -GMA.overlap(5^2,2*σ^2,ϕ*ϕ, 1.)
    lb = get_bounds(x,y,π/(2*√(3)),0,[0,0,π/4,0,0,0])[1]
    @test lb ≈ -GMA.overlap(5^2,2*σ^2,ϕ*ϕ, 1.) 
    
    # translation distance, no rotation
    # centered at origin
    lb, ub = get_bounds(x,y,0,2/√(3),zeros(6))
    @test lb ≈ -GMA.overlap(6^2,2*σ^2,ϕ*ϕ, 1.)
    @test ub ≈ -GMA.overlap(7^2,2*σ^2,ϕ*ϕ, 1.)
    # centered with translation of 1 in +x
    lb, ub = get_bounds(x,y,0,2/√(3),[0,0,0,1,0,0])
    @test lb ≈ -GMA.overlap(7^2,2*σ^2,ϕ*ϕ, 1.)
    @test ub ≈ -GMA.overlap(8^2,2*σ^2,ϕ*ϕ, 1.)
    # centered with translation of 3 in +y 
    lb, ub = get_bounds(x,y,0,2/√(3),[0,0,0,0,3,0])
    @test lb ≈ -GMA.overlap((√(58)-1)^2,2*σ^2,ϕ*ϕ, 1.)
    @test ub ≈ -GMA.overlap(58,2*σ^2,ϕ*ϕ, 1.)

end

@testset "divide block" begin
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

@testset "GOGMA" begin
    # two sets of points, each forming a 3-4-5 triangle
    xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]] 
    ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
    σ = ϕ = 1.
    gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
    gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

    # aligning a GMM to itself
    bigblock = fullBlock(gmmx, gmmx)
    # @show bigblock.lowerbound, bigblock.upperbound
    @test bigblock.lowerbound ≈ -length(gmmx.gaussians)^2 # / √(4π)^3

    blk = fullBlock(gmmx, gmmx, NTuple{6,Tuple{Float64,Float64}}(((0,π), (0,π), (0,π), (0,2), (0,2), (0,2))))
    lb = blk.lowerbound
    ub = blk.upperbound
    for i = 1:20
        blk = fullBlock(gmmx, gmmx, subranges(blk.ranges, 2)[1])
        @test blk.lowerbound >= lb
        @test blk.upperbound <= ub
        lb = blk.lowerbound
        ub = blk.upperbound
    end

    # make sure this runs without an error
    res = gogma_align(gmmx, gmmy, maxblocks=1E5)
    res = tiv_gogma_align(gmmx, gmmy)

    mgmmx = IsotropicMultiGMM(Dict(:x => gmmx, :y => gmmy))
    mgmmy = IsotropicMultiGMM(Dict(:y => gmmx, :x => gmmy))
    res = gogma_align(mgmmx, mgmmy, maxblocks=1E5)
    res = tiv_gogma_align(mgmmx, mgmmy)

    # ROCS alignment should work perfectly
    @test isapprox(rocs_align(gmmx, gmmy).minimum, -overlap(gmmx,gmmx); atol=1E-12)
end

@testset "directional GOGMA" begin
    xpt = [1.,0.,0.]
    ypt = [1.,0.,0.]
    xdir = [0.,1.,0.]
    ydir = [1.,0.,0.]
    σ = ϕ = 1.
    x = IsotropicGaussian(xpt, σ, ϕ, [xdir])
    y = IsotropicGaussian(ypt, σ, ϕ, [ydir])
    @test get_bounds(x,y,0.,0.,zeros(6)) == (-0.5,-0.5)
    @test get_bounds(x,y,π/(3*GMA.sqrt3),0.,zeros(6)) == (-0.75,-0.5)
    @test get_bounds(x,y,π/GMA.sqrt3,0.,zeros(6)) == (-1.0,-0.5)

    xpt = [1.,0.,0.]
    ypt = [1.,0.,0.]
    xdir = [-1.,0.,0.]
    ydir = [1.,0.,0.]
    σ = ϕ = 1.
    x = IsotropicGaussian(xpt, σ, ϕ, [xdir])
    y = IsotropicGaussian(ypt, σ, ϕ, [ydir])
    @test get_bounds(x,y,0.,0.,zeros(6)) == (0.,0.)
    

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

@testset "TIV-GOGMA (perfect alignment)" begin
    for i=1:10
        randpts = 10*rand(3,50)
        randtform = AffineMap(10*rand(6)...)
        gmmx = IsotropicGMM([IsotropicGaussian(randpts[:,i],1,1) for i=1:size(randpts,2)])
        gmmy = randtform(gmmx)
        min_overlap_score = -overlap(gmmx,gmmx)
        res = tiv_gogma_align(gmmx,gmmy,0.5,0.5; maxstagnant=1E3)
        @test isapprox(res.upperbound, min_overlap_score; rtol=0.01)
        @test isapprox(overlap(res.tform(gmmx), gmmy), -min_overlap_score; rtol=0.01)
    end

    for i=1:10
        randpts = 10*rand(3,50)
        randtform = AffineMap(10*rand(6)...)
        mgmmx = IsotropicMultiGMM(Dict([Symbol(j)=>IsotropicGMM([IsotropicGaussian(randpts[:,i+10*(j-1)],1,1) for i=1:Int(size(randpts,2)/5)]) for j=1:5]))
        mgmmy = randtform(mgmmx)
        min_overlap_score = -overlap(mgmmx,mgmmx)
        res = tiv_gogma_align(mgmmx,mgmmy,0.5,0.5; maxstagnant=1E3)
        @test isapprox(res.upperbound, min_overlap_score; rtol=0.01)
        @test isapprox(overlap(res.tform(mgmmx), mgmmy), -min_overlap_score; rtol=0.01)
    end
end

@testset "ROCS (perfect alignment)" begin
    for i=1:10
        randpts = 10*rand(3,50)
        randtform = AffineMap(10*rand(6)...)
        gmmx = IsotropicGMM([IsotropicGaussian(randpts[:,i],1,1) for i=1:size(randpts,2)])
        gmmy = randtform(gmmx)
        min_overlap_score = -overlap(gmmx,gmmx)
        rocs_res = rocs_align(gmmx,gmmy)
        ovlp, tform  = rocs_res.minimum, rocs_res.tform
        @test isapprox(ovlp, min_overlap_score; atol=1E-12)
        @test isapprox(overlap(tform(gmmx), gmmy), -min_overlap_score; atol=1E-12)
    end

    for i=1:10
        randpts = 10*rand(3,50)
        randtform = AffineMap(10*rand(6)...)
        mgmmx = IsotropicMultiGMM(Dict([Symbol(j)=>IsotropicGMM([IsotropicGaussian(randpts[:,i+10*(j-1)],1,1) for i=1:Int(size(randpts,2)/5)]) for j=1:5]))
        mgmmy = randtform(mgmmx)
        min_overlap_score = -overlap(mgmmx,mgmmx)
        rocs_res = rocs_align(mgmmx,mgmmy)
        ovlp, tform  = rocs_res.minimum, rocs_res.tform
        @test isapprox(ovlp, min_overlap_score; atol=1E-12)
        @test isapprox(overlap(tform(mgmmx), mgmmy), -min_overlap_score; atol=1E-12)
    end
end

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