using GOGMA
using IntervalSets
using LinearAlgebra
using Test
# using SteroidComputationalChemistry
# using Rotations

# const SCC = SteroidComputationalChemistry

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
    @test lb ≈ GOGMA.objectivefun(1,2*σ^2,ϕ*ϕ, 1.) atol=1e-16
    @test ub ≈ GOGMA.objectivefun(7^2,2*σ^2,ϕ*ϕ, 1.)
    lb, ub = get_bounds(x,y,2π,0,[0,0,π,0,0,0])
    @test lb ≈ ub ≈ GOGMA.objectivefun(1,2*σ^2,ϕ*ϕ, 1.)
    # spheres with closest alignment at 90 degree rotation
    lb = get_bounds(x,y,π/√(3),0,zeros(6))[1]
    @test lb ≈ GOGMA.objectivefun(5^2,2*σ^2,ϕ*ϕ, 1.)
    lb = get_bounds(x,y,π/(2*√(3)),0,[0,0,π/4,0,0,0])[1]
    @test lb ≈ GOGMA.objectivefun(5^2,2*σ^2,ϕ*ϕ, 1.) 
    
    # translation distance, no rotation
    # centered at origin
    lb, ub = get_bounds(x,y,0,2/√(3),zeros(6))
    @test lb ≈ GOGMA.objectivefun(6^2,2*σ^2,ϕ*ϕ, 1.)
    @test ub ≈ GOGMA.objectivefun(7^2,2*σ^2,ϕ*ϕ, 1.)
    # centered with translation of 1 in +x
    lb, ub = get_bounds(x,y,0,2/√(3),[0,0,0,1,0,0])
    @test lb ≈ GOGMA.objectivefun(7^2,2*σ^2,ϕ*ϕ, 1.)
    @test ub ≈ GOGMA.objectivefun(8^2,2*σ^2,ϕ*ϕ, 1.)
    # centered with translation of 3 in +y 
    lb, ub = get_bounds(x,y,0,2/√(3),[0,0,0,0,3,0])
    @test lb ≈ GOGMA.objectivefun((√(58)-1)^2,2*σ^2,ϕ*ϕ, 1.)
    @test ub ≈ GOGMA.objectivefun(58,2*σ^2,ϕ*ϕ, 1.)

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

@testset "gogma" begin
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
    objmin, lowerbound, bestloc, ndivisions = branch_bound(gmmx, gmmy, maxblocks=1E5)
    objmin, lowerbound, bestloc, ndivisions = tiv_branch_bound(gmmx, gmmy)

    mgmmx = MultiGMM(Dict(:x => gmmx, :y => gmmy))
    mgmmy = MultiGMM(Dict(:y => gmmx, :x => gmmy))
    objmin, lowerbound, bestloc, ndivisions = branch_bound(mgmmx, mgmmy, maxblocks=1E5)
    @show objmin
    objmin, lowerbound, bestloc, ndivisions = tiv_branch_bound(mgmmx, mgmmy)
    @show objmin
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
    @test get_bounds(x,y,π/(3*GOGMA.sqrt3),0.,zeros(6)) == (-0.75,-0.5)
    @test get_bounds(x,y,π/GOGMA.sqrt3,0.,zeros(6)) == (-1.0,-0.5)

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
    objminxy, lowerbound, bestlocxy, ndivisions = tiv_branch_bound(dgmmx, dgmmy)
    objminxx, lowerbound, bestlocxx, ndivisions = tiv_branch_bound(dgmmx, dgmmx)
    objminyy, lowerbound, bestlocyy, ndivisions = tiv_branch_bound(dgmmy, dgmmy)
    @test objminxy ≈ objminxx ≈ objminyy

    randdgmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ, [2*rand(3).-1]) for (i,y) in enumerate(ypts)])
    randobjminxy = tiv_branch_bound(dgmmx, randdgmmy)[1]
    @test randobjminxy > objminxy

    σ = 0.001
    ddgmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ, [xdirs[i], 2*rand(3).-1]) for (i,x) in enumerate(xpts)])
    ddgmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ, [ydirs[i], 2*rand(3).-1, 2*rand(3).-1]) for (i,y) in enumerate(ypts)])
    ddobjmin, ddlb, ddbestloc, ddndivisions = tiv_branch_bound(ddgmmx, ddgmmy)
    @test ddobjmin ≈ -3.0

    mgmmx = MultiGMM(Dict(:one => IsotropicGMM([IsotropicGaussian(xpts[1], σ, ϕ, [xdirs[1]])]),
                          :two => IsotropicGMM([IsotropicGaussian(xpts[2], σ, ϕ, [xdirs[2]])]),
                          :three => IsotropicGMM([IsotropicGaussian(xpts[3], σ, ϕ, [xdirs[3]])])))
    mgmmy = MultiGMM(Dict(:one => IsotropicGMM([IsotropicGaussian(ypts[1], σ, ϕ, [ydirs[1]])]),
                          :two => IsotropicGMM([IsotropicGaussian(ypts[2], σ, ϕ, [ydirs[2]])]),
                          :three => IsotropicGMM([IsotropicGaussian(ypts[3], σ, ϕ, [ydirs[3]])])))
    mobjminxy = tiv_branch_bound(mgmmx, mgmmy)[1]
    @test mobjminxy ≈ -3.0
end
