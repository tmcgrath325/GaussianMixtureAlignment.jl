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

    x, y = IsotropicGaussian(μx, σ, ϕ), IsotropicGaussian(μy, σ, ϕ)

    # rotation distances, no translation
    # anti-aligned (no rotation) and aligned (180 degree rotation)
    lb, ub = get_bounds(x,y,2π,0,zeros(6))
    @test lb ≈ GOGMA.objectivefun(1,σ,σ,ϕ,ϕ) atol=1e-16
    @test ub ≈ GOGMA.objectivefun(7,σ,σ,ϕ,ϕ)
    lb, ub = get_bounds(x,y,2π,0,[0,0,π,0,0,0])
    @test lb ≈ ub ≈ GOGMA.objectivefun(1,σ,σ,ϕ,ϕ)
    # spheres with closest alignment at 90 degree rotation
    lb = get_bounds(x,y,π/√(3),0,zeros(6))[1]
    @test lb ≈ GOGMA.objectivefun(5,σ,σ,ϕ,ϕ)
    lb = get_bounds(x,y,π/(2*√(3)),0,[0,0,π/4,0,0,0])[1]
    @test lb ≈ GOGMA.objectivefun(5,σ,σ,ϕ,ϕ) 
    
    # translation distance, no rotation
    # centered at origin
    lb, ub = get_bounds(x,y,0,2/√(3),zeros(6))
    @test lb ≈ GOGMA.objectivefun(6,σ,σ,ϕ,ϕ)
    @test ub ≈ GOGMA.objectivefun(7,σ,σ,ϕ,ϕ)
    # centered with translation of 1 in +x
    lb, ub = get_bounds(x,y,0,2/√(3),[0,0,0,1,0,0])
    @test lb ≈ GOGMA.objectivefun(7,σ,σ,ϕ,ϕ)
    @test ub ≈ GOGMA.objectivefun(8,σ,σ,ϕ,ϕ)
    # centered with translation of 3 in +y 
    lb, ub = get_bounds(x,y,0,2/√(3),[0,0,0,0,3,0])
    @test lb ≈ GOGMA.objectivefun(√(58)-1,σ,σ,ϕ,ϕ)
    @test ub ≈ GOGMA.objectivefun(√(58),σ,σ,ϕ,ϕ)

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
    @show subblks1[length(subblks1)]

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

    # xposmat = fill(NaN, 3, length(xpts))
    # for (i,x) in enumerate(xpts)
    #     xposmat[1:3,i] = x
    # end
    # yposmat = fill(NaN, 3, length(xpts))
    # for (i,y) in enumerate(ypts)
    #     yposmat[1:3,i] = y
    # end
    # tform = SCC.align(yposmat, xposmat)[1]
    # ϕrot = rotation_angle(RotMatrix(tform.linear))
    # vrot = rotation_axis(RotMatrix(tform.linear))

    # rmin = vrot/norm(vrot) * ϕrot
    # tmin = tform.translation
    # @show rmin, tmin

    # aligning a GMM to itself
    bigblock = Block(gmmx, gmmx)
    @show bigblock.lowerbound, bigblock.upperbound
    @test bigblock.lowerbound == -length(gmmx.gaussians)^2

    blk = Block(gmmx, gmmx, NTuple{6,Tuple{Float64,Float64}}(((0,π), (0,π), (0,π), (0,2), (0,2), (0,2))))
    lb = blk.lowerbound
    ub = blk.upperbound
    for i = 1:20
        blk = Block(gmmx, gmmx, subranges(blk.ranges, 2)[1])
        @test blk.lowerbound >= lb
        @test blk.upperbound <= ub
        lb = blk.lowerbound
        ub = blk.upperbound
        if i%5 == 0
            @show i, blk.lowerbound, blk.upperbound
        end
    end

    # make sure this runs without an error
    @time min, lowerbound, bestloc, ndivisions = branch_bound(gmmx, gmmy, maxblocks=1E5)

end
