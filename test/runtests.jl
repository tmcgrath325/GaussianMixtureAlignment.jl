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
    t = Float64
    center = NTuple{6,t}(zeros(t,2*dims))
    rwidth = t(2*π)
    twidth = one(t)
    scntrs = GOGMA.subcenters(center, rwidth, twidth, 4)
    @test length(unique(scntrs)) == 4^6
end

@testset "gogma" begin
    # two sets of points, each forming a 3-4-5 triangle
    xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]] 
    ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
    σ = ϕ = 1.
    gmmx = IsotropicGMM([IsotropicGaussian(x, σ, ϕ) for x in xpts])
    gmmy = IsotropicGMM([IsotropicGaussian(y, σ, ϕ) for y in ypts])

    # make sure this runs without an error
    min, lowerbound, bestloc, ndivisions = branch_bound(gmmx, gmmy)

end
