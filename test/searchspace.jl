using GOGMA

function testsplit(blocks, gmmx, gmmy)
    sblocks = []
    lb = Inf
    ub = -Inf
    for (i,blk) in enumerate(blocks)
        for rng in subranges(blk.ranges, 2)
            sblk = Block(gmmx, gmmy, rng)
            if sblk.lowerbound < globalmin
                push!(sblocks, sblk)
                if sblk.lowerbound < lb
                    lb = sblk.lowerbound
                end
                if sblk.upperbound > ub
                    ub = sblk.upperbound
                end
            end
        end
    end
    println(length(sblocks)," valid blocks,   ", 100*length(sblocks)/(length(blocks)*64),"% of previous space left,   lower bound: ", lb, "   upper bound: ", ub)
    return sblocks, lb, ub
end

# looking at how many regions in rigid transformation space are excluded as each region
# is split and hypercubes with no possible improvement are discarded

# two sets of 3 points, each forming a 3-4-5 triangle
xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]] 
ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
gmmx = IsotropicGMM([IsotropicGaussian(x, 1, 1) for x in xpts])
gmmy = IsotropicGMM([IsotropicGaussian(y, 1, 1) for y in ypts])
globalmin = get_bounds(gmmx, gmmx, 0, 0, zeros(6))[2]   # overlap between gmmx and gmmy after alignment is the same as between gmmx with itself

# the entire rigid transformation space
bigblock = Block(gmmx,gmmy)
lowestlb = bigblock.lowerbound      # lower bound is case where all points have the same μ after alignment, so lb = -(3^2)
@show globalmin

# first divisions along each dimension, giving 2^6 subcubes
blocks = []
lb1 = Inf
@time blocks, lb1, ub1 = testsplit([bigblock], gmmx, gmmy)

# further subdividing each subcube, giving (2^6)^2 subcubes if none are excluded
@time sblocks2, lb2, ub2 = testsplit(blocks, gmmx, gmmy)

# further subdividing each subcube, giving (2^6)^3 subcubes if none are excluded
@time sblocks3, lb3, ub3 = testsplit(sblocks2, gmmx, gmmy)

# ... and so on
@time sblocks4, lb4, ub4 = testsplit(sblocks3, gmmx, gmmy)
@time sblocks5, lb5, ub5 = testsplit(sblocks4, gmmx, gmmy)
@time sblocks6, lb6, ub6 = testsplit(sblocks5, gmmx, gmmy)
@time sblocks7, lb7, ub7 = testsplit(sblocks6, gmmx, gmmy)
@time sblocks8, lb8, ub8 = testsplit(sblocks7, gmmx, gmmy)
@time sblocks9, lb9, ub9 = testsplit(sblocks8, gmmx, gmmy)
@time sblocks10, lb10, ub10 = testsplit(sblocks9, gmmx, gmmy);