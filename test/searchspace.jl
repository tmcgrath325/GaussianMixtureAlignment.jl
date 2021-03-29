using GOGMA

function testsplit(blocks, gmmx, gmmy, ub=Inf)
    lb=Inf
    sblocks = []
    for (i,blk) in enumerate(blocks)
        for rng in subranges(blk.ranges, 2)
            sblk = Block(gmmx, gmmy, rng)
            if sblk.lowerbound < ub
                push!(sblocks, sblk)
                if sblk.lowerbound < lb
                    lb = sblk.lowerbound
                end
                if sblk.upperbound < ub
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
blocks, lb, ub = testsplit([bigblock], gmmx, gmmy)

# further subdividing each subcube, giving (2^6)^2 subcubes if none are excluded
sblocks2, lb, ub = testsplit(blocks, gmmx, gmmy, ub)

# further subdividing each subcube, giving (2^6)^3 subcubes if none are excluded
sblocks3, lb, ub = testsplit(sblocks2, gmmx, gmmy, ub)

# ... and so on
sblocks4, lb, ub = testsplit(sblocks3, gmmx, gmmy, ub)
sblocks5, lb, ub = testsplit(sblocks4, gmmx, gmmy, ub)
sblocks6, lb, ub = testsplit(sblocks5, gmmx, gmmy, ub)
sblocks7, lb, ub = testsplit(sblocks6, gmmx, gmmy, ub)
sblocks8, lb, ub = testsplit(sblocks7, gmmx, gmmy, ub)
sblocks9, lb, ub = testsplit(sblocks8, gmmx, gmmy, ub)
sblocks10, lb, ub = testsplit(sblocks9, gmmx, gmmy, ub);