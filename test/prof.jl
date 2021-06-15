using GOGMA
using ProfileView

# small problem
xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]]
ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
gmmx = IsotropicGMM([IsotropicGaussian(x, 1, 1) for x in xpts])
gmmy = IsotropicGMM([IsotropicGaussian(y, 1, 1) for y in ypts])

@time branch_bound(gmmx, gmmy, maxblocks=1e3)
@time tiv_branch_bound(gmmx, gmmy, maxblocks=1e3)

@ProfileView.profview branch_bound(gmmx, gmmy, maxblocks=1e5)
@ProfileView.profview tiv_branch_bound(gmmx, gmmy)

# steroid-sized problem
xpts = [10*rand(3).-6 for i in 1:50]
ypts = [10*rand(3).-4 for i in 1:50]
gmmx = IsotropicGMM([IsotropicGaussian(x, 1, 1) for x in xpts])
gmmy = IsotropicGMM([IsotropicGaussian(y, 1, 1) for y in ypts])

@time branch_bound(gmmx, gmmy, maxblocks=1e3)
@time tiv_branch_bound(gmmx, gmmy, 2, 2, maxblocks=1e3)

@ProfileView.profview branch_bound(gmmx, gmmy, maxblocks=1e4)
@ProfileView.profview tiv_branch_bound(gmmx, gmmy, 2, 2, maxblocks=1e4)

@time branch_bound(gmmx, gmmy, maxblocks=1e3, threads=true)
@time tiv_branch_bound(gmmx, gmmy, 2, 2, maxblocks=1e3, threads=true)

@ProfileView.profview branch_bound(gmmx, gmmy, maxblocks=1e4, threads=true)
@ProfileView.profview tiv_branch_bound(gmmx, gmmy, 2, 2, maxblocks=1e4, threads=true)