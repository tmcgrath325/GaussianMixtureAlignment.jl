using GOGMA
using ProfileView

xpts = [[0.,0.,0.], [3.,0.,0.,], [0.,4.,0.]] 
ypts = [[1.,1.,1.], [1.,-2.,1.], [1.,1.,-3.]]
gmmx = IsotropicGMM([IsotropicGaussian(x, 1, 1) for x in xpts])
gmmy = IsotropicGMM([IsotropicGaussian(y, 1, 1) for y in ypts])

branch_bound(gmmx, gmmy, maxblocks=1e4)

@ProfileView.profview branch_bound(gmmx, gmmy, maxblocks=1e6)