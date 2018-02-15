# coding: utf-8

"""
**EMAG2_V3_20170530.csv**

- Column 1: i ; grid column/longitude index
- Column 2: j ; grid row/latitude index
- Column 3: LON ; Geographic Longitude WGS84 (decimal degrees)
- Column 4: LAT ; Geographic Latitude WGS84 (decimal degrees)
- Column 5: SeaLevel ; Magnetic Anomaly Value at Sea Level(nT)
- Column 6: UpCont ; Magnetic Anomaly Value at continuous 4km altitude (nT)
- Column 7: Code ; Data Source Code (see table below)
- Column 8: Error ; Error estimate (nT)

Code 888 is assigned in certain cells on grid edges where the data source is ambiguous and assigned an error of -888 nT

Code 999 is assigned in cells where no data is reported with the anomaly value assigned 99999 nT and an error of -999 nT



## Optimise

We have 4 parameters we wish to optimise:
- beta : fractal parameter
- z_t  : the top of magnetic sources
- dz   : the thickness of magnetic sources
- C    : a field constant

We minimise this using `scipy.optimize.minimize` which defaults
to `L-BFGS-B` for constrained minimisation.


### Optimisation strategy

The problem is that $dz$ in Bouligand *et al.* (2009) [eq. 4] is
least sensitive to the problem, but the most important parameter
for evaluating Curie depth.

If we revisit the problem, however, we see that beta and C mostly
vary at long wavelengths, thus if we determine what their average
value is over the whole field then we can fix them as constants.
Our strategy is now a two-stage inversion method.

**First pass**

Ensure there is no *a priori* information and commence with
suitable starting values.

**Second pass**

Take the mean and standard deviation of all parameters and add them
as priors within the misfit function, then run the optimisation again.

beta, C, and z_t have the lowest standard deviation therefore the
effect of perturbing these greatly increases the misfit compared to dz.
"""

import numpy as np
import matplotlib.pyplot as plt
import pycurious

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank



filedir = '/mnt/ben_raid/'

try:
    with np.load(filedir+'EMAG2_V3_20170530.npz') as f:
        mag_data = f['data']
except:
    mag_data = np.loadtxt(filedir+'EMAG2_V3_20170530.csv', delimiter=',', usecols=(2,3,4,5,7))
    lon_mask = mag_data[:,0] > 180.0
    mag_data[lon_mask,0] -= 360.0
    np.savez_compressed(filedir+'EMAG2_V3_20170530.npz', data=mag_data.astype(np.float32))


# filter NaNs
mag_data = mag_data[mag_data[:,3] != 99999.]
mag_data = mag_data[mag_data[:,4] != -888.]
mag_data = mag_data[mag_data[:,4] != -999.]

# print min/max
mincols = mag_data.min(axis=0)
maxcols = mag_data.max(axis=0)


if rank == 0:
    fmt = "min/max {:.2f} -> {:.2f}"
    for col in xrange(mag_data.shape[1]):
        print(fmt.format(mincols[col], maxcols[col]))





xmin = 200000.0
xmax = 1450000.0
ymin = 150000.0
ymax = 1550000.0
extent = [xmin, xmax, ymin, ymax]

dx, dy = 1e3, 1e3 # 1km resolution
nx, ny = int(round((xmax-xmin)/dx)), int(round((ymax-ymin)/dy))


# grid data
mag_grid = pycurious.grid(mag_data[:,:2], mag_data[:,3], extent, shape=(ny,nx), epsg_in=4326, epsg_out=2157)



# initialise CurieOptimise object
max_window = 200e3

cpd = pycurious.CurieParallel(mag_grid, xmin, xmax, ymin, ymax, max_window)

print("{:3d} old shape {} new shape {}".format(rank, mag_grid.shape, cpd.data.shape))
print("{:3d} old extent {}".format(rank, (xmin, xmax, ymin, ymax)))
print("{:3d} new extent {}".format(rank, (cpd.xmin, cpd.xmax, cpd.ymin, cpd.ymax)))



xc_list, yc_list = cpd.create_centroid_list(max_window, spacingX=10e3, spacingY=10e3)
print("{} number of centroids = {}".format(rank, len(xc_list)))




# first pass - no priors
betaN, ztN, dzN, CN = cpd.optimise_routine(max_window, xc_list, yc_list, taper=np.hanning)



xc = np.unique(xc_list)
yc = np.unique(yc_list)

nr, nc = yc.size, xc.size


# these parameters did well based on previous experience...
# cpd.add_prior(beta=(5.44,0.62), zt=(0.14,0.28), dz=(22.73,29.63), C=(11.88,1.38))

beta_mu, beta_std = np.mean(betaN), np.std(betaN)
zt_mu, zt_std = np.mean(ztN), np.std(ztN)
dz_mu, dz_std = np.mean(dzN), np.std(dzN)
C_mu, C_std = np.mean(CN), np.std(CN)

cpd.reset_priors()
cpd.distribute_prior('mean', beta=(beta_mu, beta_std), zt=(zt_mu, zt_std), dz=(20.0, 80.), C=(C_mu,C_std))

beta_mu, beta_std = cpd.prior['beta']
zt_mu, zt_std = cpd.prior['zt']
dz_mu, dz_std = cpd.prior['dz']
C_mu, C_std = cpd.prior['C']

for p in ['beta', 'zt', 'dz', 'C']:
    prior = cpd.prior[p]
    if type(prior) != type(None):
        mu, sigma = prior # local
    else:
        mu, sigma = None, None
    print("{:3d} prior {:5} mean={:.2f} std={:.2f}".format(rank, p, mu, sigma))



lonc, latc = pycurious.transform_coordinates(xc_list, yc_list, epsg_in=2157, epsg_out=4326)
posterior = np.empty((lonc.size, 6))
posterior[:,0] = lonc
posterior[:,1] = latc

# second pass - with distributed priors
nsim = 1000

prior = np.zeros((nsim,4))
prior[:,0] = np.random.normal(beta_mu, beta_std, nsim)
prior[:,1] = np.random.normal(zt_mu, zt_std, nsim)
prior[:,2] = np.random.normal(dz_mu, dz_std, nsim)
prior[:,3] = np.random.normal(C_mu, C_std, nsim)

# prior will be identical across all processors
comm.Bcast([prior, MPI.DOUBLE], root=0)


for i in xrange(0, nsim):
    # random priors from root processor
    beta, zt, dz, C = prior[i]
    cpd.reset_priors()
    cpd.add_prior(beta=(beta,beta_std), zt=(zt,zt_std), dz=(dz,dz_std))

    betaOpt, ztOpt, dzOpt, COpt = cpd.optimise_routine(max_window, xc_list, yc_list, taper=np.hanning)


    # Write to file
    posterior[:,2] = betaOpt
    posterior[:,3] = ztOpt
    posterior[:,4] = dzOpt
    posterior[:,5] = COpt
    np.savetxt('{:03d}-british_isles_cpd_s{:06d}.csv'.format(rank, i), posterior,\
               fmt='%.6f', delimiter=',', header='lon, lat, beta, zt, dz, C')

    if i % 10 == 0 and i > 0:
        print("{:3d} {:3d}/{} simulations complete".format(rank, i, nsim))

if rank == 0:
    np.savetxt('british_isles_cpd_sx.csv', prior, fmt='%.6f', delimiter=',', header='beta, zt, dz, C')
