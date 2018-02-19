# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pycurious

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

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

fmt = "min/max {:.2f} -> {:.2f}"
if rank == 0:
    for col in xrange(mag_data.shape[1]):
        print(fmt.format(mincols[col], maxcols[col]))


# In[7]:


xmin = 250000.0
xmax = 1400000.0
ymin = 200000.0
ymax = 1500000.0
extent = [xmin, xmax, ymin, ymax]

dx, dy = 1e3, 1e3 # 1km resolution
nx, ny = int(round((xmax-xmin)/dx)), int(round((ymax-ymin)/dy))

# also get WGS84 extent
tlon, tlat = pycurious.transform_coordinates([xmin,xmin,xmax,xmax], [ymin,ymax,ymin,ymax], 2157, 4326)
lonmin, lonmax = min(tlon), max(tlon)
latmin, latmax = min(tlat), max(tlat)
extent_sphere = [lonmin, lonmax, latmin, latmax]

# grid data
mag_grid = pycurious.grid(mag_data[:,:2], mag_data[:,3], extent, shape=(ny,nx), epsg_in=4326, epsg_out=2157)
mag_sphere = pycurious.grid(mag_data[:,:2], mag_data[:,3], extent_sphere, shape=(ny,nx), epsg_in=4326, epsg_out=4326)





# 2 arc minutes is approximately 4km
max_window = 350e3
window_sizes = np.arange(100e3, max_window+50e3, 50e3)



# first pass
# use the largest window size first to determine priors
window = window_sizes[-1]
cpd = pycurious.CurieParallel(mag_grid, xmin, xmax, ymin, ymax, window)
xc_list, yc_list = cpd.create_centroid_list(window, spacingX=10e3, spacingY=10e3)


betaN, ztN, dzN, CN = cpd.optimise_routine(window, xc_list, yc_list)


beta_mu, beta_std = np.mean(betaN), np.std(betaN)
zt_mu, zt_std = np.mean(ztN), np.std(ztN)
dz_mu, dz_std = np.mean(dzN), np.std(dzN)
C_mu, C_std = np.mean(CN), np.std(CN)

cpd.reset_priors()
cpd.distribute_prior('mean', beta=(beta_mu, beta_std), zt=(zt_mu, zt_std), dz=(dz_mu, dz_std))


for p in ['beta', 'zt', 'dz', 'C']:
    prior = cpd.prior[p]
    if type(prior) != type(None):
        mu, sigma = prior # local
    else:
        mu, sigma = -1, -1
    print("{:3d} prior {:5} mean={:.2f} std={:.2f}".format(rank, p, mu, sigma))



# now priors will remain idential for all iterations
prior_dict = cpd.prior.copy()

for window in window_sizes:
    cpd = pycurious.CurieParallel(mag_grid, xmin, xmax, ymin, ymax, window)
    cpd.prior = prior_dict
    xc_list, yc_list = cpd.create_centroid_list(window, spacingX=4e3, spacingY=4e3)
    print("{} window {:.1f} km, number of centroids = {}".format(rank, window/1e3, len(xc_list)))

    # second pass
    betaOpt, ztOpt, dzOpt, COpt = cpd.optimise_routine(window, xc_list, yc_list)

    # Write to file
    lonc, latc = pycurious.transform_coordinates(xc_list, yc_list, epsg_in=2157, epsg_out=4326)
    out = np.column_stack([lonc, latc, betaOpt, ztOpt, dzOpt, COpt])
    np.savetxt('{}-british_isles_cpd_w{:06d}.csv'.format(rank, int(window/1e3)),\
               out, fmt='%.6f', delimiter=',', header='lon, lat, beta, zt, dz, C')

