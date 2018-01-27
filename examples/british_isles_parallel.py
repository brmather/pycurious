# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pycurious

from mpi4py import MPI

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

# print min/max
mincols = mag_data.min(axis=0)
maxcols = mag_data.max(axis=0)

fmt = "min/max {:.2f} -> {:.2f}"
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



# initialise CurieOptimise object
max_window = 150e3

grd = pycurious.CurieParallel(mag_grid, xmin, xmax, ymin, ymax, max_window)

comm = grd.comm
rank = comm.rank
print("{} old shape {} new shape {}".format(rank, mag_grid.shape, grd.data.shape))
print("{} old extent {}".format(rank, (xmin, xmax, ymin, ymax)))
print("{} new extent {}".format(rank, (grd.xmin, grd.xmax, grd.ymin, grd.ymax)))



"""
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
"""

xc_list, yc_list = grd.create_centroid_list(max_window, spacingX=5e3, spacingY=5e3)
# xc_list, yc_list = grd.create_centroid_list(max_window)

print("{} number of centroids = {}".format(rank, len(xc_list)))




grd.reset_priors()
betaN, ztN, dzN, CN = grd.optimise_routine(max_window, xc_list, yc_list, 3.0, 1.0, 20.0, 5.0, taper=None)




xc = np.unique(xc_list)
yc = np.unique(yc_list)

nr, nc = yc.size, xc.size

plot_helper = [(betaN, 'beta'), (ztN, 'zt'), (dzN, 'dz'), (CN, 'C')]

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, sharey=True, figsize=(16,3.5))
for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    title = plot_helper[i][1]
    data  = plot_helper[i][0]
    mask  = np.isfinite(data)
    
    ax.set_title(title)
    sci = ax.scatter(xc_list[mask]/1e3, yc_list[mask]/1e3, c=data[mask])
    fig.colorbar(sci, ax=ax)
    fig.savefig('{}-CPD_initial.png'.format(rank), bbox_inches='tight')
    
    print "{} {:5} mean={:.2f} std={:.2f}".format(rank, title, data.mean(), np.std(data))

"""
**Second pass**

Take the mean and standard deviation of all parameters and add them
as priors within the misfit function, then run the optimisation again.

beta, C, and z_t have the lowest standard deviation therefore the
effect of perturbing these greatly increases the misfit compared to dz.
"""


# these parameters did well based on previous experience...
# grd.add_prior(beta=(5.44,0.62), zt=(0.14,0.28), dz=(22.73,29.63), C=(11.88,1.38))

beta_mu, beta_std = np.mean(betaN), np.std(betaN)
zt_mu, zt_std = np.mean(ztN), np.std(ztN)
dz_mu, dz_std = np.mean(dzN), np.std(dzN)
C_mu, C_std = np.mean(CN), np.std(CN)

grd.reset_priors()
grd.add_prior(beta=(beta_mu, beta_std))
grd.add_prior(zt=(zt_mu, zt_std))
grd.add_prior(dz=(20.0, 100.))
grd.add_prior(C=(C_mu, C_std))

g_sigma = np.array(0.0)
g_mu = np.array(0.0)

for p in ['beta', 'zt', 'dz', 'C']:
    prior = grd.prior[p]
    if type(prior) != type(None):
        mu, sigma = prior # local
        comm.Allreduce([np.array(mu), MPI.DOUBLE], [g_mu, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([np.array(sigma), MPI.DOUBLE], [g_sigma, MPI.DOUBLE], op=MPI.SUM)
        grd.prior[p] = (g_mu/comm.size, g_sigma/comm.size)
        mu, sigma = grd.prior[p]
    else:
        mu, sigma = 0,0
    print("{} prior {:5} mean={:.2f} std={:.2f}".format(rank, p, mu, sigma))

betaOpt, ztOpt, dzOpt, COpt = grd.optimise_routine(max_window, xc_list, yc_list, taper=None)



plot_helper = [(betaOpt, 'beta'), (ztOpt, 'zt'), (dzOpt, 'dz'), (COpt, 'C')]

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, sharey=True, figsize=(16,3.5))
for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    title = plot_helper[i][1]
    data  = plot_helper[i][0]
    mask  = np.isfinite(data)
    
    ax.set_title(title)
    sci = ax.scatter(xc_list[mask]/1e3, yc_list[mask]/1e3, c=data[mask])
    fig.colorbar(sci, ax=ax)
    fig.savefig('{}-CPD_optimised.png'.format(rank), bbox_inches='tight')
    
    print "{} {:5} mean={:.2f} std={:.2f}".format(rank, title, data.mean(), np.std(data))

"""
Not bad!
The Curie depth can be refined further if an average geotherm is assumed,
but then it is no longer an independent constraint on the temperature field.
"""

# Write to file
curie_depth = ztOpt + dzOpt

lonc, latc = pycurious.transform_coordinates(xc_list, yc_list, epsg_in=2157, epsg_out=4326)
np.savetxt('{}-british_isles_cpd.csv'.format(rank), np.column_stack([lonc, latc, curie_depth]),\
           fmt='%.6f', delimiter=',', header='lon, lat, curie_depth')

