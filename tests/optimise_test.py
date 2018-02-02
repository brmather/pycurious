# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pycurious
import time


# initialise CurieOptimise object
max_window = 100e3


# square grid 100km x 100km

xmin = 0.0
xmax = 100e3
ymin = 0.0
ymax = 100e3

nx = 100
ny = 100

# random noise
mag_grid = np.random.randn(ny,nx)

cpd = pycurious.CurieOptimise(mag_grid, xmin, xmax, ymin, ymax)


# compute centroid
xc = 0.5*(xmin + xmax)
yc = 0.5*(ymin + ymax)

xc_list, yc_list = cpd.create_centroid_list(0.5*max_window, spacingX=5e3, spacingY=5e3)
print("number of centroids = {}".format(len(xc_list)))



t = time.time()
betaN, ztN, dzN, CN = cpd.optimise_routine(0.5*max_window, xc_list, yc_list, taper=None)
print("\nfirst optimisation - no priors\ntime = {:.3f} sec".format(time.time() - t))



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
    fig.savefig('CPD_initial.png', bbox_inches='tight')
    
    print("{:5} mean={:.2f} std={:.2f}".format(title, data.mean(), np.std(data)))



beta_mu, beta_std = np.mean(betaN), np.std(betaN)
zt_mu, zt_std = np.mean(ztN), np.std(ztN)
dz_mu, dz_std = np.mean(dzN), np.std(dzN)
C_mu, C_std = np.mean(CN), np.std(CN)

cpd.reset_priors()
cpd.add_prior(beta=(beta_mu, beta_std))
cpd.add_prior(zt=(zt_mu, zt_std))
cpd.add_prior(dz=(dz_mu, dz_std))
cpd.add_prior(C=(C_mu, C_std))


t = time.time()
betaOpt, ztOpt, dzOpt, COpt = cpd.optimise_routine(0.5*max_window, xc_list, yc_list, taper=None)
print("\nsecond optimisation - with priors\ntime = {:.3f} sec".format(time.time() - t))



plot_helper = [(betaOpt, 'beta'), (ztOpt, 'zt'), (dzOpt, 'dz'), (COpt, 'C')]

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, sharey=True, figsize=(16,3.5))
for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    title = plot_helper[i][1]
    data  = plot_helper[i][0]
    mask  = np.isfinite(data)
    
    ax.set_title(title)
    sci = ax.scatter(xc_list[mask]/1e3, yc_list[mask]/1e3, c=data[mask])
    fig.colorbar(sci, ax=ax)
    fig.savefig('CPD_optimised.png', bbox_inches='tight')
    
    print("{:5} mean={:.2f} std={:.2f}".format(title, data.mean(), np.std(data)))
