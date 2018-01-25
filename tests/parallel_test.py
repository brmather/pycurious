import matplotlib.pyplot as plt
import numpy as np
import pycurious

xmin = -1.0
xmax = 1.0
ymin = -1.0
ymax = 1.0

nx = 10
ny = 10

xcoords = np.linspace(xmin, xmax, nx)
ycoords = np.linspace(ymin, ymax, ny)
xq, yq = np.meshgrid(xcoords, ycoords)

grid = xq**2 + yq**2

max_window_size = 0.5
print("window size {}".format(max_window_size))


cpd = pycurious.CurieParallel(grid, xmin, xmax, ymin, ymax, max_window_size)

fig = plt.figure()
ax1 = fig.add_subplot(111)
im1 = ax1.imshow(grid, extent=[xmin, xmax, ymin, ymax])
fig.colorbar(im1)
fig.savefig('{}_whole_grid.png'.format(cpd.comm.rank))





print("{} ghost range {}".format(cpd.comm.rank, cpd.dm.getGhostRanges()))
print("{} min {}".format(cpd.comm.rank, cpd.coords.min(axis=0)))
print("{} max {}".format(cpd.comm.rank, cpd.coords.max(axis=0)))


fig = plt.figure()
ax1 = fig.add_subplot(111)
im1 = ax1.imshow(cpd.data, extent=[cpd.xmin, cpd.xmax, cpd.ymin, cpd.ymax], origin='lower')
fig.colorbar(im1)
fig.savefig('{}_decomposed_grid.png'.format(cpd.comm.rank))



xc_list, yc_list = cpd.create_centroid_list(max_window_size)

cpd.reset_priors()
betaN, ztN, dzN, CN = cpd.optimise_routine(max_window_size, xc_list, yc_list, 3.0, 1.0, 20.0, 5.0, taper=None)

print("{} min/max dz {}".format(cpd.comm.rank, (dzN.min(), dzN.max())))