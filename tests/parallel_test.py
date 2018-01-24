import numpy as np
import pycurious

xmin = 0.0
xmax = 1.0
ymin = 0.0
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

print("{} ghost range {}".format(cpd.comm.rank, cpd.dm.getGhostRanges()))
print("{} min {}".format(cpd.comm.rank, cpd.coords.min(axis=0)))
print("{} max {}".format(cpd.comm.rank, cpd.coords.max(axis=0)))