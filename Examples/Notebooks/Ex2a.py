import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import pycurious


# load x,y,anomaly
mag_data = np.loadtxt("../data/test_mag_data.txt")

nx, ny = 305, 305

x = mag_data[:,0]
y = mag_data[:,1]
d = mag_data[:,2].reshape(ny,nx)

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

# initialise CurieOptimise object
#grid = pycurious.CurieOptimise(d, xmin, xmax, ymin, ymax)
grid = pycurious.CurieGrid(d, xmin, xmax, ymin, ymax)

# pick the centroid
xpt = 0.5*(xmin + xmax)
ypt = 0.5*(ymin + ymax)

window_size = 304e3
subgrid = grid.subgrid(xpt, ypt, window_size)


S, k, sigma2 = grid.radial_spectrum(subgrid)
#print(S)
S, k, sigma2 = grid.radial_spectrum_log(subgrid)
#print(S)
#S, k, sigma2 = grid.radial_spectrum_log(subgrid)


zb=pycurious.tanaka1999(S, k, sigma2)
print(zb)