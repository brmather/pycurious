import numpy as np
import pycurious

# load x,y,anomaly
mag_data = np.loadtxt("../pycurious/Examples/data/test_mag_data.txt")

nx, ny = 305, 305

x = mag_data[:,0]
y = mag_data[:,1]
d = mag_data[:,2].reshape(ny,nx)

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

# initialise CurieOptimise object
grid = pycurious.CurieOptimise(d, xmin, xmax, ymin, ymax)


# get centroids
window_size = 200e3
xc_list, yc_list = grid.create_centroid_list(window_size, spacingX=10e3, spacingY=10e3)

print("number of centroids = {}".format(len(xc_list)))


beta, zt, dz, C = grid.optimise_routine(window_size, xc_list, yc_list)
print "done with optimise_routine"

result = grid.parallelise_routine(window_size, xc_list, yc_list, grid.sensitivity, 5)
print result
result1 = np.array(result)
print result1.shape
# print result1[:,0]
# print result1[:,1]
# print result1[:,:,0]

# result2 = np.hstack(result)
# print result2.shape
# print result2[0]

# print grid.sensitivity(window_size, xc_list[0], yc_list[0], 5)



import matplotlib.pyplot as plt
# get dimensions of domain
xcoords = np.unique(xc_list)
ycoords = np.unique(yc_list)
nc, nr = xcoords.size, ycoords.size

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(17,3.))

im1 = ax1.imshow(beta.reshape(nr,nc))
im2 = ax2.imshow(zt.reshape(nr,nc))
im3 = ax3.imshow(dz.reshape(nr,nc))
im4 = ax4.imshow(C.reshape(nr,nc))

fig.colorbar(im1, ax=ax1, label=r"$\beta$")
fig.colorbar(im2, ax=ax2, label=r"$z_t$")
fig.colorbar(im3, ax=ax3, label=r"$\Delta z$")
fig.colorbar(im4, ax=ax4, label=r"$C$")

fig.savefig("curie_depth.png")