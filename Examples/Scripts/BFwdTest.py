import numpy as np
import matplotlib.pyplot as plt
import pycurious
# %matplotlib inline

#xmin = 600000.
#xmax = 660000.
#ymin = 780000.
#ymax = 835000.

#nx = int((xmax - xmin)/200.)
#ny = int((ymax - ymin)/200.)
#xcoords, dx = np.linspace(xmin, xmax, nx, retstep=True)
#ycoords, dy = np.linspace(ymin, ymax, ny, retstep=True)
#xq, yq = np.meshgrid(xcoords, ycoords)
# importing the data and gridding is time consuming
# see if the grid exists first, otherwise proceed gridding
# ORIGINAL
# try:
    # with np.load('mag_grid.npz') as f:
        # mag_grid = f['data']
# except:
    # from scipy.interpolate import griddata
    # # import data  X | Y | Z | nT
    # xc, yc, zc, mag = np.loadtxt('./mag2.xyz', dtype=np.float32, unpack=True)
    # mag_grid = griddata(np.column_stack([xc, yc]), mag, (xq, yq))
    # # save grid
    # np.savez_compressed('mag_grid.npz', data=mag_grid)

# NEW
try:
    # with np.load('mag_grid.npz') as f:
    # with np.load('mag2.npz') as f:
    with np.load('tst_data.npz') as f:
        print('tst_data.npz found...')
        mag_grid = f['data']
        xmin = f['xm']
        xmin = xmin[()]
        xmax = f['xM']
        xmax = xmax[()]
        ymin = f['ym']
        ymin = ymin[()]
        ymax = f['yM']
        ymax = ymax[()]
    
        # print(mag2[:,0])
        # xc, yc, zc, mag = mag2
        # mag_grid = griddata(np.column_stack([mag2[:,0], mag2[:,1]]), mag2[:,3], (xq, yq))
except:
    print('making tst_data.npz...')
    from scipy.interpolate import griddata
    # import data  X | Y | Z | nT    
    xc, yc, mag = np.loadtxt('./test_mag_data.txt', dtype=np.float32, unpack=True)
    xmin = np.nanmin(xc[0])
    xmax = np.nanmax(xc[-1])
    ymin = np.nanmin(yc[0])
    ymax = np.nanmax(yc[-1])
    
    nx = int((xmax - xmin)/1000.)
    ny = int((ymax - ymin)/1000.)
    #print(xmin,xmax,ymin,ymax,nx,ny)
    xcoords, dx = np.linspace(xmin, xmax, nx, retstep=True)
    ycoords, dy = np.linspace(ymin, ymax, ny, retstep=True)
    xq, yq = np.meshgrid(xcoords, ycoords)
    mag_grid = griddata(np.column_stack([xc, yc]), mag, (xq, yq),method='nearest')
    #mag_grid = griddata(np.column_stack([mag2[:,0], mag2[:,1]]), mag2[:,3], (xq, yq),method='nearest')
    # save grid
    np.savez_compressed('tst_data.npz', data=mag_grid,xm=xmin,xM=xmax,ym=ymin,yM=ymax)

#print(mag_grid.shape,xmin,xmax,ymin,ymax)
#plt.imshow(mag_grid, extent=(xmin, xmax, ymin, ymax), origin='lower')
#plt.show()
#exit()
    

## plt.imshow(mag_grid, extent=(xmin, xmax, ymin, ymax), origin='lower')
## initialise CurieGrid object
grid = pycurious.CurieGrid(mag_grid, xmin, xmax, ymin, ymax)


## Let's not!'
## let's pick a point in the centre of the study area
xpt = xmin + (xmax-xmin)/2
ypt = ymin + (ymax-ymin)/2

subgrid = grid.subgrid(xpt, ypt, 125000.)
S, k, sigma2 = grid.radial_spectrum(subgrid, taper=None)

DK=0.025
nb=20
hmap=np.zeros((nb,nb))
ZT=np.zeros((nb,1))
ZO=np.zeros((nb,1))
for j in range(1,nb,1):
    for l in range(1,nb,1):
        print((j-1)*0.05, (j)*0.05,(l-1)*0.05, (l)*0.05)
        Zb,eZb, Ztr, Zor = grid.Tanakac(S, k, sigma2,(j-1)*DK, (j)*DK,(l-1)*DK, (l)*DK)
        hmap[j,l]=Zb
        ZO[l]=Zor
    ZT[j]=Ztr

fig2 = plt.figure(figsize=(8,8))
ax2 = fig2.add_subplot(111, xlabel=r'$\nu$ window of $\Phi/|k|$ (km$^{-1}$)', ylabel=r'$\nu$ window of $\Phi$')
#im2 = ax2.imshow(np.clip(hmap,-30,-10), extent=[0., (nb+1)*DK, 0., (nb+1)*DK], aspect=1)
im2 = ax2.imshow(np.clip(hmap,-30,-10), extent=[0., 0.5, 0., 0.5], aspect=1)
fig2.colorbar(im2, label='CPD')
plt.title('Heatmap of CPD')
plt.savefig('Heatmap.jpg')
plt.show()




#f.write(str(xpt))
# compute radial spectrum

Zb,eZb = grid.Tanaka(S, k, sigma2,kmin=0.05, kmax=0.25)

print(Zb,eZb)
#exit()

#print(xpt,ypt)
## get a subsection of the grid
#subgrid = grid.subgrid(xpt, ypt, 50000.)
#print(subgrid.shape)
## compute radial spectrum
#S, k, sigma2 = grid.radial_spectrum(subgrid, taper=None)
## fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
## ax1.plot(k, abs(S))
##ax2.plot(sf, abs(S))
#Zb,eZb = grid.Tanaka(S, k, sigma2,kmin=0.05, kmax=0.2)
#print(Zb,eZb)

## INSTEAD
## Let's read a central point from file pts.xy
#cx,cy=np.loadtxt('pts.xy',unpack=True)
#f=open("CPD_pts.txt","w+")
#for xpt, ypt in zip(cx,cy):
	### get a subsection of the grid
    #subgrid = grid.subgrid(xpt, ypt, 75000.)
    #if subgrid.shape[0] != subgrid.shape[1]:
        #continue
    ##f.write(str(xpt))
    ## compute radial spectrum
    #try:
        #S, k, sigma2 = grid.radial_spectrum(subgrid, taper=None)
        #Zb,eZb = grid.Tanaka(S, k, sigma2,kmin=0.05, kmax=0.2)
        ##print(xpt,ypt,Zb,eZb)
        #f.write(' '.join([str(xpt),str(ypt),str(Zb),str(eZb),"\n"]))
    #except:
        #print('Error for ',xpt,ypt)
    ##break

#f.close()
