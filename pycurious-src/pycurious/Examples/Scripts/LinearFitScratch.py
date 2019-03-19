import numpy as np
import matplotlib.pyplot as plt
import pycurious
try:
    # with np.load('mag_grid.npz') as f:
    # with np.load('mag2.npz') as f:
    with np.load('mag_grid2.npz') as f:
        print('mag_grid2.npz found...')
        mag_grid = f['data']
        xmin = f['xm']
        xmin = xmin[()]
        xmax = f['xM']
        xmax = xmax[()]
        ymin = f['ym']
        ymin = ymin[()]
        ymax = f['yM']
        ymax = ymax[()]
except:
    print('making mag_grid2.npz...')
    from scipy.interpolate import griddata
    # import data  X | Y | Z | nT    
    with np.load('mag2.npz') as f:
        print('mag2.npz found...')
        mag2 = f['mag2']
    print(mag2[0,])
    xmin = np.nanmin(mag2[:,0],axis=0)
    xmax = np.nanmax(mag2[:,0],axis=0)
    ymin = np.nanmin(mag2[:,1],axis=0)
    ymax = np.nanmax(mag2[:,1],axis=0)
    
    nx = int((xmax - xmin)/200.)
    ny = int((ymax - ymin)/200.)
    print(xmin,xmax,ymin,ymax,nx,ny)
    xcoords, dx = np.linspace(xmin, xmax, nx, retstep=True)
    ycoords, dy = np.linspace(ymin, ymax, ny, retstep=True)
    xq, yq = np.meshgrid(xcoords, ycoords)
    mag_grid = griddata(np.column_stack([mag2[:,0], mag2[:,1]]), mag2[:,3], (xq, yq),method='nearest')
    # save grid
    np.savez_compressed('mag_grid2.npz', data=mag_grid,xm=xmin,xM=xmax,ym=ymin,yM=ymax)

grid = pycurious.CurieGrid(mag_grid, xmin, xmax, ymin, ymax)




xpt=632781
ypt=805755
        ## get a subsection of the grid
subgrid = grid.subgrid(xpt, ypt, 75000.)

S, k, sigma2 = grid.radial_spectrum(subgrid, taper=None)


#ax1.plot(k, abs(S))
# ok, let's plot the interval derivatives of dS/dk
Sn=np.log(np.exp(S)/(k/(2.0*np.pi)))
dS = np.zeros((S.size-1, 1))
dSn = np.zeros((S.size-1, 1))
dk = np.zeros((S.size-1, 1))
#print(S.size,dS.size)
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
for x in range(dS.size):
    dS[x]=(abs(S[x+1])-abs(S[x]))/(k[x+1]-k[x])
    dSn[x]=(abs(Sn[x+1])-abs(Sn[x]))/(k[x+1]-k[x])
    dk[x]=(k[x+1]+k[x])/2
    #print(dS)
    #break
    #print(str((S(x+1)-S(x))/(k(x+1)-k(1))))
# let's compute variance over the same 5 pts...
dSm=running_mean(dS,5)
dSnm=running_mean(dSn,5)
var=np.zeros(dSm.size)
varn=np.zeros(dSm.size)
for x in range(var.size):
    var[x]=np.var(dS[x:x+5])
    varn[x]=np.var(dSn[x:x+5])
#print(var)
#ax2.plot(dk,dS,dk[2:-2],dSm)
plt.figure(figsize=(20,12))
plt.subplot(2,3,1)
plt.plot(k/(2.0*np.pi),S/(2.0*np.pi))
plt.title('$\Phi$')
plt.subplot(2,3,2)
plt.plot(dk/(2.0*np.pi),dS/(2.0*np.pi),dk[2:-2]/(2.0*np.pi),dSm/(2.0*np.pi))
plt.title('$\partial\Phi/\partial k$')
plt.legend(['Original','5pt mean'])
plt.subplot(2,3,3)
plt.plot(dk[2:-2]/(2.0*np.pi),var)
plt.title('$\sigma^2(\partial\Phi/\partial k($')
plt.subplot(2,3,4)
plt.plot(k/(2.0*np.pi),Sn/(2.0*np.pi))
plt.title('$\Phi/|k|$')
plt.xlabel('Spatial Frequency')
plt.subplot(2,3,5)
plt.plot(dk/(2.0*np.pi),dSn/(2.0*np.pi),dk[2:-2]/(2.0*np.pi),dSnm/(2.0*np.pi))
plt.title('$\partial(\Phi/|k|)/\partial k$')
plt.legend(['Original','5pt mean'])
plt.xlabel('Spatial Frequency')
plt.subplot(2,3,6)
plt.plot(dk[2:-2]/(2.0*np.pi),varn)
plt.title('$\sigma^2(\partial(\Phi/|k|)/\partial k($')
plt.xlabel('Spatial Frequency')
plt.savefig('dSpectra.jpg')

f2, ax2 = plt.subplots(1,1,figsize=(12,4))
ax2.plot(k/(2.0*np.pi),S/(2.0*np.pi),k/(2.0*np.pi),np.log(np.exp(S)/(k/(2.0*np.pi)))/(2.0*np.pi))
plt.xlabel('Spatial Frequency$')
plt.legend(['$\Phi$','$\Phi/|k|$'])
plt.title('Spectra')
plt.savefig('Spectra.jpg')
#plt.show()
#exit()
#ax4.plot(k/(2.0*np.pi),np.log(np.exp(S)/(k/(2.0*np.pi)))/(2.0*np.pi))


#print(k/(2.0*np.pi))
DK=0.05
nb=48
hmap=np.zeros((nb,nb))
ZT=np.zeros((nb,1))
ZO=np.zeros((nb,1))
for j in range(1,nb,1):
    for l in range(1,nb,1):
        #print((j-1)*0.05, (j)*0.05)
        Zb,eZb, Ztr, Zor = grid.Tanakac(S, k, sigma2,(j-1)*DK, (j)*DK,(l-1)*DK, (l)*DK)
        hmap[j,l]=Zb
        ZO[l]=Zor
    ZT[j]=Ztr

fig2 = plt.figure(figsize=(8,8))
ax2 = fig2.add_subplot(111, xlabel='$k$ window of $\Phi/|k|$', ylabel='$k$ window of $\Phi$')
#im2 = ax2.imshow(np.clip(hmap,-30,-10), extent=[0., (nb+1)*DK, 0., (nb+1)*DK], aspect=1)
im2 = ax2.imshow(np.clip(hmap,-30,-10), extent=[0., 0.5, 0., 0.5], aspect=1)
fig2.colorbar(im2, label='CPD')
plt.title('Heatmap of CPD')
plt.savefig('Heatmap.jpg')

fig4 = plt.figure(figsize=(12,6))
ax5 = fig4.add_subplot(111,xlabel='k', ylabel='Z (km)')
nk=np.linspace(0, nb*DK, nb)
ax5.plot(nk,ZT,nk,ZO)
plt.legend(['Z_t','Z_o'])
plt.ylim(-40,0)
plt.savefig('Z_dist.jpg')

plt.show()
#plt.clim(-50,0)
#exit()











### INSTEAD
### Let's read a central point from file pts.xy
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
