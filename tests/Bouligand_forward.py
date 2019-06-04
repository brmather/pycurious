## Code to generate test magnetic data for testing of PyCurious methods
## Converted from MATLAB code by R Delhaye. 
#Original code:
#Claire Bouligand
#03-19-2007
#05-09-2014
#Modified to use less memory

import numpy as np
import matplotlib.pyplot as plt
import pycurious
import time

start_time=time.time()
cm=1e-7

##########################################################
#A VERIFIER: multiplier facteur cm par 4*pi
## TO VERIFY: multiplication factor "cm for 4*pi"
###########################################################
## Degree of fractal
b=-3.0
## Dimension of pixels (km), should be multiples of the smallest
dlx=1.0
dly=1.0
dlz=1.0
dl=min([dlx,dly,dlz])
dx=dlx/dl
dy=dly/dl
dz=dlz/dl

#nombre de pixels
## Number of pixels.
nx=305
ny=305
nz=10
nmax=int(max([nx*dx,ny*dy,nz*dz]))

#texte="dimensions x : %d / y : %d / z : %d" % (nx*dlx,ny*dly,nz*dlz)
#print(texte)
#disp(texte)

#Parametres de l'aimantation
## Parameters of magnetisation
#M1=1         # aimantation moyenne = 1 A/m ## Average magnetisation
#M2=10^(0.25) # std de l'aimantation = 10^(0.25) A/m ## Standard dev of
#magnetisation
M1=0
M2=0.2
#prof sommet zt # Depth Zt
zt=0.305
#prof base zb ## Depth Zb
zb=zt+nz*dlz

############################################################
#GENERATION DE LA DISTRIB 3D D'AIMANTATION FRACTAL 
#A PARTIR VOLUME DE DIM ID. (nmax,dl) SELON X, Y ET Z

M=np.zeros((nmax,nmax,nmax))
l=[0,(nmax-1)*dl,dl]

#x=np.arange(0,dlx*(nx-1),dlx)
#y=np.arange(0,dly*(nx-1),dly)
#z=np.arange(0,dlz*(nx-1),dlz)

df=1.0/(nmax*dl)
dfx=1.0/(nx*dlx)
dfy=1.0/(ny*dly)
dfz=1.0/(nz*dlz)

#Center indices of matrix = Nyquist
c=(nmax+1)/2
f=np.zeros((int(np.floor((nmax+1))),1))
fx=np.zeros((int(np.floor((nmax+1))),1))
fy=np.zeros((int(np.floor((nmax+1))),1))
fz=np.zeros((int(np.floor((nmax+1))),1))
for i in range(0,int(np.floor((nmax+1)/2))-1):
    f[i]=(i)*df
for i in range(int(np.floor((nmax+1)/2))-1,int(nmax)):
    f[i]=-1.0*(nmax-i)*df
for i in range(0,int(np.floor((nx+1)/2))-1):
    fx[i]=(i)*dfx
for i in range(int(np.floor((nx+1)/2))-1,int(nx)):
    fx[i]=-1.0*(nx-i)*dfx
for i in range(1,int(np.floor((ny+1)/2))-1):
    fy[i]=(i)*dfy
for i in range(int(np.floor((ny+1)/2))-1,int(ny)):
    fy[i]=-1.0*(ny-i)*dfy
for i in range(1,int(np.floor((nz+1)/2))-1):
    fz[i]=(i)*dfz
for i in range(int(np.floor((nz+1)/2))-1,int(nz)):
    fz[i]=-1.0*(nz-i)*dfz

# Normal distribution
M=M2*np.random.randn(nmax,nmax,nmax)+M1

Mf=np.fft.fftn(M)
for i in range(0,nmax-1):
    for k in range(0,nmax-1):
        for l in range(0,nmax-1):
            Mf[i,k,l]=Mf[i,k,l]*((f[i])**2+(f[k])**2+(f[l])**2+0.0000001)**(b/4)
## DC correction - not needed in python, req. in MATLAB        
##Mf(1,1,1)=mean([Mf(2,1,1) Mf(1,2,1) Mf(1,1,2)])

Mi=np.fft.ifftn(Mf)
TFANO=np.zeros((nx,ny), dtype=np.complex)
ANO=np.zeros((nx,ny), dtype=np.complex)
#input("Press Enter to continue.")
## executes fine up to here.
for k in range(0,nz):
    TFHM=np.fft.fft2(Mi[:,:,k])
    z1=zt+k*dlz
    z2=z1+dlz
#    disp(z1)
#    disp(z2)
    for i in range(0,nx):
        for j in range(0,ny):
            fH=float(np.sqrt((fx[i])**2+(fy[j])**2))
            TFANO[i,j]=TFHM[i,j]*2*np.pi*cm*(np.exp(-2*np.pi*fH*z1)-np.exp(-2*np.pi*fH*z2))
    ANO=ANO+np.fft.ifft2(TFANO)

print("--- %s seconds ---" % (time.time() - start_time))

x=np.arange(0.5,0.5+dlx*(nx),dlx)*1000
y=np.arange(0.5,0.5+dly*(nx),dly)*1000
fid=open("test_mag_data.txt","w+")
for i in range(0,nx):
    for j in range(0,ny):
        fid.write(' '.join([str(x[i]),str(y[j]),str(np.real(ANO[i,j])),"\n"]))
        
fid.close()
#plt.pcolor(np.real(ANO))
#plt.show()
#clear TFHM TFANO

#figure (7)
## RD - PROBLEM! "Data inputs must be real". Trying abs(), or real()
##pcolor(ANO)
#pcolor(log10(abs(ANO)))
#shading flat
#colorbar

##Sauvegarde carte :
#ncol=nx  #Number of columns
#nrow=ny  #Number of rows
#xleft=x(1) #x coordinate for left hand corner of grid
#dx=dlx    #delta x
#yleft=y(1) #y coordinate for left hand corner of grid
#dy=dly    #delta y
#gridC=ANO

##save carte_synth_format.mat ncol nrow xleft yleft dx dy gridC
##save carte_synth.mat b M1 M2 dlx dly dlz zt zb MM2 ANO 


