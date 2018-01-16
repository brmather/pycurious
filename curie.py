# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma, kv, lambertw
from scipy.optimize import minimize
from scipy.signal import turkey
import time

try: range = xrange
except: pass


class CurieGrid(object):
    """
    Accepts a 2D array and Cartesian coordinates specifying the
    bounding box of the array

    Parameters
    ----------
     data     : 2D array of magnetic data
     xmin     : minimum x bound
     xmax     : maximum x bound
     ymin     : minimum y bound
     ymax     : maximum y bound
    """
    def __init__(self, data, xmin, xmax, ymin, ymax):
        
        self.data = np.array(data)
        self.ny, self.nx = self.data.shape
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.xcoords, self.dx = np.linspace(xmin, xmax, nx, retstep=True)
        self.ycoords, self.dy = np.linspace(ymin, ymax, ny, retstep=True)


    def subgrid(self, xc, yc, window):
        """
        Extract a subgrid from the data at a window around
        the point (xc,yc)
        
        Parameters
        ----------
         xc     : x coordinate
         yc     : y coordinate
         window : size of window in metres

        Returns
        -------
         data   : subgrid
         nw     : size of window in integers
        """

        # find nearest index to xc,yc
        ix = np.abs(self.xcoords - xc).argmin()
        iy = np.abs(self.ycoords - yc).argmin()

        nw = int(round(window/self.dx))
        n2w = nw//2

        # extract a square window from the data
        data = self.data[ix-n2w:ix+n2w+1, iy-n2w:iy+n2w+1]

        return data, nw

    def radial_spectrum(self, xc, yc, window, taper=np.hanning, scale=0.001, *args):
        """
        Compute the radial spectrum for a point (xc,yc)
        for a square window

        Parameters
        ----------
         xc     : x coordinate
         yc     : y coordinate
         window : size of window
         taper  : taper function (np.hanning is default)
         scale  : scaling factor to get k into rad/km
                : (0.001 by default)
         args   : arguments to pass to taper

        Returns
        -------
         S      : Radial spectrum
         k      : wavenumber in rad/km
         sigma2 : variance of S
        """

        # check in coordinate in grid
        if xc < self.xmin or xc > self.xmax or yc < self.ymin or yc > self.ymax:
            raise ValueError("Point xc,yc outside data range")

        data, nw = self.subgrid(xc, yc, window)
        nr, nc = data.shape

        # control taper
        if taper is None:
            vtaper = 1.0
        else:
            vtaper = np.ones((nr, nc))
            rt = taper(nr)
            ct = taper(nc)

            for col in range(0, nc):
                vtaper[:,col] *= rt
            for row in range(0, nr):
                vtaper[row,:] *= ct

        dx_scale = self.dx*scale
        dk = 2.0*np.pi/(nw - 1)/dx_scale

        # fast Fourier transform and shift
        FT = np.abs(np.fft.fft2(data*vtaper))
        FT = np.fft.fftshift(FT)

        kbins = np.arange(dk, dk*nw/2, dk)
        nbins = kbins.size - 1

        S = np.zeros(nbins)
        k = np.zeros(nbins)
        sigma2 = np.zeros(nbins)

        i0 = int((nw - 1)//2)
        nw_range = np.arange(0, nw)
        iy, ix = np.meshgrid(nw_range, nw_range)
        kk = np.hypot((ix - i0)*dk, (iy - i0)*dk)

        for i in range(0, nbins):
            mask = np.logical_and(kk >= kbins[i], kk <= kbins[i+1])
            rr = 2.0*np.log(FT[mask])
            S[i] = rr.mean()
            k[i] = kk[mask].mean()
            sigma2[i] = np.std(rr)/np.sqrt(rr.size)

        return S, k, sigma2