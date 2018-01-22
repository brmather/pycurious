# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma, kv, lambertw
from scipy.optimize import minimize
from scipy.signal import tukey
import warnings

try: range = xrange
except: pass


class CurieGrid(object):
    """
    Accepts a 2D array and Cartesian coordinates specifying the
    bounding box of the array

    Grid must be projected in metres

    Parameters
    ----------
     grid     : 2D array of magnetic data
     xmin     : minimum x bound in metres
     xmax     : maximum x bound in metres
     ymin     : minimum y bound in metres
     ymax     : maximum y bound in metres
    """
    def __init__(self, grid, xmin, xmax, ymin, ymax):
        
        self.data = np.array(grid)
        ny, nx = self.data.shape
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.xcoords, dx = np.linspace(xmin, xmax, nx, retstep=True)
        self.ycoords, dy = np.linspace(ymin, ymax, ny, retstep=True)
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy

        if not np.isclose(dx, dy):
            warnings.warn("node spacing should be identical {}".format((dx,dy)), RuntimeWarning)


    def subgrid(self, xc, yc, window):
        """
        Extract a subgrid from the data at a window around
        the point (xc,yc)
        
        Parameters
        ----------
         xc      : x coordinate
         yc      : y coordinate
         window  : size of window in metres

        Returns
        -------
         data    : subgrid
        """

        # check in coordinate in grid
        if xc < self.xmin or xc > self.xmax or yc < self.ymin or yc > self.ymax:
            raise ValueError("Point ({},{}) outside data range".format(xc,yc))

        # find nearest index to xc,yc
        ix = np.abs(self.xcoords - xc).argmin()
        iy = np.abs(self.ycoords - yc).argmin()

        nw = int(round(window/self.dx))
        n2w = nw//2

        # extract a square window from the data
        imin = ix - n2w
        imax = ix + n2w + 1
        jmin = iy - n2w
        jmax = iy + n2w + 1

        # safeguard if window size is larger than grid
        imin = max(imin, 0)
        imax = min(imax, self.nx)
        jmin = max(jmin, 0)
        jmax = min(jmax, self.ny)
        
        data = self.data[jmin:jmax, imin:imax]

        return data


    def create_centroid_list(self, window, spacingX=None, spacingY=None):
        """
        Create a list of xc,yc values to extract subgrids.

        Parameters
        ----------
         window   : size of the windows in metres
         spacingX : specify spacing in metres in the X direction
                  : (optional) will default to maximum X resolution
         spacingY : specify spacing in metres in the Y direction
                    (optional) will default to maximum Y resolution

        Returns
        -------
         xc_list  : list of x coordinates
         yc_list  : list of y coordinates
        """
        
        nx, ny = grd.nx, grd.ny
        xcoords = self.xcoords
        ycoords = self.ycoords
        
        nw = int(round(window/self.dx))
        n2w = nw//2
        
        # this is the densest spacing possible given the data
        xc = xcoords[n2w:-n2w]
        yc = ycoords[n2w:-n2w]
        
        # but we can alter it if required
        if type(spacingX) != type(None):
            xc = np.arange(xc.min(), xc.max(), spacingX)
        if type(spacingY) != type(None):
            yc = np.arange(yc.min(), yc.max(), spacingY)

        xq, yq = np.meshgrid(xc, yc)
            
        return xq.ravel(), yq.ravel()


    def remove_trend_linear(self, data):
        """
        Remove the best-fitting linear trend from the data

        This may come in handy if the magnetic data has not been
        reduced to the pole.

        Parameters
        ----------
         data    : 2D numpy array

        Returns
        -------
         data    : 2D numpy array
        """
        data = data.copy()
        nr, nc = data.shape
        xq, yq = np.meshgrid(np.arange(0, nc), np.arange(0, nr))

        A = np.column_stack((x.flatten(), y.flatten(), np.ones(x.size)))
        c, resid, rank, sigma = np.linalg.lstsq(A, data.flatten())

        data.flat[:] -= np.dot(A, c)
        return data

    def radial_spectrum(self, subgrid, taper=np.hanning, scale=0.001, **kwargs):
        """
        Compute the radial spectrum for a square grid.

        Parameters
        ----------
         subgrid : window of the original data
                 : (see subgrid method)
         taper   : taper function (np.hanning is default)
                 : set to None for no taper function
         scale   : scaling factor to get k into rad/km
                 : (0.001 by default)
         args    : keyword arguments to pass to taper

        Returns
        -------
         S       : Radial spectrum
         k       : wavenumber in rad/km
         sigma2  : variance of S
        """

        data = subgrid
        nr, nc = data.shape
        nw = nr

        if nr != nc:
            warnings.warn("subgrid is not square {}".format((nr,nc)), RuntimeWarning)


        # control taper
        if taper is None:
            vtaper = 1.0
        else:
            vtaper = np.ones((nr, nc))
            rt = taper(nr, **kwargs)
            ct = taper(nc, **kwargs)

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


    def azimuthal_spectrum(self, subgrid, taper=np.hanning, scale=0.001, theta=5.0, **kwargs):
        """
        Compute azimuthal spectrum for a square grid.

        Parameters
        ----------
         subgrid : window of the original data
                 : (see subgrid method)
         taper   : taper function (np.hanning is default)
                 : set to None for no taper function
         scale   : scaling factor to get k into rad/km
                 : (0.001 by default)
         theta   : angle increment in degrees
         args    : arguments to pass to taper

        Returns
        -------
         S       : Azimuthal spectrum
         k       : wavenumber [rad/km]
         theta   : angles
        """
        import radon
        data = subgrid
        nr, nc = data.shape
        nw = nr

        if nr != nc:
            raise RuntimeWarning("subgrid is not square {}".format((nr,nc)))

        dx_scale = self.dx*scale
        dk = 2.0*np.pi/(nw - 1)/dx_scale

        kbins = np.arange(dk, dk*nw/2, dk)

        dtheta = np.arange(0.0, 180.0, theta)
        sinogram = radon.radon2d(data, np.pi*dtheta/180.0)
        S = np.zeros((dtheta.size, kbins.size))

        # control taper
        if taper is None:
            vtaper = 1.0
        else:
            vtaper = taper(sinogram.shape[0], **kwargs)

        nk = 1 + 2*kbins.size
        for i in range(0, dtheta.size):
            PSD = np.abs(np.fft.fft(vtaper*sinogram[:,i], n=nk))
            S[i,:] = 2.0*np.log( PSD[1:kbins.size+1] )

        return S, kbins, dtheta


# Helper functions to calculate Curie depth

def bouligand2009(beta, zt, dz, kh, C=0.0):
    """
    Calculate the synthetic radial power spectrum of
    magnetic anomalies

    Equation (4) of Bouligand et al. (2009)

    Parameters
    ----------
     beta  : fractal parameter
     zt    : top of magnetic sources
     dz    : thickness of magnetic sources
     kh    : norm of the wave number in the horizontal plane
     C     : field constant (Maus et al., 1997)

    Returns
    -------
     Phi  : radial power spectrum of magnetic anomalies

    References
    ----------
     Bouligand, C., J. M. G. Glen, and R. J. Blakely (2009), Mapping Curie
       temperature depth in the western United States with a fractal model for
       crustal magnetization, J. Geophys. Res., 114, B11104,
       doi:10.1029/2009JB006494
     Maus, S., D. Gordon, and D. Fairhead (1997), Curie temperature depth
       estimation using a self-similar magnetization model, Geophys. J. Int.,
       129, 163–168, doi:10.1111/j.1365-246X.1997.tb00945.x
    """
    # from scipy.special import kv
    khdz = kh*dz
    coshkhdz = np.cosh(khdz)

    Phi1d = C - 2.0*kh*zt - (beta-1.0)*np.log(kh) - khdz
    A = np.sqrt(np.pi)/gamma(1.0+0.5*beta) * \
        (0.5*coshkhdz*gamma(0.5*(1.0+beta)) - \
        kv((-0.5*(1.0+beta)), khdz) * np.power(0.5*khdz,(0.5*(1.0+beta)) ))
    Phi1d += np.log(A)
    return Phi1d


def maus1995(beta, zt, kh, C=0.0):
    """
    Calculate the synthetic radial power spectrum of
    magnetic anomalies (Maus and Dimri; 1995)

    This is not all that useful except when testing
    overflow errors which occur for the second term
    in Bouligand et al. (2009).

    Parameters
    ----------
     beta  : fractal parameter
     zt    : top of magnetic sources
     kh    : norm of the wave number in the horizontal plane
     C     : field constant (Maus et al., 1997)

    Returns
    -------
     Phi  : radial power spectrum of magnetic anomalies

    References
    ----------
     Bouligand, C., J. M. G. Glen, and R. J. Blakely (2009), Mapping Curie
       temperature depth in the western United States with a fractal model for
       crustal magnetization, J. Geophys. Res., 114, B11104,
       doi:10.1029/2009JB006494
     Maus, S., D. Gordon, and D. Fairhead (1997), Curie temperature depth
       estimation using a self-similar magnetization model, Geophys. J. Int.,
       129, 163–168, doi:10.1111/j.1365-246X.1997.tb00945.x
    """
    return C - 2.0*kh*zt - (beta-1.0)*np.log(kh)
