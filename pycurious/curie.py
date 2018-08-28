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

        if abs(dx - dy) > 1.0:
            warnings.warn("node spacing should be identical {}".format((dx,dy)), RuntimeWarning)


    def subgrid(self, window, xc, yc):
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
        
        nx, ny = self.nx, self.ny
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
        if type(taper) == type(None):
            vtaper = 1.0
        else:
            rt = taper(nr, **kwargs)
            ct = taper(nc, **kwargs)
            xq, yq = np.meshgrid(ct, rt)
            vtaper = xq*yq

        dx_scale = self.dx*scale
        dk = 2.0*np.pi/(nw - 1)/dx_scale

        # fast Fourier transform and shift
        FT = np.abs(np.fft.fft2(data*vtaper))
        FT = np.fft.fftshift(FT)

        kbins = np.arange(dk, dk*nw/2, dk)
        nbins = kbins.size - 1

        S = np.empty(nbins)
        k = np.empty(nbins)
        sigma2 = np.empty(nbins)

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


        sigma_S = np.sqrt(sigma2)

        return k, S, sigma_S


    def radial_spectrum_log(self, subgrid, taper=np.hanning, scale=0.001, **kwargs):
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
        if type(taper) == type(None):
            vtaper = 1.0
        else:
            rt = taper(nr, **kwargs)
            ct = taper(nc, **kwargs)
            xq, yq = np.meshgrid(ct, rt)
            vtaper = xq*yq

        dx_scale = self.dx*scale
        dk = 2.0*np.pi/(nw - 1)/dx_scale

        # fast Fourier transform and shift
        FT = np.abs(np.fft.fft2(data*vtaper))
        FT = np.fft.fftshift(FT)

        kbins = np.arange(dk, dk*nw/2, dk)
        nbins = kbins.size - 1

        S = np.empty(nbins)
        k = np.empty(nbins)
        sigma2 = np.empty(nbins)

        i0 = int((nw - 1)//2)
        ix, iy = np.mgrid[0:nw,0:nw]
        kk = np.hypot((ix - i0)*dk, (iy - i0)*dk)

        for i in range(0, nbins):
            mask = np.logical_and(kk >= kbins[i], kk <= kbins[i+1])
            rr = np.log(np.sqrt(FT[mask]))
            S[i] = rr.mean()
            k[i] = kk[mask].mean()
            sigma2[i] = np.std(rr)/np.sqrt(rr.size)

        sigma_S = np.sqrt(sigma2)

        return k, S, sigma_S
        
        
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
        from pycurious import radon
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
        sigma2 = np.zeros((dtheta.size, kbins.size))

        # control taper
        if taper is None:
            vtaper = 1.0
        else:
            vtaper = taper(sinogram.shape[0], **kwargs)

        nk = 1 + 2*kbins.size
        for i in range(0, dtheta.size):
            PSD = np.abs(np.fft.fft(vtaper*sinogram[:,i], n=nk))
            S[i,:] = np.log( np.sqrt(PSD[1:kbins.size+1] ))
        
        return kbins, S, dtheta


    def reduce_to_pole(self, data, inc, dec, sinc=None, sdec=None):
        """
        Reduce total field magnetic anomaly data to the pole.

        The reduction to the pole if a phase transformation that can be
        applied to total field magnetic anomaly data. It simulates how
        the data would be if both the Geomagnetic field and the
        magnetization of the source were vertical (Blakely, 1996).

        Parameters
        ----------
         data : 1d-array
            The total field anomaly data at each point.
         inc, dec : floats
            The inclination and declination of the inducing Geomagnetic field
         sinc, sdec : floats (optional)
            The inclination and declination of the total magnetization of the
            anomaly source. The total magnetization is the vector sum of the
            induced and remanent magnetization. If there is only induced
            magnetization, use the *inc* and *dec* of the Geomagnetic field.

        Returns
        -------
         rtp : 2d-array
            The data reduced to the pole.
        
        References
        ----------
         Blakely, R. J. (1996), Potential Theory in Gravity and Magnetic
         Applications, Cambridge University Press.

        Notes
        -----
         This functions performs the reduction in the frequency domain
         (using the FFT). The transform filter is (in the freq domain):
             RTP(k_x, k_y) = \frac{|k|}{
                 a_1 k_x^2 + a_2 k_y^2 + a_3 k_x k_y +
                 i|k|(b_1 k_x + b_2 k_y)}
         in which k_x and k_y are the wave-numbers in the x and y
         directions and
             |k| = \sqrt{k_x^2 + k_y^2} \\
             a_1 = m_z f_z - m_x f_x \\
             a_2 = m_z f_z - m_y f_y \\
             a_3 = -m_y f_x - m_x f_y \\
             b_1 = m_x f_z + m_z f_x \\
             b_2 = m_y f_z + m_z f_y
         \mathbf{m} = (m_x, m_y, m_z)` is the unit-vector of the total
         magnetization of the source and
         \mathbf{f} = (f_x, f_y, f_z)` is the unit-vector of the
         Geomagnetic field.
        """
        nr, nc = data.shape

        if nr != nc:
            warnings.warn("subgrid is not square {}".format((nr,nc)), RuntimeWarning)

        fx, fy, fz = ang2vec(1.0, inc, dec)
        if sinc is None or sdec is None:
            mx, my, mz = fx, fy, fz
        else:
            mx, my, mz = ang2vec(1.0, sinc, sdec)

        kx, ky = [k for k in _fftfreqs(self.dx, self.dy, data.shape)]
        kz = np.hypot(kx, ky)

        a1 = mz*fz - mx*fx
        a2 = mz*fz - my*fy
        a3 = -my*fx - mx*fy
        b1 = mx*fz + mz*fx
        b2 = my*fz + mz*fy

        # The division gives a RuntimeWarning because of the zero frequency term.
        # This suppresses the warning.
        with np.errstate(divide='ignore', invalid='ignore'):
            rtp = (kz)/(a1*kx**2 + a2*ky**2 + a3*kx*ky + \
                            1j*np.sqrt(kz)*(b1*kx + b2*ky))
        
        rtp[0, 0] = 0
        ft_pole = rtp*np.fft.fft2(data)
        return np.real(np.fft.ifft2(ft_pole))


    def upward_continuation(self, data, height):
        """
        Upward continuation of potential field data.

        Calculates the continuation through the Fast Fourier Transform in
        the wavenumber domain (Blakely, 1996):

            F\{h_{up}\} = F\{h\} e^{-\Delta z |k|}

        and then transformed back to the space domain. :math:`h_{up}` is the
        upward continue data, \Delta z is the height increase, F denotes
        the Fourier Transform, |k| is the wavenumber modulus.

        Parameters
        ----------
         data : 2D grid
            The potential field at the grid points
         height : float
            The height increase (delta z) in meters.
        
        Returns
        -------
         cont : array
            The upward continued data

        Notes
        -----
        It is not possible to get the FFT of a masked grid. The default
        :func:`fatiando.gridder.interp` call using minimum curvature will
        not be suitable.  Use extrapolate=True or algorithm='nearest' to
        get an unmasked grid.

        References
        ----------
        Blakely, R. J. (1996), Potential Theory in Gravity and Magnetic
        Applications, Cambridge University Press.
        """
        nr, nc = data.shape

        if nr != nc:
            warnings.warn("subgrid is not square {}".format((nr,nc)), RuntimeWarning)

        if height <= 0:
            warnings.warn("Using 'height' <= 0 means downward continuation, " +
                          "which is known to be unstable.")

        fx = 2.0*np.pi*np.fft.fftfreq(nr, self.dx)
        fy = 2.0*np.pi*np.fft.fftfreq(nc, self.dy)

        kx, ky = np.meshgrid(fy, fx)[::-1]
        kz = np.hypot(kx, ky)

        upcont_ft = np.fft.fft2(data)*np.exp(-height*kz)
        cont = np.real(np.fft.ifft2(upcont_ft))
        return cont


# Helper functions to calculate Curie depth

def bouligand2009(kh, beta, zt, dz, C):
    """
    Calculate the synthetic radial power spectrum of
    magnetic anomalies

    Equation (4) of Bouligand et al. (2009)

    Parameters
    ----------
     kh    : norm of the wave number in the horizontal plane
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
       129, 163-168, doi:10.1111/j.1365-246X.1997.tb00945.x
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


def tanaka1999(k, Phi, sigma_Phi, kmin_range=(0.05, 0.2), kmax_range=(0.05, 0.2)):
    """
 Compute weighted linear fit of S over spatial frequency window kmin:kmax

    Parameters
    ----------
     S      : Power spectrum of signal (see radial spectrum), expected in ln(sqrt(S)) form
     k      : Wavenumber
     sigma2 : Errors of S
     kmin,kmax  : Minimum and maximum spatial frequencies to fit. Ideally low frequency, straight-ish line

    Returns
    -----------
     Zb     : Estimated Curie point depth, as per Tanaka et al, 1999
     eZb    : Error of Zb
            : No geothermal gradient, trivial to estimate.
            : Possible to extend, return fitted components?
    """
    # for now...
    S = Phi
    sigma2 = sigma_Phi**2

    def compute_coefficients(X, Y, E):
        X2 = X**2
        Y2 = Y**2
        E2 = E**2

        XY = np.multiply(X, Y)
        XE2sum = np.sum(X/E2)
        YE2sum = np.sum(Y/E2)
        rE2sum = np.sum(1.0/E2)
        X2E2sum = np.sum(X2/E2)
        Y2E2sum = np.sum(Y2/E2)

        #TL = XE2sum*YE2sum - np.sum(XY/E2*rE2sum)
        # I think summation in second TL term needed to be split
        TL = XE2sum*YE2sum - np.sum(XY/E2)*rE2sum
        BL = XE2sum**2 - X2E2sum*rE2sum

        Z  = TL/BL
        b  = (np.sum(XY/E2) - Z*X2E2sum)/XE2sum
        #dZ = np.sqrt( rE2sum/(X2E2sum*rE2sum - XE2sum) )
        ## There was a missing **2 term at end of error term.
        dZ = np.sqrt( rE2sum/(X2E2sum*rE2sum - XE2sum**2) )
        return Z, b, dZ


    sf=k/(2.0*np.pi)
    S2=np.log(np.exp(S)/sf)

    # mask low wavenumbers
    kmin, kmax = kmin_range
    mask1 = np.logical_and(sf >=kmin, sf <=kmax)
    X1 = sf[mask1]
    Y1 = S[mask1]
    E1 = sigma2[mask1]
    
    # mask high wavenumbers
    kmin, kmax = kmax_range
    mask2 = np.logical_and(sf >=kmin, sf <=kmax)
    X2 = sf[mask2]
    Y2 = np.log(np.exp(S[mask2])/(X2*2*np.pi))
    E2 = np.log(np.exp(sigma2[mask2])/(X2*2*np.pi))

    # compute top and bottom of magnetic layer
    Ztr, btr, dZtr = compute_coefficients(X1, Y1, E1)
    Zor, bor, dZor = compute_coefficients(X2, Y2, E2)
    return (Ztr,btr,dZtr), (Zor, bor, dZor)
    
    
def ComputeTanaka(Ztr, dZtr, Zor, dZor):
    Zb = 2.0*Zor - Ztr
    dZb  = 2.0*dZor - dZtr
    return abs(Zb), dZb


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
       129, 163-168, doi:10.1111/j.1365-246X.1997.tb00945.x
    """
    return C - 2.0*kh*zt - (beta-1.0)*np.log(kh)


def _fftfreqs(dx, dy, shape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    fx = 2.0*np.pi*np.fft.fftfreq(shape[0], dx)
    fy = 2.0*np.pi*np.fft.fftfreq(shape[1], dy)
    return np.meshgrid(fy, fx)[::-1]

def ang2vec(intensity, inc, dec):
    """
    Convert intensity, inclination and  declination to a 3-component vector

    Parameter
    ---------
     intensity : float or array
        The intensity (norm) of the vector
     inc : float
        The inclination of the vector (in degrees)
     dec : float
        The declination of the vector (in degrees)

    Returns
    -------
     vec : array = [x, y, z]
        The vector

    Notes
    -----
     Coordinate system is assumed to be x->North, y->East, z->Down.
     Inclination is positive down and declination is measured with respect
     to x (North).

    Examples
    --------
        >>> import numpy
        >>> print ang2vec(3, 45, 45)
        [ 1.5         1.5         2.12132034]
        >>> print ang2vec(numpy.arange(4), 45, 45)
        [[ 0.          0.          0.        ]
         [ 0.5         0.5         0.70710678]
         [ 1.          1.          1.41421356]
         [ 1.5         1.5         2.12132034]]

    """
    return np.transpose([intensity * i for i in dircos(inc, dec)])


def dircos(inc, dec):
    """
    Returns the 3 coordinates of a unit vector given its inclination and
    declination.

    Parameters
    ----------
     inc : float
        The inclination of the vector (in degrees)
     dec : float
        The declination of the vector (in degrees)

    Returns
    -------
     vect : list = [x, y, z]
        The unit vector

    Notes
    -----
     Coordinate system is assumed to be x->North, y->East, z->Down.
     Inclination is positive down and declination is measured with respect
     to x (North).
    """
    d2r = np.pi/180.0
    vect = [np.cos(d2r * inc) * np.cos(d2r * dec), \
            np.cos(d2r * inc) * np.sin(d2r * dec), \
            np.sin(d2r * inc)]
    return vect
