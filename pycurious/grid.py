# Copyright 2018-2019 Ben Mather, Robert Delhaye
#
# This file is part of PyCurious.
#
# PyCurious is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# PyCurious is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PyCurious.  If not, see <http://www.gnu.org/licenses/>.

"""
This PyCurious module contains the `pycurious.grid.CurieGrid` class,
which can be initialised with a magnetic grid of equal spacing in the x and y direction.
It contains methods for the following functionality:

- Decomposition of subgrids for processing square windows of the magnetic anomaly
- Removing linear trends from the magnetic anomaly
- Upward continuation
- Reduction to the pole

Other functions within this module are useful to compute analytical solutions
of the radial power spectrum, \\( \\Phi \\) according to Bouligand *et al.* (2009),
Maus and Dimri (1995), and the decomposition of \\( \\Phi \\) from the magnetic
anomaly according to Tanaka *et al.* (1999):

- `bouligand2009`: analytic solution used in `pycurious.optimise.CurieOptimise`
- `maus1995`: simplified version of `bouligand2009` without higher order integration.
- `tanaka1999`: to be used in conjunction with `ComputeTanaka`

"""

# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma, kv
import warnings

try:
    range = xrange
except:
    pass


class CurieGrid(object):
    """
    Accepts a 2D array and Cartesian coordinates specifying the
    bounding box of the array

    Grid must be projected in metres.

    Args:
        grid : 2D numpy array
            2D array of magnetic data
        xmin : float
            minimum x bound in metres
        xmax : float
            maximum x bound in metres
        ymin : float
            minimum y bound in metres
        ymax : float
            maximum y bound in metres

    Attributes:
        grid : 2D numpy array
            2D array of magnetic data
        xmin : float
            minimum x bound in metres
        xmax : float
            maximum x bound in metres
        ymin : float
            minimum y bound in metres
        ymax : float
            maximum y bound in metres
        dx : float
            grid spacing in the x-direction in metres
        dy : float
            grid spacing in the y-direction in metres
        nx : int
            number of nodes in the x-direction
        ny : int
            number of nodes in the y-direction
        xcoords : 1D numpy array
            1D numpy array of coordinates in the x-direction
        ycoords : 1D numpy array
            1D numpy array of coordinates in the y-direction

    Notes:
        In all instances `x` indicates eastings in metres and `y` indicates northings.
        Using a grid of longitude / latitudinal coordinates (degrees) will result
        in incorrect Curie depth calculations.
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

        if not np.allclose(dx, dy, 1.0):
            raise ValueError("node spacing should be identical {}".format((dx, dy)))

    def subgrid(self, window, xc, yc):
        """
        Extract a subgrid from the data at a window around
        the point (xc,yc)

        Args:
            xc : float
                x coordinate
            yc : float
                y coordinate
            window : float
                size of window in metres

        Returns:
            data : 2D array
                subgrid encompassing window size
        """

        # check whether coordinate is inside grid
        if xc < self.xmin or xc > self.xmax or yc < self.ymin or yc > self.ymax:
            raise ValueError("Point {} outside data range".format((xc, yc)))

        # find nearest index to xc,yc
        ix = np.abs(self.xcoords - xc).argmin()
        iy = np.abs(self.ycoords - yc).argmin()

        nw = int(round(window / self.dx))
        n2w = nw // 2

        # extract a square window from the data
        imin = ix - n2w
        imax = ix + n2w + 1
        jmin = iy - n2w
        jmax = iy + n2w + 1

        # check whether window fits inside grid
        if imin < 0 or imax > self.nx or jmin < 0 or jmax > self.ny:
            raise ValueError(
                "Window size {} at centroid {} exceeds the data range".format(
                    window, (xc, yc)
                )
            )

        data = self.data[jmin:jmax, imin:imax]

        return data

    def create_centroid_list(self, window, spacingX=None, spacingY=None):
        """
        Create a list of xc,yc values to extract subgrids.

        Args:
            window : float
                size of the windows in metres
            spacingX : float (optional)
                specify spacing in metres in the X direction
                will default to maximum X resolution
            spacingY : float (optional)
                specify spacing in metres in the Y direction
                will default to maximum Y resolution

        Returns:
            xc_list : 1D array
                array of x coordinates
            yc_list : 1D array
                array of y coordinates
        """
        xcoords = self.xcoords
        ycoords = self.ycoords

        nw = int(round(window / self.dx))
        n2w = nw // 2

        # this is the densest spacing possible given the data
        xc = xcoords[n2w:-n2w]
        yc = ycoords[n2w:-n2w]

        # but we can alter it if required
        if spacingX is not None:
            xc = np.arange(xc.min(), xc.max(), spacingX)
        if spacingY is not None:
            yc = np.arange(yc.min(), yc.max(), spacingY)

        xq, yq = np.meshgrid(xc, yc)

        return xq.ravel(), yq.ravel()

    def remove_trend_linear(self, data):
        """
        Remove the best-fitting linear trend from the data

        This may come in handy if the magnetic data has not been
        reduced to the pole.

        Args:
            data : 2D numpy array

        Returns:
            data : 2D numpy array
        """
        nr, nc = data.shape
        yq, xq = np.mgrid[0:nc, 0:nr]
        A = np.c_[xq.ravel(), yq.ravel(), np.ones(xq.size)]
        c, resid, rank, sigma = np.linalg.lstsq(A, data.ravel(), rcond=None)
        return data - np.dot(A, c).reshape(data.shape)

    def _taper_spectrum(self, subgrid, taper=np.hanning, scale=0.001, **kwargs):
        """
        Template for tapering the power spectrum used in:

        - `radial_spectrum`
        - `radial_spectrum_log`
        - `azimuthal_spectrum`
        """
        data = subgrid
        nr, nc = data.shape

        if nr != nc:
            warnings.warn("subgrid is not square {}".format((nr, nc)), RuntimeWarning)

        # control taper
        if taper is None:
            vtaper = 1.0
        else:
            rt = taper(nr, **kwargs)
            ct = taper(nc, **kwargs)
            xq, yq = np.meshgrid(ct, rt)
            vtaper = xq * yq

        # scaling factor to transform wavenumber into units of rad/km
        dx_scale = self.dx * scale
        dk = 2.0 * np.pi / (nr - 1) / dx_scale

        kbins = np.arange(dk, dk * nr / 2, dk)
        return vtaper, dk, kbins

    def _FFT_spectrum(self, subgrid, vtaper, dk, kbins, const):
        """
        Template for computing the (fast) Fourier transform used in:

        - `radial_spectrum`
        - `radial_spectrum_log`

        A constant `const` should be applied to the FFT of the magnetic anomaly
        to convert `S` and `sigma` to specific units for further analysis.

        It is useful to remember that:

        ```python
        2*log(FFT) == log(FFT**2)
        ```
        """
        data = subgrid
        nr, nc = data.shape

        nbins = kbins.size - 1

        # fast Fourier transform and shift
        FT = np.abs(np.fft.fft2(data * vtaper))
        FT = np.fft.fftshift(FT)

        S = np.empty(nbins)
        k = np.empty(nbins)
        sigma = np.empty(nbins)

        i0 = int((nr - 1) // 2)
        ix, iy = np.mgrid[0:nr, 0:nr]
        kk = np.hypot((ix - i0) * dk, (iy - i0) * dk)

        for i in range(nbins):
            mask = np.logical_and(kk >= kbins[i], kk <= kbins[i + 1])
            rr = const * np.log(FT[mask])
            S[i] = rr.mean()
            k[i] = kk[mask].mean()
            sigma[i] = np.std(rr)

        return k, S, sigma

    def radial_spectrum(self, subgrid, taper=np.hanning, power=2.0, **kwargs):
        """
        Compute the radial spectrum for a square grid.

        > Wavenumber is returned in values of __rad/km__

        Args:
            subgrid : 2D array
                window of the original data (see subgrid method)
            taper : function (default=np.hanning)
                taper function, set to None for no taper function
            power : float
                raise the FFT of the magnetic anomaly to the power.
                - 2.0 for Bouligand _et al._ (2009) use cases
                - 0.5 for Tanaka _et al.__ (1999) use cases
            kwargs : keyword arguments
                keyword arguments to pass to `taper`

        Returns:
            k : 1D array shape (n,)
                wavenumber in rad/km
            Phi : 1D array shape (n,)
                Radial power spectrum
            sigma_Phi : 1D array shape (n,)
                Standard deviation of Phi

        Notes:
            While `subgrid` is projected in eastings / northings (in metres),
            the wavenumber, \\( k \\), is returned in units of rad/km.
            This is because both Bouligand *et al.* (2009) and Tanaka *et al.*
            (1999) require the computation of Curie depth in these units.

        References:
            Bouligand, C., J. M. G. Glen, and R. J. Blakely (2009), Mapping Curie
            temperature depth in the western United States with a fractal model for
            crustal magnetization, J. Geophys. Res., 114, B11104,
            doi:10.1029/2009JB006494

            Tanaka, A., Okubo, Y., & Matsubayashi, O. (1999). Curie point depth
            based on spectrum analysis of the magnetic anomaly data in East and
            Southeast Asia. Tectonophysics, 306(3–4), 461–470.
            doi:10.1016/S0040-1951(99)00072-4
        """

        # bin the spectrum and compute the taper
        vtaper, dk, kbins = self._taper_spectrum(subgrid, taper, **kwargs)

        # calculate the Fourier transform and apply scaling constant to retrieve
        # values compatible with Bouligand or Tanaka analysis
        return self._FFT_spectrum(subgrid, vtaper, dk, kbins, power)

    def azimuthal_spectrum(
        self, subgrid, taper=np.hanning, power=2.0, theta=5.0, **kwargs
    ):
        """
        Compute azimuthal spectrum for a square grid.

        > Wavenumber is returned in values of __rad/km__

        Args:
            subgrid : 2D array
                window of the original data (see subgrid method)
            taper : function (default=np.hanning)
                taper function, set to None for no taper function
            theta : float
                angle increment in degrees
            args : arguments
                arguments o pass to taper

        Returns:
            k : 1D array shape (n,)
                wavenumber in rad/km
            Phi : 1D array shape (n,)
                Radial power spectrum
            sigma_Phi : 1D array shape (n,)
                Standard deviation of Phi

        Notes:
            While `subgrid` is projected in eastings / northings (in metres),
            the wavenumber, \\( k \\), is returned in units of rad/km.
            This is because both Bouligand *et al.* (2009) and Tanaka *et al.*
            (1999) require the computation of Curie depth in these units.

        References:
            Bouligand, C., J. M. G. Glen, and R. J. Blakely (2009), Mapping Curie
            temperature depth in the western United States with a fractal model for
            crustal magnetization, J. Geophys. Res., 114, B11104,
            doi:10.1029/2009JB006494

            Tanaka, A., Okubo, Y., & Matsubayashi, O. (1999). Curie point depth
            based on spectrum analysis of the magnetic anomaly data in East and
            Southeast Asia. Tectonophysics, 306(3–4), 461–470.
            doi:10.1016/S0040-1951(99)00072-4
        """
        from pycurious import radon

        vtaper, dk, kbins = self._taper_spectrum(subgrid, taper, **kwargs)

        dtheta = np.arange(0.0, 180.0, theta)
        sinogram = radon.radon2d(subgrid, np.pi * dtheta / 180.0)
        S = np.zeros((dtheta.size, kbins.size))

        # control taper
        if taper is None:
            vtaper = 1.0
        else:
            vtaper = taper(sinogram.shape[0], **kwargs)

        nk = 1 + 2 * kbins.size
        for i in range(0, dtheta.size):
            PSD = np.abs(np.fft.fft(vtaper * sinogram[:, i], n=nk))
            S[i, :] = power * np.log(PSD[1 : kbins.size + 1])

        return kbins, S, dtheta

    def reduce_to_pole(self, data, inc, dec, sinc=None, sdec=None):
        """
        Reduce total field magnetic anomaly data to the pole.

        The reduction to the pole if a phase transformation that can be
        applied to total field magnetic anomaly data. It simulates how
        the data would be if both the Geomagnetic field and the
        magnetization of the source were vertical (Blakely, 1996).

        Args:
            data : 1D array
                the total field anomaly data at each point.
            inc : float / 1D array
                inclination of the inducing Geomagnetic field
            dec : float / 1D array
                declination of the inducing Geomagnetic field
            sinc : float / 1D array (optional)
                inclination of the total magnetization of the anomaly source
            sdec : float / 1D array (optional)
                declination of the total magnetization of the anomaly source
                The total magnetization is the vector sum of the
                induced and remanent magnetization. If there is only induced
                magnetization, use the *inc* and *dec* of the Geomagnetic field.

        Returns:
            rtp : 2D array
                the data reduced to the pole.

        References:
            Blakely, R. J. (1996), Potential Theory in Gravity and Magnetic
            Applications, Cambridge University Press.

        Notes:
            This functions performs the reduction in the frequency domain
            (using the FFT). The transform filter is (in the freq domain):

            \\( RTP(k_x, k_y) = \\frac{|k|}{
                a_1 k_x^2 + a_2 k_y^2 + a_3 k_x k_y +
                i|k|(b_1 k_x + b_2 k_y)}    \\)

            in which \\( k_x, k_y \\) are the wave-numbers in the x and y
            directions and

            \\( |k| = \\sqrt{k_x^2 + k_y^2} \\)

            \\( a_1 = m_z f_z - m_x f_x     \\)

            \\( a_2 = m_z f_z - m_y f_y     \\)

            \\( a_3 = -m_y f_x - m_x f_y    \\)

            \\( b_1 = m_x f_z + m_z f_x     \\)

            \\( b_2 = m_y f_z + m_z f_y     \\)

            \\( \\mathbf{m} = (m_x, m_y, m_z) \\) is the unit-vector of the total
            magnetization of the source and
            \\( \\mathbf{f} = (f_x, f_y, f_z) \\) is the unit-vector of the
            Geomagnetic field.
        """
        nr, nc = data.shape

        if nr != nc:
            warnings.warn("subgrid is not square {}".format((nr, nc)), RuntimeWarning)

        fx, fy, fz = ang2vec(1.0, inc, dec)
        if sinc is None or sdec is None:
            mx, my, mz = fx, fy, fz
        else:
            mx, my, mz = ang2vec(1.0, sinc, sdec)

        kx, ky = [k for k in _fftfreqs(self.dx, self.dy, data.shape)]
        kz = np.hypot(kx, ky)

        a1 = mz * fz - mx * fx
        a2 = mz * fz - my * fy
        a3 = -my * fx - mx * fy
        b1 = mx * fz + mz * fx
        b2 = my * fz + mz * fy

        # The division gives a RuntimeWarning because of the zero frequency term.
        # This suppresses the warning.
        with np.errstate(divide="ignore", invalid="ignore"):
            rtp = (kz) / (
                a1 * kx ** 2
                + a2 * ky ** 2
                + a3 * kx * ky
                + 1j * np.sqrt(kz) * (b1 * kx + b2 * ky)
            )

        rtp[0, 0] = 0
        ft_pole = rtp * np.fft.fft2(data)
        return np.real(np.fft.ifft2(ft_pole))

    def upward_continuation(self, data, height):
        """
        Upward continuation of potential field data.

        Calculates the continuation through the Fast Fourier Transform in
        the wavenumber domain (Blakely, 1996):

        \\( F\\{h_{up}\\} = F\\{h\\} e^{-\\Delta z |k|} \\)

        and then transformed back to the space domain. \\( h_{up} \\) is the
        upward continue data, \\( \\Delta z \\) is the height increase,
        \\( F \\) denotes the Fourier Transform,
        \\( |k| \\) is the wavenumber modulus.

        Args:
            data : 2D array
                potential field at the grid points
            height : float
                height increase (delta z) in meters.

        Returns:
            cont : array
                upward continued data

        References:
            Blakely, R. J. (1996), Potential Theory in Gravity and Magnetic
            Applications, Cambridge University Press.
        """
        nr, nc = data.shape

        if nr != nc:
            warnings.warn("subgrid is not square {}".format((nr, nc)), RuntimeWarning)

        if height <= 0:
            warnings.warn(
                "Using 'height' <= 0 means downward continuation, "
                + "which is known to be unstable."
            )

        fx = 2.0 * np.pi * np.fft.fftfreq(nr, self.dx)
        fy = 2.0 * np.pi * np.fft.fftfreq(nc, self.dy)

        kx, ky = np.meshgrid(fy, fx)[::-1]
        kz = np.hypot(kx, ky)

        upcont_ft = np.fft.fft2(data) * np.exp(-height * kz)
        cont = np.real(np.fft.ifft2(upcont_ft))
        return cont


# Helper functions to calculate Curie depth


def bouligand2009(kh, beta, zt, dz, C):
    """
    Calculate the synthetic radial power spectrum of
    magnetic anomalies

    Equation (4) of Bouligand et al. (2009)

    Args:
        kh : float / 1D array
            wavenumber in rad/km
        beta : float / 1D array
            fractal parameter
        zt : float / 1D array
            top of magnetic sources
        dz : float / 1D array
            thickness of magnetic sources
        C : float 1D array
            field constant (Maus et al., 1997)

    Returns:
        Phi : float / 1D array
            radial power spectrum of magnetic anomalies

    References:
        Bouligand, C., J. M. G. Glen, and R. J. Blakely (2009), Mapping Curie
        temperature depth in the western United States with a fractal model for
        crustal magnetization, J. Geophys. Res., 114, B11104,
        doi:10.1029/2009JB006494

        Maus, S., D. Gordon, and D. Fairhead (1997), Curie temperature depth
        estimation using a self-similar magnetization model, Geophys. J. Int.,
        129, 163-168, doi:10.1111/j.1365-246X.1997.tb00945.x
    """
    # from scipy.special import kv
    khdz = kh * dz
    coshkhdz = np.cosh(khdz)

    Phi1d = C - 2.0 * kh * zt - (beta - 1.0) * np.log(kh) - khdz
    A = (
        np.sqrt(np.pi)
        / gamma(1.0 + 0.5 * beta)
        * (
            0.5 * coshkhdz * gamma(0.5 * (1.0 + beta))
            - kv((-0.5 * (1.0 + beta)), khdz)
            * np.power(0.5 * khdz, (0.5 * (1.0 + beta)))
        )
    )
    Phi1d += np.log(A)
    return Phi1d


def tanaka1999(k, lnPhi, sigma_lnPhi, kmin_range=(0.05, 0.2), kmax_range=(0.05, 0.2)):
    """
    Compute weighted linear fit of Phi over spatial frequency window kmin:kmax

    Args:
        k : float / 1D-array
            wavenumber in rad/km
        lnPhi : float / 1D array
            log of the radial power spectrum (see power_spectrum_log)
            expected in ln(sqrt(S)) form
        sigma_lnPhi : standard deviation of lnPhi
        kmin_range : tuple (default:(0.05, 0.2))
            minimum and maximum range of spatial frequencies to fit for the
            top of magnetic sources - ideally low frequency, straight line
        kmax_range : tuple (default:(0.05, 0.2))
            minimum and maximum range of spatial frequencies to fit for the
            bottom of magnetic source - ideally low frequency, straight line

    Returns:
        upper_source : tuple
            (Ztr,btr,dZtr) gradient, intercept, error for the top of magnetic sources
        lower_source : tuple
            (Zor,bor,dZor) gradient, intercept, error for the bottom of magnetic sources

    """
    # for now...
    S = lnPhi
    sigma2 = sigma_lnPhi ** 2

    def compute_coefficients(X, Y, E):
        X2 = X ** 2
        Y2 = Y ** 2
        E2 = E ** 2

        XY = np.multiply(X, Y)
        XE2sum = np.sum(X / E2)
        YE2sum = np.sum(Y / E2)
        rE2sum = np.sum(1.0 / E2)
        X2E2sum = np.sum(X2 / E2)

        # TL = XE2sum*YE2sum - np.sum(XY/E2*rE2sum)
        # I think summation in second TL term needed to be split
        TL = XE2sum * YE2sum - np.sum(XY / E2) * rE2sum
        BL = XE2sum ** 2 - X2E2sum * rE2sum

        Z = TL / BL
        b = (np.sum(XY / E2) - Z * X2E2sum) / XE2sum
        # dZ = np.sqrt( rE2sum/(X2E2sum*rE2sum - XE2sum) )
        ## There was a missing **2 term at end of error term.
        dZ = np.sqrt(rE2sum / (X2E2sum * rE2sum - XE2sum ** 2))
        return Z, b, dZ

    sf = k / (2.0 * np.pi)

    # mask low wavenumbers
    kmin, kmax = kmin_range
    mask1 = np.logical_and(sf >= kmin, sf <= kmax)
    X1 = sf[mask1]
    Y1 = S[mask1]
    E1 = sigma2[mask1]

    # mask high wavenumbers
    kmin, kmax = kmax_range
    mask2 = np.logical_and(sf >= kmin, sf <= kmax)
    X2 = sf[mask2]
    Y2 = np.log(np.exp(S[mask2]) / (X2 * 2 * np.pi))
    E2 = np.log(np.exp(sigma2[mask2]) / (X2 * 2 * np.pi))

    # compute top and bottom of magnetic layer
    Ztr, btr, dZtr = compute_coefficients(X1, Y1, E1)
    Zor, bor, dZor = compute_coefficients(X2, Y2, E2)
    return (Ztr, btr, dZtr), (Zor, bor, dZor)


def ComputeTanaka(Ztr, dZtr, Zor, dZor):
    """
    Compute the Curie depth from the results of tanaka1999

    Args:
        Ztr : float / 1D array
            top of the magnetic source
        dZtr : float / 1D array
            error of Ztr
        Zor : float / 1D array
            centroid depth of the magnetic source
        dZor : float / 1D array
            error of Zor

    Returns:
        Zb : float / 1D array
            estimated Curie point depth at bottom of magnetic source
        eZb : float / 1D array
            error of `Zb`
    """
    Zb = 2.0 * Zor - Ztr
    dZb = 2.0 * dZor + dZtr
    return abs(Zb), dZb


def maus1995(beta, zt, kh, C=0.0):
    """
    Calculate the synthetic radial power spectrum of
    magnetic anomalies (Maus and Dimri; 1995)

    This is not all that useful except when testing
    overflow errors which occur for the second term
    in Bouligand et al. (2009).

    Args:
        beta : float / 1D array
            fractal parameter
        zt : float / 1D array
            top of magnetic sources
        kh : float / 1D array
            norm of the wave number in the horizontal plane
        C : float / 1D array
            field constant (Maus et al., 1997)

    Returns:
        Phi : float / 1D array
            radial power spectrum of magnetic anomalies

    References:
        Bouligand, C., J. M. G. Glen, and R. J. Blakely (2009), Mapping Curie
        temperature depth in the western United States with a fractal model for
        crustal magnetization, J. Geophys. Res., 114, B11104,
        doi:10.1029/2009JB006494

        Maus, S., D. Gordon, and D. Fairhead (1997), Curie temperature depth
        estimation using a self-similar magnetization model, Geophys. J. Int.,
        129, 163-168, doi:10.1111/j.1365-246X.1997.tb00945.x
    """
    return C - 2.0 * kh * zt - (beta - 1.0) * np.log(kh)


def _fftfreqs(dx, dy, shape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    fx = 2.0 * np.pi * np.fft.fftfreq(shape[0], dx)
    fy = 2.0 * np.pi * np.fft.fftfreq(shape[1], dy)
    return np.meshgrid(fy, fx)[::-1]


def ang2vec(intensity, inc, dec):
    """
    Convert intensity, inclination and  declination to a 3-component vector

    Args:
        intensity : float or 1D array
            The intensity (norm) of the vector
        inc : float
            The inclination of the vector (in degrees)
        dec : float
            The declination of the vector (in degrees)

    Returns:
        vec : array = [x, y, z]
            3-component vector

    Notes:
        Coordinate system is assumed to be x->North, y->East, z->Down.
        Inclination is positive down and declination is measured with respect
        to x (North).

    Examples:
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

    Args:
        inc : float
            The inclination of the vector (in degrees)
        dec : float
            The declination of the vector (in degrees)

    Returns:
        vect : list
            The unit vector = [x, y, z]

    Notes:
        Coordinate system is assumed to be x->North, y->East, z->Down.
        Inclination is positive down and declination is measured with respect
        to x (North).
    """
    d2r = np.pi / 180.0
    vect = [
        np.cos(d2r * inc) * np.cos(d2r * dec),
        np.cos(d2r * inc) * np.sin(d2r * dec),
        np.sin(d2r * inc),
    ]
    return vect
