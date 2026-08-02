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
The ``CurieGrid`` class and the shared spectrum and covariance machinery.

``CurieGrid`` is initialised with a magnetic grid of equal spacing in x and y and
provides:

- decomposition of subgrids for processing square windows of the anomaly;
- radially averaged spectra, either raw (``radial_spectrum``) or weighted ready
  for fitting (``window_spectrum``);
- removing linear trends, upward continuation, and reduction to the pole.

The module also holds the analytic spectra used by the optimisers --
``bouligand2009`` and its simplified form ``maus1995`` -- and the covariance
machinery both methods share. ``tanaka1999`` and ``ComputeTanaka`` implement the
centroid method without uncertainties and are **deprecated** in favour of
``pycurious.optimise_tanaka.CurieOptimiseTanaka``.
"""

# -*- coding: utf-8 -*-
from .parallel import CurieParallel
import numpy as np
from scipy.linalg import solveh_banded
from scipy.special import gamma, kv
import warnings


# How much larger the scatter of the binned mean is than sigma_Phi/sqrt(N),
# because the FFT cells in an annulus are not independent. There are two
# effects, and the deflation is stored as (dof_inf, lost):
#
#     dof(N) = dof_inf * N / max(N - lost, 1)
#
# `dof_inf` is the asymptotic redundancy. A real field has Hermitian symmetry,
# so about half of its cells repeat -- the factor of 2 with no taper, which is
# exact -- and tapering correlates neighbours further.
#
# `lost` is a fixed number of cells given up to correlation however many the
# annulus holds, so it only matters for the innermost bins. It matters a lot
# there: at N = 8 it costs another factor of 2.6 under np.hanning, and those
# are the bins a centroid depth leans on hardest.
#
# Measured by Monte Carlo over 400 realisations at n = 128, 256 and 512.
# Both terms are stable to about 6% across that range, and depend on the bin
# count rather than on the grid size.
_TAPER_DOF = {
    None: (2.0, 3.4),
    "hanning": (3.3, 4.9),
    "hamming": (3.0, 4.8),
}


# How many off-diagonals of the residual correlation to estimate. The measured
# correlation length is about 1.4 bins, so two is enough.
_CORRELATION_BANDS = 2

# Cap on the total off-diagonal weight. By Gershgorin, keeping it below one
# leaves the banded matrix positive definite and so factorisable.
_CORRELATION_LIMIT = 0.95


def _banded_correlation(r):
    """
    Correlation of neighbouring fit residuals, as a band.

    Neighbouring radial bins are not independent: a taper spreads each
    wavenumber over a main lobe several bins wide, so their residuals
    correlate. Treating them as independent understates the uncertainty of
    every fitted parameter by around 30% under `numpy.hanning`.

    Estimating from the residuals rather than tabulating per taper means this
    holds for a taper that has not been calibrated. On a correct model it
    recovers the taper: 0.008 with no taper against a measured 0.003, and 0.383
    under `numpy.hanning` against a measured 0.363.

    Args:
        r : 1D array
            residuals, already whitened by their own uncertainties

    Returns:
        ab : 2D array shape (`_CORRELATION_BANDS` + 1, len(r))
            lower-form band of the correlation matrix, as
            `scipy.linalg.solveh_banded` takes it

    Notes:
        The estimate also picks up smooth model error, which is likewise
        correlated between neighbours. That is a feature rather than a flaw --
        a model that cannot follow the data genuinely leaves its parameters
        less well determined -- but it does mean the result describes the fit
        as a whole and not the taper alone.
    """
    r = np.asarray(r, dtype=float)
    n = r.size

    rho = np.zeros(_CORRELATION_BANDS + 1)
    rho[0] = 1.0

    denominator = np.sum(r * r)
    if denominator > 0.0:
        for lag in range(1, _CORRELATION_BANDS + 1):
            if n > lag:
                rho[lag] = np.sum(r[:-lag] * r[lag:]) / denominator

    # a negative estimate is noise about zero, and would not describe a taper
    # spreading power into its neighbours
    rho[1:] = np.clip(rho[1:], 0.0, None)

    total = 2.0 * rho[1:].sum()
    if total > _CORRELATION_LIMIT:
        rho[1:] *= _CORRELATION_LIMIT / total

    ab = np.zeros((_CORRELATION_BANDS + 1, n))
    for lag in range(_CORRELATION_BANDS + 1):
        if n > lag:
            ab[lag, : n - lag] = rho[lag]

    return ab


def _gls_covariance(J, r, ncorrelated):
    """
    Parameter covariance allowing for correlation between residuals.

    Generalised least squares, :math:`(J^T R^{-1} J)^{-1}`, with `R` the banded
    correlation of the first `ncorrelated` residuals from
    `_banded_correlation`. Any rows beyond that -- the prior terms of a fit --
    are independent of the spectrum and of each other, so they keep unit weight
    and never enter the solve.

    Both optimisers use this. They differ in how `J` is obtained, analytically
    for a straight line and by finite differences for the four-parameter
    spectral model, but not in what is done with it.

    Args:
        J : 2D array shape (n, m)
            Jacobian of the whitened residuals
        r : 1D array shape (n,)
            those residuals
        ncorrelated : int
            how many leading rows are the correlated spectrum

    Returns:
        cov : 2D array shape (m, m), or None if the fit is singular
    """
    RiJ = np.array(J, dtype=float)
    ab = _banded_correlation(r[:ncorrelated])

    try:
        RiJ[:ncorrelated] = solveh_banded(ab, J[:ncorrelated], lower=True)
        return np.linalg.inv(J.T.dot(RiJ))
    except (np.linalg.LinAlgError, ValueError):
        return None


def _dof_factor(taper, counts=None, dof_factor=None):
    """
    Effective-degrees-of-freedom deflation for a taper.

    Returns the asymptotic scalar when `counts` is None, and the per-bin
    factor otherwise. An explicit `dof_factor` overrides both.

    An uncalibrated taper falls back to the untapered entry, which is
    conservative in the sense that Hermitian redundancy holds exactly whatever
    the taper does.
    """
    if dof_factor is not None:
        return float(dof_factor)

    dof_inf, lost = _TAPER_DOF.get(getattr(taper, "__name__", None), _TAPER_DOF[None])

    if counts is None:
        return dof_inf

    counts = np.asarray(counts, dtype=float)
    return dof_inf * counts / np.maximum(counts - lost, 1.0)


class CurieGrid(CurieParallel):
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

    def __init__(self, grid, xmin, xmax, ymin, ymax, **kwargs):

        super(CurieGrid, self).__init__()

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

        The trend is the least-squares plane. Over a regular grid the centred
        row and column indices are mutually orthogonal and both orthogonal to
        the constant, so the normal equations decouple and the plane's three
        coefficients are one mean and two 1-D inner products -- no design
        matrix and no SVD. This is an order of magnitude cheaper than the
        equivalent ``lstsq`` fit (the trend is subtracted once per window when
        computing a spectrum), and unlike an ``(nr, nc)`` vs ``(nc, nr)``
        design matrix it stays correct when the grid is not square.

        Args:
            data : 2D numpy array

        Returns:
            data : 2D numpy array
        """
        nr, nc = data.shape
        i = np.arange(nr) - (nr - 1) / 2.0  # centred row index
        j = np.arange(nc) - (nc - 1) / 2.0  # centred column index
        mean = data.mean()
        # Centring the means before the inner product drops the constant
        # term's contribution (sum(i) == sum(j) == 0) and keeps the sum well
        # conditioned, so the fit matches lstsq to machine precision.
        # A slope is only identifiable along an axis with more than one node;
        # a singleton axis carries no trend and its (i*i).sum() is zero, so
        # take a zero slope there rather than dividing by zero into a NaN.
        sii = (i * i).sum()
        sjj = (j * j).sum()
        ci = (i * (data.mean(axis=1) - mean)).sum() / sii if sii else 0.0
        cj = (j * (data.mean(axis=0) - mean)).sum() / sjj if sjj else 0.0
        return data - (mean + ci * i[:, None] + cj * j[None, :])

    def _taper_spectrum(self, subgrid, taper=np.hanning, scale=0.001, **kwargs):
        """
        Template for tapering the power spectrum used in `radial_spectrum`.
        """
        data = subgrid
        nr, nc = data.shape

        if nr != nc:
            warnings.warn("subgrid is not square {}".format((nr, nc)), RuntimeWarning)

        # control taper
        if taper is None:
            # nothing downstream consumes kwargs once there is no taper to pass
            # them to, so an unrecognised one would be silently ignored rather
            # than raising as it does when a taper is present
            if kwargs:
                raise TypeError(
                    "unexpected keyword argument(s) {} -- with taper=None there "
                    "is no taper function to pass them to".format(
                        ", ".join(repr(key) for key in sorted(kwargs))
                    )
                )
            vtaper = 1.0
        else:
            rt = taper(nr, **kwargs)
            ct = taper(nc, **kwargs)
            # the separable taper is an outer product. Building it through
            # meshgrid instead materialises two further (nr, nc) arrays and
            # multiplies them: 16.4 ms against 0.51 ms at nr = nc = 1025, for a
            # result identical to the bit. The end-to-end gain to `optimise` is
            # smaller than that and mostly within run-to-run noise -- the
            # allocations this saves are partly paid back as page faults in
            # `_FFT_spectrum` -- so this is here for doing less work, not for a
            # measured speed-up.
            vtaper = np.outer(rt, ct)

        # scaling factor to transform wavenumber into units of rad/km
        dx_scale = self.dx * scale
        # the DFT fundamental is 2*pi/(N*dx). Using (N-1) overstates every
        # wavenumber by N/(N-1) and so understates every depth by (N-1)/N.
        dk = 2.0 * np.pi / nr / dx_scale

        kbins = np.arange(dk, dk * nr / 2, dk)
        return vtaper, dk, kbins

    def _FFT_spectrum(self, subgrid, vtaper, dk, kbins, const):
        """
        Template for computing the (fast) Fourier transform used in
        `radial_spectrum`.

        A constant `const` should be applied to the FFT of the magnetic anomaly
        to convert `S` and `sigma` to specific units for further analysis.

        It is useful to remember that::

            2*log(FFT) == log(FFT**2)

        Returns `(k, S, sigma, counts)`, where `counts` is the number of FFT
        cells averaged into each radial bin.

        The transform is `numpy.fft.rfft2`, not the full `fft2`. The anomaly is
        real, so `|FFT|` is symmetric under reflection through the origin and
        the discarded half carries no new information; using it halves both the
        transform and the number of cells to bin, for around twice the speed
        with no change to the result. To keep `counts` the full-spectrum count
        -- which `window_spectrum` deflates `sigma` by, and whose Hermitian
        factor of two is baked into the `_TAPER_DOF` calibration -- each
        retained cell is weighted by how many full-spectrum cells it stands in
        for: the columns whose mirror was dropped count twice, the self-mirrored
        DC and (for even `nc`) Nyquist columns count once. Dropping that weight
        would halve `counts` and inflate every reported uncertainty by ~sqrt(2).

        .. note::

            This method returned three values prior to v2. Subclasses that
            override it must now also return `counts`.
        """
        data = subgrid
        nr, nc = data.shape

        nbins = kbins.size - 1

        # real-input transform: only the non-redundant half plane, nc//2 + 1
        # columns wide, is computed. No fftshift -- rows stay in FFT order and
        # columns are the non-negative frequencies 0 .. nc//2.
        FT = np.abs(np.fft.rfft2(data * vtaper))
        ncol = nc // 2 + 1

        # signed integer row frequencies (0, 1, .. -1), built exactly rather
        # than via fftfreq so a cell on the kx axis lands on the same bin edge
        # as the old centred grid did, to the bit -- counts are a partition and
        # must match a full fft2 exactly. Only |k| is binned, so the sign of the
        # row frequency does not matter.
        row_freq = np.arange(nr)
        row_freq[row_freq > (nr - 1) // 2] -= nr
        ix = (row_freq * dk)[:, np.newaxis]
        iy = (np.arange(ncol) * dk)[np.newaxis, :]
        kk = np.hypot(ix, iy).ravel()

        # a dropped mirror means every interior column stands for two cells of
        # the full spectrum; the DC column and, when nc is even, the Nyquist
        # column are their own mirror and stand for one.
        weight = np.full(ncol, 2.0)
        weight[0] = 1.0
        if nc % 2 == 0:
            weight[-1] = 1.0
        weight = np.broadcast_to(weight, (nr, ncol)).ravel()

        # bin every cell once and reduce with bincount. Masking the whole array
        # per bin instead is O(nbins * nr * nc), which is cubic in the window
        # and dominates a large run -- over 20x slower at nr = 2001.
        idx = np.digitize(kk, kbins) - 1
        # digitize is half-open above, matching the annuli, so a cell landing on
        # an edge is counted once rather than in both neighbours. The final bin
        # is the exception: it is closed, so a cell exactly at kbins[-1] belongs
        # to it rather than falling off the end.
        idx[(idx == nbins) & (kk <= kbins[-1])] = nbins - 1
        keep = (idx >= 0) & (idx < nbins)
        idx = idx[keep]
        kk = kk[keep]
        weight = weight[keep]
        # log only the cells that land in a bin, so a zero outside the binned
        # range cannot raise a divide-by-zero that the per-bin masking never saw
        rr = const * np.log(FT.ravel()[keep])

        counts = np.bincount(idx, weights=weight, minlength=nbins)
        with np.errstate(invalid="ignore", divide="ignore"):
            S = np.bincount(idx, weights=weight * rr, minlength=nbins) / counts
            k = np.bincount(idx, weights=weight * kk, minlength=nbins) / counts
            # two-pass variance. The one-pass E[x^2] - E[x]^2 form is cheaper
            # but cancels: ln|FFT| is O(10) with O(1) scatter, so it loses
            # three digits of sigma, and more on a near-constant bin.
            dev = rr - S[idx]
            sigma = np.sqrt(
                np.bincount(idx, weights=weight * dev * dev, minlength=nbins) / counts
            )

        # an empty annulus averages nothing -- mirrors the mean of an empty
        # slice the per-bin form produced, without the warning
        empty = counts == 0
        S[empty] = k[empty] = sigma[empty] = np.nan

        # counts are integer-valued (sums of the weights 1 and 2); round before
        # casting so a float sum landing a hair below the integer is not
        # truncated downward.
        return k, S, sigma, np.rint(counts).astype(int)

    def radial_spectrum(self, subgrid, taper=np.hanning, power=2.0, return_counts=False, **kwargs):
        """
        Compute the radial spectrum for a square grid.

        Wavenumber is returned in units of **rad/km**.

        Args:
            subgrid : 2D array
                window of the original data (see subgrid method)
            taper : function (default=np.hanning)
                taper function, set to None for no taper function
            power : float
                raise the FFT of the magnetic anomaly to the power:

                - 2.0 for Bouligand *et al.* (2009) use cases, which gives
                  the log power spectrum :math:`\\ln \\Phi_{\\Delta T}`
                - 1.0 for Tanaka *et al.* (1999) use cases, which gives the
                  log amplitude spectrum :math:`\\ln \\Phi_{\\Delta T}^{1/2}`
            return_counts : bool (default=False)
                also return the number of FFT cells averaged into each bin
            kwargs : keyword arguments
                keyword arguments to pass to `taper`

        Returns:
            k : 1D array shape (n,)
                wavenumber in rad/km
            Phi : 1D array shape (n,)
                Radial power spectrum
            sigma_Phi : 1D array shape (n,)
                Standard deviation of Phi within each radial bin
            counts : 1D array shape (n,)
                number of FFT cells in each radial bin.
                Only returned if `return_counts=True`.

        Notes:
            `Phi` is the mean of :math:`\\ln |FFT|` over each annulus, so
            `sigma_Phi` describes the scatter of the individual cells, not
            the uncertainty of that mean. Dividing by the square root of
            `counts` gives the standard error, though note the cells are not
            independent -- a real field has Hermitian symmetry, so roughly
            half of them are redundant, and tapering correlates neighbours.

            While `subgrid` is projected in eastings / northings (in metres),
            the wavenumber, :math:`k`, is returned in units of rad/km.
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
        k, Phi, sigma_Phi, counts = self._FFT_spectrum(
            subgrid, vtaper, dk, kbins, power
        )

        if return_counts:
            return k, Phi, sigma_Phi, counts
        return k, Phi, sigma_Phi

    def window_spectrum(
        self,
        window,
        xc,
        yc,
        taper=np.hanning,
        power=2.0,
        process_subgrid=None,
        dof_factor=None,
        **kwargs
    ):
        """
        Radial spectrum of one window, weighted ready for fitting.

        Extracts the subgrid, computes its radial spectrum, and converts the
        within-annulus scatter into the uncertainty of the annulus *mean*,
        which is what a fit needs. Both `pycurious.optimise_bouligand` and
        `pycurious.optimise_tanaka` build on this.

        Args:
            window : float
                size of the window in metres
            xc, yc : float
                centroid of the window
            taper : function (default=np.hanning)
                taper function, or None for no taper
            power : float
                raise the FFT of the anomaly to this power -- 2.0 for the log
                power spectrum that Bouligand *et al.* (2009) fit, 1.0 for the
                log amplitude spectrum of Tanaka *et al.* (1999)
            process_subgrid : function, optional
                applied to the subgrid before the spectrum is computed
            dof_factor : float, optional
                override the effective-degrees-of-freedom deflation (see Notes)
            kwargs : keyword arguments
                passed to `taper`

        Returns:
            k : 1D array
                wavenumber in rad/km
            Phi : 1D array
                log spectrum, raised to `power`
            sigma : 1D array
                uncertainty of the binned mean

        Usage:
            >>> k, Phi, sigma = grid.window_spectrum(200e3, xc, yc, power=2)

        Notes:
            `radial_spectrum` returns the scatter of the FFT cells within each
            annulus, whereas a fit needs the uncertainty of the annulus mean.
            That is the standard error, except that the cells are not
            independent: Hermitian symmetry makes about half of them redundant,
            and tapering correlates neighbours. The correction is calibrated
            per taper and varies with the number of cells in the bin, since a
            fixed number of them is lost to correlation however few there are.
            `dof_factor` overrides it with a constant.

            The cells of *neighbouring* annuli are correlated too, which this
            does not address -- it inflates the uncertainty of a fitted
            parameter rather than of any individual bin. See
            `pycurious.optimise_bouligand.CurieOptimiseBouligand.optimise`.
        """
        if process_subgrid is None:
            # dummy function
            def process_subgrid(subgrid):
                return subgrid

        subgrid = self.subgrid(window, xc, yc)
        subgrid = process_subgrid(subgrid)

        kwargs.pop("return_counts", None)
        k, Phi, sigma_Phi, counts = self.radial_spectrum(
            subgrid, taper=taper, power=power, return_counts=True, **kwargs
        )

        sigma = sigma_Phi / np.sqrt(counts / _dof_factor(taper, counts, dof_factor))

        return k, Phi, sigma

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

            .. math::

                RTP(k_x, k_y) = \\frac{|k|}
                {a_1 k_x^2 + a_2 k_y^2 + a_3 k_x k_y +
                i|k|(b_1 k_x + b_2 k_y)}

            in which :math:`k_x, k_y` are the wave-numbers in the x and y
            directions and

            :math:`|k| = \\sqrt{k_x^2 + k_y^2}`

            :math:`a_1 = m_z f_z - m_x f_x`

            :math:`a_2 = m_z f_z - m_y f_y`

            :math:`a_3 = -m_y f_x - m_x f_y`

            :math:`b_1 = m_x f_z + m_z f_x`

            :math:`b_2 = m_y f_z + m_z f_y`

            :math:`\\mathbf{m} = (m_x, m_y, m_z)` is the unit-vector of the total
            magnetization of the source and
            :math:`\\mathbf{f} = (f_x, f_y, f_z)` is the unit-vector of the
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

        :math:`F\\{h_{up}\\} = F\\{h\\} e^{-\\Delta z |k|}`

        and then transformed back to the space domain. :math:`h_{up}` is the
        upward continue data, :math:`\\Delta z` is the height increase,
        :math:`F` denotes the Fourier Transform,
        :math:`|k|` is the wavenumber modulus.

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
            log of the radial power spectrum, expected in ln(sqrt(S)) form
            (as returned by `radial_spectrum` with `power=1`)
        sigma_lnPhi : float / 1D array
            standard deviation of lnPhi
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

    Notes:
        .. deprecated:: 2.0
            Use `pycurious.optimise_tanaka.CurieOptimiseTanaka.optimise`,
            which fits both bands with `scipy.optimize.curve_fit` and returns
            depths positive downwards with their uncertainties.

        This hand-rolled weighted least squares squares an already-squared
        error term, so it weights by 1/sigma**4 rather than 1/sigma**2, and it
        subtracts ln(k) from a standard deviation. Its uncertainties are
        therefore not meaningful. It is retained only so existing scripts keep
        running.
    """
    warnings.warn(
        "tanaka1999 is deprecated, use CurieOptimiseTanaka.optimise instead. "
        "Its uncertainties are not meaningful -- see the docstring.",
        FutureWarning,
        stacklevel=2,
    )

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


def ComputeTanaka(zT, dzT, z0, dz0):
    """
    Compute the Curie depth from the results of `tanaka1999`.

    .. deprecated:: 2.0
        Use `pycurious.optimise_tanaka.CurieOptimiseTanaka.calculate_CPD`.

    Args:
        zT : float / 1D array
            top of the magnetic source
        dzT : float / 1D array
            standard deviation of zT
        z0 : float / 1D array
            centroid depth of the magnetic source
        dz0 : float / 1D array
            standard deviation of z0

    Returns:
        CPD : float / 1D array
            estimated Curie point depth at bottom of magnetic source
        CPD_stdev : float / 1D array
            standard deviation

    Notes:
        The arguments interleave the depths with their standard deviations,
        whereas `calculate_CPD` groups them. Renaming a call without also
        reordering the arguments computes nonsense.
    """
    warnings.warn(
        "ComputeTanaka is deprecated, use "
        "CurieOptimiseTanaka.calculate_CPD(zt, z0, sigma_zt, sigma_z0) "
        "instead. Note the argument order differs: ComputeTanaka takes "
        "(zt, sigma_zt, z0, sigma_z0), interleaving each depth with its "
        "standard deviation. Note also that the returned standard deviation "
        "changed in v2, from 2*dz0 + dzT to the quadrature sum.",
        FutureWarning,
        stacklevel=2,
    )
    CPD = abs(2.0 * z0 - zT)
    CPD_stdev = np.sqrt(dzT ** 2 + (dz0 * 2) ** 2)
    return CPD, CPD_stdev


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
