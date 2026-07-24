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
The centroid method of Tanaka *et al.* (1999), with uncertainties.

``CurieOptimiseTanaka`` fits two straight lines to separate wavenumber bands of
the log amplitude spectrum -- a short-wavelength band whose slope gives the depth
to the top of the source, and a long-wavelength band whose slope gives the
centroid depth -- and propagates the fit covariance into an uncertainty on the
Curie depth. The bands have no defaults and must be chosen for the data; use
``check_bands`` to test a choice. The derivation, the conditions each band relies
on, and the fractal-magnetisation correction are documented in the online theory
guide.

References:
    Tanaka, A., Okubo, Y., & Matsubayashi, O. (1999). Curie point depth based on
    spectrum analysis of the magnetic anomaly data in East and Southeast Asia.
    Tectonophysics, 306(3-4), 461-470. doi:10.1016/S0040-1951(99)00072-4
"""

import warnings
from multiprocessing import cpu_count

import numpy as np
from scipy.optimize import curve_fit

from .grid import CurieGrid, _gls_covariance
from .parallel import stochastic

# Below this many points in a band the residual autocorrelation is estimated
# from too little to be worth anything, and a noisy inflation is worse than
# none. The bands used in practice hold two or three times this.
_MIN_POINTS_FOR_CORRELATION = 8

# Truncating sinh(|k|d) at |k|d biases the centroid, and hence the Curie
# depth, low. Measured against exact layer spectra the shortfall is
# proportional to both the thickness and |k|d:
#
#     Z_b bias = -0.17 * thickness * |k|d
#
# holding to within 6% over thicknesses of 10-40 km and |k|d of 0.2-1.0. That
# lets check_bands quote a bias in km rather than leave the user to judge
# whether |k|d is small enough.
_CENTROID_BIAS = -0.17

# Warn once the bias reaches about 5% of the source thickness. The previous
# threshold of 1.0 only fired when the Curie depth was already 3.3 km out on a
# 20 km layer, having said nothing at |k|d = 0.5 where it is 1.7 km out.
_KD_LIMIT = 0.3

# The zt fit wants wavelengths short compared with the source thickness. The
# previous rule warned above twice the thickness, where the measured zt error
# is 0.05% of it -- so it complained about a 10 m error while staying silent
# about the kilometres above. At four times the thickness the error is about
# 1%, which is the point at which it starts to matter.
_HALF_SPACE_RATIO = 4.0

# A zt band fits the short-wavelength end of the spectrum, so its upper edge
# sitting this far down the available range means the numbers are much more
# likely to be cycles/km left over from before bands were given in rad/km.
# Judged relative to the spectrum rather than absolutely, since the Nyquist
# wavenumber depends on the grid spacing.
_CYCLES_PER_KM_SUSPICION = 0.1


def _linear_func(x, a, b):
    """Straight line, fitted to each band."""
    return a * x + b


class CurieOptimiseTanaka(CurieGrid):
    """
    Extends `pycurious.grid.CurieGrid` with the centroid method of
    Tanaka *et al.* (1999).

    Two straight lines are fitted to separate bands of the radial amplitude
    spectrum, giving the top and centroid depths of the magnetic source and,
    from those, the Curie point depth with an uncertainty.

    Args:
        grid : 2D array
            2D array of magnetic data
        xmin, xmax : float
            minimum/maximum x bounds of the grid
        ymin, ymax : float
            minimum/maximum y bounds of the grid
        max_processors : int, optional
            processors to use in `optimise_routine` (default=all)

    Attributes:
        max_processors : int
            processors used by the parallel routines

    Notes:
        All attributes of `pycurious.grid.CurieGrid` are inherited as well.

        The grid must be projected in eastings/northings (metres), not
        degrees, since depths are computed in km.
    """

    def __init__(self, grid, xmin, xmax, ymin, ymax, **kwargs):

        super(CurieOptimiseTanaka, self).__init__(grid, xmin, xmax, ymin, ymax)

        self.max_processors = kwargs.pop("max_processors", cpu_count())

    def check_bands(self, k, zt_range, z0_range, thickness=None, verbose=True):
        """
        Report whether two fitting bands are usable, before relying on them.

        Both of Tanaka's straight-line approximations hold only over part of
        the spectrum. A band outside that range still produces a confident
        looking fit, so this is worth checking explicitly.

        Args:
            k : 1D array
                wavenumbers from `radial_spectrum`, in rad/km
            zt_range : tuple
                (min, max) wavenumber of the :math:`Z_t` band, rad/km
            z0_range : tuple
                (min, max) wavenumber of the :math:`Z_0` band, rad/km
            thickness : float, optional
                estimated source thickness in km. Without it the validity of
                each approximation cannot be assessed, only the point counts.
            verbose : bool (default=True)
                print a summary. Set False to receive any problems as
                `UserWarning` instead, for use in a script.

        Returns:
            diagnostics : dict
                `dk`, `n_zt`, `n_z0`, `lambda_zt`, `lambda_z0` and, if
                `thickness` was given, `kd_max`, `CPD_bias` and
                `lambda_zt_min_required`. `CPD_bias` is an estimate in km of
                how far the :math:`|k|d` approximation drags the Curie depth,
                and is negative.

        Usage:
            >>> k, Phi, sigma_Phi = grid.radial_spectrum(subgrid, power=1)
            >>> grid.check_bands(k, (0.2, 0.6), (0.0, 0.05), thickness=20.0)
        """
        k = np.asarray(k)
        _warn_if_cycles_per_km(zt_range, k)

        mask_zt = np.logical_and(k >= zt_range[0], k <= zt_range[1])
        mask_z0 = np.logical_and(k >= z0_range[0], k <= z0_range[1])

        n_zt = int(np.count_nonzero(mask_zt))
        n_z0 = int(np.count_nonzero(mask_z0))
        dk = float(np.min(np.diff(k))) if k.size > 1 else np.nan

        diagnostics = {
            "dk": dk,
            "n_zt": n_zt,
            "n_z0": n_z0,
            "lambda_zt": _wavelength_range(k[mask_zt]),
            "lambda_z0": _wavelength_range(k[mask_z0]),
        }

        messages = []
        for name, count in (("zt_range", n_zt), ("z0_range", n_z0)):
            if count < 3:
                messages.append(
                    "{} holds {} points, need at least 3".format(name, count)
                )
            elif count < 8:
                messages.append(
                    "{} holds only {} points, so its gradient will be poorly "
                    "determined. A wider window resolves the spectrum more "
                    "finely.".format(name, count)
                )

        if thickness is not None:
            half = 0.5 * float(thickness)
            kd_max = float(k[mask_z0].max() * half) if n_z0 else np.nan
            cpd_bias = _CENTROID_BIAS * float(thickness) * kd_max
            diagnostics["kd_max"] = kd_max
            diagnostics["CPD_bias"] = cpd_bias
            diagnostics["lambda_zt_min_required"] = _HALF_SPACE_RATIO * float(thickness)

            if n_z0 and kd_max > _KD_LIMIT:
                messages.append(
                    "z0_range reaches |k|d = {:.2f}, which biases the Curie "
                    "depth low by about {:.1f} km. The centroid fit assumes "
                    "|k|d << 1. Lower the upper edge of z0_range, or use a "
                    "wider window so there are enough points below "
                    "it.".format(kd_max, abs(cpd_bias))
                )
            if n_zt and diagnostics["lambda_zt"][1] > _HALF_SPACE_RATIO * thickness:
                messages.append(
                    "zt_range reaches a wavelength of {:.1f} km, more than {:g} "
                    "times the source thickness ({:.1f} km). The half-space "
                    "approximation behind the zt fit weakens there.".format(
                        diagnostics["lambda_zt"][1],
                        _HALF_SPACE_RATIO,
                        thickness,
                    )
                )

        if verbose:
            print("spectral resolution dk = {:.4f} rad/km".format(dk))
            print(
                "zt band: {} points, wavelengths {:.1f}-{:.1f} km".format(
                    n_zt, *diagnostics["lambda_zt"]
                )
            )
            print(
                "z0 band: {} points, wavelengths {:.1f}-{:.1f} km".format(
                    n_z0, *diagnostics["lambda_z0"]
                )
            )
            if thickness is not None:
                print(
                    "z0 band reaches |k|d = {:.2f}, biasing the Curie depth by "
                    "about {:+.1f} km".format(kd_max, cpd_bias)
                )
            for message in messages:
                print("WARNING: {}".format(message))
            if not messages:
                print("both bands look usable")
        else:
            # when verbose the messages have already been printed; warning as
            # well just prints everything twice
            for message in messages:
                warnings.warn(message, UserWarning, stacklevel=2)

        return diagnostics

    def _fit_band(self, k, Phi, sigma, band, absolute_sigma=True):
        """
        Fit a straight line over one band and return the depth it implies.

        Returns `(depth, intercept, depth_stdev)`, where `depth` is the
        negated gradient and so is positive downwards.

        The gradient is the ordinary weighted least-squares one. Its
        uncertainty is not: neighbouring spectral bins are correlated, and
        `curve_fit` assumes they are not, so the covariance it returns is too
        small -- on the legacy fixture the centroid band reports 0.126 km where
        the residuals imply 0.282. The correlation is estimated from the fit
        residuals and folded in, as it is on the Bouligand side.
        """
        mask = np.logical_and(k >= band[0], k <= band[1])
        mask &= np.isfinite(Phi) & np.isfinite(sigma) & (sigma > 0.0)

        n = int(np.count_nonzero(mask))
        if n < 3:
            raise ValueError(
                "only {} usable points in the band {}-{} rad/km, need at least "
                "3. Widen the band, or use a larger window to resolve the "
                "spectrum more finely.".format(n, band[0], band[1])
            )

        (gradient, intercept), covariance = curve_fit(
            _linear_func,
            k[mask],
            Phi[mask],
            sigma=sigma[mask],
            absolute_sigma=absolute_sigma,
        )

        stdev = np.sqrt(np.diag(covariance))[0]
        if absolute_sigma:
            stdev = self._correlated_gradient_stdev(
                k[mask], Phi[mask], sigma[mask], gradient, intercept, stdev
            )

        return -gradient, intercept, stdev

    @staticmethod
    def _correlated_gradient_stdev(k, Phi, sigma, gradient, intercept, fallback):
        """
        Standard deviation of a fitted gradient, allowing for correlation
        between neighbouring spectral bins.

        :math:`(X^T R^{-1} X)^{-1}` for the whitened design matrix `X` of the
        straight line, with `R` the banded correlation of the residuals.

        Falls back to the uncorrelated value when the band holds too few points
        to estimate a correlation from, or when the result is singular. Below
        about eight points the estimate is noise, and a noisy inflation is
        worse than none.
        """
        if k.size < _MIN_POINTS_FOR_CORRELATION:
            return fallback

        residual = (Phi - _linear_func(k, gradient, intercept)) / sigma

        # for a straight line the Jacobian of the whitened residual is just the
        # design matrix, so there is nothing to differentiate numerically
        J = np.column_stack([k, np.ones_like(k)]) / sigma[:, None]

        cov = _gls_covariance(J, residual, k.size)
        if cov is None:
            return fallback

        stdev = np.sqrt(np.diag(cov))[0]
        return stdev if np.isfinite(stdev) else fallback

    def _spectrum(self, window, xc, yc, taper, beta, process_subgrid, dof_factor,
                  **kwargs):
        """
        Radial amplitude spectrum of one window, prepared for both fits.

        Returns `(k, Phi, Phi_n, sigma)` where `Phi` is the log amplitude
        spectrum, `Phi_n` is that divided by `|k|`, and `sigma` is the
        uncertainty of the binned mean -- shared by both, since `ln|k|` is
        deterministic and so does not alter it.
        """
        # power=1 gives ln of the amplitude spectrum, Tanaka's ln(Phi^1/2)
        k, Phi, sigma = self.window_spectrum(
            window,
            xc,
            yc,
            taper=taper,
            power=1,
            process_subgrid=process_subgrid,
            dof_factor=dof_factor,
            **kwargs
        )

        if beta is not None:
            # remove the fractal contribution -0.5*(beta-1)*ln|k|, matching the
            # parameterisation of pycurious.grid.bouligand2009
            Phi = Phi + 0.5 * (beta - 1.0) * np.log(k)

        # ln|k| is deterministic, so subtracting it leaves sigma untouched
        Phi_n = Phi - np.log(k)

        return k, Phi, Phi_n, sigma

    def optimise(
        self,
        window,
        xc,
        yc,
        zt_range,
        z0_range,
        taper=np.hanning,
        beta=None,
        process_subgrid=None,
        absolute_sigma=True,
        dof_factor=None,
        **kwargs
    ):
        """
        Estimate the top and centroid depths of the magnetic source for one
        centroid, with their uncertainties.

        Args:
            window : float
                size of the window in metres
            xc, yc : float
                centroid of the window
            zt_range : tuple
                (min, max) wavenumber in **rad/km** over which to fit
                :math:`Z_t`. Should cover wavelengths shorter than twice the
                source thickness.
            z0_range : tuple
                (min, max) wavenumber in **rad/km** over which to fit
                :math:`Z_0`. Must satisfy :math:`|k| d \\ll 1` -- check with
                `check_bands`.
            taper : function (default=np.hanning)
                taper function, or None for no taper
            beta : float, optional
                fractal parameter of the magnetisation. If given, its
                contribution is removed before fitting. Leave unset for the
                method exactly as Tanaka published it; `beta=1` is equivalent.
            process_subgrid : function, optional
                applied to the subgrid before the spectrum is computed
            absolute_sigma : bool (default=True)
                treat the spectral uncertainties as absolute, so the reported
                errors carry their units. Set False to rescale the covariance
                by the reduced chi-squared instead.
            dof_factor : float, optional
                override the effective-degrees-of-freedom deflation applied to
                the spectral uncertainties (see Notes)
            kwargs : keyword arguments
                passed to `radial_spectrum`

        Returns:
            zt : float
                depth to the top of the magnetic source, km
            z0 : float
                centroid depth of the magnetic source, km
            zt_intercept : float
                intercept of the :math:`Z_t` fit
            z0_intercept : float
                intercept of the :math:`Z_0` fit
            sigma_zt : float
                standard deviation of `zt`
            sigma_z0 : float
                standard deviation of `z0`

        Usage:
            >>> zt, z0, zt_i, z0_i, sigma_zt, sigma_z0 = grid.optimise(
            ...     200e3, xc, yc, (0.2, 0.6), (0.0, 0.05))
            >>> CPD, sigma_CPD = grid.calculate_CPD(
            ...     zt, z0, sigma_zt=sigma_zt, sigma_z0=sigma_z0)

        Notes:
            Depths are returned positive downwards, i.e. the negated gradient
            of each fit.

            The reported uncertainties describe the scatter of the spectrum
            only. They do not include the systematic error from the choice of
            band, which is usually larger -- see `sensitivity`.

            `radial_spectrum` returns the scatter of the FFT cells within each
            annulus, whereas the fit needs the uncertainty of the annulus
            mean. That is the standard error, except that the cells are not
            independent: Hermitian symmetry makes about half of them
            redundant, and tapering correlates neighbours. The correction is
            calibrated per taper; `dof_factor` overrides it.

            Cells in *neighbouring* annuli are correlated too, which `_fit_band`
            allows for. Against 80 independent synthetics the ratio of the true
            spread to the reported `sigma_zt` improves from 1.48 to 1.12 with
            that correction in place.

            `sigma_z0` remains understated, at a ratio of about 1.34, and no
            covariance can fix it. The centroid gradient is fitted over a
            handful of the longest wavelengths the window resolves, and its
            distribution is heavy-tailed: on a 4000 km grid with a true
            :math:`Z_0` of 11 km, the middle 90% of estimates spanned 4.2 to
            25.3 km. Treat `sigma_z0`, and the Curie depth that follows from
            it, as a lower bound.

            There is no profile-likelihood alternative here, as there is on the
            Bouligand side. Each band is a straight-line fit, so its misfit is
            exactly quadratic in the gradient and the profile interval is
            provably the same as the covariance one -- verified, a deviance of
            3.841459 against a threshold of 3.841459 on both bands. It would
            return the number `_fit_band` already returns. The equivalence
            holds only for `absolute_sigma=True`; setting it False rescales the
            covariance by the reduced chi-squared afterwards, which the profile
            construction does not do.
        """
        k, Phi, Phi_n, sigma = self._spectrum(
            window, xc, yc, taper, beta, process_subgrid, dof_factor, **kwargs
        )

        _warn_if_cycles_per_km(zt_range, k)

        zt, zt_intercept, sigma_zt = self._fit_band(
            k, Phi, sigma, zt_range, absolute_sigma
        )
        z0, z0_intercept, sigma_z0 = self._fit_band(
            k, Phi_n, sigma, z0_range, absolute_sigma
        )

        return (zt, z0, zt_intercept, z0_intercept, sigma_zt, sigma_z0)

    def optimise_routine(
        self,
        window,
        xc_list,
        yc_list,
        zt_range,
        z0_range,
        taper=np.hanning,
        beta=None,
        process_subgrid=None,
        absolute_sigma=True,
        dof_factor=None,
        **kwargs
    ):
        """
        Iterate `optimise` over a list of centroids, in parallel.

        Takes the same arguments as `optimise`, with lists of centroids in
        place of a single one. See
        `pycurious.parallel.CurieParallel.parallelise_routine` for the
        `on_error` and `seed` keywords.

        Returns:
            zt, z0, zt_intercept, z0_intercept, sigma_zt, sigma_z0 :
                1D arrays, one entry per centroid

        Usage:
            >>> xc_list, yc_list = grid.create_centroid_list(window, 10e3, 10e3)
            >>> zt, z0, zt_i, z0_i, sigma_zt, sigma_z0 = grid.optimise_routine(
            ...     window, xc_list, yc_list, (0.2, 0.6), (0.0, 0.05))
        """
        return self.parallelise_routine(
            window,
            xc_list,
            yc_list,
            self.optimise,
            zt_range,
            z0_range,
            taper,
            beta,
            process_subgrid,
            absolute_sigma,
            dof_factor,
            **kwargs
        )

    @stochastic
    def sensitivity(
        self,
        window,
        xc,
        yc,
        nsim,
        zt_range,
        z0_range,
        taper=np.hanning,
        beta=None,
        band_scale=0.1,
        process_subgrid=None,
        absolute_sigma=True,
        dof_factor=None,
        seed=None,
        **kwargs
    ):
        """
        Sample the uncertainty of the Curie depth by perturbing both the
        spectrum and the fitting bands.

        The fit covariance alone understates the uncertainty, because where
        the band edges are placed usually matters more than the scatter of the
        spectrum. Each simulation therefore redraws the spectrum within its
        uncertainty *and* jitters both band edges. The remaining arguments are
        as `optimise`.

        Args:
            window : float
                size of the window in metres
            xc, yc : float
                centroid of the window
            nsim : int
                number of simulations
            zt_range, z0_range : tuple
                as `optimise`, in rad/km. These are the centres about which
                the band edges are jittered.
            band_scale : float (default=0.1)
                standard deviation of the jitter applied to each band edge, as
                a fraction of that band's width. Set to 0 to perturb only the
                spectrum, which recovers the analytic covariance.
            seed : int, optional
                seed for reproducibility

        Returns:
            zt : 1D array shape (nsim,)
                sampled top depths
            z0 : 1D array shape (nsim,)
                sampled centroid depths
            CPD : 1D array shape (nsim,)
                sampled Curie point depths

        Usage:
            >>> zt, z0, CPD = grid.sensitivity(
            ...     200e3, xc, yc, 500, (0.2, 0.6), (0.0, 0.05))
            >>> print(CPD.mean(), CPD.std())

        Notes:
            This samples the statistical uncertainty and the sensitivity to
            band placement. It does not capture the systematic error from
            fitting outside the range where each approximation holds, nor from
            unmodelled fractal magnetisation -- both bias the two fits
            coherently rather than scattering them. Use `check_bands` and
            `beta` for those.
        """
        rng = np.random.default_rng(seed)

        # the spectrum is computed once and resampled, as in
        # CurieOptimiseBouligand.sensitivity
        k, Phi, Phi_n, sigma = self._spectrum(
            window, xc, yc, taper, beta, process_subgrid, dof_factor, **kwargs
        )

        _warn_if_cycles_per_km(zt_range, k)

        zt_width = zt_range[1] - zt_range[0]
        z0_width = z0_range[1] - z0_range[0]

        zt_samples = np.empty(nsim)
        z0_samples = np.empty(nsim)

        i = 0
        attempts = 0
        max_attempts = 100 * nsim
        while i < nsim:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "only {} of {} simulations produced a usable fit. The "
                    "bands are probably too narrow to survive jittering -- "
                    "widen them or lower band_scale.".format(i, nsim)
                )

            rPhi = rng.normal(Phi, sigma)
            rPhi_n = rPhi - np.log(k)

            zt_band = _jitter(zt_range, band_scale * zt_width, rng)
            z0_band = _jitter(z0_range, band_scale * z0_width, rng)

            try:
                zt, _, _ = self._fit_band(k, rPhi, sigma, zt_band, absolute_sigma)
                z0, _, _ = self._fit_band(k, rPhi_n, sigma, z0_band, absolute_sigma)
            except (ValueError, RuntimeError):
                # a jittered band can fall off the end of the spectrum
                continue

            zt_samples[i] = zt
            z0_samples[i] = z0
            i += 1

        CPD, _ = self.calculate_CPD(zt_samples, z0_samples)
        return [zt_samples, z0_samples, CPD]

    def calculate_CPD(self, zt, z0, sigma_zt=0.0, sigma_z0=0.0):
        """
        Compute the Curie depth from the results of `optimise`.

        Args:
            zt : float / 1D array
                depth to the top of the magnetic source
            z0 : float / 1D array
                centroid depth of the magnetic source
            sigma_zt : float / 1D array
                standard deviation of `zt`
            sigma_z0 : float / 1D array
                standard deviation of `z0`

        Returns:
            CPD : float / 1D array
                estimated Curie point depth at the base of the magnetic source
            CPD_stdev : float / 1D array
                standard deviation of `CPD`

        Notes:
            :math:`Z_b = 2 Z_0 - Z_t`, so the uncertainties combine as
            :math:`\\sqrt{\\sigma_{Z_t}^2 + 4\\sigma_{Z_0}^2}`. This assumes
            the two fits are independent, which holds well enough in practice
            -- they use disjoint bands, and the measured correlation between
            them is about 0.05.

            `zt` and `z0` are expected positive downwards, as returned by
            `optimise`.
        """
        CPD = 2.0 * z0 - zt
        CPD_stdev = np.sqrt(np.asarray(sigma_zt) ** 2 + (2.0 * np.asarray(sigma_z0)) ** 2)
        return (CPD, CPD_stdev)


def _jitter(band, scale, rng):
    """Perturb both edges of a band, keeping it ordered and non-negative."""
    lo, hi = rng.normal(band[0], scale), rng.normal(band[1], scale)
    lo, hi = min(lo, hi), max(lo, hi)
    return (max(0.0, lo), hi)


def _wavelength_range(k):
    """Wavelengths spanned by a set of wavenumbers, shortest first."""
    if k.size == 0:
        return (np.nan, np.nan)
    kmax = k.max()
    kmin = k[k > 0].min() if np.any(k > 0) else np.nan
    return (2.0 * np.pi / kmax, 2.0 * np.pi / kmin if kmin == kmin else np.inf)


def _warn_if_cycles_per_km(zt_range, k):
    """
    Catch bands left over from when they were specified in cycles/km.

    Such a call still runs, and at a large enough window returns a plausible
    looking number rather than raising, so it is worth flagging. A cycles/km
    value is 2*pi too small, which drags a zt band -- which should sit at the
    short-wavelength end -- down to the bottom of the spectrum.
    """
    kmax = np.max(k)
    upper = np.max(zt_range)

    # only complain if reading them as rad/km puts the band implausibly low
    # *and* the cycles/km reading would land somewhere sensible
    if upper < _CYCLES_PER_KM_SUSPICION * kmax and upper * 2.0 * np.pi <= kmax:
        warnings.warn(
            "zt_range={} may be in cycles/km. Bands are specified in rad/km, "
            "and this one covers only the lowest {:.0%} of the spectrum, which "
            "is unusual for a zt fit. Multiplying by 2*pi would give "
            "{}.".format(
                tuple(zt_range),
                upper / kmax,
                tuple(np.round(np.asarray(zt_range) * 2.0 * np.pi, 3)),
            ),
            UserWarning,
            stacklevel=3,
        )
