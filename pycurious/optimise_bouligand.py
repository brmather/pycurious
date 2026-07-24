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
Fit the fractal spectrum of Bouligand *et al.* (2009), with uncertainties.

``CurieOptimiseBouligand`` inherits ``pycurious.grid.CurieGrid`` and recovers the
four parameters of ``bouligand2009`` (beta, zt, dz, C) by optimisation, posing
the recovery as a Bayesian inverse problem with a flexible objective function
that accepts *a priori* and likelihood terms. Beyond the fit it offers
profile-deviance intervals, Metropolis-Hastings posterior sampling, and a
sensitivity analysis, and it decomposes the computation across CPUs to map Curie
depth over a grid. The model and the uncertainty machinery are documented in the
online theory guide.
"""

# -*- coding: utf-8 -*-
from .grid import CurieGrid, bouligand2009, _gls_covariance
from .parallel import stochastic
import numpy as np
import warnings
from scipy.optimize import minimize, brentq
from scipy import stats
from multiprocessing import cpu_count

# bouligand2009 evaluates cosh(|k| dz), which overflows a float64 above about
# |k| dz = 710. `dz` is bounded just inside that, at `_max_thickness`, so the
# optimiser and the sampler cannot wander into the region where the forward
# model stops returning a number.
#
# This is a numerical guard, not a physical prior, and the distinction matters.
# A Curie depth on Earth sits in the mid crust, so anything past a few tens of
# km is already meaningless -- but bounding it *there* would clip the upper
# tail of a genuinely skewed posterior and pile probability against the wall,
# which misrepresents the distribution rather than reporting it. The bound is
# therefore set far beyond any physical value, where it never binds on data
# that constrain the base at all. A fit that does reach it is not a deep
# source; it is a window too small to see one, and `_warn_on_bounds` says so.
_COSH_OVERFLOW = 700.0

# Stand-in for a residual that came back non-finite. Inside the bound above
# that cannot arise from the forward model, so this is a backstop for anyone
# calling `residuals` directly rather than something the fit relies on. One
# such residual contributes its square to the misfit, a penalty of 1e6 per
# unusable bin: big enough to dominate, small enough that prior terms added
# alongside it stay representable, which a flat 1e99 would not be.
_OVERFLOW_RESIDUAL = 1.0e3

# Parameters of bouligand2009, in the order the optimiser sees them.
_PARAMETERS = ("beta", "zt", "dz", "C")

# Relative step for the finite-difference Jacobian behind the covariance.
_JACOBIAN_STEP = 1.0e-6

# Acceptance rate the burn-in tunes the proposal towards, the usual optimum
# for a random walk over a smooth multivariate target.
_TARGET_ACCEPTANCE = 0.234

# Tolerance for the root find that refines a profile interval, in the units of
# whatever is being profiled. A depth is wanted to a metre at most, and each
# evaluation is a full re-fit, so the default 2e-12 would spend a dozen fits
# resolving digits far past anything the uncertainty supports.
_PROFILE_XTOL = 1.0e-3

# `profile` can also work on the Curie depth, which is not a parameter of the
# forward model but a sum of two of them.
_CPD = "CPD"


def _prior_loc_scale(pdf):
    """
    Return `(loc, scale)` of a frozen `scipy.stats` distribution, however it was
    constructed.

    `objective_function` takes the prior as `(x0, sigma_x0)`, so a prior is only
    ever stored as its centre and width. Reading `pdf.args` alone is not enough:
    a distribution built with keywords -- `stats.norm(loc=p, scale=s)` -- has an
    empty `args` and carries the values in `kwds` instead.
    """

    loc = pdf.kwds.get("loc")
    scale = pdf.kwds.get("scale")

    if loc is None or scale is None:
        # scipy orders positional arguments as (*shapes, loc, scale)
        nshapes = len(pdf.dist.shapes.split(",")) if pdf.dist.shapes else 0
        positional = pdf.args[nshapes:]
        if loc is None and len(positional) > 0:
            loc = positional[0]
        if scale is None and len(positional) > 1:
            scale = positional[1]

    # fall back to the scipy defaults for a standard distribution
    if loc is None:
        loc = 0.0
    if scale is None:
        scale = 1.0

    # a tuple, so a caller cannot perturb a stored prior in place
    return (loc, scale)


class CurieOptimiseBouligand(CurieGrid):
    """
    Extends the `pycurious.grid.CurieGrid` class to include
    optimisation routines see `scipy.optimize.minimize` for
    a description of the algorithm.

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
        bounds : list of tuples
            lower and upper bounds for \\( \\beta, z_t, \\Delta z, C \\).
            \\( \\Delta z \\) is capped where the forward model stops
            evaluating, which depends on the grid spacing and is hundreds of km
            -- far beyond any Curie depth on Earth, so it never binds on data
            that constrain the base. Reassign this attribute to impose a
            tighter one, bearing in mind that a bound near the physical range
            will truncate the upper tail of a skewed posterior rather than
            report it.
        prior : dict
            dictionary of priors for \\( \\beta, z_t, \\Delta z, C \\)
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

        super(CurieOptimiseBouligand, self).__init__(grid, xmin, xmax, ymin, ymax)

        # initialise prior dictionary
        self.reset_priors()

        # lower / upper bounds for [beta, zt, dz, C]. Only the thickness gets
        # a ceiling: it is the one the data routinely fail to constrain, and
        # the one that can therefore run away far enough to stop the forward
        # model evaluating. beta and zt are pinned by the bulk of the spectrum.
        lb = [0.0, 0.0, 0.0, None]
        ub = [None, None, self._max_thickness(), None]
        self.bounds = list(zip(lb, ub))

        self.max_processors = kwargs.pop("max_processors", cpu_count())

    def _max_thickness(self):
        """
        Largest `dz` this grid can evaluate, in km.

        The radial spectrum reaches the Nyquist wavenumber,
        \\( \\pi/\\Delta x \\) in rad/km, whatever the window size, so the depth at which
        `pycurious.grid.bouligand2009` overflows is fixed by the grid spacing
        alone: 446 km at 2 km spacing, 111 km at 500 m. Both are far beyond any
        Curie depth on Earth, which is the point -- see `_COSH_OVERFLOW`.
        """
        return _COSH_OVERFLOW * (self.dx * 1.0e-3) / np.pi

    def add_prior(self, **kwargs):
        """
        Add a prior to the dictionary (tuple)
        Available priors are \\( \\beta, z_t, \\Delta z, C \\)

        Assumes a normal distribution or
        define another distribution from `scipy.stats`

        Usage:
            >>> add_prior(beta=(p, sigma_p))

            >>> add_prior(beta=scipy.stats.norm(p, sigma_p))
        """

        for key in kwargs:
            if key in self.prior:
                prior = kwargs[key]
                if isinstance(prior, tuple):
                    p, sigma_p = prior
                    pdf = stats.norm(p, sigma_p)
                elif isinstance(prior, stats.distributions.rv_frozen):
                    pdf = prior
                else:
                    raise ValueError("Use a distribution from scipy.stats module")

                # add prior PDF to dictionary
                self.prior_pdf[key] = pdf
                self.prior[key] = _prior_loc_scale(pdf)

            else:
                raise ValueError("prior must be one of {}".format(self.prior.keys()))

    def reset_priors(self):
        """
        Reset priors to uniform distribution
        """
        self.prior = {"beta": None, "zt": None, "dz": None, "C": None}
        self.prior_pdf = {"beta": None, "zt": None, "dz": None, "C": None}

    def objective_routine(self, **kwargs):
        """
        Evaluate the objective routine to find the misfit with priors
        Only keys carrying a prior will be added to the total misfit

        Args:
            kwargs : parameter values to test against their priors

        Usage:
            >>> objective_routine(beta=2.5)

        Returns:
            misfit : float
                misfit integrated over all observations and priors
        """
        prior = self.prior

        c = 0.0

        for key in kwargs:
            val = kwargs[key]
            if key in prior:
                prior_args = prior[key]
                if prior_args is not None:
                    c += self.objective_function(val, *prior_args)
        return c

    def objective_function(self, x, x0, sigma_x0, *args):
        """
        Objective function used in `objective_routine`
        Evaluates the l2-norm misfit

        Args:
            x : float, ndarray
            x0 : float, ndarray
            sigma_x0 : float, ndarray

        Returns:
            misfit : float
        """
        return 0.5 * np.sum((x - x0) ** 2 / sigma_x0 ** 2)

    def residuals(self, x, kh, Phi, sigma_Phi, prior=None):
        """
        Whitened residuals of the fit: the spectrum first, then one entry per
        prior.

        `min_func` is the half sum of squares of this vector, and the fit
        covariance comes from its Jacobian, so the two cannot drift apart.
        A Gaussian prior \\( N(p, \\sigma_p) \\) on a parameter \\( m \\) is
        just another observation, contributing a residual
        \\( (m - p)/\\sigma_p \\).

        Args:
            x : array shape (4,)
                \\( \\beta, z_t, \\Delta z, C \\)
            kh : array shape (n,)
                wavenumbers (rad/km)
            Phi : array shape (n,)
                radial power spectrum \\( \\Phi \\)
            sigma_Phi : array shape (n,)
                uncertainty of \\( \\Phi \\), as returned by
                `pycurious.grid.CurieGrid.window_spectrum`
            prior : dict, optional
                priors to use in place of `self.prior`

        Returns:
            residuals : array shape (n + number of priors,)

        Notes:
            Warnings from `pycurious.grid.bouligand2009` are suppressed because
            some combinations of parameters overflow, which would otherwise
            crash the minimiser. Any residual that comes back non-finite is
            replaced by a large finite value, so that one unusable bin costs
            the fit a fixed penalty rather than poisoning the whole vector.
        """
        beta, zt, dz, C = x

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Phi_syn = bouligand2009(kh, beta, zt, dz, C)

        r = (Phi_syn - Phi) / sigma_Phi
        r = np.where(np.isfinite(r), r, _OVERFLOW_RESIDUAL)

        if prior is None:
            prior = self.prior

        rows = [r]
        for key, value in zip(_PARAMETERS, x):
            prior_args = prior.get(key)
            if prior_args is not None:
                loc, scale = prior_args
                rows.append(np.array([(value - loc) / scale]))

        return np.concatenate(rows)

    def min_func(self, x, kh, Phi, sigma_Phi, prior=None):
        """
        Function to minimise: the negative log posterior, up to a constant.

        Args:
            x : array shape (4,)
                array of variables \\( \\beta, z_t, \\Delta z, C \\)
            kh : array shape (n,)
                wavenumbers (rad/km)
            Phi : array shape (n,)
                radial power spectrum \\( \\Phi \\)
            sigma_Phi : array shape (n,)
                uncertainty of \\( \\Phi \\), as returned by
                `pycurious.grid.CurieGrid.window_spectrum`
            prior : dict, optional
                priors to use in place of `self.prior`

        Returns:
            misfit : float
                sum of misfit (scalar)

        Notes:
            `sigma_Phi` should be the uncertainty of the binned *mean*, not the
            scatter of the cells within each annulus. Weighting by the latter
            recovers parameters no better than not weighting at all, because
            it is nearly flat across the spectrum and so carries almost no
            information -- see `pycurious.grid.CurieGrid.window_spectrum`.
        """
        return 0.5 * np.sum(self.residuals(x, kh, Phi, sigma_Phi, prior) ** 2)

    def _spectrum(self, window, xc, yc, taper, process_subgrid, dof_factor, **kwargs):
        """
        Radial power spectrum of one window, weighted ready for fitting.

        Pins `power=2`, which is what `pycurious.grid.bouligand2009` describes,
        so the four routines that need a spectrum cannot drift apart on it.
        """
        return self.window_spectrum(
            window,
            xc,
            yc,
            taper=taper,
            power=2.0,
            process_subgrid=process_subgrid,
            dof_factor=dof_factor,
            **kwargs
        )

    def _jacobian(self, x, r, args):
        """
        Central-difference Jacobian of `residuals` at `x`, given `r` there.

        Eight evaluations for four parameters, so it costs nothing beside the
        fit itself. The step is relative, since the parameters differ in scale
        by more than an order of magnitude.
        """
        x = np.asarray(x, dtype=float)
        J = np.empty((r.size, x.size))

        for i in range(x.size):
            h = _JACOBIAN_STEP * max(abs(x[i]), 1.0)
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            J[:, i] = (self.residuals(xp, *args) - self.residuals(xm, *args)) / (2.0 * h)

        return J

    def _covariance(self, x, kh, Phi, sigma_Phi):
        """
        Covariance of the fitted parameters at `x`.

        The spectral residuals are correlated between neighbouring bins, so
        this is generalised least squares rather than \\( (J^T J)^{-1} \\) --
        see `pycurious.grid._gls_covariance`, which the Tanaka sibling shares.
        """
        args = (kh, Phi, sigma_Phi)
        r = self.residuals(x, *args)
        J = self._jacobian(x, r, args)

        cov = _gls_covariance(J, r, np.size(kh))
        if cov is None:
            warnings.warn(
                "the fit is singular, so no covariance could be formed. This "
                "usually means a parameter is unconstrained by the data.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.full((np.size(x), np.size(x)), np.nan)
        return cov

    def optimise(
        self,
        window,
        xc,
        yc,
        beta=3.0,
        zt=1.0,
        dz=10.0,
        C=5.0,
        taper=np.hanning,
        process_subgrid=None,
        dof_factor=None,
        return_cov=False,
        **kwargs
    ):
        """
        Find the optimal parameters of \\( \\beta, z_t, \\Delta z, C \\)
        for a given centroid (xc,yc) and window size, with their
        uncertainties.

        Args:
            window : float
                size of window in metres
            xc : float
                centroid x values
            yc : float
                centroid y values
            beta : float
                fractal parameter (starting value)
            zt : float
                top of magnetic layer (starting value)
            dz : float
                thickness of magnetic layer (starting value)
            C : float
                field constant (starting value)
            taper : taper (default=`numpy.hanning`)
                taper function, set to None for no taper function
            process_subgrids : function
                a custom function to process the subgrid
            dof_factor : float, optional
                override the effective-degrees-of-freedom deflation applied to
                the spectral uncertainties, see
                `pycurious.grid.CurieGrid.window_spectrum`
            return_cov : bool (default=False)
                also return the 4x4 parameter covariance matrix
            kwargs : keyword arguments
                to pass to radial_spectrum.

        Returns:
            beta : float
                fractal parameter
            zt : float
                top of magnetic layer
            dz : float
                thickness of magnetic layer
            C : float
                field constant
            sigma_beta, sigma_zt, sigma_dz, sigma_C : float
                standard deviation of each of the above
            cov : 2D array shape (4,4)
                parameter covariance. Only returned if `return_cov=True`.

        Usage:
            >>> beta, zt, dz, C, s_beta, s_zt, s_dz, s_C = grid.optimise(
            ...     200e3, xc, yc)
            >>> CPD, sigma_CPD = grid.calculate_CPD(zt, dz, s_zt, s_dz)

        Notes:
            The uncertainties come from the curvature of the misfit at the
            solution, corrected for the correlation between neighbouring
            spectral bins -- see `_covariance`. They describe the scatter of
            the spectrum at this window and this model. They do **not** include
            the systematic error from the choice of window size or centroid,
            which on a small grid is larger: sweeping those over
            `tests/test_mag_data.txt` moves \\( \\Delta z \\) by 5.5 km, where
            the fit reports about 3.3 km.

            `sigma_dz` in particular should be read as a lower bound. The
            likelihood in \\( \\Delta z \\) has a long upper tail (Mather &
            Fullea, 2019), so a symmetric interval is the wrong shape for it.
            Measured over 200 independent synthetics, `sigma_beta`, `sigma_zt`
            and `sigma_C` reproduce the true spread to within 1%, while
            `sigma_dz` understates it by about 40%. Use `profile` for an honest
            interval on \\( \\Delta z \\) and on the Curie depth.
        """

        x0 = np.array([beta, zt, dz, C])

        k, Phi, sigma_Phi = self._spectrum(
            window, xc, yc, taper, process_subgrid, dof_factor, **kwargs
        )

        # minimise function
        res = minimize(self.min_func, x0, args=(k, Phi, sigma_Phi), bounds=self.bounds)
        x = res.x

        self._warn_on_bounds(x)
        cov = self._covariance(x, k, Phi, sigma_Phi)
        sigma = np.sqrt(np.diag(cov))

        if return_cov:
            return tuple(x) + tuple(sigma) + (cov,)
        return tuple(x) + tuple(sigma)

    def _warn_on_bounds(self, x, rtol=1.0e-6):
        """
        Warn when a parameter has been driven onto one of `self.bounds`.

        The covariance is derived from the curvature of an interior minimum,
        so at an active bound it describes a solution the optimiser was not
        free to find, and the corresponding uncertainty is meaningless.
        """
        for name, value, (lb, ub) in zip(_PARAMETERS, x, self.bounds):
            for edge in (lb, ub):
                if edge is not None and np.isclose(value, edge, rtol=rtol, atol=rtol):
                    warnings.warn(
                        "{} converged onto its bound at {:g}, so its "
                        "uncertainty is not meaningful -- the fit was not free "
                        "to move it.".format(name, edge),
                        RuntimeWarning,
                        stacklevel=3,
                    )

    def optimise_routine(
        self,
        window,
        xc_list,
        yc_list,
        beta=3.0,
        zt=1.0,
        dz=10.0,
        C=5.0,
        taper=np.hanning,
        process_subgrid=None,
        dof_factor=None,
        **kwargs
    ):
        """
        Iterate through a list of centroids to compute the optimal values
        of \\( \\beta, z_t, \\Delta z, C \\) for a given window size.
        
        Args:
            window : float
                size of window in metres
            xc_list : ndarray shape (l,)
                centroid x values 
            yc_list : ndarray shape (l,)
                centroid y values 
            beta : float
                fractal parameter 
            zt : float
                top of magnetic layer
            dz : float
                thickness of magnetic layer
            C : float
                field constant
            taper : function
                taper function (default=`numpy.hanning`)
                set to None for no taper function
            process_subgrids : func
                a custom function to process the subgrid
            kwargs : keyword arguments
                to pass to radial_spectrum.

        Returns:
            beta : ndarray shape (l,)
                fractal parameters
            zt : ndarray shape (l,)
                top of magnetic layer
            dz : ndarray shape (l,)
                thickness of magnetic layer
            C : ndarray shape (l,)
                field constant
            sigma_beta, sigma_zt, sigma_dz, sigma_C : ndarray shape (l,)
                standard deviation of each of the above, so a map of the
                uncertainty comes out alongside the map of the parameter

        Notes:
            The covariance matrix is deliberately not available here.
            `pycurious.parallel.CurieParallel.parallelise_routine` collects one
            array per returned quantity, which a 4x4 matrix per centroid does
            not fit. Call `optimise` directly with `return_cov=True` for that.
        """
        return self.parallelise_routine(
            window,
            xc_list,
            yc_list,
            self.optimise,
            beta,
            zt,
            dz,
            C,
            taper,
            process_subgrid,
            dof_factor,
            **kwargs
        )

    def _profiled_misfit(self, target, value, x_hat, args):
        """
        Smallest misfit attainable with `target` held at `value`.

        For a parameter of the forward model that means fixing it and
        minimising over the other three. The Curie depth is the same thing one
        step along: \\( \\Delta z \\) becomes the constrained coordinate
        through \\( \\Delta z = \\mathrm{CPD} - z_t \\), leaving
        \\( \\beta, z_t, C \\) free -- so it is profiled directly rather than
        propagated from `dz`, whose uncertainty is not symmetric.
        """
        curie = target == _CPD
        fixed = _PARAMETERS.index("dz" if curie else target)
        free = [j for j in range(len(_PARAMETERS)) if j != fixed]

        def expand(y):
            x = np.empty(len(_PARAMETERS))
            x[free] = y
            # for the Curie depth the constraint depends on zt, which is free
            x[fixed] = value - x[1] if curie else value
            return x

        res = minimize(
            lambda y: self.min_func(expand(y), *args),
            np.asarray(x_hat)[free],
            bounds=[self.bounds[j] for j in free],
        )
        return res.fun

    def profile(
        self,
        window,
        xc,
        yc,
        target,
        level=0.95,
        npoints=21,
        bracket=None,
        beta=3.0,
        zt=1.0,
        dz=10.0,
        C=5.0,
        taper=np.hanning,
        process_subgrid=None,
        dof_factor=None,
        **kwargs
    ):
        """
        Confidence interval for one parameter, or for the Curie depth, without
        assuming the posterior is symmetric.

        Each point of the scan holds `target` fixed and re-optimises everything
        else, tracing the deviance \\( 2(F - F_{min}) \\). The interval is
        where that crosses \\( \\chi^2_1 \\) at the requested level, which is
        the usual likelihood-ratio construction.

        This matters most for \\( \\Delta z \\) and hence the Curie depth. Both
        have a long upper tail (Mather & Fullea, 2019), so the symmetric
        \\( \\pm \\sigma \\) that `optimise` reports understates how far the
        parameter can plausibly reach -- by about 40% on synthetics.

        Args:
            window : float
                size of window in metres
            xc, yc : float
                centroid of the window
            target : str
                one of `"beta"`, `"zt"`, `"dz"`, `"C"` or `"CPD"`
            level : float (default=0.95)
                confidence level
            npoints : int (default=21)
                nodes in the scan. The endpoints are then refined by root
                finding, so this sets the resolution of the returned curve
                rather than of the interval.
            bracket : tuple, optional
                (min, max) of the scan. Defaults to a range either side of the
                fitted value, wider above than below because of the tail.
            beta, zt, dz, C : float
                starting values for the underlying fit
            taper : function (default=np.hanning)
                taper function, or None for no taper
            process_subgrid : function, optional
                applied to the subgrid before the spectrum is computed
            dof_factor : float, optional
                see `pycurious.grid.CurieGrid.window_spectrum`
            kwargs : keyword arguments
                passed to `radial_spectrum`

        Returns:
            values : 1D array shape (npoints,)
                where the target was held
            deviance : 1D array shape (npoints,)
                \\( 2(F - F_{min}) \\) at each of those
            lower : float
                lower end of the interval, `-inf` if the scan never crossed
            upper : float
                upper end of the interval, `inf` if the scan never crossed

        Usage:
            >>> values, deviance, lo, hi = grid.profile(200e3, xc, yc, "CPD")
            >>> print("Curie depth {:.1f} ({:.1f} to {:.1f}) km".format(cpd, lo, hi))

        Notes:
            This is a profile *posterior* deviance rather than a profile
            likelihood: any priors added with `add_prior` contribute to `F`. A
            Gaussian prior is one more observation, so the calibration still
            holds, but it is not the marginal an MCMC would report -- profiling
            takes the ridge of the posterior rather than integrating over it,
            and so is a little narrower for a skewed one.

            Like the covariance from `optimise`, the interval describes the
            scatter of the spectrum at a fixed window, centroid and model. It
            does not cover the systematic error from choosing those: on
            `tests/test_mag_data.txt` the interval for \\( \\Delta z \\) is
            about 3.3 km wide, where sweeping the window size and centroid
            moves \\( \\Delta z \\) over 5.5 km.
        """
        if target not in _PARAMETERS + (_CPD,):
            raise ValueError(
                "target must be one of {}, not {!r}".format(
                    _PARAMETERS + (_CPD,), target
                )
            )

        k, Phi, sigma_Phi = self._spectrum(
            window, xc, yc, taper, process_subgrid, dof_factor, **kwargs
        )
        args = (k, Phi, sigma_Phi)

        x0 = np.array([beta, zt, dz, C])
        res = minimize(self.min_func, x0, args=args, bounds=self.bounds)
        x_hat, F_min = res.x, res.fun

        # every constrained fit is cached, so the root finding below reuses the
        # scan nodes it lands on rather than paying for them twice
        cache = {}

        def constrained(value):
            value = float(value)
            if value not in cache:
                cache[value] = self._profiled_misfit(target, value, x_hat, args)
            return cache[value]

        if bracket is None:
            bracket = self._profile_bracket(target, x_hat, args)
        lo, hi = float(bracket[0]), float(bracket[1])

        values = np.linspace(lo, hi, int(npoints))

        # carry the fitted value itself as a node. Without it a coarse or
        # badly placed bracket can step over the minimum entirely, leaving
        # every node above the threshold and the interval collapsed onto a
        # single point with nothing to say so.
        hat = x_hat[1] + x_hat[2] if target == _CPD else x_hat[_PARAMETERS.index(target)]
        if lo < hat < hi:
            values = np.unique(np.append(values, hat))

        misfit = np.array([constrained(v) for v in values])

        # a constrained fit can land below the unconstrained one when the
        # latter stopped early, which would put the deviance negative
        F_min = min(F_min, misfit.min())
        deviance = 2.0 * (misfit - F_min)

        threshold = stats.chi2.ppf(level, 1)
        centre = values[np.argmin(deviance)]

        if deviance.min() > threshold:
            warnings.warn(
                "the {} scan over {:.4g} to {:.4g} never came within the "
                "threshold of the best fit, so the interval it returns is "
                "meaningless. Widen `bracket` around {:.4g}, or raise "
                "`npoints`.".format(target, values[0], values[-1], hat),
                RuntimeWarning,
                stacklevel=2,
            )

        def gap(value):
            return 2.0 * (constrained(value) - F_min) - threshold

        # walk outwards from the best fit in each direction
        below = values <= centre
        above = values >= centre
        lower = self._profile_root(
            values[below][::-1], deviance[below][::-1], threshold, gap,
            -np.inf, target, level,
        )
        upper = self._profile_root(
            values[above], deviance[above], threshold, gap, np.inf, target, level
        )

        return values, deviance, lower, upper

    def _profile_bracket(self, target, x_hat, args):
        """
        Default scan range: a few standard deviations either side of the fit,
        reaching further above than below because the tail is on that side.
        """
        cov = self._covariance(x_hat, *args)
        sigma = np.sqrt(np.abs(np.diag(cov)))

        if target == _CPD:
            centre = x_hat[1] + x_hat[2]
            width = np.hypot(sigma[1], sigma[2])
        else:
            index = _PARAMETERS.index(target)
            centre, width = x_hat[index], sigma[index]

        if not np.isfinite(width) or width <= 0.0:
            width = max(abs(centre), 1.0)

        lo, hi = centre - 5.0 * width, centre + 12.0 * width

        # depths are positive, and the scan must stay inside the range the
        # forward model can be evaluated over
        if target in ("dz", _CPD):
            lo, hi = max(lo, 0.0), min(hi, self._max_thickness())
        elif target in ("zt", "beta"):
            lo = max(lo, 0.0)

        return lo, hi

    @staticmethod
    def _profile_root(values, deviance, threshold, gap, unbounded, target, level):
        """
        Where the deviance crosses `threshold`, refined off the scan grid.

        `values` and `deviance` run outwards from the best fit, so this is the
        same walk in either direction and the caller supplies the reflection
        and the sign of `unbounded`.

        Reading the crossing off the nearest node would quantise the interval
        at the node spacing, which is a large fraction of its width for a
        sensible `npoints`; root finding between the two bracketing nodes costs
        a handful of extra fits and removes that.
        """
        crossed = np.nonzero(deviance > threshold)[0]

        if crossed.size == 0:
            warnings.warn(
                "the {} profile never reached the {:.0%} threshold within "
                "{:.4g} to {:.4g}, so that side of the interval is unbounded. "
                "The data do not constrain it; widen `bracket` to confirm."
                .format(target, level, min(values), max(values)),
                RuntimeWarning,
                stacklevel=3,
            )
            return unbounded

        j = crossed[0]
        if j == 0:
            # already over the threshold at the best fit: nothing to bracket
            return float(values[0])

        return float(brentq(gap, values[j - 1], values[j], xtol=_PROFILE_XTOL))

    @stochastic
    def metropolis_hastings(
        self,
        window,
        xc,
        yc,
        nsim,
        burnin,
        x_scale=None,
        beta=3.0,
        zt=1.0,
        dz=10.0,
        C=5.0,
        taper=np.hanning,
        process_subgrid=None,
        dof_factor=None,
        adapt=True,
        seed=None,
        return_diagnostics=False,
        **kwargs
    ):
        """
        MCMC algorithm using a Metropolis-Hastings sampler.

        Evaluates a Markov chain for starting values of
        \\( \\beta, z_t, \\Delta z, C \\) and returns the ensemble of model
        realisations.

        Args:
            window : float
                size of window in metres
            xc : float
                centroid x values
            yc : float
                centroid y values
            nsim : int
                number of simulations
            burnin : int
                number of burn-in simulations before to nsim
            x_scale : float(4), optional
                initial width of the proposal in each parameter
                (default=`[1,1,1,1]` for `[beta, zt, dz, C]`). With
                `adapt=True` this is only a starting point.
            beta : float
                fractal parameter (starting value for the search)
            zt : float
                top of magnetic layer (starting value for the search)
            dz : float
                thickness of magnetic layer (starting value for the search)
            C : float
                field constant (starting value for the search)
            taper : function (default=np.hanning)
                taper function, or None for no taper
            process_subgrid : function, optional
                applied to the subgrid before the spectrum is computed
            dof_factor : float, optional
                see `pycurious.grid.CurieGrid.window_spectrum`
            adapt : bool (default=True)
                tune the proposal during burn-in -- see Notes. Turning this off
                is only sensible if you have a good `x_scale` already.
            seed : int, optional
                seed for reproducibility
            return_diagnostics : bool (default=False)
                also return a dict of `acceptance`, `burnin_acceptance`,
                `x_scale`

        Returns:
            beta : ndarray shape (nsim,)
                fractal parameter
            zt : ndarray shape (nsim,)
                top of magnetic layer
            dz : ndarray shape (nsim,)
                thickness of magnetic layer
            C : ndarray shape (nsim,)
                field constant
            diagnostics : dict
                only if `return_diagnostics=True`

        Usage:
            >>> posterior, info = grid.metropolis_hastings(
            ...     200e3, xc, yc, 10000, 2000, seed=1, return_diagnostics=True)
            >>> print("acceptance {:.2f}".format(info["acceptance"]))

        Notes:
            Acceptance is decided in log space. Comparing
            \\( e^{-F} \\) directly underflows to zero for any real spectrum --
            \\( F \\) runs to hundreds -- at which point every proposal is
            rejected and the chain returns a handful of distinct states
            dressed up as a posterior.

            The chain starts at the mode, found with the same minimiser
            `optimise` uses and from the same starting values. That costs a
            fraction of a second and removes the job the burn-in is worst at.

            There is no tempering. It was tried -- annealing the burn-in
            after Sambridge (2013), doi:10.1093/gji/ggt342 -- and made every
            case worse. What motivated it was that large parts of the posterior
            evaluated to zero, and that was the \\( e^{-F} \\) underflow rather
            than a property of the problem, so log-space acceptance removes the
            reason for it. It also fights the proposal tuning below: a high
            temperature makes almost everything acceptable, driving the scale
            up, and the scale then collapses as the temperature falls, freezing
            the chain wherever the hot phase left it.

            The shape of the proposal matters more than any of the above. The
            four parameters are strongly correlated -- \\( \\beta \\) with
            \\( z_t \\) at about -0.92, \\( z_t \\) with \\( C \\) at about
            0.87 -- and their marginal widths differ by a factor of thirty, so
            a proposal with one width per parameter cannot move along the ridge
            they lie on, and the chain sits still. The proposal is therefore
            drawn along the fit covariance, the same one `optimise` reports,
            with the burn-in tuning only a scalar multiplier on it towards an
            acceptance rate of 0.234 (Robbins-Monro). `x_scale` sets where that
            multiplier starts, and `adapt=False` fixes it there.

            The chain respects `self.bounds`, which the optimiser has always
            done but the sampler previously did not.

            Both this and `sensitivity` treat the spectral bins as independent,
            which they are not, so the posterior is narrower than the spread
            over independent realisations of the field. See `optimise`.
        """
        rng = np.random.default_rng(seed)
        ndim = len(_PARAMETERS)

        k, Phi, sigma_Phi = self._spectrum(
            window, xc, yc, taper, process_subgrid, dof_factor, **kwargs
        )

        lower = np.array(
            [-np.inf if b[0] is None else b[0] for b in self.bounds], dtype=float
        )
        upper = np.array(
            [np.inf if b[1] is None else b[1] for b in self.bounds], dtype=float
        )

        def log_posterior(x):
            if np.any(x < lower) or np.any(x > upper):
                return -np.inf
            return -self.min_func(x, k, Phi, sigma_Phi)

        def step(x, F, scale, chol):
            """One Metropolis move."""
            proposal = x + scale * chol.dot(rng.normal(size=ndim))

            F1 = log_posterior(proposal)
            accepted = np.isfinite(F1) and np.log(rng.random()) < F1 - F

            if accepted:
                return proposal, F1, True
            return x, F, False

        # Start the chain at the mode rather than at the caller's guess. The
        # optimiser finds it in a fraction of the time a random walk takes to
        # wander there, and a chain started away from it spends its whole
        # burn-in travelling instead of tuning. Measured on a synthetic, the
        # posterior mean from a default start sits at a misfit of 121 against
        # the mode's 50; started here it lands on 50.1.
        start = minimize(
            self.min_func,
            np.array([beta, zt, dz, C], dtype=float),
            args=(k, Phi, sigma_Phi),
            bounds=self.bounds,
        )

        x = start.x
        F = log_posterior(x)

        # Propose along the fit covariance. That already describes the ridge
        # the parameters lie on -- it is what `optimise` reports -- so there is
        # no reason to rediscover it by watching the chain, and every reason
        # not to: a burn-in started at the mode with too small a step learns a
        # covariance narrower than the truth, proposes from it, and confirms
        # itself. Measured that way the chain reported a sigma on dz of 0.8
        # against a true 8.7.
        chol = self._proposal_cholesky(self._covariance(x, k, Phi, sigma_Phi))

        # `scale` is a scalar multiplier on an already correctly shaped
        # proposal, so there is one thing for the burn-in to tune
        scale = 1.0 if x_scale is None else float(np.mean(x_scale))

        burnin_accepted = 0

        for i in range(int(burnin)):
            x, F, accepted = step(x, F, scale, chol)
            burnin_accepted += accepted

            if adapt:
                # Robbins-Monro: nudge towards 0.234, with a decaying step so
                # the scale settles rather than rattling around
                scale = scale * np.exp((accepted - _TARGET_ACCEPTANCE) / (i + 1.0) ** 0.6)

        samples = np.empty((int(nsim), ndim))
        accepted_total = 0
        for i in range(int(nsim)):
            x, F, accepted = step(x, F, scale, chol)
            accepted_total += accepted
            samples[i] = x

        if return_diagnostics:
            diagnostics = {
                "acceptance": accepted_total / max(int(nsim), 1),
                "burnin_acceptance": burnin_accepted / max(int(burnin), 1),
                "x_scale": scale,
            }
            return list(samples.T), diagnostics

        # the default return shape has to stay a plain list of arrays:
        # pycurious.parallel dispatches on the dimensionality of the result
        return list(samples.T)

    @staticmethod
    def _proposal_cholesky(cov):
        """
        Scaled Cholesky factor of a covariance, for use as a proposal.

        Proposing along the covariance lets the chain move down the correlated
        ridge the parameters lie on, which no proposal with one width per
        parameter can follow: `beta` and `zt` correlate at about -0.92 and
        their marginal widths differ by a factor of thirty. The factor of
        \\( 2.38/\\sqrt{d} \\) is the usual optimal scaling for a Gaussian
        target.

        Falls back to the identity when the covariance is unusable, so the
        chain still runs -- isotropically, and badly -- rather than drawing
        from a degenerate distribution or failing outright.
        """
        ndim = np.shape(cov)[0] if cov is not None else len(_PARAMETERS)
        scaling = 2.38 / np.sqrt(ndim)

        if cov is not None and np.all(np.isfinite(cov)):
            try:
                return np.linalg.cholesky(cov) * scaling
            except np.linalg.LinAlgError:
                pass

        return np.eye(ndim) * scaling

    @stochastic
    def sensitivity(
        self,
        window,
        xc,
        yc,
        nsim,
        beta=3.0,
        zt=1.0,
        dz=10.0,
        C=5.0,
        taper=np.hanning,
        process_subgrid=None,
        dof_factor=None,
        seed=None,
        **kwargs
    ):
        """
        Sample the uncertainty of \\( \\beta, z_t, \\Delta z, C \\) by
        resampling the spectrum, and the centre of each prior distribution
        (if provided by the user - see add_prior).

        Args:
            window : float
                size of window in metres
            xc : float
                centroid x values
            yc : float
                centroid y values
            nsim : int
                number of Monte Carlo simulations
            beta : float
                starting fractal parameter
            zt : float
                starting top of magnetic layer
            dz : float
                starting thickness of magnetic layer
            C : float
                starting field constant
            dof_factor : float, optional
                override the effective-degrees-of-freedom deflation, see
                `pycurious.grid.CurieGrid.window_spectrum`
            seed : int, optional
                seed for reproducibility

        Returns:
            beta : ndarray shape (nsim,)
                fractal parameters
            zt : ndarray shape (nsim,)
                top of magnetic layer
            dz : ndarray shape (nsim,)
                thickness of magnetic layer
            C : ndarray shape (nsim,)
                field constant

        Notes:
            Each bin of the spectrum is resampled independently, so this shares
            the assumption behind the fit covariance that the bins are
            independent. They are not -- a taper correlates neighbouring
            annuli -- so agreement between the two is not evidence that either
            is right. Only an ensemble over independent realisations of the
            field calibrates that.
        """
        rng = np.random.default_rng(seed)

        samples = np.empty((nsim, 4))
        x0 = np.array([beta, zt, dz, C])

        use_keys = [key for key, pdf in self.prior_pdf.items() if pdf is not None]

        k, Phi, sigma_Phi = self._spectrum(
            window, xc, yc, taper, process_subgrid, dof_factor, **kwargs
        )

        # Every resampled spectrum lands in the same basin, so start each
        # simulation from the unresampled solution rather than from the
        # caller's guess. One extra fit up front, and about a third off the
        # total for any useful `nsim`.
        x0 = minimize(
            self.min_func, x0, args=(k, Phi, sigma_Phi), bounds=self.bounds
        ).x

        for sim in range(0, nsim):
            # a fresh set of prior centres, drawn without disturbing the ones
            # stored on the instance
            prior = dict(self.prior)
            for key in use_keys:
                loc = self.prior_pdf[key].rvs(random_state=rng)
                prior[key] = (loc, self.prior[key][1])

            rPhi = rng.normal(Phi, sigma_Phi)
            res = minimize(
                self.min_func,
                x0,
                args=(k, rPhi, sigma_Phi, prior),
                bounds=self.bounds,
            )
            samples[sim] = res.x

        return list(samples.T)

    def calculate_CPD(self, zt, dz, sigma_zt=0.0, sigma_dz=0.0):
        """
        Compute the Curie depth from the results of `optimise`.

        Args:
            zt : float / 1D array
                depth to the top of the magnetic source
            dz : float / 1D array
                thickness of the magnetic source
            sigma_zt : float / 1D array
                standard deviation of `zt`
            sigma_dz : float / 1D array
                standard deviation of `dz`

        Returns:
            CPD : float / 1D array
                estimated Curie point depth at the base of the magnetic source
            CPD_stdev : float / 1D array
                standard deviation of `CPD`

        Usage:
            >>> beta, zt, dz, C, s_beta, s_zt, s_dz, s_C = grid.optimise(
            ...     200e3, xc, yc)
            >>> CPD, sigma_CPD = grid.calculate_CPD(zt, dz, s_zt, s_dz)

        Notes:
            \\( Z_b = z_t + \\Delta z \\), so the uncertainties combine as
            \\( \\sqrt{\\sigma_{z_t}^2 + \\sigma_{\\Delta z}^2} \\). The two are
            correlated -- about 0.6 -- but \\( \\sigma_{\\Delta z} \\) exceeds
            \\( \\sigma_{z_t} \\) by four orders of magnitude, so including the
            covariance changes the answer by around 1%. `optimise` will hand
            over the full matrix with `return_cov=True` for anyone who wants it.

            The far larger effect is that this is symmetric and the Curie depth
            is not: \\( \\Delta z \\) has a long upper tail, so `CPD_stdev`
            understates how deep the base can plausibly lie. Use `profile` with
            `target="CPD"` for an interval that does not assume symmetry.

            Matches the signature and return of
            `pycurious.optimise_tanaka.CurieOptimiseTanaka.calculate_CPD`, so
            code written against one behaves the same against the other.
        """
        CPD = zt + dz
        CPD_stdev = np.sqrt(np.asarray(sigma_zt) ** 2 + np.asarray(sigma_dz) ** 2)
        return (CPD, CPD_stdev)
