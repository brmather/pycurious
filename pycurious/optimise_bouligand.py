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
from .grid import (
    CurieGrid,
    bouligand2009,
    _correlation_inflation,
    _gls_covariance,
)
from .parallel import stochastic
import numpy as np
import warnings
from collections import namedtuple
from scipy.optimize import least_squares, brentq
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

# Relative step for the finite-difference Jacobian, both in `_fit` and behind
# the covariance.
_JACOBIAN_STEP = 1.0e-6

# `zt`, which the Curie constraint dz = CPD - zt differentiates through.
_ZT = _PARAMETERS.index("zt")

# `beta` and `dz`, the two the ladder scan works in, and `C`, the other half of
# the linear pair `_solve_linear` recovers.
_BETA = _PARAMETERS.index("beta")
_DZ = _PARAMETERS.index("dz")
_C = _PARAMETERS.index("C")

# Width given to a bound the caller collapsed to a point. Relative, so it pins
# a parameter of any magnitude to its own last few bits.
_DEGENERATE_BOUND = 1.0e-12

# Two of the four columns of the fit Jacobian are exact. Writing the forward
# model out,
#
#     Phi_syn = C - 2 |k| zt - (beta - 1) ln|k| - |k| dz + ln A(beta, dz, |k|)
#
# `zt` and `C` enter linearly and A depends on neither, so the derivatives of
# the whitened residual (Phi_syn - Phi)/sigma are closed form. Supplying them
# removes two of the four finite differences per Jacobian, and in `profile`
# -- where `dz` is the coordinate being held -- two of the three.
#
# The other two stay numerical. `beta` enters through the *order* of the Bessel
# term, which has no closed-form derivative. `dz` enters through its argument
# and could be differentiated via d/dx K_v = -(K_{v-1} + K_{v+1})/2, one extra
# `kv` call given the recurrence -- but that is exactly what the finite
# difference it would replace costs, so there is nothing to gain.
_ANALYTIC_COLUMNS = {
    "zt": lambda kh, sigma: -2.0 * kh / sigma,
    "C": lambda kh, sigma: 1.0 / sigma,
}

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

# Nodes per axis of the posterior mesh. `C` and `zt` integrate out in closed
# form, leaving a posterior over `beta` and `dz` alone -- two dimensions, which
# is small enough to evaluate rather than sample.
#
# 32 is 24 with a margin. Measured against a 192-node reference, the moments
# still move at 16 nodes (`dz` mean out by 0.02 of a standard deviation, its sd
# by 1.5%), are exact to four decimal places at 24, and do not move again after.
# There is very little room between "not enough" and "exact", so the margin
# matters more than the tuning does.
_MESH_NODES = 32

# Half-width of the first mesh box, in marginal standard deviations, and the
# extra stretch given to the upper `dz` edge because that tail is the long one.
_MESH_SPAN = 6.0
_MESH_SKEW = 2.0

# Posterior mass allowed outside the mesh before an edge is widened. A mesh that
# has clipped a tail is wrong in the direction of a *tighter* interval, and
# silently, which is the one failure mode a sampler does not have -- so this is
# checked rather than assumed.
_MESH_EDGE_TOL = 1.0e-6

# How many times an edge may be widened before the posterior is declared
# unbounded on that side. Each doubling costs one whole mesh, and a posterior
# that is still spilling after four is not one a wider box would contain.
_MESH_EXPANSIONS = 4

# Beyond this many conditional standard deviations above the `zt` bound, the
# truncated mass is 1 to double precision and the normal CDF is pure overhead.
_NEGLIGIBLE_TRUNCATION = 8.0

# Points on the 1-D grid an interval is read off. The density is smooth, so this
# sets the resolution of the endpoints rather than their accuracy -- 512 puts
# them well inside a metre on any depth this package reports.
_INTERVAL_POINTS = 512

# Nodes per axis the mesh is interpolated up to before `C`, `z_t` or the Curie
# depth are pushed through it. Costs no evaluation of the forward model -- the
# density and the conditional mean are smooth in `(beta, dz)` -- and it is what
# separates a density from a comb of one spike per node.
_MARGINAL_REFINE = 192

# Largest `dz` the data can speak about, as a multiple of `1/k_min`. The layer
# rolloff sits at `|k| dz ~ 1`, so past this the rolloff is below the longest
# wavelength the window measured and one thickness is indistinguishable from
# any larger one. The same inequality that sets the ladder's range in
# `_thickness_ladder`, used here to decide whether an interval is a measurement
# or a restatement of the forward model's numerical ceiling.
_MESH_IDENTIFIABLE = 1.0

# What a starting value falls back to when the spectrum cannot supply one --
# either because the high-`k` band is too short to regress a `beta` out of, or,
# for all four, because the spectrum has no usable bins at all.
#
# These are the constants `optimise` started every fit at before the start was
# derived, and keeping them is deliberate. A window poisoned by a NaN comes
# back with an empty spectrum, and the fit then returns its start untouched --
# which is exactly how a caller detects that nothing happened
# (`~/Global_CPD` raises `STARTING_GUESS` on it). Falling back to anything else
# would move a value whose only job is to be recognisable.
_FALLBACK_START = (3.0, 1.0, 10.0, 5.0)

#: Where `beta` lands when the high band cannot supply one. Named separately
#: because it is reached far more often than the whole-vector fallback.
_BETA_START = _FALLBACK_START[0]

# Fewest usable bins the linear solve can run on: two columns, `1` and `-2k`,
# need two rows to determine them.
_MIN_BINS = 2

# Range within which a `beta` read off the high-`k` asymptote is believed. It is
# a regression slope on a handful of noisy bins, so it can come back at
# anything; outside this it is noise rather than a measurement, and the fallback
# is the better guess. The same range the earlier prototype bounded `beta` to.
_BETA_SANE = (0.0, 8.0)

# Number of nodes on the `dz` ladder `_initial_guess` scans. Each node costs one
# `bouligand2009` evaluation plus a 2x2 solve, so this is the whole price of
# seeding -- against a fit that spends 15 to 45 of the same evaluations.
#
# Eight is where measurement put it, and the surprise is how flat the choice is:
# over 20 synthetics at a 4000 km window the recovered `dz` is *identical* at 5,
# 6, 8, 10, 12 and 16 nodes. The ladder only has to name the basin; the fit
# resolves it. So the node count is chosen on cost, and more nodes buy nothing.
_LADDER_NODES = 8

# Ends of the ladder, as multiples of `1/k_max` and `1/k_min`.
#
# The layer rolloff sits at `|k| dz ~ 1`, so a `dz` outside `[1/k_max, 1/k_min]`
# puts the rolloff outside the band this window measured. Past the top of that
# range the spectrum is in its low-`k` asymptote across the whole band, where
# `dz` survives only as `2 ln dz` inside the constant and is degenerate with `C`
# -- so nodes above it are not a wider search, they are a search of a direction
# the data cannot see. The bottom is extended half a rolloff because the
# fractal half-space asymptote beyond it still constrains `zt`.
_LADDER_MARGIN = (0.5, 1.0)

# Where the two straight-line bands of the asymptote estimate meet, as a
# fraction of the **logarithmic** wavenumber range: 0.5 is the geometric mean of
# `k_min` and `k_max`. A fraction rather than a wavenumber, so it works at every
# window size without being told anything about the window.
#
# Logarithmic, and this is the whole of it. The high band has to separate
# `(1-beta) ln k` from `-2 k z_t`, and over a narrow range in `k` those two
# regressors are nearly collinear. A *linear* split leaves the high band
# spanning a factor of ~2.8 in `k` whatever the window, which is not enough: at
# a 200 km window it returned beta = 3.25 +/- 1.13 against a truth of 3, wild
# enough on one realisation in eight to send the ladder scan to the wrong
# surface. In log space the same split gives 3.5 to 6 octaves and
# 3.12 +/- 0.48 there, 2.98 +/- 0.05 at 1000 km.
_ASYMPTOTE_SPLIT = 0.5

# Smallest `dz` the seeder will propose, in km. The reduced misfit is flat as
# `dz -> 0` -- the layer becomes a half-space and `dz` stops being identifiable
# -- so a node there is a start with no gradient to descend.
_MIN_THICKNESS = 0.5

# Deviance within which two ladder nodes count as indistinguishable, and the
# thinner layer wins. One unit is about one standard error on a single
# coordinate, and the competing basin that motivated the ladder sits 25% of the
# misfit away -- tens of units -- so this separates a tie from a real second
# minimum by a wide margin.
_LADDER_TIE = 1.0

# Relative determinant below which `_solve_small` gives up on the closed form
# and decomposes instead. Forming the normal equations squares the condition
# number, so this sits well above the double-precision floor.
_SINGULAR_RTOL = 1.0e-12

# How far the per-parameter correlation corrections may disagree before
# `_temperature` stops calling itself one number and says so. Measured across
# free fits and production-pinned ones they sit within 0.6-3.1% of each other,
# which is what makes a scalar temperature a statement about degrees of freedom
# rather than a convenience. Ten percent is several times the observed spread,
# so this fires on a change of kind rather than on noise.
_TEMPER_SPREAD_LIMIT = 0.10


#: The 2-D marginal posterior of `(beta, dz)`, with `C` and `zt` integrated out.
#:
#: * `beta`, `dz` -- the mesh axes, shape (n,) each
#: * `density`    -- shape (n, n), indexed [beta, dz], summing to 1
#: * `mean`       -- shape (n, n, 2), the conditional mean of `(C, zt)` at each
#:   node. Not a marginal: `(C, zt)` are Gaussian *given* a node, and their
#:   posterior is the mixture of those Gaussians weighted by `density`.
#: * `cov`        -- shape (2, 2), the conditional covariance of `(C, zt)`. One
#:   matrix for the whole mesh, because it does not depend on `(beta, dz)` --
#:   which is the fact the whole construction rests on.
#: * `edge_mass`  -- mass on each of the four edges, in the order
#:   (beta low, beta high, dz low, dz high). What says whether the mesh
#:   contained the posterior.
#:
#: A plain namedtuple so it pickles, which `parallelise_routine` needs.
#: * `mass`       -- shape (n, n), the fraction of each node's conditional that
#:   survives the bound on `zt`. `density` already carries it; it is kept so an
#:   interval on `zt` or on the Curie depth can undo it and integrate the
#:   untruncated Gaussian over the half line instead.
#: * `kmin`       -- the smallest wavenumber the spectrum carries. `1/kmin` is
#:   the thickest layer whose rolloff is still inside the measured band, and so
#:   the largest `dz` the data can say anything about at all.
Posterior = namedtuple(
    "Posterior", "beta dz density mean cov mass edge_mass kmin"
)


def _bilinear(x, y, values, fine_x, fine_y):
    """
    Bilinear interpolation of `values` onto the outer product of the fine axes.

    Two `numpy.interp` passes, because the mesh is a rectangular grid. Used to
    refine a posterior before pushing it through a map to `C`, `z_t` or the
    Curie depth: the density and the conditional mean are both smooth in
    `(beta, dz)`, so this recovers the continuous posterior the mesh is a
    quadrature rule for, without evaluating the forward model again.
    """
    along_y = np.empty((values.shape[0], fine_y.size))
    for row in range(values.shape[0]):
        along_y[row] = np.interp(fine_y, y, values[row])

    out = np.empty((fine_x.size, fine_y.size))
    for column in range(fine_y.size):
        out[:, column] = np.interp(fine_x, x, along_y[:, column])
    return out


def _solve_small(A, b):
    """
    Solve a symmetric 1x1 or 2x2 system in closed form.

    `_solve_linear` runs this once per node of a `dz` ladder, and the whole
    argument for seeding a fit rather than multi-starting it is that the ladder
    costs a few percent of one fit. At that size the answer is three
    multiplications, and `numpy.linalg.solve` -- let alone `lstsq`, which
    decomposes -- is dominated by its own dispatch. Measured across a twelve
    node ladder that overhead is comparable to the forward-model evaluations
    the nodes exist to spend.

    Falls back to `lstsq` when the system is singular, which is the case where
    the shortcut has nothing to offer anyway: a spectrum too short, or one
    whose two columns have collapsed onto each other.
    """
    if A.shape[0] == 1:
        if A[0, 0] > 0.0:
            return np.array([b[0] / A[0, 0]])
    else:
        determinant = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        # scale free: the columns differ by orders of magnitude, so an absolute
        # threshold would call a healthy system singular at one window size and
        # a singular one healthy at another
        if abs(determinant) > _SINGULAR_RTOL * abs(A[0, 0] * A[1, 1]):
            return np.array([
                (A[1, 1] * b[0] - A[0, 1] * b[1]) / determinant,
                (A[0, 0] * b[1] - A[1, 0] * b[0]) / determinant,
            ])

    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    return solution


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
    optimisation routines see `scipy.optimize.least_squares` for
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
            lower and upper bounds for :math:`\\beta, z_t, \\Delta z, C`.
            :math:`\\Delta z` is capped where the forward model stops
            evaluating, which depends on the grid spacing and is hundreds of km
            -- far beyond any Curie depth on Earth, so it never binds on data
            that constrain the base. Reassign this attribute to impose a
            tighter one, bearing in mind that a bound near the physical range
            will truncate the upper tail of a skewed posterior rather than
            report it.
        prior : dict
            dictionary of priors for :math:`\\beta, z_t, \\Delta z, C`
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

        # the starting point most recently derived, alongside `last_spectrum`.
        # A start that is computed rather than declared cannot be recovered from
        # the arguments a caller passed, and it is what says whether a fit that
        # came back looking untouched ever moved -- see `_initial_guess`.
        self.last_x0 = None

        # the mesh most recently evaluated by `posterior`, so intervals on
        # several targets can be read off one density rather than paying for it
        # per target -- the same reasoning as `last_spectrum`
        self.last_posterior = None

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
        :math:`\\pi/\\Delta x` in rad/km, whatever the window size, so the depth at which
        `pycurious.grid.bouligand2009` overflows is fixed by the grid spacing
        alone: 446 km at 2 km spacing, 111 km at 500 m. Both are far beyond any
        Curie depth on Earth, which is the point -- see `_COSH_OVERFLOW`.
        """
        return _COSH_OVERFLOW * (self.dx * 1.0e-3) / np.pi

    def add_prior(self, **kwargs):
        """
        Add a prior to the dictionary (tuple)
        Available priors are :math:`\\beta, z_t, \\Delta z, C`

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
        A Gaussian prior :math:`N(p, \\sigma_p)` on a parameter :math:`m` is
        just another observation, contributing a residual
        :math:`(m - p)/\\sigma_p`.

        Args:
            x : array shape (4,)
                :math:`\\beta, z_t, \\Delta z, C`
            kh : array shape (n,)
                wavenumbers (rad/km)
            Phi : array shape (n,)
                radial power spectrum :math:`\\Phi`
            sigma_Phi : array shape (n,)
                uncertainty of :math:`\\Phi`, as returned by
                `pycurious.grid.CurieGrid.window_spectrum`
            prior : dict, optional
                priors to use in place of `self.prior`

        Returns:
            residuals : array shape (n + number of priors,)

        Notes:
            Floating-point errors from `pycurious.grid.bouligand2009` are
            silenced because the fit legitimately probes parameters the forward
            model cannot evaluate, which would otherwise crash the minimiser.
            Any residual that comes back non-finite is replaced by a large
            finite value, so that one unusable bin costs the fit a fixed
            penalty rather than poisoning the whole vector.

            In practice the two that occur are both `invalid`, not overflow:
            `np.power` of a negative base when the step takes
            :math:`\\Delta z` below zero -- which the Curie profile does at
            every scan node below :math:`z_t`, since
            :math:`\\Delta z = CPD - z_t` -- and `inf * 0` from
            :math:`K_\\nu(0)` when it lands exactly on zero.

            This is `numpy.errstate` rather than `warnings.catch_warnings`
            deliberately. Both silence those, but `catch_warnings` swallows
            *every* warning raised in the block, including real ones from
            elsewhere in the library, and it is the more expensive of the two
            to enter several hundred times per fit.
        """
        beta, zt, dz, C = x

        with np.errstate(all="ignore"):
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
                array of variables :math:`\\beta, z_t, \\Delta z, C`
            kh : array shape (n,)
                wavenumbers (rad/km)
            Phi : array shape (n,)
                radial power spectrum :math:`\\Phi`
            sigma_Phi : array shape (n,)
                uncertainty of :math:`\\Phi`, as returned by
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

    # see `pycurious.grid.CurieGrid._resolve_spectrum`
    _SPECTRUM_ARGS = ("window", "xc", "yc", "taper", "process_subgrid", "dof_factor")
    _SPECTRUM_PROVENANCE = ("window", "xc", "yc", "dof_factor")
    _SPECTRUM_RETURNS = ("k", "Phi", "sigma_Phi")

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

    def _bound_arrays(self, free=None):
        """
        `self.bounds` as a pair of arrays, with `None` meaning infinite.

        Both the fit and the sampler need the bounds this way, and the `None`
        sentinel is the sort of convention that goes stale in one copy.

        Args:
            free : list of int, optional
                indices of `_PARAMETERS` to include, default all four

        Returns:
            lower, upper : arrays shape (len(free),)
        """
        if free is None:
            free = range(len(_PARAMETERS))
        pairs = [self.bounds[i] for i in free]
        return (
            np.array([-np.inf if lo is None else lo for lo, _ in pairs], dtype=float),
            np.array([np.inf if hi is None else hi for _, hi in pairs], dtype=float),
        )

    def _prior_misfit(self, x, prior=None):
        """
        The prior rows of `min_func` alone, at `x`.

        `residuals` appends one row per prior, so `min_func` is
        :math:`F_{spectral} + F_{prior}`. Splitting them is what lets the
        likelihood be tempered where it is read without touching the priors --
        see `_temperature`. This costs no evaluation of the forward model,
        because a prior row depends on the parameter value and nothing else.
        """
        if prior is None:
            prior = self.prior

        total = 0.0
        for name, value in zip(_PARAMETERS, np.asarray(x, dtype=float)):
            args = prior.get(name)
            if args is not None:
                total += ((value - args[0]) / args[1]) ** 2
        return 0.5 * float(total)

    def _solve_linear(self, kh, Phi, sigma_Phi, beta, dz, prior=None, zt=None,
                      C=None, clamp=True, temper=1.0):
        """
        Exact weighted :math:`C, z_t` at fixed :math:`\\beta, \\Delta z`, and the
        misfit there.

        Writing `pycurious.grid.bouligand2009` out,

        .. math::
            \\Phi = C \\cdot 1 + z_t \\cdot (-2k) + h(k; \\beta, \\Delta z)

        :math:`C` appears once, additively, and :math:`z_t` once, as
        :math:`-2 k z_t`; the Bessel term depends on neither. So at any
        :math:`\\beta, \\Delta z` the best :math:`C, z_t` is a two-column
        weighted linear solve -- exact, one step, no iteration. That is variable
        projection (Golub & Pereyra, 1973), and it is the same manoeuvre
        `_ANALYTIC_COLUMNS` already relies on for the Jacobian.

        A Gaussian prior is one more row of the same system, so pinning
        :math:`z_t` -- which is what a production run does -- is not lost.
        Either parameter can also be held: pass it and it becomes a known
        offset rather than an unknown, and the solve drops to one column.

        Args:
            kh : array shape (n,)
                wavenumbers (rad/km)
            Phi : array shape (n,)
                radial power spectrum :math:`\\Phi`
            sigma_Phi : array shape (n,)
                uncertainty of :math:`\\Phi`
            beta, dz : float
                the two parameters this solve is conditional on
            prior : dict, optional
                priors to use in place of `self.prior`
            zt, C : float, optional
                hold this parameter rather than solving for it
            temper : float (default=1.0)
                divide the **spectral** part of the misfit by this, leaving any
                prior rows alone. `1.0` is `min_func` exactly. See
                `_temperature` for where the value comes from and why the
                prior rows are excluded; the default is what every caller that
                is choosing a starting point wants, since a seed has to be the
                argmin of the objective `_fit` will actually minimise.

        Returns:
            C, zt : float
                the solution, clipped into `self.bounds`
            cost : float
                `min_func` at :math:`\\beta, z_t, \\Delta z, C` when `temper`
                is 1, which is what makes this comparable across a ladder of
                :math:`\\Delta z`

        Notes:
            The 2x2 normal equations are solved in closed form rather than by
            `numpy.linalg.lstsq`. The whole reason to seed a fit rather than
            multi-start it is that a scan costs a few percent of one fit, and
            `lstsq` does an SVD whose Python overhead alone is comparable to
            the forward-model evaluation the node exists to spend. The
            condition number is squared by forming the normal equations, but
            the two columns are :math:`1/\\sigma` and :math:`-2k/\\sigma` --
            smooth, well separated, and never near collinear -- so this is safe
            in double precision. `lstsq` is kept as a fallback for the singular
            case rather than as the default.

            Only one bound can be active at a time in practice, because `C` is
            unbounded by default. Where one is, clamping it and re-solving for
            the other is the exact constrained solution; where both would be,
            this returns the corner, which is not.
        """
        if prior is None:
            prior = self.prior

        # `bouligand2009` at C = zt = 0 is exactly the part of the model those
        # two add onto, so this is a decomposition rather than an approximation.
        with np.errstate(all="ignore"):
            shape = bouligand2009(kh, beta, 0.0, dz, 0.0)

        weight = 1.0 / sigma_Phi
        columns = {_C: weight, _ZT: -2.0 * kh * weight}
        base = (Phi - shape) * weight

        usable = np.isfinite(base) & np.isfinite(weight) & np.isfinite(columns[_ZT])

        held = {_C: C, _ZT: zt}
        solved = self._solve_columns(columns, base, usable, held, prior)

        # Clamp into the bounds. `zt >= 0` is the one that binds: a thick enough
        # layer can otherwise be paid for with a negative depth to the top, and
        # the scan would then prefer a basin no fit is allowed to reach.
        # Re-solving whatever is left is exact while only one bound is active,
        # which is the only case that arises -- `C` is unbounded by default.
        #
        # `clamp=False` is for `posterior`, which needs the *unconstrained*
        # conditional mean: a bound on a Bayesian conditional is a truncation,
        # carried as the mass below it, and clamping the mean onto the bound
        # instead would put a point mass where a truncated tail belongs.
        lower, upper = self._bound_arrays()
        for index in (_ZT, _C) if clamp else ():
            if held[index] is not None:
                continue
            clamped = float(np.clip(solved[index], lower[index], upper[index]))
            if clamped == solved[index]:
                continue
            held[index] = clamped
            solved = self._solve_columns(columns, base, usable, held, prior)

        C, zt = solved[_C], solved[_ZT]

        # The misfit from the pieces already in hand rather than from
        # `min_func`, which would evaluate the forward model a second time --
        # `bouligand2009(k, beta, zt, dz, C)` is `shape + C - 2 k zt` exactly,
        # so there is nothing left to compute. That halves what a ladder node
        # costs, and the ladder's whole claim on being worth running is that a
        # node is cheap next to a fit. `test_solve_linear_cost_is_min_func`
        # holds the two together.
        residual = np.where(
            usable, C * columns[_C] + zt * columns[_ZT] - base, _OVERFLOW_RESIDUAL
        )
        spectral = 0.5 * float(np.dot(residual, residual))

        x = np.array([beta, zt, dz, C], dtype=float)
        return C, zt, spectral / float(temper) + self._prior_misfit(x, prior)

    @staticmethod
    def _solve_columns(columns, base, usable, held, prior):
        """
        Weighted least squares over whichever of `columns` is not in `held`.

        The normal equations, with a Gaussian prior contributing
        :math:`1/\\sigma_p^2` to its own diagonal and
        :math:`p/\\sigma_p^2` to the right-hand side -- which is the same row it
        contributes to `residuals`, written in these coordinates rather than in
        the residual vector.

        Returns a full mapping, so a held parameter comes back alongside a
        solved one and the caller does not have to reassemble them.
        """
        unknowns = [index for index in (_C, _ZT) if held[index] is None]

        target = base
        for index in (_C, _ZT):
            if held[index] is not None:
                target = target - held[index] * columns[index]

        if not unknowns:
            return dict(held)

        if np.count_nonzero(usable) < len(unknowns):
            # nothing to fit against; fall back to the prior centre, or to zero
            return {
                index: (
                    held[index]
                    if held[index] is not None
                    else (prior.get(_PARAMETERS[index]) or (0.0,))[0]
                )
                for index in (_C, _ZT)
            }

        size = len(unknowns)
        A = np.zeros((size, size))
        b = np.zeros(size)
        for row, index in enumerate(unknowns):
            column = columns[index][usable]
            for col, other in enumerate(unknowns):
                A[row, col] = np.dot(column, columns[other][usable])
            b[row] = np.dot(column, target[usable])
            prior_args = prior.get(_PARAMETERS[index])
            if prior_args is not None:
                loc, scale = prior_args
                A[row, row] += 1.0 / scale ** 2
                b[row] += loc / scale ** 2

        solution = _solve_small(A, b)
        out = dict(held)
        for row, index in enumerate(unknowns):
            out[index] = float(solution[row])
        return out

    @staticmethod
    def _asymptote_estimates(kh, Phi):
        """
        :math:`\\beta` and :math:`\\Delta z` read off the two straight-line
        limits of the model.

        Either side of the layer rolloff the spectrum is a line in
        :math:`\\ln k` and :math:`k` together
        (`notes/bouligand-rederivation.md`):

        .. math::
            k \\Delta z \\gg 1: \\quad \\ln \\Phi = C' + (1-\\beta)\\ln k - 2 k z_t

            k \\Delta z \\ll 1: \\quad \\ln \\Phi = C'' + (3-\\beta)\\ln k - 2 k z_0

        so one three-column regression per band gives :math:`\\beta` and
        :math:`z_t` from the high band, the centroid :math:`z_0` from the low
        one, and :math:`\\Delta z = 2(z_0 - z_t)`. No evaluation of the forward
        model at all, and read off the `power=2` spectrum already in hand rather
        than by computing Tanaka's `power=1` version of the same window.

        Returns:
            beta : float or None
                the useful one. Measured against synthetics at a 1000 km
                window it lands within about 0.2 of the truth at
                :math:`\\beta` of both 2 and 3, which is close enough to put
                the ladder scan on the right surface -- and the scan is
                sensitive to it, because a ladder run at :math:`\\beta = 3`
                against a spectrum whose :math:`\\beta` is 2 ranks the
                thicknesses wrongly.
            dz : float or None
                the weak one. The low-band slope only gives the centroid once
                :math:`k \\Delta z \\ll 1`, which a window merely large enough
                to fit does not reach: at a true 15 km this comes back near 4.
                It is offered as one more ladder node, where being wrong costs
                nothing, and never used on its own.

        Notes:
            Each is None where its band is too short to regress or the result
            is not physical, and the caller falls back rather than propagating
            a nonsense value.
        """
        kmin, kmax = float(np.min(kh)), float(np.max(kh))
        if not (kmin > 0.0 and kmax > kmin):
            return None, None
        cut = kmin * (kmax / kmin) ** _ASYMPTOTE_SPLIT
        high, low = kh > cut, kh <= cut

        if high.sum() < 4 or low.sum() < 4:
            return None, None

        def line(mask):
            # [1, ln k, k] against ln Phi, which is both asymptotes' shape
            design = np.column_stack(
                [np.ones(int(mask.sum())), np.log(kh[mask]), kh[mask]]
            )
            coefficients, *_ = np.linalg.lstsq(design, Phi[mask], rcond=None)
            return coefficients

        with np.errstate(all="ignore"):
            upper = line(high)
            lower = line(low)

        beta = 1.0 - upper[1]
        if not (np.isfinite(beta) and _BETA_SANE[0] <= beta <= _BETA_SANE[1]):
            beta = None

        # -2 k zt in the high band, -2 k z0 in the low one
        dz = 2.0 * (-0.5 * lower[2] - -0.5 * upper[2])
        if not (np.isfinite(dz) and dz > 0.0):
            dz = None

        return beta, dz

    def _thickness_ladder(self, kh, asymptote=None):
        """
        The :math:`\\Delta z` the scan tries, in km.

        The layer rolloff sits at :math:`k \\Delta z \\sim 1`, so a thickness
        outside :math:`[1/k_{max}, 1/k_{min}]` puts the rolloff outside the band
        this window measured, and the spectrum has nothing to say about it. The
        ladder is that range scaled by `_LADDER_MARGIN`, log spaced because it
        spans three orders of magnitude, and clipped to `self.bounds`.

        `asymptote`, when the straight-line limits gave one, joins as one more
        node. It is biased low and is never trusted on its own -- but a node
        costs one forward-model evaluation and the scan ranks it against the
        rest, so a wrong one is harmless and a right one is free.
        """
        lower, upper = self._bound_arrays()
        kmin, kmax = float(np.min(kh)), float(np.max(kh))

        low = max(_LADDER_MARGIN[0] / kmax, _MIN_THICKNESS, lower[_DZ])
        high = min(_LADDER_MARGIN[1] / kmin, upper[_DZ])
        if not (high > low):
            return np.array([max(low, _MIN_THICKNESS)])

        ladder = np.geomspace(low, high, _LADDER_NODES)
        if asymptote is not None:
            ladder = np.append(ladder, np.clip(asymptote, low, high))

        return np.unique(ladder)

    def _initial_guess(self, args, beta=None, zt=None, dz=None, C=None, prior=None):
        """
        Starting point for the fit, derived from the spectrum where it can be.

        Anything the caller supplied is used as given; anything left `None` is
        derived. :math:`C` and :math:`z_t` come from the exact linear solve in
        `_solve_linear`, so they are never guessed at all.
        :math:`\\Delta z` comes from a scan over `_thickness_ladder` of the
        objective with those two profiled out.

        That reduced objective *is* `min_func` minimised over :math:`C, z_t`, so
        a scan of it is a strictly better basin detector than multi-starting the
        fit itself over the same nodes -- and costs a few percent of one fit
        rather than one fit per node. :math:`\\beta` is left at `_BETA_START`:
        it is not the direction the misfit is multimodal in.

        Args:
            args : tuple
                `(kh, Phi, sigma_Phi)`, as `residuals` takes them
            beta, zt, dz, C : float, optional
                hold this parameter rather than deriving it
            prior : dict, optional
                priors to use in place of `self.prior`

        Returns:
            x0 : array shape (4,)
                :math:`\\beta, z_t, \\Delta z, C`. Also left on the instance as
                `last_x0`, because a derived start is no longer something a
                caller can reconstruct from the arguments it passed -- and a
                fit that returned exactly its start is the signature of a
                window the optimiser could not move in, which is worth being
                able to detect. `sensitivity` derives one per realisation, so
                there it is the last of them.

        Notes:
            The misfit is genuinely bimodal in :math:`\\Delta z`, and the
            constant this replaced sat in the wrong basin on about one
            thick-layer synthetic in five -- at a misfit 25% higher, so not the
            flat-likelihood tie where both answers are equally good. See
            `notes/spectrum-binning-weighting-multitaper.md`.

            Measured over 20 synthetics per cell with
            `notes/bench/score_starting_values.py`, against a dense ten-point
            multi-start standing in for the global minimum:

            =============== ===================== =====================
            regime          constant              derived
            =============== ===================== =====================
            4000 km, dz 45  32.61 +/- 15.32, 7/20 46.08 +/- 5.85, 0/20
            4000 km, dz 30  28.93 +/- 4.83,  1/20 30.42 +/- 3.32, 0/20
            1000 km, any    identical             identical
            200 km, any     unusable              unusable
            =============== ===================== =====================

            where the count is realisations that missed the global minimum.
            The gain is concentrated where a window resolves the layer at all
            and the constant is furthest from it. At 200 km neither start is
            worth anything: both land tens of km from a truth of 30, because
            that window cannot see a layer that thick.

            **Cost.** The ladder is a flat `_LADDER_NODES + 2` evaluations of
            the forward model, and it earns them back in iterations the fit
            does not then spend -- 44.5 evaluations down to 18.1 at
            4000 km and dz 45, so 27.9 in total against 44.5. It does not
            always earn them back: on a *thin* layer at a large window the
            constant was already almost exactly right, and there the derived
            start costs 48.5 against 18.6 for the same answer. Two ways of
            avoiding that were measured and both made the average worse -- a
            coarse `beta` axis on the scan, and scanning at the measured
            `beta` and 3.0 together, each buying identical accuracy for 1.4x
            to 2x the evaluations.

            Where two nodes are within `_LADDER_TIE` of each other in deviance
            the smaller :math:`\\Delta z` wins. A thicker layer that fits no
            better is the unidentifiable regime -- past the rolloff
            :math:`\\Delta z` survives only as :math:`2 \\ln \\Delta z` inside
            the constant -- and the thinner one is the more conservative reading
            of a spectrum that cannot tell them apart.

            **A lower misfit is not always a better answer.** Where the window
            does not constrain the fit the likelihood has a second minimum at
            high :math:`\\beta` and low :math:`\\Delta z`, and a start that
            finds it reports a confident, worse number. The old constant
            avoided that by anchoring rather than by being right. Use
            `add_prior` on :math:`\\beta` where the window is small, which is
            what a production run does -- and which this then starts from,
            ahead of the spectrum's own estimate.

            Nothing here is random, and nothing is read from the instance
            except `bounds` and `prior`, so the same spectrum always gives the
            same start. Routines that hand a spectrum straight to a second call
            depend on that.
        """
        if beta is not None and zt is not None and dz is not None and C is not None:
            self.last_x0 = np.array([beta, zt, dz, C], dtype=float)
            return self.last_x0

        if prior is None:
            prior = self.prior

        kh, Phi, sigma_Phi = args
        kh = np.asarray(kh, dtype=float)

        # A window carrying a NaN comes back with an empty spectrum, and a band
        # cut can empty one too. There is nothing to derive from, so fall back
        # to the constants rather than reducing over an empty axis -- and the
        # fit then returns its start untouched, which is how a caller sees that
        # the window was unusable.
        if np.count_nonzero(np.isfinite(kh) & (kh > 0.0)) < _MIN_BINS:
            supplied = (beta, zt, dz, C)
            self.last_x0 = np.array(
                [_FALLBACK_START[i] if supplied[i] is None else supplied[i]
                 for i in range(len(_PARAMETERS))], dtype=float
            )
            return self.last_x0

        # Both come from the straight-line limits, and neither costs an
        # evaluation of the forward model. `beta` is the one that matters: the
        # ladder is scanned at a fixed `beta`, and at the wrong one it ranks the
        # thicknesses wrongly -- a spectrum whose `beta` is 2, scanned at 3,
        # sends the fit into a basin 14% worse in misfit than the one it should
        # have found. Reading `beta` off the high band instead of assuming 3.0
        # is what makes the scan meaningful on a spectrum that is not the
        # textbook one.
        # A prior on `beta` outranks the spectrum's own estimate. `C` and
        # `zt` are linear, so their priors enter `_solve_linear` as extra rows
        # and shape the answer directly; `beta` is not, and starting it
        # anywhere other than where the prior says is asking the optimiser to
        # walk back to it. It also guards the one case the straight-line
        # estimate cannot handle: where the high-`k` band carries something the
        # model does not contain, the regression reads that as `beta` and comes
        # back low. On WDMAM -- whose high-`k` band is an unmodelled ~4 km
        # resolution rolloff -- it returns 1.7 to 2.0 against a fitted 3.0 to
        # 3.5, and a `beta` prior of sigma 0.15 is precisely how that dataset
        # says so.
        prior_beta = prior.get(_PARAMETERS[_BETA])
        if beta is None and prior_beta is not None:
            beta = prior_beta[0]

        asymptote_beta, asymptote_dz = (
            self._asymptote_estimates(kh, Phi)
            if beta is None or dz is None
            else (None, None)
        )

        if beta is None:
            beta = _BETA_START if asymptote_beta is None else asymptote_beta
        lower, upper = self._bound_arrays()
        beta = float(np.clip(beta, lower[_BETA], upper[_BETA]))

        if dz is None:
            ladder = self._thickness_ladder(kh, asymptote_dz)
            costs = np.array([
                self._solve_linear(
                    kh, Phi, sigma_Phi, beta, node, prior, zt=zt, C=C
                )[2]
                for node in ladder
            ])
            # deviance is twice the misfit, and `_LADDER_TIE` is in its units
            close = costs <= costs.min() + 0.5 * _LADDER_TIE
            dz = float(ladder[np.argmax(close)])

        C_hat, zt_hat, _ = self._solve_linear(
            kh, Phi, sigma_Phi, beta, dz, prior, zt=zt, C=C
        )

        self.last_x0 = np.array([beta, zt_hat, dz, C_hat], dtype=float)
        return self.last_x0

    def _linear_precision(self, kh, sigma_Phi, prior=None):
        """
        Precision of the conditional posterior of :math:`(C, z_t)`.

        :math:`A = G^T G` with :math:`G = [1/\\sigma, -2k/\\sigma]`, plus
        :math:`1/\\sigma_p^2` on the diagonal for a Gaussian prior. **This does
        not depend on** :math:`\\beta` **or** :math:`\\Delta z`, which is the
        fact `posterior` rests on: it makes the Gaussian integral over
        :math:`(C, z_t)` contribute a constant :math:`\\tfrac12 \\log \\det A`
        to the marginal, so the marginal *is* the reduced misfit
        `_solve_linear` already returns.

        Returns `(A, A^-1)` in :math:`(C, z_t)` order, matching `_solve_linear`.
        """
        if prior is None:
            prior = self.prior

        weight = 1.0 / sigma_Phi
        columns = np.column_stack([weight, -2.0 * kh * weight])
        usable = np.all(np.isfinite(columns), axis=1)
        columns = columns[usable]

        A = columns.T @ columns
        for index in (_C, _ZT):
            args = prior.get(_PARAMETERS[index])
            if args is not None:
                # (C, zt) order, so C is row 0 and zt row 1
                row = 0 if index == _C else 1
                A[row, row] += 1.0 / args[1] ** 2

        return A, np.linalg.inv(A)

    def _mesh_box(self, args, x_hat, bracket):
        """
        First guess at a box containing the posterior, in `(beta, dz)`.

        Taken from the curvature at the mode, stretched further above in
        :math:`\\Delta z` because that tail is the long one -- which is a
        Gaussian approximation to a posterior that `profile` exists precisely
        because it distrusts. It is a *starting* box for that reason: `posterior`
        widens whichever edge is still carrying mass, and says so when widening
        stops helping.
        """
        if bracket is not None:
            (beta_lo, beta_hi), (dz_lo, dz_hi) = bracket
            return [float(beta_lo), float(beta_hi)], [float(dz_lo), float(dz_hi)]

        covariance = self._covariance(x_hat, *args)
        sigma = np.sqrt(np.abs(np.diag(covariance)))
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            # a singular fit says nothing about where the posterior is; fall
            # back to a wide box and let the expansion do the work
            sigma = np.array([1.0, 1.0, max(x_hat[_DZ], 1.0), 1.0])

        lower, upper = self._bound_arrays()
        beta = [x_hat[_BETA] - _MESH_SPAN * sigma[_BETA],
                x_hat[_BETA] + _MESH_SPAN * sigma[_BETA]]
        dz = [x_hat[_DZ] - _MESH_SPAN * sigma[_DZ],
              x_hat[_DZ] + _MESH_SKEW * _MESH_SPAN * sigma[_DZ]]

        beta[0] = max(beta[0], lower[_BETA])
        beta[1] = min(beta[1], upper[_BETA])
        dz[0] = max(dz[0], lower[_DZ], _MIN_THICKNESS * 0.1)
        dz[1] = min(dz[1], upper[_DZ])
        return beta, dz

    def _mesh_density(self, args, beta_box, dz_box, nodes, prior, sd_zt):
        """
        Evaluate the marginal posterior on one box.

        Returns axes, density, the unconstrained conditional mean of
        :math:`(C, z_t)`, the truncated mass at each node, and the mass on each
        of the four edges.

        Two things happen here that do not happen in `_initial_guess`, and both
        are the difference between an argmin and a density:

        * the conditional mean is taken **unclamped** (`clamp=False`), because
          the bound on :math:`z_t` is a truncation of a distribution rather
          than a wall for a point estimate;
        * the mass below that bound is subtracted from the log density. Unlike
          :math:`\\tfrac12 \\log \\det A`, which is constant over the mesh and
          drops out, this term moves with the conditional mean and so does not.
        """
        kh, Phi, sigma_Phi = args
        beta_axis = np.linspace(beta_box[0], beta_box[1], nodes)
        dz_axis = np.linspace(dz_box[0], dz_box[1], nodes)
        floor = self._bound_arrays()[0][_ZT]

        cost = np.empty((nodes, nodes))
        mean = np.empty((nodes, nodes, 2))
        mass = np.ones((nodes, nodes))

        for i, beta in enumerate(beta_axis):
            for j, dz in enumerate(dz_axis):
                C, zt, value = self._solve_linear(
                    kh, Phi, sigma_Phi, beta, dz, prior, clamp=False
                )
                mean[i, j] = (C, zt)
                reduced = (zt - floor) / sd_zt if sd_zt > 0.0 else np.inf
                if reduced < _NEGLIGIBLE_TRUNCATION:
                    truncated = float(stats.norm.cdf(reduced))
                    mass[i, j] = truncated
                    value = (value - np.log(truncated) if truncated > 0.0
                             else np.inf)
                cost[i, j] = value

        # The cost is minus the log of the marginal, up to the constant
        # `0.5 log det A`. Subtracting the minimum before exponentiating is what
        # keeps a spectrum of several hundred bins from underflowing to zero
        # everywhere.
        finite = np.isfinite(cost)
        blank = np.full((nodes, nodes), np.nan)
        if not finite.any():
            return beta_axis, dz_axis, blank, mean, mass, np.full(4, np.nan)

        density = np.where(finite, np.exp(-(cost - cost[finite].min())), 0.0)
        total = density.sum()
        if total <= 0.0 or not np.isfinite(total):
            return beta_axis, dz_axis, blank, mean, mass, np.full(4, np.nan)
        density /= total

        edge_mass = np.array([
            density[0, :].sum(), density[-1, :].sum(),
            density[:, 0].sum(), density[:, -1].sum(),
        ])
        return beta_axis, dz_axis, density, mean, mass, edge_mass

    def _fit(self, y0, args, prior=None, free=None, fixed=None, curie=False):
        """
        Bounded least-squares fit of `residuals`, with the Jacobian supplied.

        Every fit in this module goes through here, so they cannot drift apart
        on method, bounds or Jacobian.

        Args:
            y0 : array shape (len(free),)
                starting point, in the free coordinates
            args : tuple
                `(kh, Phi, sigma_Phi)`, as `residuals` takes them
            prior : dict, optional
                priors to use in place of `self.prior`
            free : list of int, optional
                indices of `_PARAMETERS` allowed to vary, default all four
            fixed : tuple, optional
                `(index, value)` for a coordinate held constant, as `profile`
                holds one
            curie : bool (default=False)
                the held coordinate is :math:`\\Delta z = \\mathrm{CPD} - z_t`
                rather than a constant, so it moves with :math:`z_t`

        Returns:
            res : `scipy.optimize.OptimizeResult`
                `res.cost` is `min_func` at the solution by construction --
                both are half the sum of squares of the same vector.

        Notes:
            This is `least_squares`, not `minimize`. The problem is a sum of
            squares and `residuals` already exposes the vector, so the
            trust-region reflective method can use structure L-BFGS-B cannot
            see. Measured over six synthetic 200 km windows with the BLAS
            pinned to one thread, `optimise` costs 12.5 ms of CPU against
            5.0 ms and a `dz` profile 102 ms against 33 ms -- about 3x -- on
            **359** evaluations of the forward model per vertex rather than
            2012.

            The evaluation count is the durable number; the timings are not.
            Unpinned, L-BFGS-B calls a threaded BLAS whose workers spin-wait,
            and on a busy machine the same comparison reads anywhere from 20x
            to 400x depending on how many cores are already contended. That is
            an artefact of the measurement, not a property of the algorithms.
            Pin `OMP_NUM_THREADS` and friends before timing anything here.

            Parameters agree with L-BFGS-B to three or four figures on windows
            that constrain the fit, and exactly on real band-limited spectra.
            Where the likelihood is flat they can disagree by much more --
            `dz` of 310 km against 139 km for a misfit difference of 2e-6 on
            one 100 km synthetic -- because both answers are equally good and
            neither optimiser has anything to descend. That is a property of
            the window, not of the change.
        """
        if free is None:
            free = list(range(len(_PARAMETERS)))
        if prior is None:
            prior = self.prior

        kh, sigma = args[0], args[2]
        nk = np.size(kh)

        # The map from the free coordinates to the full parameter vector is
        # affine, `x = transform @ y + offset`, in all three cases the callers
        # use. Writing it as a matrix once means `expand` and the chain rule in
        # `jacobian` are the same statement rather than two hand-maintained
        # ones -- an inconsistency between them is close to invisible in the
        # output, since the fit converges from a wrong Jacobian anyway.
        transform = np.zeros((len(_PARAMETERS), len(free)))
        for column, index in enumerate(free):
            transform[index, column] = 1.0
        offset = np.zeros(len(_PARAMETERS))
        if fixed is not None:
            offset[fixed[0]] = fixed[1]
            if curie:
                # dz = CPD - zt, so the dz row differentiates through zt
                transform[fixed[0], free.index(_ZT)] = -1.0

        def expand(y):
            return transform @ y + offset

        # `least_squares` evaluates the residual at a point and then asks for
        # the Jacobian there, which is the only reuse there is: a rejected step
        # is retried somewhere else, never at the same point. One entry.
        last = [None, None]

        def residuals_at(x):
            key = x.tobytes()
            if last[0] != key:
                last[0] = key
                last[1] = self.residuals(x, *args, prior=prior)
            return last[1]

        # Everything in the Jacobian except the differenced columns is constant
        # over the fit, so build it once. The analytic columns depend only on
        # `kh` and `sigma`; a prior contributes a row whose derivative is
        # `1/scale` against its own parameter and zero elsewhere, in the order
        # `residuals` appends them. Columns the transform discards are filled
        # anyway -- harmless, and cheaper than deciding not to.
        prior_rows = [
            index
            for index, key in enumerate(_PARAMETERS)
            if prior.get(key) is not None
        ]
        constant = np.zeros((nk + len(prior_rows), len(_PARAMETERS)))
        for index, key in enumerate(_PARAMETERS):
            analytic = _ANALYTIC_COLUMNS.get(key)
            if analytic is not None:
                constant[:nk, index] = analytic(kh, sigma)
        for row, index in enumerate(prior_rows):
            constant[nk + row, index] = 1.0 / prior[_PARAMETERS[index]][1]

        # what is left to difference: the free columns with no closed form,
        # plus the held one when the Curie constraint makes it move with `zt`
        differenced = [i for i in free if _PARAMETERS[i] not in _ANALYTIC_COLUMNS]
        if curie:
            differenced.append(fixed[0])

        def jacobian(y):
            x = expand(y)
            J = constant.copy()
            base = residuals_at(x)
            for index in differenced:
                step = _JACOBIAN_STEP * max(abs(x[index]), 1.0)
                xp = x.copy()
                xp[index] += step
                # spectral rows only -- the prior rows are already exact in
                # `constant`, and differencing them would count them twice
                J[:nk, index] = (residuals_at(xp)[:nk] - base[:nk]) / step
            return J @ transform

        lower, upper = self._bound_arrays(free)

        # L-BFGS-B accepted an equality bound and simply pinned the parameter.
        # Trust-region reflective needs a box with an interior to reflect
        # inside and refuses one without, so give it the narrowest box that is
        # numerically distinct. That pins the parameter just as effectively,
        # and leaves it near enough the edge that `_warn_on_bounds` still says
        # the uncertainty there is meaningless.
        #
        # Equality only. An *inverted* bound is a typo, not an intention --
        # `self.bounds` is documented as reassignable, so it is a typo a user
        # can make -- and widening it would silently rewrite it into whichever
        # of the two numbers happened to be first. Let `least_squares` raise.
        # An infinite bound cannot compare equal to the other, which is always
        # its opposite sign, so this needs no finiteness test.
        collapsed = upper == lower
        if collapsed.any():
            upper[collapsed] = lower[collapsed] + _DEGENERATE_BOUND * np.maximum(
                np.abs(lower[collapsed]), 1.0
            )

        res = least_squares(
            lambda y: residuals_at(expand(y)),
            # trust-region reflective requires a feasible start, which a
            # caller passing the unconstrained solution into a constrained fit
            # cannot guarantee
            np.clip(np.asarray(y0, dtype=float), lower, upper),
            jac=jacobian,
            bounds=(lower, upper),
        )

        # `res.x` is in the free coordinates, which for a constrained fit are
        # not the four parameters. Carrying the expanded vector saves every
        # caller reconstructing the affine map, and `_profiled_misfit` needs it
        # to split `res.cost` into its spectral and prior parts without
        # evaluating the forward model a second time.
        res.x_full = expand(res.x)
        return res

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
        this is generalised least squares rather than :math:`(J^T J)^{-1}` --
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
        beta=None,
        zt=None,
        dz=None,
        C=None,
        taper=np.hanning,
        process_subgrid=None,
        dof_factor=None,
        return_cov=False,
        spectrum=None,
        **kwargs
    ):
        """
        Find the optimal parameters of :math:`\\beta, z_t, \\Delta z, C`
        for a given centroid (xc,yc) and window size, with their
        uncertainties.

        Args:
            window : float
                size of window in metres
            xc : float
                centroid x values
            yc : float
                centroid y values
            beta : float, optional
                starting fractal parameter, derived from the spectrum if None
            zt : float, optional
                starting top of magnetic layer, derived if None
            dz : float, optional
                starting thickness of magnetic layer, derived if None
            C : float, optional
                starting field constant, derived if None
            taper : taper (default=`numpy.hanning`)
                taper function, set to None for no taper function
            process_subgrid : function
                a custom function to process the subgrid
            dof_factor : float, optional
                override the effective-degrees-of-freedom deflation applied to
                the spectral uncertainties, see
                `pycurious.grid.CurieGrid.window_spectrum`
            return_cov : bool (default=False)
                also return the 4x4 parameter covariance matrix
            spectrum : tuple (k, Phi, sigma_Phi), optional
                a spectrum already in hand, as returned by
                `pycurious.grid.CurieGrid.window_spectrum` with `power=2` or
                read from `last_spectrum`. Skips computing one, and `window`,
                `xc`, `yc`, `taper`, `process_subgrid` and `dof_factor` are
                then unused.
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
            `tests/test_mag_data.txt` moves :math:`\\Delta z` by 5.5 km, where
            the fit reports about 3.3 km.

            `sigma_dz` in particular should be read as a lower bound. The
            likelihood in :math:`\\Delta z` has a long upper tail (Mather &
            Fullea, 2019), so a symmetric interval is the wrong shape for it.
            Measured over 200 independent synthetics, `sigma_beta`, `sigma_zt`
            and `sigma_C` reproduce the true spread to within 1%, while
            `sigma_dz` understates it by about 40%. Use `profile` for an honest
            interval on :math:`\\Delta z` and on the Curie depth.

            Computing the spectrum is most of what this costs at a large
            window, so hand the same one to `profile` rather than paying for it
            twice::

                >>> beta, zt, dz, C = grid.optimise(2000e3, xc, yc)[:4]
                >>> _, _, lo, hi = grid.profile(
                ...     2000e3, xc, yc, "CPD", spectrum=grid.last_spectrum,
                ...     beta=beta, zt=zt, dz=dz, C=C)

            which takes 41% off the pair at a 2049-cell window.

            A starting value left at None is derived from the spectrum by
            `_initial_guess` rather than guessed. Supply one to override it --
            they are independent, so `dz=30` alone starts the other three from
            the best they can be at that thickness.
        """

        k, Phi, sigma_Phi = self._resolve_spectrum(
            spectrum, window, xc, yc, taper, process_subgrid, dof_factor, **kwargs
        )

        x0 = self._initial_guess((k, Phi, sigma_Phi), beta, zt, dz, C)

        x = self._fit(x0, (k, Phi, sigma_Phi)).x

        self._warn_on_bounds(x)
        cov = self._covariance(x, k, Phi, sigma_Phi)
        sigma = np.sqrt(np.diag(cov))

        if return_cov:
            return tuple(x) + tuple(sigma) + (cov,)
        return tuple(x) + tuple(sigma)

    def posterior(
        self,
        window,
        xc,
        yc,
        nodes=_MESH_NODES,
        bracket=None,
        beta=None,
        zt=None,
        dz=None,
        C=None,
        taper=np.hanning,
        process_subgrid=None,
        dof_factor=None,
        spectrum=None,
        **kwargs
    ):
        """
        The posterior over :math:`\\beta` and :math:`\\Delta z`, on a mesh, with
        :math:`C` and :math:`z_t` integrated out exactly.

        The forward model is
        :math:`\\Phi = C \\cdot 1 + z_t \\cdot (-2k) + h(k; \\beta, \\Delta z)`,
        so :math:`C` and :math:`z_t` are linear coefficients on basis vectors
        that do not involve the other two. Their conditional posterior is
        therefore an exact Gaussian, and integrating it out leaves a posterior
        over :math:`(\\beta, \\Delta z)` alone -- two dimensions, which is small
        enough to *evaluate* rather than sample.

        Args:
            window : float
                size of window in metres
            xc, yc : float
                centroid
            nodes : int (default=32)
                mesh nodes per axis. The moments are exact by 24; see
                `_MESH_NODES`.
            bracket : tuple, optional
                `((beta_min, beta_max), (dz_min, dz_max))`. Default is a box
                from the curvature at the mode, widened on any edge still
                carrying mass.
            beta, zt, dz, C : float, optional
                starting values for the fit that locates the mode, as `optimise`
                takes them. They set where the mesh is centred, not what it
                contains.
            taper, process_subgrid, dof_factor, spectrum, kwargs
                as `optimise`

        Returns:
            posterior : `Posterior`
                axes, density, the conditional mean of :math:`(C, z_t)` at each
                node, their constant conditional covariance, and the mass left
                on each of the four edges

        Usage:
            >>> p = grid.posterior(2000e3, xc, yc)
            >>> dz_marginal = p.density.sum(axis=0)     # over beta
            >>> _, _, lo, hi = grid.profile(
            ...     2000e3, xc, yc, "CPD", method="mesh", posterior=p)

        Notes:
            **What makes this exact rather than approximate.** Writing
            :math:`p = (C, z_t)`, the whitened residual is affine in `p`, so

            .. math::
                \\tfrac12 \\|Gp - y\\|^2
                = \\tfrac12 (p - \\hat p)^T A (p - \\hat p) + F(\\beta, \\Delta z)

            and the Gaussian integral over `p` contributes
            :math:`\\tfrac12 \\log \\det A`. The design matrix
            :math:`G = [1/\\sigma, -2k/\\sigma]` is built from the wavenumbers,
            the uncertainties and the prior widths only -- **nothing in it
            depends on** :math:`\\beta` **or** :math:`\\Delta z` -- so that term
            is an additive constant and the marginal posterior *is* the reduced
            misfit `_solve_linear` returns. A Gaussian prior on :math:`C` or
            :math:`z_t` adds a row to `G` and changes nothing about the
            argument; priors on the other two ride along inside `F`.

            This is the same identity behind the two exact Jacobian columns in
            `_fit` and behind the derived starting point in `_initial_guess`,
            read as a density rather than as a derivative or an argmin.

            **The bound on** :math:`z_t` **does not drop out.** It is bounded
            below, so the conditional is a Gaussian truncated to a half plane
            whose mass *does* vary over the mesh. `mean` and `cov` describe the
            untruncated conditional; anything integrating them -- `profile` with
            `method="mesh"` -- applies the truncation. On windows that determine
            :math:`z_t` at all the correction is inert, and it is not where
            `_warn_on_bounds` fires.

            **Read `edge_mass`.** A mesh that has clipped a tail is wrong in the
            direction of a *tighter* interval, and silently, which is the one
            failure a sampler does not have. Edges above `_MESH_EDGE_TOL` are
            widened and retried; what is reported is what remained.

            **This posterior is too sharp, and so is the profile deviance.**
            `_gls_covariance` corrects the reported covariance for correlation
            between neighbouring spectral bins; `min_func` does not, so the
            likelihood both this and `profile` are built on still treats them as
            independent. Measured over 100 synthetic realisations, a nominal
            68.27% interval on :math:`\\beta` covers 0.58 here and 0.52 through
            the scan, where the Gaussian :math:`\\sigma` `optimise` reports --
            which *is* corrected -- covers 0.65. The inflation the correction
            applies is 1.30, and a likelihood too sharp by exactly that would
            cover 0.56, which is what both do. Neither reading of the likelihood
            is at fault and no change to the reading fixes it; the correction
            has to reach the objective. Until it does, treat an interval from
            either method as a lower bound on the uncertainty.
        """
        args = self._resolve_spectrum(
            spectrum, window, xc, yc, taper, process_subgrid, dof_factor, **kwargs
        )
        prior = self.prior

        x_hat = self._fit(self._initial_guess(args, beta, zt, dz, C), args).x
        beta_box, dz_box = self._mesh_box(args, x_hat, bracket)

        # A caller who supplied a box asked for that box, so it is evaluated
        # once and reported as it stands.
        lower, upper = self._bound_arrays()
        remaining = 0 if bracket is not None else _MESH_EXPANSIONS

        precision, covariance = self._linear_precision(args[0], args[2], prior)
        sd_zt = float(np.sqrt(covariance[1, 1]))

        while True:
            beta_axis, dz_axis, density, mean, mass, edge_mass = \
                self._mesh_density(
                    args, beta_box, dz_box, int(nodes), prior, sd_zt
                )
            if remaining <= 0 or not np.all(np.isfinite(edge_mass)):
                break

            # Widen only the edges still carrying mass, and only where there is
            # room: an edge already sitting on a parameter bound is as wide as
            # it goes, and the posterior beyond it does not exist.
            widened = False
            for spilled, box, side, limit in (
                (edge_mass[0], beta_box, 0, lower[_BETA]),
                (edge_mass[1], beta_box, 1, upper[_BETA]),
                (edge_mass[2], dz_box, 0, lower[_DZ]),
                (edge_mass[3], dz_box, 1, upper[_DZ]),
            ):
                if spilled <= _MESH_EDGE_TOL:
                    continue
                span = box[1] - box[0]
                target = box[side] - span if side == 0 else box[side] + span
                target = max(target, limit) if side == 0 else min(target, limit)
                if not np.isclose(target, box[side]):
                    box[side] = target
                    widened = True

            if not widened:
                break
            remaining -= 1

        finite_k = args[0][np.isfinite(args[0]) & (args[0] > 0.0)]
        self.last_posterior = Posterior(
            beta_axis, dz_axis, density, mean, covariance, mass, edge_mass,
            float(finite_k.min()) if finite_k.size else np.nan,
        )
        return self.last_posterior

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
        beta=None,
        zt=None,
        dz=None,
        C=None,
        taper=np.hanning,
        process_subgrid=None,
        dof_factor=None,
        **kwargs
    ):
        """
        Iterate through a list of centroids to compute the optimal values
        of :math:`\\beta, z_t, \\Delta z, C` for a given window size.

        Args:
            window : float
                size of window in metres
            xc_list : ndarray shape (l,)
                centroid x values
            yc_list : ndarray shape (l,)
                centroid y values
            beta : float, optional
                starting fractal parameter, derived per centroid if None
            zt : float, optional
                starting top of magnetic layer, derived if None
            dz : float, optional
                starting thickness of magnetic layer, derived if None
            C : float, optional
                starting field constant, derived if None
            taper : function
                taper function (default=`numpy.hanning`)
                set to None for no taper function
            process_subgrid : func
                a custom function to process the subgrid
            dof_factor : float, optional
                override the effective-degrees-of-freedom deflation applied to
                the spectral uncertainties, see
                `pycurious.grid.CurieGrid.window_spectrum`
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

            A starting value left at None is derived at *each* centroid from
            that centroid's own spectrum, which is what a map wants: one
            constant good for the middle of a grid is a poor start at its
            edges.
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

    def _mesh_samples(self, posterior, target):
        """
        The mesh as a weighted sample of one target, plus the conditional width.

        Every target is a function of position on the mesh: :math:`\\beta` and
        :math:`\\Delta z` are the coordinates themselves, :math:`C` and
        :math:`z_t` are the conditional mean there, and the Curie depth is
        :math:`\\bar z_t + \\Delta z`. The first two are exact at a node; the
        rest carry a Gaussian of fixed width on top.

        The mesh is refined by interpolation first. It is a quadrature rule over
        a continuous posterior, not a discrete distribution, and treating its
        nodes as atoms puts one spike per node where a density belongs. The
        refinement costs no evaluation of the forward model, because both the
        density and the conditional mean are smooth in :math:`(\\beta, \\Delta z)`.
        """
        floor = self._bound_arrays()[0][_ZT]
        fine_beta = np.linspace(
            posterior.beta[0], posterior.beta[-1], _MARGINAL_REFINE
        )
        fine_dz = np.linspace(posterior.dz[0], posterior.dz[-1], _MARGINAL_REFINE)
        weight = _bilinear(
            posterior.beta, posterior.dz, posterior.density, fine_beta, fine_dz
        )

        if target == "beta":
            return np.broadcast_to(fine_beta[:, None], weight.shape), weight, 0.0
        if target == "dz":
            return np.broadcast_to(fine_dz[None, :], weight.shape), weight, 0.0

        index = 0 if target == "C" else 1
        scale = float(np.sqrt(posterior.cov[index, index]))
        centre = posterior.mean[..., index]
        if target == _CPD:
            centre = centre + posterior.dz[None, :]
        values = _bilinear(
            posterior.beta, posterior.dz, centre, fine_beta, fine_dz
        )

        if target in ("zt", _CPD):
            # A node the bound has truncated has its conditional mean below the
            # bound, and a Gaussian truncated well below its mean piles up at
            # the boundary -- so that is where its mass goes. Exact in the limit
            # and inert wherever this has been measured; `posterior.mass` says
            # when it is not.
            limit = floor if target == "zt" else floor + fine_dz[None, :]
            values = np.maximum(values, limit)

        return values, weight, scale

    def _mesh_marginal(self, posterior, target):
        """
        The 1-D posterior of one target: values, density and CDF.

        Built from the **exact** weighted pushforward -- sort the target's value
        over the refined mesh, accumulate the weights, and interpolate that
        cumulative onto a uniform grid. No binning, and so none of a histogram's
        noise: a binned density needed a 1536-node refinement before its
        interval stopped moving, and reported spurious multimodality below that,
        which is the one diagnostic here that has to be trustworthy.

        The conditional Gaussian is applied afterwards, and only where it is
        resolvable. On a 1000 km window `z_t`'s conditional width is 0.007 km
        against a 2 km node spacing, so for the Curie depth it is far below the
        grid and smearing by it would invent structure rather than represent it.

        Returns `(values, density, cdf)`.
        """
        values, weight, scale = self._mesh_samples(posterior, target)

        weight = np.clip(np.asarray(weight, dtype=float).ravel(), 0.0, None)
        values = np.asarray(values, dtype=float).ravel()
        good = np.isfinite(values) & np.isfinite(weight) & (weight > 0.0)
        if not good.any():
            blank = np.zeros(_INTERVAL_POINTS)
            return blank, blank, blank
        values, weight = values[good], weight[good]
        weight = weight / weight.sum()

        order = np.argsort(values)
        sorted_values = values[order]
        # midpoint of each step, so the empirical CDF is centred rather than
        # biased half a step high
        cumulative = np.cumsum(weight[order])
        cumulative = cumulative - 0.5 * weight[order]

        mean = float(np.dot(weight, values))
        spread = float(np.sqrt(np.dot(weight, (values - mean) ** 2) + scale ** 2))
        low = min(sorted_values[0] - 4.0 * scale, mean - _MESH_SPAN * spread)
        high = max(sorted_values[-1] + 4.0 * scale, mean + _MESH_SPAN * spread)
        if not (high > low):
            low, high = mean - 1.0, mean + 1.0

        grid = np.linspace(low, high, _INTERVAL_POINTS)
        cdf = np.interp(grid, sorted_values, cumulative, left=0.0, right=1.0)

        spacing = grid[1] - grid[0]
        if scale > 0.5 * spacing:
            half = int(np.ceil(4.0 * scale / spacing))
            offsets = np.arange(-half, half + 1) * spacing
            kernel = np.exp(-0.5 * (offsets / scale) ** 2)
            kernel /= kernel.sum()
            padded = np.concatenate([
                np.zeros(half), cdf, np.ones(half)
            ])
            cdf = np.convolve(padded, kernel, mode="same")[half:half + grid.size]

        cdf = np.clip(cdf, 0.0, 1.0)
        cdf = np.maximum.accumulate(cdf)
        density = np.gradient(cdf, grid)
        return grid, np.clip(density, 0.0, None), cdf

    @staticmethod
    def _mesh_interval(values, cdf, level):
        """
        Equal-tailed credible interval, by inverting the CDF.

        Equal-tailed rather than highest-density, and the reason is numerical
        rather than philosophical: a highest-density region is read off the
        density, which here is a derivative of an interpolated cumulative and
        is noisy enough that the endpoints move with the refinement. Inverting
        the CDF is stable at any refinement -- the endpoints move by 0.05 km
        between a 192-node and a 768-node refinement, against 1.5 km for the
        density-based version.

        On a Gaussian the two constructions coincide. On the skewed
        :math:`\\Delta z` posterior the equal-tailed interval sits higher,
        because it is centred on the median where a deviance level set is
        centred on the mode. Which of those covers the truth at its nominal
        rate is a question for measurement, not for argument.
        """
        if cdf.size == 0 or cdf[-1] <= 0.0:
            return np.nan, np.nan

        tail = 0.5 * (1.0 - float(level))
        return (float(np.interp(tail, cdf, values)),
                float(np.interp(1.0 - tail, cdf, values)))

    def _temperature(self, x_hat, args, prior=None, calibrate=True):
        """
        The factor the likelihood is too sharp by, measured once at the mode.

        `optimise` reports :math:`(J^T R^{-1} J)^{-1}`, which allows for
        correlation between neighbouring spectral bins. `min_func` does not, so
        an interval read off the likelihood -- a profile deviance, a posterior
        density, a Metropolis chain -- is narrower than the covariance the same
        fit reports, by a factor this returns. Dividing the **spectral** part
        of the misfit by it puts the correction back where the likelihood is
        read, without changing anything that is minimised.

        Args:
            x_hat : array shape (4,)
                the fitted parameters -- the mode, not a scan node
            args : tuple
                `(kh, Phi, sigma_Phi)`
            prior : dict, optional
                priors to use in place of `self.prior`
            calibrate : bool (default=True)
                `False` returns 1.0, which reproduces a pre-v2 interval exactly

        Returns:
            t2 : float
                the **variance** inflation. An interval widens by
                :math:`\\sqrt{t_2}`.

        Notes:
            **Measured once, at the mode, and then held fixed.**
            `pycurious.grid._banded_correlation` estimates the correlation from
            the residuals, and it also absorbs smooth model mismatch -- which
            is right for a covariance at the solution, where `_gls_covariance`
            already trusts it, and wrong anywhere it could move. Re-estimating
            it per scan node or per mesh node would make the objective a
            function of its own residuals, which a fit can lower by making them
            look correlated. One number, computed here, passed down.

            **The prior rows are excluded, on both sides.** They are not
            spectral, they are not correlated, and `_gls_covariance` already
            refuses to let them into its solve. Tempering them would widen a
            pin the caller set deliberately: at :math:`\\sigma_{z_t} = 0.05` km
            that is a 27% loosening of a constraint that carries the depth
            scale. Measured on the same synthetic, `t2` taken from the
            spectral block alone agrees between a free fit and a production-
            pinned one to 0.6-3.9%, where taking it over the whole Jacobian
            disagrees by up to 14%.

            **One scalar is defensible because the four agree.** The
            per-parameter inflations sit within 0.6-3.1% of each other across
            free and pinned fits, so the correction is a reduction in effective
            degrees of freedom rather than a reshaping of the covariance. Where
            they stop agreeing, `_TEMPER_SPREAD_LIMIT` says so rather than
            letting one number stand for four that disagree.
        """
        if not calibrate:
            return 1.0

        kh, Phi, sigma_Phi = args
        r = self.residuals(np.asarray(x_hat, dtype=float), *args, prior=prior)
        J = self._jacobian(np.asarray(x_hat, dtype=float), r, args)

        t2, spread = _correlation_inflation(J, r, np.size(kh))

        if t2 is None or not np.isfinite(t2) or t2 <= 0.0:
            warnings.warn(
                "the correlation between neighbouring spectral bins could not "
                "be measured, so the likelihood is left uncorrected and this "
                "interval is narrower than the covariance the same fit "
                "reports. Usually a singular fit or too few usable bins.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 1.0

        if spread is not None and spread > _TEMPER_SPREAD_LIMIT:
            warnings.warn(
                "the correlation correction differs by {:.0%} between "
                "parameters, so one number does not describe all four and "
                "this interval carries their geometric mean. A correction "
                "this direction-dependent is not a loss of degrees of freedom "
                "and usually means the model is not following the "
                "data.".format(spread),
                RuntimeWarning,
                stacklevel=3,
            )

        return float(t2)

    def _profiled_misfit(self, target, value, x_hat, args, temper=1.0):
        """
        Smallest misfit attainable with `target` held at `value`.

        For a parameter of the forward model that means fixing it and
        minimising over the other three. The Curie depth is the same thing one
        step along: :math:`\\Delta z` becomes the constrained coordinate
        through :math:`\\Delta z = \\mathrm{CPD} - z_t`, leaving
        :math:`\\beta, z_t, C` free -- so it is profiled directly rather than
        propagated from `dz`, whose uncertainty is not symmetric.

        `temper` divides the spectral part and leaves the prior rows alone, as
        `_solve_linear` does; `1.0` is `min_func` at the constrained optimum.
        The minimisation itself is untempered, which is deliberate: a positive
        scalar on part of the objective would move the constrained argmin
        wherever a prior is active, and the point of tempering is to widen an
        interval rather than to relocate it.
        """
        curie = target == _CPD
        fixed = _PARAMETERS.index("dz" if curie else target)
        free = [j for j in range(len(_PARAMETERS)) if j != fixed]

        res = self._fit(
            np.asarray(x_hat)[free],
            args,
            free=free,
            fixed=(fixed, value),
            curie=curie,
        )
        # `cost` is half the sum of squares, which is what `min_func` returns
        if temper == 1.0:
            return res.cost

        prior_part = self._prior_misfit(res.x_full)
        return (res.cost - prior_part) / float(temper) + prior_part

    def profile(
        self,
        window,
        xc,
        yc,
        target,
        level=0.95,
        npoints=21,
        bracket=None,
        calibrate=True,
        beta=None,
        zt=None,
        dz=None,
        C=None,
        taper=np.hanning,
        process_subgrid=None,
        dof_factor=None,
        spectrum=None,
        **kwargs
    ):
        """
        Confidence interval for one parameter, or for the Curie depth, without
        assuming the posterior is symmetric.

        Each point of the scan holds `target` fixed and re-optimises everything
        else, tracing the deviance :math:`2(F - F_{min})`. The interval is
        where that crosses :math:`\\chi^2_1` at the requested level, which is
        the usual likelihood-ratio construction.

        This matters most for :math:`\\Delta z` and hence the Curie depth. Both
        have a long upper tail (Mather & Fullea, 2019), so the symmetric
        :math:`\\pm \\sigma` that `optimise` reports understates how far the
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
            calibrate : bool (default=True)
                correct the likelihood for correlation between neighbouring
                spectral bins before reading the interval off it -- see
                `_temperature`, and the note below. `False` reproduces the
                pre-v2 interval exactly, which is what an archive written
                before this existed has to be compared against.
            beta, zt, dz, C : float, optional
                starting values for the underlying fit; each is derived from
                the spectrum where it is left None
            taper : function (default=np.hanning)
                taper function, or None for no taper
            process_subgrid : function, optional
                applied to the subgrid before the spectrum is computed
            dof_factor : float, optional
                see `pycurious.grid.CurieGrid.window_spectrum`
            spectrum : tuple (k, Phi, sigma_Phi), optional
                a spectrum already in hand, typically `last_spectrum` from the
                `optimise` at this same centroid -- see `optimise`
            kwargs : keyword arguments
                passed to `radial_spectrum`

        Returns:
            values : 1D array shape (npoints,)
                where the target was held
            deviance : 1D array shape (npoints,)
                :math:`2(F - F_{min})` at each of those
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
            and so is a little narrower for a skewed one. `posterior` is the
            integrated reading of the same surface.

            **The likelihood is corrected before it is read.** Neighbouring
            spectral bins are correlated, `optimise`'s covariance allows for it
            and `min_func` does not, so a deviance read off `min_func` is too
            sharp by the factor `_temperature` measures -- about 1.6 in
            variance under `numpy.hanning` at a 1000 km window, so about 1.27
            on a width. Only the spectral part is corrected; a prior keeps the
            width the caller gave it. This *widens* every interval relative to
            pre-v2 and does not move the fitted value it is centred on, since
            nothing about what is minimised changed. `calibrate=False` turns it
            off.

            Like the covariance from `optimise`, the interval describes the
            scatter of the spectrum at a fixed window, centroid and model. It
            does not cover the systematic error from choosing those: on
            `tests/test_mag_data.txt` the interval for :math:`\\Delta z` is
            about 3.3 km wide, where sweeping the window size and centroid
            moves :math:`\\Delta z` over 5.5 km.
        """
        if target not in _PARAMETERS + (_CPD,):
            raise ValueError(
                "target must be one of {}, not {!r}".format(
                    _PARAMETERS + (_CPD,), target
                )
            )
        k, Phi, sigma_Phi = self._resolve_spectrum(
            spectrum, window, xc, yc, taper, process_subgrid, dof_factor, **kwargs
        )
        args = (k, Phi, sigma_Phi)

        x0 = self._initial_guess(args, beta, zt, dz, C)
        res = self._fit(x0, args)
        x_hat = res.x

        # Measured once, at the mode, and held for the whole scan. Estimating
        # it again at each node would make the objective a function of its own
        # residuals -- see `_temperature`.
        temper = self._temperature(x_hat, args, calibrate=calibrate)
        prior_part = self._prior_misfit(x_hat)
        F_min = (res.cost - prior_part) / temper + prior_part

        # every constrained fit is cached, so the root finding below reuses the
        # scan nodes it lands on rather than paying for them twice
        cache = {}

        def constrained(value):
            value = float(value)
            if value not in cache:
                cache[value] = self._profiled_misfit(
                    target, value, x_hat, args, temper=temper
                )
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

    def _profile_on_mesh(self, window, xc, yc, target, level, nodes, bracket,
                         posterior, beta, zt, dz, C, taper, process_subgrid,
                         dof_factor, spectrum, **kwargs):
        """
        `profile` with `method="mesh"`: the interval read off the posterior.

        Returns the same 4-tuple the scan does, so a caller unpacking
        `values, deviance, lower, upper` does not care which produced it.
        `deviance` here is :math:`-2 \\log(p/p_{max})` of the marginal, which is
        the same scale as the profile deviance and plots against the same
        threshold.
        """
        if bracket is not None and np.ndim(bracket) != 2:
            raise ValueError(
                "with method='mesh' the bracket is a box in the two parameters "
                "the posterior lives in, ((beta_min, beta_max), (dz_min, "
                "dz_max)), not a range in {!r}. A range in one target cannot "
                "say where the other should be evaluated.".format(target)
            )

        if posterior is None:
            posterior = self.posterior(
                window, xc, yc, nodes=nodes, bracket=bracket, beta=beta, zt=zt,
                dz=dz, C=C, taper=taper, process_subgrid=process_subgrid,
                dof_factor=dof_factor, spectrum=spectrum, **kwargs
            )

        values, density, cdf = self._mesh_marginal(posterior, target)
        lower, upper = self._mesh_interval(values, cdf, level)

        with np.errstate(divide="ignore", invalid="ignore"):
            peak = np.max(density) if np.any(density > 0.0) else np.nan
            deviance = -2.0 * np.log(np.where(density > 0.0, density, np.nan)
                                     / peak)
        deviance = np.where(np.isfinite(deviance), deviance, np.inf)

        # An endpoint is only a number if the mesh contained the tail it sits
        # in. `_MESH_EDGE_TOL` is the accounting tolerance; what matters for an
        # interval is whether more mass escaped than the interval is allowed to
        # miss, because then the endpoint is set by the edge of the box rather
        # than by the data.
        # the mass beyond *one* endpoint, which is what that endpoint is
        # placed to cut off -- not the mass outside the whole interval
        escaped = 0.5 * (1.0 - float(level))
        low_edge, high_edge = (
            (posterior.edge_mass[0], posterior.edge_mass[1])
            if target == "beta"
            else (posterior.edge_mass[2], posterior.edge_mass[3])
        )
        for side, mass, name in ((0, low_edge, "lower"), (1, high_edge, "upper")):
            if mass > escaped:
                warnings.warn(
                    "the {} posterior still carries {:.2e} of its mass at the "
                    "edge of the mesh, more than the {:.3g} a {:.4g} interval "
                    "may miss, so the {} side of the interval is unbounded. "
                    "The data do not constrain it; widen `bracket` to "
                    "confirm.".format(target, mass, escaped, level, name),
                    RuntimeWarning,
                    stacklevel=3,
                )
                if side == 0:
                    lower = -np.inf
                else:
                    upper = np.inf

        # A thickness whose rolloff sits below the longest wavelength measured
        # is not something this window can distinguish from any larger one --
        # `dz` survives there only as `2 ln dz` inside the constant, degenerate
        # with `C`. Where the interval runs past that, its upper end is set by
        # the numerical ceiling on the forward model rather than by the data,
        # and reporting the resulting number is worse than reporting nothing:
        # at a 250 km window it comes back as (190, 943) km.
        #
        # The scan reaches the same verdict a different way, by never crossing
        # its threshold. Reporting it the same way keeps the two comparable.
        limit = (_MESH_IDENTIFIABLE / posterior.kmin
                 if np.isfinite(posterior.kmin) and posterior.kmin > 0.0
                 else np.inf)
        if target in ("dz", _CPD) and np.isfinite(upper) and upper > limit:
            warnings.warn(
                "the {} interval reaches {:.4g} km, past the {:.4g} km whose "
                "rolloff is still inside this window's band, so its upper side "
                "is unbounded -- beyond that the spectrum cannot tell one "
                "thickness from another. The data do not constrain it.".format(
                    target, upper, limit
                ),
                RuntimeWarning,
                stacklevel=3,
            )
            upper = np.inf
            # and the lower end is then a quantile of a posterior most of whose
            # mass sits in the region the data cannot speak about. Take it from
            # the part they can, which is what makes it comparable to the
            # scan's finite lower endpoint rather than an artefact of the
            # ceiling.
            inside = values <= limit
            if np.count_nonzero(inside) > 1 and cdf[inside][-1] > 0.0:
                restricted = cdf[inside] / cdf[inside][-1]
                lower = float(np.interp(escaped, restricted, values[inside]))

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
        beta=None,
        zt=None,
        dz=None,
        C=None,
        taper=np.hanning,
        process_subgrid=None,
        dof_factor=None,
        adapt=True,
        seed=None,
        return_diagnostics=False,
        calibrate=True,
        spectrum=None,
        **kwargs
    ):
        """
        MCMC algorithm using a Metropolis-Hastings sampler.

        Evaluates a Markov chain for starting values of
        :math:`\\beta, z_t, \\Delta z, C` and returns the ensemble of model
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
                number of burn-in simulations discarded before the `nsim`
                recorded samples
            x_scale : float(4), optional
                initial width of the proposal in each parameter
                (default=`[1,1,1,1]` for `[beta, zt, dz, C]`). With
                `adapt=True` this is only a starting point.
            beta : float, optional
                fractal parameter (starting value for the search), derived
                from the spectrum if None
            zt : float, optional
                top of magnetic layer (starting value), derived if None
            dz : float, optional
                thickness of magnetic layer (starting value), derived if None
            C : float, optional
                field constant (starting value), derived if None
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
            calibrate : bool (default=True)
                sample the likelihood corrected for correlation between
                neighbouring spectral bins, which is the posterior `profile`
                and `posterior` also read -- see `_temperature`. The chain is
                then wider than a pre-v2 one by about 1.27 under
                `numpy.hanning`. `False` reproduces the old chain.
            spectrum : tuple (k, Phi, sigma_Phi), optional
                a spectrum already in hand, typically `last_spectrum` from
                the `optimise` at this same centroid -- see `optimise`

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
            :math:`e^{-F}` directly underflows to zero for any real spectrum --
            :math:`F` runs to hundreds -- at which point every proposal is
            rejected and the chain returns a handful of distinct states
            dressed up as a posterior.

            The chain starts at the mode, found with the same minimiser
            `optimise` uses and from the same starting values. That costs a
            fraction of a second and removes the job the burn-in is worst at.

            There is no tempering. It was tried -- annealing the burn-in
            after Sambridge (2013), doi:10.1093/gji/ggt342 -- and made every
            case worse. What motivated it was that large parts of the posterior
            evaluated to zero, and that was the :math:`e^{-F}` underflow rather
            than a property of the problem, so log-space acceptance removes the
            reason for it. It also fights the proposal tuning below: a high
            temperature makes almost everything acceptable, driving the scale
            up, and the scale then collapses as the temperature falls, freezing
            the chain wherever the hot phase left it.

            The shape of the proposal matters more than any of the above. The
            four parameters are strongly correlated -- :math:`\\beta` with
            :math:`z_t` at about -0.92, :math:`z_t` with :math:`C` at about
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

        k, Phi, sigma_Phi = self._resolve_spectrum(
            spectrum, window, xc, yc, taper, process_subgrid, dof_factor, **kwargs
        )

        lower, upper = self._bound_arrays()

        # Start the chain at the mode rather than at the caller's guess. The
        # optimiser finds it in a fraction of the time a random walk takes to
        # wander there, and a chain started away from it spends its whole
        # burn-in travelling instead of tuning. Measured on a synthetic, the
        # posterior mean from a default start sits at a misfit of 121 against
        # the mode's 50; started here it lands on 50.1.
        args = (k, Phi, sigma_Phi)
        start = self._fit(self._initial_guess(args, beta, zt, dz, C), args)

        # The same correction `profile` and `posterior` read their intervals
        # through, so all three describe one posterior rather than three. The
        # chain is the one place it could have been left out and not noticed --
        # a sampler has no interval to compare against -- which is exactly why
        # it is here. Measured once at the mode, as everywhere else.
        temper = self._temperature(start.x, args, calibrate=calibrate)

        def log_posterior(x):
            if np.any(x < lower) or np.any(x > upper):
                return -np.inf
            if temper == 1.0:
                return -self.min_func(x, k, Phi, sigma_Phi)
            prior_part = self._prior_misfit(x)
            spectral = self.min_func(x, k, Phi, sigma_Phi) - prior_part
            return -(spectral / temper + prior_part)

        def step(x, F, scale, chol):
            """One Metropolis move."""
            proposal = x + scale * chol.dot(rng.normal(size=ndim))

            F1 = log_posterior(proposal)
            accepted = np.isfinite(F1) and np.log(rng.random()) < F1 - F

            if accepted:
                return proposal, F1, True
            return x, F, False

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
        :math:`2.38/\\sqrt{d}` is the usual optimal scaling for a Gaussian
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
        beta=None,
        zt=None,
        dz=None,
        C=None,
        taper=np.hanning,
        process_subgrid=None,
        dof_factor=None,
        seed=None,
        spectrum=None,
        **kwargs
    ):
        """
        Sample the uncertainty of :math:`\\beta, z_t, \\Delta z, C` by
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
            beta : float, optional
                starting fractal parameter, derived per realisation if None
            zt : float, optional
                starting top of magnetic layer, derived if None
            dz : float, optional
                starting thickness of magnetic layer, derived if None
            C : float, optional
                starting field constant, derived if None
            taper : function (default=`numpy.hanning`)
                taper function, set to None for no taper function
            process_subgrid : function, optional
                a custom function to process the subgrid
            dof_factor : float, optional
                override the effective-degrees-of-freedom deflation, see
                `pycurious.grid.CurieGrid.window_spectrum`
            seed : int, optional
                seed for reproducibility
            spectrum : tuple (k, Phi, sigma_Phi), optional
                a spectrum already in hand, typically `last_spectrum` from
                the `optimise` at this same centroid -- see `optimise`

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

            Every realisation starts from one fit to the *unresampled*
            spectrum, not from its own. That warm start is worth about a third
            of the total, and it was measured rather than assumed: re-deriving
            a start per realisation costs 1.4x the forward-model evaluations
            and gives a spread of 13.28 against the warm start's 13.28.
            Resampling :math:`\\Phi` within :math:`\\sigma_\\Phi` does not move
            a realisation across a basin boundary, so there is nothing for the
            extra work to find.

            What that warm start does depend on is the fit it starts from. This
            routine used to strand its whole ensemble in one basin and report a
            confident spread about it -- on a synthetic whose true
            :math:`\\Delta z` is 45 km the ensemble came back with a median of
            10.3 -- and the cause was the old constant :math:`\\Delta z = 10`,
            not the sharing. Deriving the start (`_initial_guess`) is what
            fixed it. Supplying all four puts the anchoring fit wherever the
            caller says, which is the way to ask the narrower question of how
            the spectrum's scatter alone moves the answer from a chosen point.

            The result is not calibrated either way: measured against the
            spread over independent realisations of the *field*, this reports
            about 0.5 to 0.6 of it. That is the bin-independence assumption
            above, and no choice of starting point touches it.
        """
        rng = np.random.default_rng(seed)

        samples = np.empty((nsim, 4))

        use_keys = [key for key, pdf in self.prior_pdf.items() if pdf is not None]

        k, Phi, sigma_Phi = self._resolve_spectrum(
            spectrum, window, xc, yc, taper, process_subgrid, dof_factor, **kwargs
        )
        args = (k, Phi, sigma_Phi)

        # Every resampled spectrum lands in the same basin, so start each
        # simulation from the unresampled solution rather than from the start
        # itself. One extra fit up front, and about a third off the total for
        # any useful `nsim`.
        x0 = self._fit(self._initial_guess(args, beta, zt, dz, C), args).x

        for sim in range(0, nsim):
            # a fresh set of prior centres, drawn without disturbing the ones
            # stored on the instance
            prior = dict(self.prior)
            for key in use_keys:
                loc = self.prior_pdf[key].rvs(random_state=rng)
                prior[key] = (loc, self.prior[key][1])

            rPhi = rng.normal(Phi, sigma_Phi)
            samples[sim] = self._fit(x0, (k, rPhi, sigma_Phi), prior=prior).x

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
            :math:`Z_b = z_t + \\Delta z`, so the uncertainties combine as
            :math:`\\sqrt{\\sigma_{z_t}^2 + \\sigma_{\\Delta z}^2}`. The two are
            correlated -- about 0.6 -- but :math:`\\sigma_{\\Delta z}` exceeds
            :math:`\\sigma_{z_t}` by four orders of magnitude, so including the
            covariance changes the answer by around 1%. `optimise` will hand
            over the full matrix with `return_cov=True` for anyone who wants it.

            The far larger effect is that this is symmetric and the Curie depth
            is not: :math:`\\Delta z` has a long upper tail, so `CPD_stdev`
            understates how deep the base can plausibly lie. Use `profile` with
            `target="CPD"` for an interval that does not assume symmetry.

            Matches the signature and return of
            `pycurious.optimise_tanaka.CurieOptimiseTanaka.calculate_CPD`, so
            code written against one behaves the same against the other.
        """
        CPD = zt + dz
        CPD_stdev = np.sqrt(np.asarray(sigma_zt) ** 2 + np.asarray(sigma_dz) ** 2)
        return (CPD, CPD_stdev)
