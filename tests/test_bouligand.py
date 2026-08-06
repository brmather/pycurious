"""
Behaviour of CurieOptimiseBouligand, and guards against defects fixed in v2.

Each test names the defect it protects, because several of them are the kind
that leave the code producing plausible numbers rather than failing.
"""

import copy
import warnings
from multiprocessing import cpu_count

import numpy as np
import pytest

import pycurious
from pycurious.optimise_bouligand import _JACOBIAN_STEP

from conftest import synthetic_grid

TRUTH = dict(beta=3.0, zt=1.0, dz=20.0)
WINDOW = 1000e3


def _grid(n=512, seed=1):
    return synthetic_grid(
        pycurious.CurieOptimiseBouligand, n=n, seed=seed, **TRUTH
    )[:3]


@pytest.fixture(scope="module")
def bouligand():
    return _grid()


def _uncorrelated_sigma(grid, x, k, Phi, sigma):
    """What inv(J^T J) would report, i.e. treating the bins as independent."""
    args = (k, Phi, sigma)
    r = grid.residuals(x, *args)
    J = grid._jacobian(x, r, args)
    return np.sqrt(np.diag(np.linalg.inv(J.T.dot(J))))


def test_min_func_is_weighted(bouligand):
    """
    min_func must divide by sigma_Phi.

    Before v2 it passed a literal 1.0, so every bin carried equal weight
    however well determined it was. Doubling every uncertainty must quarter
    the misfit; if the weighting is dropped again the ratio goes to 1.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, taper=np.hanning, power=2.0)
    x = np.array([3.0, 1.0, 20.0, 15.0])

    ratio = grid.min_func(x, k, Phi, 2.0 * sigma) / grid.min_func(x, k, Phi, sigma)
    assert ratio == pytest.approx(0.25)


def test_reduced_chi_squared_is_about_one(bouligand):
    """
    The weights must be the uncertainty of the binned *mean*.

    This is the strongest single guard in the suite. Weighting by the raw
    within-annulus scatter gives 0.01, dropping the per-bin degrees-of-freedom
    deflation gives 1.9, and not weighting at all is not on this scale.

    The taper is pinned because the guard only separates those cases with one:
    untapered, every plausible deflation lands inside the window.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    x = grid.optimise(WINDOW, xc, yc, taper=np.hanning)[:4]
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, taper=np.hanning, power=2.0)

    chi2_red = 2.0 * grid.min_func(x, k, Phi, sigma) / (k.size - len(x))
    assert 0.7 < chi2_red < 1.6, "chi2_red = {:.3f}".format(chi2_red)


def test_optimise_returns_uncertainties(bouligand):
    """optimise reports a sigma per parameter, not just the parameters."""
    grid, xc, yc = bouligand
    grid.reset_priors()
    out = grid.optimise(WINDOW, xc, yc, taper=np.hanning)

    assert len(out) == 8
    beta, zt, dz, C, s_beta, s_zt, s_dz, s_C = out
    assert np.abs(beta - TRUTH["beta"]) < 0.2
    assert np.abs(zt - TRUTH["zt"]) < 0.25
    for sigma in (s_beta, s_zt, s_dz, s_C):
        assert np.isfinite(sigma) and sigma > 0.0


def test_covariance_is_opt_in(bouligand):
    """
    The covariance must not ride in the default return tuple.

    parallel._collect dispatches on the dimensionality of the result, so an
    8-tuple of floats plus a 4x4 array raises "inhomogeneous shape" the moment
    optimise_routine is used.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    out = grid.optimise(WINDOW, xc, yc, taper=np.hanning, return_cov=True)

    assert len(out) == 9
    cov = out[8]
    assert cov.shape == (4, 4)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-8)
    # the sigmas reported alongside it are its diagonal
    np.testing.assert_allclose(np.array(out[4:8]), np.sqrt(np.diag(cov)), rtol=1e-10)


def test_correlation_between_bins_inflates_sigma(bouligand):
    """
    Neighbouring radial bins are correlated, and ignoring it understates every
    uncertainty by about 35% under a hanning taper.

    The covariance therefore has to be generalised least squares. Compare it
    against what the naive inv(J^T J) would report.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    x = np.array(grid.optimise(WINDOW, xc, yc, taper=np.hanning)[:4])

    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, taper=np.hanning, power=2.0)
    gls = np.sqrt(np.diag(grid._covariance(x, k, Phi, sigma)))
    iid = _uncorrelated_sigma(grid, x, k, Phi, sigma)

    assert np.all(gls > iid), "GLS must widen the interval, not narrow it"
    ratio = (gls / iid)[[0, 1, 3]]     # beta, zt, C
    assert np.all(ratio > 1.15) and np.all(ratio < 1.6), ratio

    # With no taper there is nothing to correlate the bins, so the two agree.
    # Refit first: evaluating the hanning solution against untapered data
    # leaves residuals dominated by smooth model mismatch, which the estimator
    # would correctly read as correlated.
    x = np.array(grid.optimise(WINDOW, xc, yc, taper=None)[:4])
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, taper=None, power=2.0)
    gls = np.sqrt(np.diag(grid._covariance(x, k, Phi, sigma)))
    iid = _uncorrelated_sigma(grid, x, k, Phi, sigma)
    np.testing.assert_allclose((gls / iid)[[0, 1, 3]], 1.0, rtol=0.15)


def test_profile_is_asymmetric_for_dz(bouligand):
    """
    dz has a long upper tail (Mather & Fullea, 2019), so a symmetric sigma is
    the wrong shape for it. The profile interval must show that.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    values, deviance, lower, upper = grid.profile(
        WINDOW, xc, yc, "dz", taper=np.hanning
    )

    centre = values[np.argmin(deviance)]
    assert lower < centre < upper
    assert lower <= TRUTH["dz"] <= upper, "truth outside [{:.2f}, {:.2f}]".format(
        lower, upper
    )
    assert (upper - centre) > 1.2 * (centre - lower), "interval is not skewed"


def test_profile_of_curie_depth(bouligand):
    """CPD is profiled directly, not propagated from a symmetric sigma_dz."""
    grid, xc, yc = bouligand
    grid.reset_priors()
    values, deviance, lower, upper = grid.profile(
        WINDOW, xc, yc, "CPD", taper=np.hanning
    )
    truth = TRUTH["zt"] + TRUTH["dz"]
    assert lower <= truth <= upper, "truth outside [{:.2f}, {:.2f}]".format(lower, upper)


def test_profile_result_does_not_depend_on_the_bracket(bouligand):
    """
    A coarse or badly placed bracket used to step over the minimum entirely,
    collapsing the interval onto a single node without saying so. The fitted
    value is now always carried as a node.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    _, _, lo_default, hi_default = grid.profile(WINDOW, xc, yc, "dz", taper=np.hanning)
    _, _, lo_coarse, hi_coarse = grid.profile(
        WINDOW, xc, yc, "dz", bracket=(1.0, 900.0), npoints=15, taper=np.hanning
    )
    assert lo_coarse == pytest.approx(lo_default, rel=0.02)
    assert hi_coarse == pytest.approx(hi_default, rel=0.02)


def test_profile_survives_the_overflow_region(bouligand):
    """
    bouligand2009 overflows cosh past |k|dz ~ 710. A scan that reaches there
    must return a finite deviance rather than raising.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values, deviance, _, _ = grid.profile(
            WINDOW, xc, yc, "dz", bracket=(1.0, 2000.0), npoints=11, taper=np.hanning
        )
    assert np.isfinite(deviance).all()


def test_profile_unbounded_returns_inf():
    """
    A window too small to constrain dz has no upper bound. That must come back
    as inf with a warning, not as the edge of the scan dressed up as an answer.
    """
    grid, xc, yc = _grid(n=256)
    grid.reset_priors()
    with pytest.warns(RuntimeWarning, match="unbounded"):
        _, _, lower, upper = grid.profile(60e3, xc, yc, "dz", taper=np.hanning)
    assert np.isinf(upper)
    assert np.isfinite(lower)


def test_profile_rejects_unknown_target(bouligand):
    grid, xc, yc = bouligand
    with pytest.raises(ValueError, match="target must be one of"):
        grid.profile(WINDOW, xc, yc, "curie_depth")


@pytest.mark.parametrize(
    "seed, interval",
    [
        (1, (11.12, np.inf)),
        (2, (1.695, 48.17)),
        (3, (5.094, 32.47)),
        (4, (24.06, np.inf)),
    ],
)
def test_profile_dz_interval_is_unchanged_by_the_optimiser(seed, interval):
    """
    Pinned against the values L-BFGS-B produced before `_fit` moved to
    `least_squares`, so a change of optimiser cannot quietly move a published
    interval. A better inner optimiser is not automatically safe here: it
    finds lower constrained minima, which re-anchors the deviance, and that
    can move the reported interval.

    The tolerance is 5%, which is loose because the quantity is. An interval
    endpoint is where `brentq` crosses the threshold on a deviance curve that
    is nearly flat there -- that flatness is the whole reason `dz` needs a
    profile rather than a sigma -- so a last-ulp difference in `kv` or in the
    FFT behind the synthetic moves it far more than it moves the fit. The same
    four intervals come out up to 1.3% different on macOS from Linux with
    identical code, which is what set the bound. Pinning tighter tests the
    platform's libm, not this package.

    Loose as it is, it still bites: it is what catches a sign-flipped or
    dropped Jacobian column, and a `_profiled_misfit` off by a constant
    factor. Those move an endpoint by tens of percent, not tenths.
    """
    grid, xc, yc = _grid(seed=seed)
    grid.reset_priors()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, lower, upper = grid.profile(200e3, xc, yc, "dz")
    np.testing.assert_allclose([lower, upper], interval, rtol=5e-2)


def test_analytic_jacobian_columns_match_the_finite_difference(bouligand):
    """
    `_fit` supplies the zt and C columns of the Jacobian in closed form and
    differences only beta and dz. If the forward model changes and these are
    not re-derived, every fit silently descends a slightly wrong gradient --
    it still converges, just to a worse place, and nothing complains.
    """
    from pycurious.optimise_bouligand import _ANALYTIC_COLUMNS, _PARAMETERS

    grid, xc, yc = bouligand
    grid.reset_priors()
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, power=2.0)
    x = np.array([3.0, 1.0, 20.0, 5.0])
    args = (k, Phi, sigma)
    J = grid._jacobian(x, grid.residuals(x, *args), args)

    for name, column in _ANALYTIC_COLUMNS.items():
        index = _PARAMETERS.index(name)
        np.testing.assert_allclose(
            column(k, sigma), J[: k.size, index], rtol=1e-6, err_msg=name
        )


def test_profiled_misfit_is_on_the_same_scale_as_min_func(bouligand):
    """
    `_profiled_misfit` returns `res.cost` where it used to return `min_func`,
    and every profile interval is differences of those against `F_min`, which
    still comes from `min_func`. A constant factor between the two would put
    every interval out by that factor while leaving the curve the right shape.

    Holding a parameter at its own fitted value and re-optimising the rest must
    recover the unconstrained misfit, which pins the scale. Comparing
    `res.cost` to `min_func(res.x)` does not: both are half the sum of squares
    of the same vector, so that identity holds however wrong the surrounding
    code is.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, power=2.0)
    args = (k, Phi, sigma)
    x_hat = grid._fit(np.array([3.0, 1.0, 10.0, 5.0]), args).x
    F_min = grid.min_func(x_hat, *args)

    for target, value in (("dz", x_hat[2]), ("beta", x_hat[0]),
                          ("CPD", x_hat[1] + x_hat[2])):
        held = grid._profiled_misfit(target, value, x_hat, args)
        assert held == pytest.approx(F_min, rel=1e-6), target


def _captured_jacobian(grid, monkeypatch, *fit_args, **fit_kwargs):
    """
    Hand back the `(fun, jac, y)` that `_fit` passes to scipy.

    The spy returns nothing rather than delegating: the caller wants the
    callables, not the fit, and running one would cost an optimisation whose
    result is discarded.
    """
    from pycurious import optimise_bouligand as mod

    grabbed = {}

    def spy(fun, y0, jac=None, **kw):
        grabbed.update(fun=fun, jac=jac, y0=np.asarray(y0, dtype=float))

    monkeypatch.setattr(mod, "least_squares", spy)
    grid._fit(*fit_args, **fit_kwargs)
    return grabbed["fun"], grabbed["jac"], grabbed["y0"]


@pytest.mark.parametrize(
    "free, fixed, curie",
    [
        (None, None, False),
        ([0, 1, 3], (2, 25.0), False),  # as profile("dz") does
        ([0, 1, 3], (2, 30.0), True),  # as profile("CPD") does
    ],
    ids=["unconstrained", "dz-held", "curie"],
)
def test_fit_jacobian_matches_finite_differences(bouligand, monkeypatch, free,
                                                 fixed, curie):
    """
    The Jacobian `_fit` hands scipy must be the derivative of the residual it
    hands scipy alongside it, in every configuration -- including the Curie
    one, where `dz = CPD - zt` couples the `zt` and `dz` columns by a chain
    rule that no other test reaches.

    Checking the fitted answer instead does not work: trust-region reflective
    converges from a wrong Jacobian too, just by a different route. Dropping
    the chain-rule term entirely, or flipping its sign, moves the CPD interval
    not at all and costs a handful of extra evaluations -- so only the
    derivative itself can be tested.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    grid.add_prior(beta=(3.0, 0.4), C=(5.0, 1.0))
    try:
        k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, power=2.0)
        args = (k, Phi, sigma)
        x_hat = grid._fit(np.array([3.0, 1.0, 10.0, 5.0]), args).x
        y0 = x_hat if free is None else np.asarray(x_hat)[free]

        fun, jac, y = _captured_jacobian(grid, monkeypatch, y0, args, free=free,
                                         fixed=fixed, curie=curie)

        analytic = jac(y)
        numeric = np.empty_like(analytic)
        for i in range(y.size):
            h = _JACOBIAN_STEP * max(abs(y[i]), 1.0)
            yp, ym = y.copy(), y.copy()
            yp[i] += h
            ym[i] -= h
            numeric[:, i] = (fun(yp) - fun(ym)) / (2.0 * h)

        scale = np.maximum(np.abs(numeric).max(axis=0), 1.0)
        np.testing.assert_allclose(
            analytic / scale, numeric / scale, atol=1e-6
        )
    finally:
        grid.reset_priors()


def test_fit_rejects_an_inverted_bound(bouligand):
    """
    A collapsed bound (lb == ub) is widened so trust-region reflective has an
    interior to work in. An inverted one (lb > ub) is a typo -- `bounds` is
    documented as reassignable -- and widening it would silently rewrite it
    into whichever number came first, pinning the parameter somewhere the
    caller never asked for.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, power=2.0)
    original = list(grid.bounds)
    try:
        grid.bounds[2] = (30.0, 10.0)
        with pytest.raises(ValueError, match="bound"):
            grid._fit(np.array([3.0, 1.0, 10.0, 5.0]), (k, Phi, sigma))
    finally:
        grid.bounds = original


def test_fit_honours_a_held_coordinate(bouligand):
    """
    A constrained fit must actually hold what it was told to, including the
    Curie-depth case where the held coordinate is dz = CPD - zt and so moves
    with a free parameter.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, power=2.0)
    args = (k, Phi, sigma)
    x_hat = grid._fit(np.array([3.0, 1.0, 10.0, 5.0]), args).x

    free = [0, 1, 3]

    # dz pinned at 25: the reported cost must be the misfit of the model that
    # actually has dz = 25, which is what makes the deviance meaningful
    held = grid._fit(x_hat[free], args, free=free, fixed=(2, 25.0))
    beta, zt, C = held.x
    assert held.cost == pytest.approx(grid.min_func([beta, zt, 25.0, C], *args))
    assert held.cost >= grid._fit(x_hat, args).cost

    # Curie depth pinned at 30: dz is not fixed, zt + dz is
    curie = grid._fit(x_hat[free], args, free=free, fixed=(2, 30.0), curie=True)
    beta, zt, C = curie.x
    assert curie.cost == pytest.approx(
        grid.min_func([beta, zt, 30.0 - zt, C], *args)
    )


def _analytic_spectrum(dz, seed, window_km=4000.0, nbins=158):
    """
    A spectrum that really is a `dz` layer, without transforming anything.

    The seeder is a function of `(k, Phi, sigma)` alone, so testing it against a
    field means paying for an 811x811 transform to reach the window sizes where
    a thick layer is resolvable at all. Building the spectrum from the forward
    model instead puts the same question in milliseconds, and pins the answer to
    the model rather than to the details of one realisation of a grid.

    `sigma` follows the real shape -- the per-cell scatter of a log periodogram,
    `pi/sqrt(6)`, over the square root of a cell count that grows linearly with
    `k` (`pycurious.grid.CurieGrid.window_spectrum`).
    """
    k = (2.0 * np.pi / window_km) * np.arange(1, nbins + 1)
    sigma = (np.pi / np.sqrt(6.0)) / np.sqrt(
        np.arange(1, nbins + 1) * np.pi / 1.65
    )
    truth = pycurious.bouligand2009(k, 3.0, 1.0, dz, 5.0)
    noise = np.random.default_rng(seed).normal(0.0, sigma)
    return k, truth + noise, sigma


def _spectrum_fitter(dx_km=5.0):
    """
    A fitter with no window behind it, for fitting a spectrum directly.

    The cell size is not decoration: `_max_thickness` scales the `dz` ceiling
    with `self.dx`, so a dummy spanning one metre would bound `dz` near zero.
    This is the same trick `~/Global_CPD` uses to refit cached spectra.
    """
    return pycurious.CurieOptimiseBouligand(
        np.zeros((2, 2)), 0.0, dx_km * 1e3, 0.0, dx_km * 1e3
    )


def test_solve_linear_is_the_exact_conditional_minimum(bouligand):
    """
    `C` and `zt` are not guessed, they are solved, so the claim is exactness.

    `bouligand2009` is `h(beta, dz) + C - 2 k zt`, which makes the best `(C, zt)`
    at any `(beta, dz)` a two-column weighted linear solve. Checked against a
    derivative-free minimisation of the real `min_func` over the same two
    coordinates -- if the decomposition were even slightly wrong, a search would
    beat a formula.
    """
    from scipy.optimize import minimize

    grid, xc, yc = bouligand
    grid.reset_priors()
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, power=2.0)

    for beta, dz in ((3.0, 20.0), (2.0, 5.0), (4.0, 60.0)):
        C, zt, cost = grid._solve_linear(k, Phi, sigma, beta, dz)
        searched = minimize(
            lambda p: grid.min_func([beta, p[1], dz, p[0]], k, Phi, sigma),
            [C, zt],
            method="Nelder-Mead",
            options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000),
        )
        assert cost <= searched.fun * (1.0 + 1e-9), (
            "a search beat the closed form at beta={}, dz={}".format(beta, dz)
        )
        assert C == pytest.approx(searched.x[0], rel=1e-6)
        assert zt == pytest.approx(searched.x[1], rel=1e-6)


def test_solve_linear_cost_is_min_func(bouligand):
    """
    The scan ranks basins, so its cost must be the one the fit then minimises.

    `_solve_linear` builds the misfit from pieces it already holds rather than
    calling `min_func`, which halves what a ladder node costs -- and puts two
    expressions for one quantity in the module. This is what stops them
    drifting. Equality is to floating point, not to the bit: the two sum the
    same terms in a different order.
    """
    grid, xc, yc = bouligand
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, power=2.0)

    for priors in ({}, {"zt": (1.0, 0.05)},
                   {"beta": (3.0, 0.2), "zt": (1.0, 0.05),
                    "dz": (20.0, 5.0), "C": (5.0, 1.0)}):
        grid.reset_priors()
        grid.add_prior(**priors)
        for held in ({}, {"zt": 2.0}, {"C": 4.0}, {"zt": 2.0, "C": 4.0}):
            C, zt, cost = grid._solve_linear(k, Phi, sigma, 3.0, 20.0, **held)
            assert cost == pytest.approx(
                grid.min_func([3.0, zt, 20.0, C], k, Phi, sigma), rel=1e-12
            ), "cost disagrees with min_func at priors={}, held={}".format(
                priors, held
            )

    grid.reset_priors()


def test_solve_linear_respects_the_zt_bound(bouligand):
    """
    An unconstrained solve can pay for a thick layer with a negative `zt`.

    That is a cheaper misfit than any fit is allowed to reach, so leaving it in
    would let the scan prefer a basin the optimiser cannot enter. The clamp
    holds `zt` at its bound and re-solves `C`, which is the exact constrained
    answer while only one bound is active.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, power=2.0)

    # a thickness far past what this window resolves drives zt negative
    thick = 400.0
    C, zt, cost = grid._solve_linear(k, Phi, sigma, 3.0, thick)
    assert zt >= 0.0, "zt came back at {}".format(zt)
    assert cost == pytest.approx(
        grid.min_func([3.0, zt, thick, C], k, Phi, sigma), rel=1e-12
    )

    # and C is still the best it can be with zt held there
    for nudge in (-0.5, 0.5):
        assert grid.min_func([3.0, zt, thick, C + nudge], k, Phi, sigma) >= cost


def test_solve_linear_folds_in_a_prior(bouligand):
    """
    Pinning zt is what a production run does, so the solve must see the prior.

    Without it the scan would rank thicknesses by a misfit the fit does not use.
    """
    from scipy.optimize import minimize

    grid, xc, yc = bouligand
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, power=2.0)

    grid.reset_priors()
    recovered = [grid._solve_linear(k, Phi, sigma, 3.0, 60.0)[1]]

    # Tightening the prior must walk zt monotonically from the data's answer to
    # the prior's centre, which is the conjugate compromise a Gaussian prior on
    # a linear coefficient has to produce. Nothing weaker distinguishes the
    # prior being folded in from it being ignored: at 249 bins the data term is
    # 1.8e5 against the prior's 1/sigma_p^2, so a 0.01 km prior moves zt to 2.3,
    # not to 5.
    for width in (1.0, 0.1, 0.01, 1e-3, 1e-4):
        grid.reset_priors()
        grid.add_prior(zt=(5.0, width))
        zt = grid._solve_linear(k, Phi, sigma, 3.0, 60.0)[1]
        assert zt > recovered[-1], "narrowing the prior did not pull zt further"
        recovered.append(zt)

    assert recovered[0] < 1.1, recovered[0]
    assert abs(recovered[-1] - 5.0) < 0.01, recovered[-1]

    # and it is still the exact minimum of the objective that carries the prior
    grid.reset_priors()
    grid.add_prior(zt=(5.0, 0.01))
    C, zt, cost = grid._solve_linear(k, Phi, sigma, 3.0, 60.0)
    searched = minimize(
        lambda p: grid.min_func([3.0, p[1], 60.0, p[0]], k, Phi, sigma),
        [C, zt],
        method="Nelder-Mead",
        options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000),
    )
    assert cost <= searched.fun * (1.0 + 1e-9)

    grid.reset_priors()


def test_an_explicit_start_bypasses_the_seeder(bouligand):
    """
    Supplying all four must give exactly the fit those four would have given.

    This is what makes the change to the defaults a change to the defaults and
    nothing else: anything that used to pass the old constants explicitly, or
    that reproduces an archived result, is untouched.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    spectrum = grid.window_spectrum(WINDOW, xc, yc, taper=np.hanning, power=2.0)
    old = np.array([3.0, 1.0, 10.0, 5.0])

    # no spectrum is consulted at all when there is nothing left to derive
    assert np.array_equal(grid._initial_guess(spectrum, *old), old)
    assert np.array_equal(grid._initial_guess((None, None, None), *old), old)

    direct = grid._fit(old, spectrum).x
    routed = grid.optimise(
        WINDOW, xc, yc, beta=3.0, zt=1.0, dz=10.0, C=5.0,
        taper=np.hanning, spectrum=spectrum,
    )[:4]
    assert np.array_equal(np.asarray(routed), direct)


def test_a_held_start_is_honoured_and_the_rest_derived(bouligand):
    """
    The four are independent, so `dz=30` alone means "start there, and put the
    others where they best sit at that thickness" -- which the four-constant
    form could not express.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    spectrum = grid.window_spectrum(WINDOW, xc, yc, power=2.0)

    x0 = grid._initial_guess(spectrum, dz=30.0)
    assert x0[2] == 30.0
    C, zt, _ = grid._solve_linear(*spectrum, x0[0], 30.0)
    assert x0[1] == zt and x0[3] == C

    # a held beta is used as given, and reaches the linear solve
    assert grid._initial_guess(spectrum, beta=2.0, dz=30.0)[0] == 2.0
    assert grid._initial_guess(spectrum, zt=4.0)[1] == 4.0

    # and a derived beta is the one the high band measured, not the fallback
    assert x0[0] == grid._asymptote_estimates(*spectrum[:2])[0]


@pytest.mark.parametrize(
    "spectrum",
    [
        (np.array([]), np.array([]), np.array([])),
        (np.array([0.05]), np.array([3.0]), np.array([1.0])),
        (np.array([np.nan, np.nan]), np.array([3.0, 2.0]), np.array([1.0, 1.0])),
    ],
    ids=["empty", "one bin", "all nan"],
)
def test_a_spectrum_with_nothing_in_it_falls_back(bouligand, spectrum):
    """
    A window carrying a NaN comes back with an empty spectrum, and a band cut
    can empty one too. There is nothing to derive from, so the start must fall
    back to the old constants rather than reduce over an empty axis.

    Which constants matters. `~/Global_CPD` detects an unusable window by the
    fit returning its own starting guess -- `optimise` is documented there as
    "silently returns x0 on a NaN window" -- so the fallback has to stay
    recognisable, not merely finite.
    """
    grid, _, _ = bouligand
    grid.reset_priors()

    x0 = grid._initial_guess(spectrum)
    assert np.array_equal(x0, [3.0, 1.0, 10.0, 5.0]), x0

    # anything supplied still wins over the fallback
    assert np.array_equal(
        grid._initial_guess(spectrum, zt=4.2, dz=25.0), [3.0, 4.2, 25.0, 5.0]
    )


def test_last_x0_records_where_the_fit_started(bouligand):
    """
    A derived start cannot be reconstructed from the arguments the caller
    passed, so the routines leave it on the instance the way they leave
    `last_spectrum`. `~/Global_CPD` needs it: its `STARTING_GUESS` status bit
    fires when a fit comes back exactly equal to its own start, which is the
    signature of a window the optimiser could not move in.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    spectrum = grid.window_spectrum(WINDOW, xc, yc, taper=np.hanning, power=2.0)

    fresh = synthetic_grid.__wrapped__(
        pycurious.CurieOptimiseBouligand, n=128, seed=11, **TRUTH
    )[0]
    assert fresh.last_x0 is None

    grid.optimise(WINDOW, xc, yc, taper=np.hanning, spectrum=spectrum)
    derived = np.array(grid.last_x0)
    assert derived.shape == (4,) and np.isfinite(derived).all()
    np.testing.assert_allclose(derived, grid._initial_guess(spectrum))

    # an explicit start is recorded as given, so the bit means the same thing
    # on both paths
    grid.optimise(WINDOW, xc, yc, beta=3.0, zt=1.0, dz=10.0, C=5.0,
                  taper=np.hanning, spectrum=spectrum)
    assert np.array_equal(grid.last_x0, [3.0, 1.0, 10.0, 5.0])

    # and it survives the pickling `parallelise_routine` does
    import pickle
    assert np.array_equal(pickle.loads(pickle.dumps(grid)).last_x0,
                          grid.last_x0)


def test_derived_start_finds_the_basin_the_constant_missed():
    """
    The defect this change exists to fix.

    On a thick layer the misfit is bimodal in `dz` and the old `dz = 10`
    constant sat in the wrong basin -- at a misfit about 25% higher, so not the
    flat-likelihood tie where both answers are equally good. Measured over
    clean synthetics at a 4000 km window it cost roughly one realisation in five
    (`notes/spectrum-binning-weighting-multitaper.md`), and `optimise` reported
    it without complaint.

    Seed 1 of the spectrum below is one of those: the constant stops at
    dz = 10.2 against a truth of 45.
    """
    grid = _spectrum_fitter()
    spectrum = _analytic_spectrum(dz=45.0, seed=1)

    constant = grid._fit(np.array([3.0, 1.0, 10.0, 5.0]), spectrum)
    derived = grid._fit(grid._initial_guess(spectrum), spectrum)

    assert constant.x[2] < 15.0, "the constant no longer misses: {}".format(
        constant.x[2]
    )
    assert derived.cost < 0.85 * constant.cost, (
        "derived cost {:.3f} is not clearly below the constant's {:.3f}".format(
            derived.cost, constant.cost
        )
    )
    assert abs(derived.x[2] - 45.0) < 10.0, "dz {:.2f}".format(derived.x[2])


def test_derived_start_costs_less_than_it_saves():
    """
    The ladder is only worth running if a node is cheap next to a fit.

    Counted in evaluations of the forward model, which is the measurement a
    loaded machine cannot distort. The ladder is a fixed `_LADDER_NODES + 2`;
    what it has to earn back is the iterations a fit does not then spend.
    """
    from pycurious import optimise_bouligand as ob

    grid = _spectrum_fitter()
    spectrum = _analytic_spectrum(dz=45.0, seed=1)

    calls = {"n": 0}
    real = ob.bouligand2009

    def counted(*args):
        calls["n"] += 1
        return real(*args)

    ob.bouligand2009 = counted
    try:
        x0 = grid._initial_guess(spectrum)
        seeding = calls["n"]
        grid._fit(x0, spectrum)
        derived_total = calls["n"]

        calls["n"] = 0
        grid._fit(np.array([3.0, 1.0, 10.0, 5.0]), spectrum)
        constant_total = calls["n"]
    finally:
        ob.bouligand2009 = real

    assert seeding == ob._LADDER_NODES + 2, seeding
    assert derived_total < constant_total, (
        "seeding cost {} + {} against the constant's {}".format(
            seeding, derived_total - seeding, constant_total
        )
    )


def test_sensitivity_ensemble_is_not_stranded_in_the_wrong_basin():
    """
    `sensitivity` used to strand its whole ensemble in the basin the `dz = 10`
    constant reached, and report a confident spread about it.

    Measured on the spectrum below, whose truth is 45 km: started at the
    constant the ensemble has median 10.3, started at the derived seed it has
    median 46.9. What fixes it is the seed, not anything about the ensemble --
    re-deriving a start for every realisation gives sd 13.28 against the warm
    start's 13.28, indistinguishable, because resampling `Phi` within
    `sigma_Phi` does not move a realisation across a basin boundary. So the
    warm start stays, and this guards the thing that actually mattered.
    """
    grid = _spectrum_fitter()
    spectrum = _analytic_spectrum(dz=45.0, seed=1)

    derived = np.asarray(
        grid.sensitivity(0.0, 0.0, 0.0, 24, seed=3, spectrum=spectrum)[2]
    )
    stranded = np.asarray(
        grid.sensitivity(0.0, 0.0, 0.0, 24, seed=3, beta=3.0, zt=1.0, dz=10.0,
                         C=5.0, spectrum=spectrum)[2]
    )

    assert np.median(stranded) < 20.0, (
        "the constant no longer strands the ensemble: {:.2f}".format(
            np.median(stranded))
    )
    assert abs(np.median(derived) - 45.0) < 10.0, (
        "median {:.2f} over {}".format(np.median(derived), np.round(derived, 1))
    )
    assert np.all(derived > 0.0)


def test_sensitivity_does_not_disturb_priors(bouligand):
    """
    sensitivity used to redraw each prior centre by mutating self.prior in
    place and restoring it afterwards, which left the instance corrupted if
    anything raised in between.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    grid.add_prior(beta=(3.0, 0.1), zt=(1.0, 0.2))
    before = copy.deepcopy(grid.prior)

    grid.sensitivity(300e3, xc, yc, 4, taper=np.hanning, seed=1)
    assert grid.prior == before

    # and still intact when a simulation blows up part way through. The count
    # is of fits rather than of objective evaluations: the first is the
    # unresampled solution the simulations start from, so raising on the third
    # lands inside the loop, which is where the prior is copied and where a
    # leak would show.
    calls = {"n": 0}
    original = grid._fit

    def exploding(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("boom")
        return original(*args, **kwargs)

    grid._fit = exploding
    try:
        with pytest.raises(RuntimeError, match="boom"):
            grid.sensitivity(300e3, xc, yc, 4, taper=np.hanning, seed=1)
    finally:
        del grid._fit
    assert grid.prior == before

    grid.reset_priors()


def test_sensitivity_is_reproducible(bouligand):
    """
    Seeded so a parallel sensitivity map cannot inherit one RNG state across
    workers, which under fork would have drawn the same sequence at every
    centroid and painted coherent artefacts across the map.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    kwargs = dict(taper=np.hanning)
    a = grid.sensitivity(300e3, xc, yc, 6, seed=7, **kwargs)[2]
    b = grid.sensitivity(300e3, xc, yc, 6, seed=7, **kwargs)[2]
    c = grid.sensitivity(300e3, xc, yc, 6, seed=8, **kwargs)[2]
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_prior_values_are_immutable(bouligand):
    """Stored as tuples, so a caller cannot perturb a prior in place."""
    grid, _, _ = bouligand
    grid.reset_priors()
    grid.add_prior(beta=(3.0, 0.1))
    with pytest.raises(TypeError):
        grid.prior["beta"][0] = 99.0
    grid.reset_priors()


def test_min_func_accepts_an_explicit_prior(bouligand):
    """
    Passing a prior through rather than reaching for self.prior is what lets
    sensitivity resample prior centres without touching the instance.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, taper=np.hanning, power=2.0)
    x = np.array([3.0, 1.0, 20.0, 15.0])

    flat = grid.min_func(x, k, Phi, sigma)
    tight = grid.min_func(x, k, Phi, sigma, {"beta": (1.0, 0.01), "zt": None,
                                             "dz": None, "C": None})
    assert tight > flat
    assert grid.prior["beta"] is None, "self.prior must be untouched"


def test_warns_when_a_parameter_hits_a_bound(bouligand):
    """
    The covariance describes the curvature of an interior minimum, so on an
    active bound the uncertainty it reports is meaningless.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    original = list(grid.bounds)
    try:
        grid.bounds = [(0.0, None), (0.0, 0.0), (0.0, None), (None, None)]
        with pytest.warns(RuntimeWarning, match="bound"):
            grid.optimise(WINDOW, xc, yc, taper=np.hanning)
    finally:
        grid.bounds = original


def test_taper_none_rejects_unknown_keywords(bouligand):
    """
    With a taper present an unrecognised keyword raises from inside the taper.
    With taper=None it used to be swallowed, which is how a seed= silently
    failed to reach anything on the notebooks that pass taper=None.
    """
    grid, xc, yc = bouligand
    with pytest.raises(TypeError, match="unexpected keyword"):
        grid.window_spectrum(300e3, xc, yc, taper=None, bogus=7)


@pytest.mark.slow
def test_reported_sigma_matches_the_spread_over_realisations():
    """
    The calibration that matters, and the only one that can catch a wrong
    covariance.

    sensitivity resamples each bin independently and metropolis_hastings uses
    the same diagonal weighting, so both share the assumption the covariance
    makes. They agree with it to within 10% whether or not it is right. Only an
    ensemble over independent realisations of the field is an outside check.

    Measured over 200 seeds: 0.997, 0.989 and 0.991 for beta, zt and C. dz sits
    near 1.44 because its likelihood is skewed, which no symmetric sigma can
    fix -- that is what profile() is for.
    """
    fits, reported = [], []
    for seed in range(60):
        grid, xc, yc = _grid(seed=2000 + seed)
        out = grid.optimise(WINDOW, xc, yc, taper=np.hanning)
        fits.append(out[:4])
        reported.append(out[4:])

    spread = np.array(fits).std(axis=0, ddof=1)
    mean_sigma = np.array(reported).mean(axis=0)
    ratio = spread / mean_sigma

    for i, name in enumerate(("beta", "zt", "C")):
        j = i if i < 2 else 3
        assert 0.8 < ratio[j] < 1.2, "{} ratio {:.3f}".format(name, ratio[j])

    # dz is understated, and knowingly so
    assert ratio[2] > 1.15, "dz ratio {:.3f}".format(ratio[2])


def test_calculate_CPD_matches_the_tanaka_sibling(bouligand):
    """
    Both classes must return (CPD, CPD_stdev) from the same shaped call.

    The Bouligand version used to be a bare `return zt+dz` with no uncertainty,
    so code written against one sibling misbehaved silently against the other.
    """
    grid, _, _ = bouligand
    tanaka = pycurious.CurieOptimiseTanaka(np.zeros((9, 9)), 0.0, 8e3, 0.0, 8e3)

    CPD, sigma = grid.calculate_CPD(1.0, 10.0, 0.3, 0.4)
    assert CPD == pytest.approx(11.0)
    assert sigma == pytest.approx(np.hypot(0.3, 0.4))

    # same arity and return shape as Tanaka, which parameterises by z0 instead
    assert len(grid.calculate_CPD(1.0, 10.0)) == len(tanaka.calculate_CPD(1.0, 6.0))

    # and vectorises over a map of centroids
    CPD, sigma = grid.calculate_CPD(
        np.array([1.0, 2.0]), np.array([10.0, 11.0]), 0.0, 0.0
    )
    np.testing.assert_allclose(CPD, [11.0, 13.0])
    np.testing.assert_allclose(sigma, [0.0, 0.0])


def test_metropolis_hastings_acceptance_is_in_a_usable_band(bouligand):
    """
    The chain must actually move.

    Comparing exp(-F) directly underflows to zero for any real spectrum, so
    every proposal was rejected: the old sampler returned 9 distinct states in
    2000 draws while reporting an acceptance rate of 0.004. Acceptance is
    decided in log space now, and the proposal is drawn along the fit
    covariance rather than a diagonal, which is what lets it move along the
    ridge beta and zt lie on.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    posterior, info = grid.metropolis_hastings(
        WINDOW, xc, yc, 2000, 500, taper=np.hanning, seed=1, return_diagnostics=True
    )
    chain = np.array(posterior)

    assert 0.15 < info["acceptance"] < 0.6, info["acceptance"]
    assert len(np.unique(chain[0])) > 0.1 * chain.shape[1]


def test_metropolis_hastings_is_invariant_to_a_constant_misfit(bouligand):
    """
    A log-space acceptance ratio sees only differences, so shifting the misfit
    by a constant leaves the chain where it was. The old exp(-F) form with its
    1e-99 clamp did not have that property -- which is precisely why it stalled
    once F grew to a few hundred: exp(-F) underflowed to zero and every
    proposal was rejected.

    The chain starts from a `least_squares` fit of `residuals`, which does not
    go through `min_func`, so the offset now moves nothing but the acceptance
    ratio. The comparison is left loose regardless -- what it is here to catch
    is a chain frozen by a constant, and reintroducing exp() would not miss
    this tolerance narrowly but freeze the chain outright.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    kwargs = dict(taper=np.hanning, seed=3, adapt=False)

    before = np.array(grid.metropolis_hastings(WINDOW, xc, yc, 200, 100, **kwargs))

    original = grid.min_func
    grid.min_func = lambda *a, **kw: original(*a, **kw) + 1000.0
    try:
        after = np.array(grid.metropolis_hastings(WINDOW, xc, yc, 200, 100, **kwargs))
    finally:
        del grid.min_func

    np.testing.assert_allclose(before, after, rtol=1e-4)
    # and neither chain is frozen, which is what the old form produced
    assert len(np.unique(before[0])) > 20


def test_metropolis_hastings_agrees_with_the_other_estimators(bouligand):
    """
    beta, zt and C are near-Gaussian, so the posterior width should match what
    the covariance and the resampling ensemble report.

    dz is deliberately excluded: its posterior is skewed, so its marginal is
    wider than a curvature-based sigma by construction.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    sigma = np.array(grid.optimise(WINDOW, xc, yc, taper=np.hanning)[4:8])
    chain = np.array(
        grid.metropolis_hastings(WINDOW, xc, yc, 4000, 1000, taper=np.hanning, seed=1)
    )

    for i, name in ((0, "beta"), (1, "zt"), (3, "C")):
        ratio = chain[i].std() / sigma[i]
        assert 0.5 < ratio < 1.6, "{} ratio {:.3f}".format(name, ratio)


def test_metropolis_hastings_respects_bounds_and_is_reproducible(bouligand):
    """
    The optimiser has always honoured self.bounds; the chain used to ignore
    them and could wander to a negative thickness.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    kwargs = dict(taper=np.hanning)
    a = np.array(grid.metropolis_hastings(WINDOW, xc, yc, 300, 100, seed=5, **kwargs))
    b = np.array(grid.metropolis_hastings(WINDOW, xc, yc, 300, 100, seed=5, **kwargs))
    c = np.array(grid.metropolis_hastings(WINDOW, xc, yc, 300, 100, seed=6, **kwargs))

    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)
    assert (a[0] >= 0.0).all() and (a[1] >= 0.0).all() and (a[2] >= 0.0).all()


def test_only_stochastic_routines_are_seeded(bouligand):
    """
    `parallelise_routine` decides once who gets a seed, rather than every
    routine growing a parameter to absorb one.

    Before, `optimise` had to accept a `seed` it ignored purely so the parallel
    routine could pass one uniformly -- and that was done on the Bouligand
    sibling only, so the identical call raised from inside `np.hanning` on the
    Tanaka one.
    """
    grid, xc, yc = _grid(n=128)
    grid.max_processors = 1
    window = 200e3
    xs, ys = np.array([xc]), np.array([yc])

    assert getattr(grid.sensitivity, "wants_seed", False)
    assert getattr(grid.metropolis_hastings, "wants_seed", False)
    assert not getattr(grid.optimise, "wants_seed", False)

    # a deterministic routine says so rather than failing inside the taper
    with pytest.warns(RuntimeWarning, match="deterministic"):
        grid.optimise_routine(window, xs, ys, taper=np.hanning, seed=7)

    # and a stochastic one is actually seeded, per centroid
    a = grid.parallelise_routine(window, xs, ys, grid.sensitivity, 3,
                                 taper=np.hanning, seed=11)
    b = grid.parallelise_routine(window, xs, ys, grid.sensitivity, 3,
                                 taper=np.hanning, seed=11)
    np.testing.assert_allclose(np.array(a, dtype=float), np.array(b, dtype=float))


def test_metropolis_hastings_default_return_shape_is_unchanged(bouligand):
    """
    pycurious.parallel dispatches on the dimensionality of a routine's result,
    so the diagnostics have to be opt-in or every parallel MCMC call breaks.

    Also checks a burn-in too short to condition a 4x4 covariance, which
    test_routines.py exercises at burnin=10.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()

    plain = grid.metropolis_hastings(WINDOW, xc, yc, 60, 10, taper=np.hanning, seed=1)
    assert isinstance(plain, list) and len(plain) == 4
    assert all(np.asarray(a).shape == (60,) for a in plain)

    _, info = grid.metropolis_hastings(
        WINDOW, xc, yc, 60, 10, taper=np.hanning, seed=1, return_diagnostics=True
    )
    assert set(info) == {"acceptance", "burnin_acceptance", "x_scale"}


def test_thickness_is_bounded_below_the_overflow(bouligand):
    """
    dz is bounded where the forward model stops evaluating, not where physics
    stops being plausible.

    bouligand2009 overflows around |k|dz = 710, and |k| reaches the Nyquist
    wavenumber whatever the window, so the ceiling follows from the grid
    spacing: 446 km at 2 km spacing. That is far past any Curie depth on Earth,
    which is deliberate -- a bound placed near the physical range would clip
    the upper tail of a skewed posterior and pile probability against the wall
    instead of reporting the shape. Reaching this one means the window cannot
    constrain the base at all.
    """
    from pycurious.optimise_bouligand import _COSH_OVERFLOW

    grid, xc, yc = bouligand
    grid.reset_priors()

    ceiling = grid.bounds[2][1]
    assert ceiling == pytest.approx(_COSH_OVERFLOW * grid.dx * 1e-3 / np.pi)
    # far beyond anything physical, so it never binds on usable data
    assert ceiling > 300.0
    # the parameters the spectrum does pin down are left free
    assert grid.bounds[0][1] is None and grid.bounds[1][1] is None

    # the model still evaluates at the bound, which is the whole point
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, taper=np.hanning, power=2.0)
    assert np.isfinite(grid.min_func([3.0, 1.0, ceiling, 15.0], k, Phi, sigma))

    # and a chain on an under-constrained window stays inside it
    chain = np.array(
        grid.metropolis_hastings(300e3, xc, yc, 400, 200, taper=np.hanning, seed=1)
    )
    assert chain[2].max() <= ceiling


def test_max_processors_is_honoured():
    """
    The constructor keyword must reach CurieParallel.

    It previously did not: the assignment sat after the return in
    _max_thickness, where it was unreachable, so the argument was silently
    ignored and every routine used cpu_count() regardless. Nothing caught it
    because parallelise_routine still works -- just not serially when asked.
    """
    data = np.zeros((9, 9))
    extent = (0.0, 8e3, 0.0, 8e3)

    assert pycurious.CurieOptimiseBouligand(*(data,) + extent,
                                            max_processors=1).max_processors == 1
    assert pycurious.CurieOptimiseBouligand(*(data,) + extent,
                                            max_processors=3).max_processors == 3

    # and the two optimisers agree, as they did not before
    assert (
        pycurious.CurieOptimiseBouligand(*(data,) + extent, max_processors=2).max_processors
        == pycurious.CurieOptimiseTanaka(*(data,) + extent, max_processors=2).max_processors
    )

    # the default is still every core
    assert pycurious.CurieOptimiseBouligand(*(data,) + extent).max_processors == cpu_count()
