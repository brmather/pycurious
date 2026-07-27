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
    interval. These are the numbers, not a tolerance band: the swap was
    adopted on the evidence that it reproduces them.

    A better inner optimiser is not automatically safe here. It finds lower
    constrained minima, which re-anchors the deviance, and on a multimodal
    window that moves the reported interval -- see the test above. dz was
    checked across seeds for exactly that reason.
    """
    grid, xc, yc = _grid(seed=seed)
    grid.reset_priors()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, lower, upper = grid.profile(200e3, xc, yc, "dz")
    np.testing.assert_allclose([lower, upper], interval, rtol=2e-3)


def test_analytic_jacobian_columns_match_the_finite_difference(bouligand):
    """
    `_fit` supplies the zt and C columns of the Jacobian in closed form and
    differences only beta and dz. If the forward model changes and these are
    not re-derived, every fit silently descends a slightly wrong gradient --
    it still converges, just to a worse place, and nothing complains.
    """
    from pycurious.optimise_bouligand import _ANALYTIC_COLUMNS

    grid, xc, yc = bouligand
    grid.reset_priors()
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, power=2.0)
    x = np.array([3.0, 1.0, 20.0, 5.0])
    args = (k, Phi, sigma)
    J = grid._jacobian(x, grid.residuals(x, *args), args)

    for index in (1, 3):
        np.testing.assert_allclose(
            _ANALYTIC_COLUMNS[index](k, sigma), J[: k.size, index], rtol=1e-6
        )


def test_fit_cost_is_min_func(bouligand):
    """
    `_profiled_misfit` returns `res.cost` where it used to return the value of
    `min_func`, and the deviance -- so every profile interval -- is differences
    of those. The two are the same quantity only for as long as `residuals` and
    `min_func` agree.
    """
    grid, xc, yc = bouligand
    grid.reset_priors()
    k, Phi, sigma = grid.window_spectrum(WINDOW, xc, yc, power=2.0)
    args = (k, Phi, sigma)
    res = grid._fit(np.array([3.0, 1.0, 10.0, 5.0]), args)
    assert res.cost == pytest.approx(grid.min_func(res.x, *args))


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
    assert calls["n"] > 2
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
    go through `min_func`, so the offset moves nothing but the acceptance
    ratio and the two chains should now agree to the bit. They are compared
    loosely anyway: the point of the test is that the chain is not frozen by a
    constant, and pinning it exactly would make the test fail for reasons that
    have nothing to do with that. Reintroducing exp() would not miss this
    tolerance narrowly; it would freeze the chain outright.
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
