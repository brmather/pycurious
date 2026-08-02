import pytest
import pycurious
import numpy as np
from scipy.optimize import minimize

from conftest import load_magnetic_anomaly


def test_optimisation_smoke(load_magnetic_anomaly):
    """
    The optimiser lands in a physically sensible region on the legacy fixture.

    This deliberately asserts no accuracy. `tests/test_mag_data.txt` is 305 km
    across for a 10 km layer, so it has too few low-wavenumber bins to pin dz
    down: sweeping the window size and moving the centroid by one window width
    moves dz over 6.4-11.9 km against a truth of 10.0. A tolerance tight enough
    to be meaningful would break on any legitimate change, and a tolerance
    loose enough to pass says nothing. Accuracy is asserted against generated
    synthetics in tests/test_recovery.py instead.

    Measured here for reference, hanning taper, whole grid as one window:

        beta 2.7688 +/- 0.0985   (truth 3.0)
        zt   0.3813 +/- 0.0304   (truth 0.305)
        dz   9.0703 +/- 1.9921   (truth 10.0)
        C  -17.6561 +/- 0.0551
    """
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]
    max_window = load_magnetic_anomaly["max_window"]

    grid = pycurious.CurieOptimiseBouligand(d, xmin, xmax, ymin, ymax)
    beta, zt, dz, C, s_beta, s_zt, s_dz, s_C = grid.optimise(
        max_window, xc, yc, taper=np.hanning
    )

    for name, value in [("beta", beta), ("zt", zt), ("dz", dz), ("C", C)]:
        assert np.isfinite(value), "{} = {} is not finite".format(name, value)
    for name, value in [("sigma_beta", s_beta), ("sigma_zt", s_zt),
                        ("sigma_dz", s_dz), ("sigma_C", s_C)]:
        assert np.isfinite(value) and value > 0.0, "{} = {}".format(name, value)

    # a magnetic layer with a positive thickness, below the surface, and a
    # fractal parameter in the range reported for continental crust
    assert zt >= 0.0
    assert dz > 0.0
    assert 1.0 < beta < 5.0


def test_priors(load_magnetic_anomaly):
    """A prior pulls the parameter it constrains towards its centre."""
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]
    max_window = load_magnetic_anomaly["max_window"]

    grid = pycurious.CurieOptimiseBouligand(d, xmin, xmax, ymin, ymax)
    beta0 = grid.optimise(max_window, xc, yc)[0]

    grid.add_prior(beta=(1.0, 0.1))
    beta1 = grid.optimise(max_window, xc, yc)[0]

    assert abs(beta1 - 1.0) < abs(beta0 - 1.0), "the prior did not pull beta"
    # quantitative, so the test cannot pass on a negligible shift. Measured
    # 2.7688 -> 2.3295 once the fit is weighted; it was 2.668 -> 1.228 when the
    # fit was unweighted and the data barely competed with the prior.
    assert beta0 - beta1 > 0.3, "beta moved only {:.3f}".format(beta0 - beta1)


def test_valid_numbers(load_magnetic_anomaly):
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]
    max_window = load_magnetic_anomaly["max_window"]

    # create phoney power spectrum
    S = np.array(
        [
            22.16409774,
            19.95258494,
            18.27873722,
            17.10575637,
            16.53959747,
            16.31539575,
            15.69619005,
            15.29953307,
            14.83475976,
            14.54031396,
            14.33361716,
            13.81764026,
            13.5176055,
            13.27386563,
            13.03493328,
            12.88581369,
            12.61998358,
            12.48616749,
            12.11261083,
            12.13079154,
            11.85440661,
            11.79244826,
            11.66823202,
            11.40231744,
            11.32521296,
            11.13634007,
            11.10650999,
            10.94822598,
            10.78032794,
            10.66593304,
            10.55815845,
            10.56805594,
            10.33514462,
            10.22026537,
            10.22945756,
            10.09275259,
            10.11562101,
            9.85061009,
            9.87165772,
            9.85976847,
            9.73954992,
            9.72021054,
            9.52959744,
            9.59582531,
            9.50927273,
            9.44691364,
            9.39293966,
            9.33097387,
            9.33191784,
        ]
    )

    k = np.array(
        [
            0.09237068,
            0.15443902,
            0.21486191,
            0.282031,
            0.33452175,
            0.4020083,
            0.46381582,
            0.5290359,
            0.59611689,
            0.65875047,
            0.71442694,
            0.77929408,
            0.84320843,
            0.90939865,
            0.96863721,
            1.03155349,
            1.09130906,
            1.15699532,
            1.22338685,
            1.28109263,
            1.34606572,
            1.40861721,
            1.46920824,
            1.53421046,
            1.59675833,
            1.66109235,
            1.7233503,
            1.78492941,
            1.84835837,
            1.90933735,
            1.973275,
            2.03650008,
            2.0983359,
            2.16386249,
            2.22416603,
            2.2876692,
            2.34841931,
            2.41225232,
            2.47774604,
            2.5379226,
            2.59980401,
            2.66346692,
            2.72735801,
            2.79146076,
            2.85220715,
            2.91443307,
            2.97643404,
            3.04008782,
            3.10369815,
        ]
    )

    sigma_S = np.ones_like(S)

    grid = pycurious.CurieOptimiseBouligand(d, xmin, xmax, ymin, ymax)

    beta0 = 3.0
    zt0 = 1.0
    dz0 = 40.0
    C0 = 5.0
    x0 = np.array([beta0, zt0, dz0, C0])

    lower_bound = np.zeros_like(x0)
    upper_bound = [None] * len(lower_bound)

    # xi = func([5, 0., 45., 6.], S, k)
    xi = grid.min_func(x0, k, S, sigma_S)

    res = minimize(
        grid.min_func,
        x0,
        args=(k, S, sigma_S),
        method="TNC",
        bounds=list(zip(lower_bound, upper_bound)),
    )
    print("beta={:.2f}, zt={:.2f}, dz={:.2f}, C={:.2f}".format(*res.x))

    parameters = ["beta", "zt", "dz", "C"]
    err_msg = "FAILED! {} = {} is not a finite number"

    for i in range(res.x.size):
        assert np.isfinite(res.x[i]), err_msg.format(parameters[i], res.x[i])


def _shared_spectrum_grid():
    from conftest import synthetic_grid

    return synthetic_grid(pycurious.CurieOptimiseBouligand, n=256, dx=2.0)


@pytest.mark.parametrize("target", ["dz", "CPD"])
def test_supplied_spectrum_reproduces_the_computed_one(target):
    """
    `spectrum=` must be the same fit, not merely a similar one.

    The point of routing it through `optimise`/`profile` rather than giving it
    its own code path is that the two cannot drift apart. Assert to the bit, so
    that they cannot.
    """
    grid, xc, yc, extent = _shared_spectrum_grid()
    window = 200e3

    stock = grid.optimise(window, xc, yc)
    shared = grid.optimise(window, xc, yc, spectrum=grid.last_spectrum)
    assert stock == shared

    a = grid.profile(window, xc, yc, target, npoints=9)
    b = grid.profile(window, xc, yc, target, npoints=9, spectrum=grid.last_spectrum)
    for lhs, rhs in zip(a, b):
        assert np.array_equal(lhs, rhs)


def test_supplied_spectrum_bypasses_the_spectrum_hook():
    """
    A supplied spectrum must not be routed through `_spectrum`.

    Subclasses override `_spectrum` to band limit or to reweight `sigma` (the
    Global_CPD workflow does both). Whatever they did was already applied to
    the array the caller is holding, so doing it again would compound it.
    """
    grid, xc, yc, extent = _shared_spectrum_grid()
    calls = []
    original = grid._spectrum

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    grid._spectrum = counting
    try:
        grid.optimise(200e3, xc, yc)
        assert len(calls) == 1
        grid.optimise(200e3, xc, yc, spectrum=grid.last_spectrum)
        assert len(calls) == 1, "_spectrum was called for a supplied spectrum"
    finally:
        grid._spectrum = original


def test_last_spectrum_tracks_both_paths():
    # a fresh instance, because `synthetic_grid` is cached and a grid that has
    # already been fitted carries the spectrum from whichever test got there
    # first
    data, extent = pycurious.fractal_anomaly(n=128, dx=2.0, beta=3.0, zt=1.0,
                                             dz=20.0, C=5.0, seed=1)
    grid = pycurious.CurieOptimiseBouligand(data, *extent)
    xc = 0.5 * (extent[0] + extent[1])
    yc = 0.5 * (extent[2] + extent[3])
    assert grid.last_spectrum is None

    grid.optimise(200e3, xc, yc)
    computed = grid.last_spectrum
    assert computed is not None and len(computed) == 3

    grid.optimise(200e3, xc, yc, spectrum=computed)
    for lhs, rhs in zip(grid.last_spectrum, computed):
        assert np.array_equal(lhs, rhs)


def test_supplied_spectrum_rejects_mismatched_shapes():
    grid, xc, yc, extent = _shared_spectrum_grid()
    bad = (np.ones(5), np.ones(4), np.ones(5))
    with pytest.raises(ValueError, match="same shape"):
        grid.optimise(200e3, xc, yc, spectrum=bad)


def test_residuals_do_not_swallow_unrelated_warnings():
    """
    `residuals` silences the forward model's floating-point errors, which it
    must -- the fit legitimately probes dz <= 0 -- but it used to do so with a
    blanket `catch_warnings`, which ate every other warning raised in the
    block as well.
    """
    import warnings

    grid, xc, yc, extent = _shared_spectrum_grid()
    k, Phi, sigma = grid.window_spectrum(200e3, xc, yc, power=2.0)

    # dz < 0 is what the Curie profile evaluates below zt, and what raises
    # 'invalid value encountered in power' inside bouligand2009
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = grid.residuals(np.array([3.0, 1.0, -1.0, 5.0]), k, Phi, sigma)
    assert np.all(np.isfinite(r)), "a non-finite residual escaped the fallback"
    assert not [w for w in caught if "invalid value" in str(w.message)]

    # ... while a genuine warning from elsewhere still gets through
    real = pycurious.optimise_bouligand.bouligand2009

    def noisy(*args, **kwargs):
        warnings.warn("a real warning", RuntimeWarning)
        return real(*args, **kwargs)

    pycurious.optimise_bouligand.bouligand2009 = noisy
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grid.residuals(np.array([3.0, 1.0, 10.0, 5.0]), k, Phi, sigma)
        assert [w for w in caught if "a real warning" in str(w.message)], (
            "residuals is still swallowing warnings it did not raise"
        )
    finally:
        pycurious.optimise_bouligand.bouligand2009 = real


def test_supplied_spectrum_from_another_window_warns():
    """
    A supplied spectrum makes `window`, `xc` and `yc` dead arguments, so
    passing the one from a different window answers a question the caller did
    not ask -- and it answers it plausibly. Measured on a 600-cell synthetic, a
    128 km spectrum passed to a 512 km call returns dz = 222.8 km where that
    window really gives 21.7.
    """
    grid, xc, yc, extent = _shared_spectrum_grid()

    grid.optimise(100e3, xc, yc)
    ours = grid.last_spectrum

    with pytest.warns(RuntimeWarning, match="different window"):
        grid.optimise(200e3, xc, yc, spectrum=ours)

    grid.optimise(100e3, xc, yc)
    with pytest.warns(RuntimeWarning, match="different xc"):
        grid.optimise(100e3, xc + 40e3, yc, spectrum=grid.last_spectrum)


def test_supplied_spectrum_at_matching_arguments_is_silent():
    """The documented idiom must not warn, or the guard is worse than useless."""
    import warnings

    grid, xc, yc, extent = _shared_spectrum_grid()
    grid.optimise(200e3, xc, yc)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grid.profile(200e3, xc, yc, "dz", npoints=7, spectrum=grid.last_spectrum)
    assert not [w for w in caught if "supplied spectrum" in str(w.message)]


def test_spectrum_of_unknown_provenance_is_taken_at_face_value():
    """
    Only a spectrum this instance computed can be checked. One built by the
    caller -- Global_CPD reads its archived spectra out of zarr as float32 --
    has nothing to compare against, so it must be accepted without a warning
    rather than guessed at.
    """
    import warnings

    grid, xc, yc, extent = _shared_spectrum_grid()
    grid.optimise(100e3, xc, yc)
    foreign = tuple(np.asarray(a).astype(np.float32) for a in grid.last_spectrum)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grid.optimise(200e3, xc, yc, spectrum=foreign)
    assert not [w for w in caught if "supplied spectrum" in str(w.message)]


def test_provenance_survives_being_passed_along():
    """
    Reusing a spectrum must not relabel it with the arguments of whichever call
    reused it, or the guard would go blind after one hop.
    """
    grid, xc, yc, extent = _shared_spectrum_grid()
    grid.optimise(100e3, xc, yc)
    grid.profile(100e3, xc, yc, "dz", npoints=7, spectrum=grid.last_spectrum)

    with pytest.warns(RuntimeWarning, match="different window"):
        grid.optimise(200e3, xc, yc, spectrum=grid.last_spectrum)
