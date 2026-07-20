"""Behaviour of CurieOptimiseTanaka, and guards against defects fixed in v2."""

import warnings

import numpy as np
import pytest

import pycurious

from conftest import load_magnetic_anomaly

ZT_RANGE = (1.26, 1.89)
Z0_RANGE = (0.0, 0.63)


@pytest.fixture
def tanaka(load_magnetic_anomaly):
    d = load_magnetic_anomaly["mag_data"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]
    return (
        pycurious.CurieOptimiseTanaka(d, xmin, xmax, ymin, ymax),
        load_magnetic_anomaly["xc"],
        load_magnetic_anomaly["yc"],
    )


def test_depths_are_positive_downwards(tanaka):
    """
    optimise returns depths, not gradients.

    Before v2 it returned the raw negative gradients, so callers saw a
    "top of magnetic source" of -0.95 km.
    """
    grid, xc, yc = tanaka
    zt, z0, _, _, sigma_zt, sigma_z0 = grid.optimise(
        300e3, xc, yc, ZT_RANGE, Z0_RANGE, taper=np.hanning
    )
    assert zt > 0.0 and z0 > 0.0
    assert sigma_zt > 0.0 and sigma_z0 > 0.0


def test_centroid_sigma_equals_spectrum_sigma(tanaka):
    """
    Dividing the amplitude spectrum by |k| must not change its uncertainty.

    Before v2 the centroid branch computed log(exp(sigma)/k), subtracting
    ln(k) from a standard deviation. That inflated sigma from ~0.6 to ~4.0 at
    the lowest wavenumbers -- exactly the bins the centroid fit relies on --
    and dragged z0 with it.
    """
    grid, xc, yc = tanaka
    k, Phi, Phi_n, sigma = grid._spectrum(
        300e3, xc, yc, np.hanning, None, None, None
    )

    # Phi_n is Phi shifted by a deterministic ln(k) ...
    np.testing.assert_allclose(Phi_n, Phi - np.log(k))
    # ... so a single sigma serves both fits, and it is strictly positive
    assert np.all(sigma > 0.0)
    assert np.all(np.isfinite(sigma))


def test_effective_dof_deflates_counts():
    """
    The uncertainty of the binned mean is not sigma/sqrt(N): the FFT cells are
    not independent. Hermitian symmetry alone makes half of them redundant.
    """
    from pycurious.optimise_tanaka import _dof_factor

    assert _dof_factor(None) == 2.0
    assert _dof_factor(np.hanning) > 2.0
    assert _dof_factor(np.hamming) > 2.0
    # an uncalibrated taper falls back to the exact Hermitian factor
    assert _dof_factor(np.bartlett) == 2.0
    # and an explicit override wins
    assert _dof_factor(np.hanning, dof_factor=1.0) == 1.0


def test_warns_when_bands_look_like_cycles_per_km(tanaka):
    """
    Bands were specified in cycles/km before v2. Such a call still runs and
    returns a plausible looking number, so it has to be flagged.
    """
    grid, xc, yc = tanaka
    with pytest.warns(UserWarning, match="cycles/km"):
        grid.optimise(300e3, xc, yc, (0.2, 0.3), (0.0, 0.1), taper=np.hanning)


def test_no_false_positive_on_valid_bands(tanaka):
    """The units guard must stay quiet for legitimate rad/km bands."""
    grid, xc, yc = tanaka
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        grid.optimise(300e3, xc, yc, ZT_RANGE, Z0_RANGE, taper=np.hanning)


def test_check_bands_flags_invalid_centroid_band(tanaka):
    """|k|d << 1 is the assumption most easily violated, and least visible."""
    grid, xc, yc = tanaka
    k, _, _ = grid.radial_spectrum(grid.subgrid(300e3, xc, yc), power=1)

    with pytest.warns(UserWarning, match=r"\|k\|d"):
        diagnostics = grid.check_bands(
            k, ZT_RANGE, (0.0, 2.0), thickness=10.0, verbose=False
        )
    assert diagnostics["kd_max"] > 1.0
    assert diagnostics["n_zt"] >= 3


def test_too_few_points_raises(tanaka):
    grid, xc, yc = tanaka
    with pytest.raises(ValueError, match="at least 3"):
        grid.optimise(300e3, xc, yc, (2.0, 2.001), Z0_RANGE, taper=np.hanning)


def test_sensitivity_collapses_to_covariance_without_band_jitter(tanaka):
    """
    With band_scale=0 only the spectrum is resampled, which must reproduce the
    analytic fit covariance. Jittering the bands should then widen it, since
    band placement dominates.
    """
    grid, xc, yc = tanaka
    _, _, _, _, sigma_zt, sigma_z0 = grid.optimise(
        300e3, xc, yc, ZT_RANGE, Z0_RANGE, taper=np.hanning
    )
    _, sigma_CPD = grid.calculate_CPD(0.0, 0.0, sigma_zt, sigma_z0)

    zt_s, z0_s, cpd_s = grid.sensitivity(
        300e3, xc, yc, 400, ZT_RANGE, Z0_RANGE,
        taper=np.hanning, band_scale=0.0, seed=42,
    )
    np.testing.assert_allclose(cpd_s.std(), sigma_CPD, rtol=0.25)

    _, _, cpd_jitter = grid.sensitivity(
        300e3, xc, yc, 400, ZT_RANGE, Z0_RANGE,
        taper=np.hanning, band_scale=0.1, seed=42,
    )
    assert cpd_jitter.std() > cpd_s.std()


def test_sensitivity_is_reproducible(tanaka):
    grid, xc, yc = tanaka
    kwargs = dict(taper=np.hanning, band_scale=0.1)
    a = grid.sensitivity(300e3, xc, yc, 100, ZT_RANGE, Z0_RANGE, seed=7, **kwargs)[2]
    b = grid.sensitivity(300e3, xc, yc, 100, ZT_RANGE, Z0_RANGE, seed=7, **kwargs)[2]
    c = grid.sensitivity(300e3, xc, yc, 100, ZT_RANGE, Z0_RANGE, seed=8, **kwargs)[2]
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_calculate_CPD_propagates_uncertainty():
    grid = pycurious.CurieOptimiseTanaka(np.zeros((9, 9)), 0.0, 8e3, 0.0, 8e3)

    CPD, sigma = grid.calculate_CPD(1.0, 6.0, 0.3, 0.4)
    assert CPD == pytest.approx(11.0)
    assert sigma == pytest.approx(np.sqrt(0.3 ** 2 + 0.8 ** 2))

    # vectorises over arrays of centroids
    CPD, sigma = grid.calculate_CPD(
        np.array([1.0, 2.0]), np.array([6.0, 7.0]), 0.0, 0.0
    )
    np.testing.assert_allclose(CPD, [11.0, 12.0])
    np.testing.assert_allclose(sigma, [0.0, 0.0])


def test_CurieOptimiseTanaka_routines(load_magnetic_anomaly):
    """optimise_routine returns one array per output, one entry per centroid."""
    d = load_magnetic_anomaly["mag_data"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]
    max_window = load_magnetic_anomaly["max_window"]

    grid = pycurious.CurieOptimiseTanaka(d, xmin, xmax, ymin, ymax)
    xc_list, yc_list = grid.create_centroid_list(
        0.5 * max_window, spacingX=40e3, spacingY=40e3
    )

    results = grid.optimise_routine(
        0.5 * max_window, xc_list, yc_list, ZT_RANGE, Z0_RANGE, taper=np.hanning
    )

    assert len(results) == 6
    for array in results:
        assert len(array) == len(xc_list)
        assert np.isfinite(array).all()

    zt, z0, _, _, sigma_zt, sigma_z0 = results
    CPD, sigma_CPD = grid.calculate_CPD(zt, z0, sigma_zt, sigma_z0)
    assert CPD.shape == (len(xc_list),)
    assert np.all(sigma_CPD > 0.0)
