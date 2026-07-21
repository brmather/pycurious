"""
Parameter recovery against synthetics with a known Curie depth.

Unlike the fixed `test_mag_data.txt` fixture, these synthetics are generated
with prescribed parameters (see `tests/synthetic.py`), so the tests ask whether
each method recovers what it was given rather than whether it reproduces one
hard-coded number.
"""

import numpy as np
import pytest

import pycurious

from synthetic import fractal_anomaly

# (beta, zt, dz) -- a shallow thick layer and a deeper thinner one
CASES = [(3.0, 1.0, 20.0), (2.0, 5.0, 15.0)]
SEEDS = [1, 2]


def _bouligand(beta, zt, dz, seed, n=512, dx=2.0):
    data, extent = fractal_anomaly(
        n=n, dx=dx, beta=beta, zt=zt, dz=dz, C=5.0, seed=seed
    )
    grid = pycurious.CurieOptimiseBouligand(data, *extent)
    xc = 0.5 * (extent[0] + extent[1])
    yc = 0.5 * (extent[2] + extent[3])
    return grid, xc, yc


@pytest.mark.parametrize("beta,zt,dz", CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_bouligand_recovers_beta_and_zt(beta, zt, dz, seed):
    """
    beta and zt are tightly determined and near-Gaussian, so a per-seed
    tolerance is meaningful for them. Across 200 realisations their mean errors
    are 0.05 and 0.02.

    dz is deliberately not asserted here -- see the two tests below.
    """
    grid, xc, yc = _bouligand(beta, zt, dz, seed)
    beta_r, zt_r = grid.optimise(1000e3, xc, yc, taper=np.hanning)[:2]

    assert np.abs(beta_r - beta) < 0.2, "beta {:.3f} != {}".format(beta_r, beta)
    assert np.abs(zt_r - zt) < 0.25, "zt {:.3f} != {}".format(zt_r, zt)


@pytest.mark.parametrize("beta,zt,dz", CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_bouligand_curie_depth_interval_contains_the_truth(beta, zt, dz, seed):
    """
    The right per-seed claim for a long-tailed parameter is that the interval
    covers the truth, not that the point estimate is close to it.

    dz has a long upper tail (Mather & Fullea, 2019). A single realisation can
    put it 85% high at a *lower* misfit than the truth, so any per-seed
    tolerance tight enough to be interesting fails on roughly one seed in five.
    """
    grid, xc, yc = _bouligand(beta, zt, dz, seed)
    _, _, lower, upper = grid.profile(1000e3, xc, yc, "CPD", taper=np.hanning)

    truth = zt + dz
    assert lower <= truth <= upper, "CPD interval [{:.2f}, {:.2f}] misses {}".format(
        lower, upper, truth
    )


@pytest.mark.slow
@pytest.mark.parametrize("beta,zt,dz", CASES)
def test_bouligand_dz_is_recovered_in_the_mean(beta, zt, dz):
    """
    dz is only well determined in aggregate, so that is what to assert.

    Per-seed values for the first case run 13.5 to 34.9 against a truth of
    20.0. Averaging eight of them gives 21.32 and 16.03 for the two cases,
    i.e. +6.6% and +6.9% -- both high, because the tail is on that side and it
    pulls the mean with it. The mean is the right statistic to assert, but it
    is not an unbiased one; see `profile` for the honest interval.
    """
    recovered = []
    for seed in range(1, 9):
        grid, xc, yc = _bouligand(beta, zt, dz, seed)
        recovered.append(grid.optimise(1000e3, xc, yc, taper=np.hanning)[2])

    mean_dz = float(np.mean(recovered))
    assert np.abs(mean_dz - dz) < 0.10 * dz, "mean dz {:.2f} != {} (from {})".format(
        mean_dz, dz, np.round(recovered, 1).tolist()
    )


def _tanaka_grid(beta=3.0, zt=1.0, dz=20.0, n=1024, dx=4.0, seed=1):
    """A grid wide enough for the centroid band to satisfy |k|d << 1."""
    data, extent = fractal_anomaly(n=n, dx=dx, beta=beta, zt=zt, dz=dz, C=5.0, seed=seed)
    grid = pycurious.CurieOptimiseTanaka(data, *extent)
    xc = 0.5 * (extent[0] + extent[1])
    yc = 0.5 * (extent[2] + extent[3])
    return grid, xc, yc, (n - 1) * dx * 1e3


def test_tanaka_recovers_curie_depth():
    beta, zt, dz = 3.0, 1.0, 20.0
    grid, xc, yc, window = _tanaka_grid(beta, zt, dz)

    zt_r, z0_r, _, _, sigma_zt, sigma_z0 = grid.optimise(
        window, xc, yc, (0.20, 0.60), (0.0, 0.05), taper=np.hanning, beta=beta
    )
    CPD, sigma_CPD = grid.calculate_CPD(zt_r, z0_r, sigma_zt, sigma_z0)

    assert np.abs(zt_r - zt) < 0.3, "zt {:.3f} != {}".format(zt_r, zt)
    assert np.abs(z0_r - (zt + dz / 2)) < 2.0, "z0 {:.3f}".format(z0_r)
    assert np.abs(CPD - (zt + dz)) < 3.5, "CPD {:.3f} != {}".format(CPD, zt + dz)
    assert sigma_CPD > 0.0


def test_tanaka_beta_correction_removes_fractal_bias():
    """
    Without the correction, zt is biased high by (beta-1)/(2*kbar). Supplying
    beta should remove that and leave zt near the truth.
    """
    beta, zt, dz = 3.0, 1.0, 20.0
    grid, xc, yc, window = _tanaka_grid(beta, zt, dz)
    band = (0.20, 0.60)

    zt_classic = grid.optimise(window, xc, yc, band, (0.0, 0.05), beta=None)[0]
    zt_corrected = grid.optimise(window, xc, yc, band, (0.0, 0.05), beta=beta)[0]

    k, _, _ = grid.radial_spectrum(grid.subgrid(window, xc, yc), power=1)
    kbar = k[np.logical_and(k >= band[0], k <= band[1])].mean()
    predicted_bias = (beta - 1.0) / (2.0 * kbar)

    assert zt_classic > zt_corrected, "correction should reduce zt"
    np.testing.assert_allclose(zt_classic - zt_corrected, predicted_bias, rtol=0.15)
    assert np.abs(zt_corrected - zt) < 0.3


def test_tanaka_exact_spectrum():
    """
    Fit an exact, noiseless layer spectrum. The zt band should return zt almost
    exactly, and the z0 bias should shrink as the band respects |k|d << 1.
    """
    zt, dz = 1.0, 20.0
    z0, half = zt + dz / 2, dz / 2
    k = np.arange(1, 600) * 0.0034
    Phi = -k * zt + np.log(1.0 - np.exp(-k * dz))
    sigma = np.full_like(k, 0.01)

    grid = pycurious.CurieOptimiseTanaka(np.zeros((9, 9)), 0.0, 8e3, 0.0, 8e3)

    zt_r, _, _ = grid._fit_band(k, Phi, sigma, (0.20, 0.60))
    assert np.abs(zt_r - zt) < 0.05, "zt {:.4f} != {}".format(zt_r, zt)

    Phi_n = Phi - np.log(k)
    errors = []
    for kmax in (0.20, 0.10, 0.05, 0.02):
        z0_r, _, _ = grid._fit_band(k, Phi_n, sigma, (0.0, kmax))
        errors.append(abs(z0_r - z0))
        assert z0_r < z0, "the |k|d approximation biases z0 low"

    # monotonically better as |k|d falls, and converging on the truth
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 0.5, "z0 error at |k|d=0.2 is {:.3f}".format(errors[-1])


def test_synthetic_matches_forward_model():
    """
    The generator must actually produce the spectrum it claims, otherwise the
    recovery tests above are circular.
    """
    beta, zt, dz, C = 3.0, 1.0, 20.0, 5.0
    data, extent = fractal_anomaly(n=512, dx=2.0, beta=beta, zt=zt, dz=dz, C=C, seed=1)

    grid = pycurious.CurieGrid(data, *extent)
    k, Phi, sigma_Phi = grid.radial_spectrum(grid.data, taper=None, power=2)
    model = pycurious.bouligand2009(k, beta, zt, dz, C)

    # the white-noise normalisation is an arbitrary constant, absorbed into C,
    # so compare the shape rather than the level
    residual = (Phi - model) - (Phi - model).mean()
    assert residual.std() < 0.3, "spectrum shape differs, rms {:.3f}".format(
        residual.std()
    )
