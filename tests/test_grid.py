import pytest
import pycurious
from pycurious.grid import _dof_factor
import numpy as np

from conftest import load_magnetic_anomaly


def test_subgrid(load_magnetic_anomaly):
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]

    grid = pycurious.CurieGrid(d, xmin, xmax, ymin, ymax)

    window_size = 100e3
    subgrid = grid.subgrid(window_size, xc, yc)

    error_msg = "FAILED! Subgrid is of shape {} and domain is of shape {}".format(
        subgrid.shape, grid.data.shape
    )
    assert subgrid.shape[0] < grid.data.shape[0], error_msg
    assert subgrid.shape[1] < grid.data.shape[1], error_msg


def test_wavenumber_grid_matches_dft():
    """
    The radial wavenumber grid must be the true DFT grid.

    The fundamental is 2*pi/(N*dx); using (N-1) overstates every wavenumber
    by N/(N-1) and so understates every depth by (N-1)/N.
    """
    N, dx_km = 201, 1.0
    grid = pycurious.CurieGrid(
        np.zeros((N, N)), 0.0, (N - 1) * 1e3, 0.0, (N - 1) * 1e3
    )
    _, dk, _ = grid._taper_spectrum(grid.data, None)

    np.testing.assert_allclose(dk, 2.0 * np.pi / (N * dx_km), rtol=1e-12)

    kx = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(N, d=dx_km))
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    i0 = N // 2
    ix, iy = np.mgrid[0:N, 0:N]

    np.testing.assert_allclose(
        np.hypot((ix - i0) * dk, (iy - i0) * dk), np.hypot(KX, KY), atol=1e-12
    )


def test_radial_spectrum_recovers_injected_depth():
    """
    A field built as white noise * exp(-|k|z) has a log amplitude spectrum of
    slope -z, so the depth must come back out of radial_spectrum directly.
    This pins the wavenumber scaling end to end.
    """
    from scipy.optimize import curve_fit

    N, z = 201, 4.0
    rng = np.random.default_rng(0)
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    spectrum = np.fft.fft2(rng.normal(size=(N, N))) * np.exp(-np.hypot(KX, KY) * z)
    field = np.real(np.fft.ifft2(spectrum))

    grid = pycurious.CurieGrid(field, 0.0, (N - 1) * 1e3, 0.0, (N - 1) * 1e3)
    k, Phi, _ = grid.radial_spectrum(grid.data, taper=None, power=1)

    mask = np.logical_and(k > 0.15, k < 1.5)
    (slope, _), _ = curve_fit(lambda x, a, b: a * x + b, k[mask], Phi[mask])

    assert np.abs(-slope - z) < 0.05, "recovered {:.4f} km, injected {} km".format(
        -slope, z
    )


def test_radial_spectrum_counts():
    """Bin counts must partition the wavenumber plane without double counting."""
    grid = pycurious.CurieGrid(np.random.rand(101, 101), 0.0, 100e3, 0.0, 100e3)

    assert len(grid.radial_spectrum(grid.data, taper=None)) == 3

    k, Phi, sigma_Phi, counts = grid.radial_spectrum(
        grid.data, taper=None, return_counts=True
    )
    assert counts.shape == k.shape
    assert counts.min() > 0
    # bins cover the inscribed circle, never more than the whole grid
    assert counts.sum() <= grid.data.size


def test_FFT(load_magnetic_anomaly):
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]

    grid = pycurious.CurieGrid(d, xmin, xmax, ymin, ymax)

    # Take Fourier transform
    k, Phi, sigma_Phi = grid.radial_spectrum(grid.data, taper=None)

    # radial power spectrum should decrease with wavenumber
    # divide Phi into three sections
    i3 = len(k) // 3
    Phi1 = Phi[:i3]
    Phi2 = Phi[i3 : 2 * i3]
    Phi3 = Phi[2 * i3 :]

    # also sigma_Phi should decrease with wavenumber
    sigma_Phi1 = sigma_Phi[:i3]
    sigma_Phi2 = sigma_Phi[i3 : 2 * i3]
    sigma_Phi3 = sigma_Phi[2 * i3 :]

    error_msg = "FAILED! Fast Fourier Transform did not produce a valid power spectrum"
    assert Phi1.mean() > Phi2.mean() > Phi3.mean(), error_msg
    assert sigma_Phi1.mean() > sigma_Phi2.mean() > sigma_Phi3.mean(), error_msg


def test_taper_functions(load_magnetic_anomaly):
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]

    grid = pycurious.CurieGrid(d, xmin, xmax, ymin, ymax)

    # Take Fourier transform using different taper functions
    k, Phi1, sigma_Phi1 = grid.radial_spectrum(grid.data, taper=None)
    k, Phi2, sigma_Phi2 = grid.radial_spectrum(grid.data, taper=np.hanning)
    k, Phi3, sigma_Phi3 = grid.radial_spectrum(grid.data, taper=np.hamming)

    grad_Phi1 = np.gradient(Phi1, k)
    grad_Phi2 = np.gradient(Phi2, k)
    grad_Phi3 = np.gradient(Phi3, k)

    assert (
        Phi1.mean() > Phi2.mean()
    ), "FAILED! 'taper=np.hanning' not significantly different from 'taper=None'"
    assert (
        Phi1.mean() > Phi3.mean()
    ), "FAILED! 'taper=np.hamming' not significantly different from 'taper=None'"
    assert (
        grad_Phi1.mean() > grad_Phi2.mean()
    ), "FAILED! 'taper=np.hanning' has steeper gradient from 'taper=None'"
    assert (
        grad_Phi1.mean() > grad_Phi3.mean()
    ), "FAILED! 'taper=np.hamming' has steeper gradient from 'taper=None'"


def test_Tanaka(load_magnetic_anomaly):
    """
    The centroid method returns a sane, positive Curie depth on the legacy
    fixture.

    No accuracy is asserted here. This grid is only 305 km across for a 10 km
    layer, so the z0 band cannot satisfy |k|d << 1 with enough points left to
    fit -- see tests/test_recovery.py, which checks accuracy against
    synthetics generated wide enough to support the fit.
    """
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]

    grid = pycurious.CurieOptimiseTanaka(d, xmin, xmax, ymin, ymax)

    # wavenumber bands in rad/km
    zt_range = (1.26, 1.89)
    z0_range = (0.0, 0.63)

    zt, z0, zt_i, z0_i, sigma_zt, sigma_z0 = grid.optimise(
        300e3, xc, yc, zt_range, z0_range, taper=np.hanning
    )
    CPD, sigma_CPD = grid.calculate_CPD(zt, z0, sigma_zt, sigma_z0)

    # depths are returned positive downwards
    assert zt > 0.0, "zt should be positive downwards, got {:.4f}".format(zt)
    assert z0 > zt, "centroid {:.4f} should lie below the top {:.4f}".format(z0, zt)
    assert CPD > z0, "CPD {:.4f} should lie below the centroid {:.4f}".format(CPD, z0)
    assert sigma_CPD > 0.0, "uncertainty should be positive"
    assert np.isfinite([zt, z0, CPD, sigma_CPD]).all()


def test_tanaka_deprecated_functions(load_magnetic_anomaly):
    """
    The pre-v2 module functions still run, and say they are deprecated.

    Their numbers are deliberately not asserted: tanaka1999 weights by
    1/sigma**4 and subtracts ln(k) from a standard deviation, so agreement
    with any particular value would not be meaningful.
    """
    d = load_magnetic_anomaly["mag_data"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]

    grid = pycurious.CurieGrid(d, xmin, xmax, ymin, ymax)
    k, Phi, sigma_Phi = grid.radial_spectrum(grid.data, taper=np.hanning, power=1)

    with pytest.warns(FutureWarning, match="deprecated"):
        (Ztr, btr, dZtr), (Zor, bor, dZor) = pycurious.tanaka1999(
            k, Phi, sigma_Phi, (0.005, 0.03), (0.03, 0.7)
        )

    with pytest.warns(FutureWarning, match="argument order differs"):
        Zb, eZb = pycurious.ComputeTanaka(Ztr, dZtr, Zor, dZor)

    # abs() is applied internally, so this cannot come back negative
    assert Zb > 0.0
    assert np.isfinite([Zb, eZb]).all()


def test_dof_factor_deflates_counts():
    """
    The uncertainty of the binned mean is not sigma/sqrt(N): the FFT cells are
    not independent. Hermitian symmetry alone makes half of them redundant, and
    a fixed number more is lost to correlation however few the annulus holds --
    which is why the deflation depends on the count.
    """
    assert _dof_factor(None) == 2.0
    assert _dof_factor(np.hanning) > 2.0
    assert _dof_factor(np.hamming) > 2.0
    # an uncalibrated taper falls back to the exact Hermitian factor
    assert _dof_factor(np.bartlett) == 2.0
    # and an explicit override wins
    assert _dof_factor(np.hanning, dof_factor=1.0) == 1.0

    counts = np.array([8, 16, 64, 1024])
    per_bin = _dof_factor(np.hanning, counts)
    assert per_bin.shape == counts.shape
    # sparse bins are deflated hardest, and a full one tends to the asymptote
    assert np.all(np.diff(per_bin) < 0.0)
    assert per_bin[0] > 2.0 * per_bin[-1]
    assert per_bin[-1] == pytest.approx(3.3, rel=0.02)
