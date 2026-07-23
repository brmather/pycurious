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


def _reference_FFT_spectrum(subgrid, vtaper, dk, kbins, const):
    """
    Explicit per-bin reference for `_FFT_spectrum`, kept deliberately naive.

    This is the pre-vectorised implementation. It is O(nbins * N**2) and far
    too slow to use, but it states the binning semantics unambiguously, so the
    bincount version is pinned against it rather than against stored numbers.
    """
    nr, nc = subgrid.shape
    nbins = kbins.size - 1

    FT = np.fft.fftshift(np.abs(np.fft.fft2(subgrid * vtaper)))
    i0, j0 = int(nr // 2), int(nc // 2)
    ix, iy = np.mgrid[0:nr, 0:nc]
    kk = np.hypot((ix - i0) * dk, (iy - j0) * dk)

    S = np.empty(nbins)
    k = np.empty(nbins)
    sigma = np.empty(nbins)
    counts = np.empty(nbins, dtype=int)
    for i in range(nbins):
        # half-open above, except the last bin which is closed
        if i == nbins - 1:
            mask = np.logical_and(kk >= kbins[i], kk <= kbins[i + 1])
        else:
            mask = np.logical_and(kk >= kbins[i], kk < kbins[i + 1])
        rr = const * np.log(FT[mask])
        S[i] = rr.mean()
        k[i] = kk[mask].mean()
        sigma[i] = np.std(rr)
        counts[i] = rr.size
    return k, S, sigma, counts


@pytest.mark.parametrize("n", [64, 65, 128, 301])
@pytest.mark.parametrize("taper", [None, np.hanning])
@pytest.mark.parametrize("power", [1.0, 2.0])
def test_FFT_spectrum_matches_per_bin_reference(n, taper, power):
    """
    The vectorised binning must reproduce the per-bin loop exactly.

    The bin-edge semantics are easy to break: annuli are half-open `[lo, hi)`
    so a cell landing on an edge is counted once, *except* the last bin which
    is closed. Cells sit on edges constantly -- everything on the kx or ky
    axis has `kk` an exact multiple of `dk` -- so an off-by-one here silently
    shifts counts between neighbouring annuli.
    """
    data, extent = pycurious.fractal_anomaly(n, 1.0, 3.0, 1.0, 20.0, 5.0, seed=n)
    grid = pycurious.CurieGrid(data, *extent)
    vtaper, dk, kbins = grid._taper_spectrum(data, taper)

    got = grid._FFT_spectrum(data, vtaper, dk, kbins, power)
    want = _reference_FFT_spectrum(data, vtaper, dk, kbins, power)

    # counts must be identical, not close -- they are a partition
    np.testing.assert_array_equal(got[3], want[3])
    for name, g, w in zip(("k", "S", "sigma"), got, want):
        np.testing.assert_allclose(g, w, rtol=1e-12, atol=0, err_msg=name)


def test_FFT_spectrum_sigma_is_population_std():
    """
    `sigma` is the population standard deviation (ddof=0) of `const*ln|FFT|`
    within the annulus, computed two-pass. The cheaper `E[x**2] - E[x]**2`
    form cancels badly here -- `ln|FFT|` is O(10) with O(1) scatter -- and
    loses three digits, more on a near-constant bin.
    """
    n = 128
    data, extent = pycurious.fractal_anomaly(n, 1.0, 3.0, 1.0, 20.0, 5.0, seed=2)
    grid = pycurious.CurieGrid(data, *extent)
    vtaper, dk, kbins = grid._taper_spectrum(data, np.hanning)

    k, S, sigma, counts = grid._FFT_spectrum(data, vtaper, dk, kbins, 2.0)

    FT = np.fft.fftshift(np.abs(np.fft.fft2(data * vtaper)))
    i0 = j0 = int(n // 2)
    ix, iy = np.mgrid[0:n, 0:n]
    kk = np.hypot((ix - i0) * dk, (iy - j0) * dk)

    for i in (0, 1, len(k) // 2, len(k) - 1):
        if i == len(k) - 1:
            mask = np.logical_and(kk >= kbins[i], kk <= kbins[i + 1])
        else:
            mask = np.logical_and(kk >= kbins[i], kk < kbins[i + 1])
        cells = 2.0 * np.log(FT[mask])
        assert counts[i] == cells.size
        np.testing.assert_allclose(sigma[i], np.std(cells), rtol=1e-13)
        np.testing.assert_allclose(S[i], cells.mean(), rtol=1e-13)


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


def test_remove_trend_linear():
    """
    remove_trend_linear subtracts the least-squares plane.

    It must reduce an exact plane to zero -- on non-square grids as well as
    square ones -- and agree with a correctly aligned lstsq plane fit on
    arbitrary data. The non-square case is a regression guard: a design matrix
    built as (nc, nr) instead of (nr, nc) passes the square tests but leaves a
    finite trend on a rectangular grid.
    """
    # remove_trend_linear reads only its argument's shape, so any valid grid
    # will do; the constructor just requires equal node spacing in x and y.
    grid = pycurious.CurieGrid(np.zeros((8, 8)), 0.0, 7e3, 0.0, 7e3)

    def lstsq_detrend(data):
        nr, nc = data.shape
        ii, jj = np.mgrid[0:nr, 0:nc]  # aligned with data's own (nr, nc) layout
        A = np.c_[ii.ravel(), jj.ravel(), np.ones(data.size)]
        coef, *_ = np.linalg.lstsq(A, data.ravel(), rcond=None)
        return data - (A @ coef).reshape(data.shape)

    rng = np.random.default_rng(0)
    for nr, nc in [(64, 64), (40, 25), (25, 40)]:
        ii, jj = np.mgrid[0:nr, 0:nc]
        plane = 3.0 + 0.5 * ii - 0.25 * jj

        # an exact plane is removed to zero, whatever the aspect ratio
        detrended = grid.remove_trend_linear(plane.astype(float))
        np.testing.assert_allclose(detrended, 0.0, atol=1e-9)

        # and on noisy data it matches the lstsq plane fit
        data = plane + rng.standard_normal((nr, nc))
        np.testing.assert_allclose(
            grid.remove_trend_linear(data), lstsq_detrend(data), atol=1e-9
        )
