import pytest
import pycurious
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
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]

    grid = pycurious.CurieGrid(d, xmin, xmax, ymin, ymax)

    # wavenumber bands for Z0 and Zt, respectively
    kwin_Z0 = (0.005, 0.03)
    kwin_Zt = (0.03, 0.7)

    k, Phi, sigma_Phi = grid.radial_spectrum(grid.data, taper=np.hanning, power=0.5)
    (Ztr, btr, dZtr), (Zor, bor, dZor) = pycurious.tanaka1999(
        k, Phi, sigma_Phi, kwin_Z0, kwin_Zt
    )
    Zb, eZb = pycurious.ComputeTanaka(Ztr, dZtr, Zor, dZor)

    error_msg = "FAILED! Tanaka CPD is {:.4f} different from expected, uncertainty is {:.4f}".format(
        Zb - 10.0, eZb
    )
    assert np.abs(Zb - 10.0) < 2.0 and eZb < Zb, error_msg
