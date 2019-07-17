
import pytest
import pycurious
import numpy as np

# load magnetic anomaly - i.e. random fractal noise
try:
    mag_data = np.loadtxt("test_mag_data.txt")
except:
    mag_data = np.loadtxt("tests/test_mag_data.txt")

nx, ny = 305, 305

x = mag_data[:,0]
y = mag_data[:,1]
d = mag_data[:,2].reshape(ny,nx)

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

xc = (xmax - xmin)/2
yc = (ymax - ymin)/2


def test_subgrid():
    grid = pycurious.CurieGrid(d, xmin, xmax, ymin, ymax)

    window_size = 100e3
    subgrid = grid.subgrid(window_size, xc, yc)

    if subgrid.shape[0] < grid.data.shape[0] and subgrid.shape[1] < grid.data.shape[1]:
        print("PASSED! Subgrid successfully extracted from domain")
    else:
        assert False, "FAILED! Subgrid is of shape {} and domain is of shape {}".format(subgrid.shape, grid.data.shape)

def test_FFT():
    grid = pycurious.CurieGrid(d, xmin, xmax, ymin, ymax)

    # Take Fourier transform
    k, Phi, sigma_Phi = grid.radial_spectrum(grid.data, taper=None)

    # radial power spectrum should decrease with wavenumber
    # divide Phi into three sections
    i3 = len(k)//3
    Phi1 = Phi[:i3]
    Phi2 = Phi[i3:2*i3]
    Phi3 = Phi[2*i3:]

    # also sigma_Phi should decrease with wavenumber
    sigma_Phi1 = sigma_Phi[:i3]
    sigma_Phi2 = sigma_Phi[i3:2*i3]
    sigma_Phi3 = sigma_Phi[2*i3:]

    if Phi1.mean() > Phi2.mean() > Phi3.mean() and \
       sigma_Phi1.mean() > sigma_Phi2.mean() > sigma_Phi3.mean():
        print("PASSED! Fast Fourier Transform produced a radial power spectrum")
    else:
        assert False, "FAILED! Fast Fourier Transform did not produce a valid power spectrum"

def test_taper_functions():
    grid = pycurious.CurieGrid(d, xmin, xmax, ymin, ymax)

    # Take Fourier transform using different taper functions
    k, Phi1, sigma_Phi1 = grid.radial_spectrum(grid.data, taper=None)
    k, Phi2, sigma_Phi2 = grid.radial_spectrum(grid.data, taper=np.hanning)
    k, Phi3, sigma_Phi3 = grid.radial_spectrum(grid.data, taper=np.hamming)

    grad_Phi1 = np.gradient(Phi1, k)
    grad_Phi2 = np.gradient(Phi2, k)
    grad_Phi3 = np.gradient(Phi3, k)

    if Phi1.mean() > Phi2.mean() and Phi1.mean() > Phi3.mean() and \
       grad_Phi1.mean() > grad_Phi2.mean() and grad_Phi1.mean() > grad_Phi3.mean():
        print("PASSED! Tapered power spectrum is shifted lower")
    else:
        assert False, "FAILED! Tapered power spectrum is not significantly different from 'taper=None'"


def test_Tanaka():
    grid = pycurious.CurieGrid(d, xmin, xmax, ymin, ymax)

    # wavenumber bands for Z0 and Zt, respectively
    kwin_Z0 = (0.005, 0.03)
    kwin_Zt = (0.03, 0.7)

    k, Phi, sigma_Phi = grid.radial_spectrum(grid.data, taper=np.hanning, power=0.5)
    (Ztr,btr,dZtr), (Zor, bor, dZor) = pycurious.tanaka1999(k, Phi, sigma_Phi, kwin_Z0, kwin_Zt)
    Zb, eZb = pycurious.ComputeTanaka(Ztr, dZtr, Zor, dZor)

    if np.abs(Zb - 10.0) < 2.0 and eZb < Zb:
        print("PASSED! Tanaka method returned a sensible Curie depth estimate")
    else:
        assert False, "FAILED! Tanaka CPD is {:.4f} different from expected, uncertainty is {:.4f}".format(Zb-10.0, eZb)


if __name__ == "__main__":
    test_subgrid()
    test_FFT()
    test_taper_functions()
    test_Tanaka()
