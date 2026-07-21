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
Synthetic magnetic anomalies with a known Curie depth.

Filtering white noise by the square root of `pycurious.grid.bouligand2009`
produces a field whose expected radial power spectrum *is* the analytic model,
for any choice of \\( \\beta, z_t \\) and \\( \\Delta z \\). That makes it
possible to ask whether a method recovers the parameters it was given, rather
than whether it reproduces one hard-coded number.

Because the field is a single random realisation its measured spectrum scatters
about the model, so recovery is approximate. \\( \\beta \\) and \\( z_t \\) come
back tightly; \\( \\Delta z \\) does not, and one realisation in five puts it
tens of percent out. That is a property of the problem rather than of the
generator -- see `pycurious.optimise_bouligand.CurieOptimiseBouligand.profile`.

Averaging over seeds tightens it.
"""

import numpy as np

from .grid import bouligand2009


def fractal_anomaly(n=512, dx=2.0, beta=3.0, zt=1.0, dz=20.0, C=5.0, seed=0):
    """
    Synthesise a magnetic anomaly with a prescribed radial power spectrum.

    Args:
        n : int
            number of points per side
        dx : float
            grid spacing in km
        beta : float
            fractal parameter of the magnetisation
        zt : float
            depth to the top of the magnetic source, km
        dz : float
            thickness of the magnetic source, km
        C : float
            field constant
        seed : int
            seed for the white noise realisation

    Returns:
        data : 2D array shape (n,n)
            the magnetic anomaly
        extent : tuple
            (xmin, xmax, ymin, ymax) in metres, for `pycurious.grid.CurieGrid`

    Usage:
        >>> data, extent = pycurious.fractal_anomaly(beta=3.0, zt=1.0, dz=20.0)
        >>> grid = pycurious.CurieOptimiseBouligand(data, *extent)

    Notes:
        The Curie depth of the result is `zt + dz`, and its centroid depth
        (as sought by the Tanaka method) is `zt + dz/2`. Those two, and
        `beta`, are recoverable.

        `C` is not, quite. It is a level rather than a depth, and it absorbs
        every constant factor between the noise and the spectrum. Normalising
        the transform removes the dependence on `n`, but two offsets remain
        and both depend on how the spectrum is measured rather than on how the
        field was made:

        - the radial spectrum averages \\( \\ln |FFT| \\) rather than taking
          the log of the mean, which is lower by the Euler-Mascheroni constant,
          0.577;
        - a taper removes power, by \\( \\ln (3/8)^2 = -1.96 \\) for a
          separable `numpy.hanning`.

        So a fit through `numpy.hanning` returns `C` about 2.5 low, and through
        no taper about 0.6 low. Treat the recovered `C` as a nuisance
        parameter, not as something with a true value to check against.
    """
    rng = np.random.default_rng(seed)

    # radial wavenumber grid in rad/km, matching the DFT of an n-point series
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kh = np.hypot(*np.meshgrid(kx, kx, indexing="ij"))

    # bouligand2009 returns the log power spectrum, so the amplitude filter is
    # exp(Phi/2). kh=0 is undefined (log kh) and only sets the mean, so drop it.
    amplitude = np.zeros_like(kh)
    nonzero = kh > 0.0
    amplitude[nonzero] = np.exp(0.5 * bouligand2009(kh[nonzero], beta, zt, dz, C))

    # white real noise keeps the transform Hermitian, so the result is real.
    # Dividing by n undoes the scaling of an unnormalised fft2, without which
    # the recovered C would shift by 2*ln(n) with the size of the grid.
    noise = np.fft.fft2(rng.normal(size=(n, n))) / n
    data = np.real(np.fft.ifft2(noise * amplitude))

    extent = (0.0, (n - 1) * dx * 1e3, 0.0, (n - 1) * dx * 1e3)
    return data, extent
