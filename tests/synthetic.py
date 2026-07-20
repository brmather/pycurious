"""
Synthetic magnetic anomalies with a known Curie depth.

`tests/Bouligand_forward.py` builds an anomaly by integrating a 3D fractal
magnetisation over depth. It is faithful but slow -- triple-nested Python loops
over a 305**3 volume -- and its output is a fixed text file, so the true
parameters cannot be varied.

Here the field is synthesised directly in the Fourier domain instead. Filtering
white noise by the square root of `pycurious.bouligand2009` produces a field
whose expected radial power spectrum *is* the analytic model, for any choice of
beta, zt and dz. That makes it possible to ask whether each method recovers the
parameters it was given, rather than whether it reproduces one hard-coded
number.

Because the field is a single random realisation its measured spectrum scatters
about the model, so recovery is approximate. Averaging over seeds tightens it.
"""

import numpy as np

from pycurious import bouligand2009


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
            (xmin, xmax, ymin, ymax) in metres, for `CurieGrid`

    Notes:
        The Curie depth of the result is `zt + dz`, and its centroid depth
        (as sought by the Tanaka method) is `zt + dz/2`.
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

    # white real noise keeps the transform Hermitian, so the result is real
    noise = np.fft.fft2(rng.normal(size=(n, n)))
    data = np.real(np.fft.ifft2(noise * amplitude))

    extent = (0.0, (n - 1) * dx * 1e3, 0.0, (n - 1) * dx * 1e3)
    return data, extent
