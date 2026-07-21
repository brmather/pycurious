import pytest
from functools import lru_cache

import pycurious
import numpy as np


@pytest.fixture(scope="module")
def load_magnetic_anomaly():
    # load magnetic anomaly - i.e. random fractal noise
    try:
        mag_data = np.loadtxt("tests/test_mag_data.txt")
    except:
        mag_data = np.loadtxt("test_mag_data.txt")

    nx, ny = 305, 305

    x = mag_data[:, 0]
    y = mag_data[:, 1]
    d = mag_data[:, 2].reshape(ny, nx)

    max_window = 300e3

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)

    # store inside dictionary
    mag_dict = {
        "extent": [xmin, xmax, ymin, ymax],
        "mag_data": d,
        "xc": xc,
        "yc": yc,
        "max_window": max_window,
    }

    return mag_dict


@lru_cache(maxsize=None)
def synthetic_grid(cls, beta=3.0, zt=1.0, dz=20.0, n=512, dx=2.0, seed=1):
    """
    A synthetic anomaly with a known Curie depth, and its centre.

    Cached because generating a 512x512 or 1024x1024 field costs more than the
    fit that follows, and the recovery tests ask for the same few fields
    repeatedly -- once per parametrised case, and again for each test over the
    same case.
    """
    data, extent = pycurious.fractal_anomaly(
        n=n, dx=dx, beta=beta, zt=zt, dz=dz, C=5.0, seed=seed
    )
    grid = cls(data, *extent)
    xc = 0.5 * (extent[0] + extent[1])
    yc = 0.5 * (extent[2] + extent[3])
    return grid, xc, yc, (n - 1) * dx * 1e3
