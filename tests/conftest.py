import pytest
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
