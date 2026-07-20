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


@pytest.mark.parametrize("beta,zt,dz", CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_bouligand_recovers_parameters(beta, zt, dz, seed):
    data, extent = fractal_anomaly(
        n=512, dx=2.0, beta=beta, zt=zt, dz=dz, C=5.0, seed=seed
    )
    grid = pycurious.CurieOptimiseBouligand(data, *extent)
    xc = 0.5 * (extent[0] + extent[1])
    yc = 0.5 * (extent[2] + extent[3])

    beta_r, zt_r, dz_r, C_r = grid.optimise(1000e3, xc, yc, taper=np.hanning)

    # beta and zt are well determined; dz is the loosest parameter of the three
    assert np.abs(beta_r - beta) < 0.2, "beta {:.3f} != {}".format(beta_r, beta)
    assert np.abs(zt_r - zt) < 0.25, "zt {:.3f} != {}".format(zt_r, zt)
    assert np.abs(dz_r - dz) < 0.25 * dz, "dz {:.3f} != {}".format(dz_r, dz)


@pytest.mark.parametrize("beta,zt,dz", CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_bouligand_recovers_curie_depth(beta, zt, dz, seed):
    """The quantity the method exists to estimate is the base of the layer."""
    data, extent = fractal_anomaly(
        n=512, dx=2.0, beta=beta, zt=zt, dz=dz, C=5.0, seed=seed
    )
    grid = pycurious.CurieOptimiseBouligand(data, *extent)
    xc = 0.5 * (extent[0] + extent[1])
    yc = 0.5 * (extent[2] + extent[3])

    beta_r, zt_r, dz_r, C_r = grid.optimise(1000e3, xc, yc, taper=np.hanning)

    cpd = zt_r + dz_r
    assert np.abs(cpd - (zt + dz)) < 0.2 * (zt + dz), "CPD {:.3f} != {}".format(
        cpd, zt + dz
    )


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
