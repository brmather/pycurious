"""Fitters over a cached spectrum: the archive's own model, and alternatives.

`fit_bouligand` is a standalone reimplementation of what Global_CPD pass B
does, checked against the archive to the storage floor. Having it standalone
is what makes the alternatives comparable -- they differ from it only in the
forward model or the estimator, never in the weighting, bounds or priors.
"""
import numpy as np
from scipy.optimize import least_squares

import spectral as S

DX_KM = 5.0
DZ_MAX = 700.0 * DX_KM / np.pi          # pycurious' cosh-overflow ceiling
ZT_PIN, ZT_SIG = 1.0, 0.05
BETA_SIG = 0.15
BIG = 1.0e6


def _resid(model, Phi, sigma, extra=()):
    r = (model - Phi) / sigma
    r = np.where(np.isfinite(r), r, BIG)
    return np.concatenate([r, np.asarray(extra, dtype=float)]) if len(extra) else r


def fit_bouligand(k, Phi, sigma, beta_prior, zt_pin=ZT_PIN, zt_sig=ZT_SIG,
                  beta_sig=BETA_SIG, x0=None):
    """The archive's model: box-car fractal layer, zt pinned, beta priored."""
    def f(x):
        beta, zt, dz, C = x
        with np.errstate(all="ignore"):
            m = S.bouligand(k, beta, zt, dz, C)
        return _resid(m, Phi, sigma,
                      [(zt - zt_pin) / zt_sig, (beta - beta_prior) / beta_sig])

    y0 = x0 if x0 is not None else [beta_prior, zt_pin, 10.0, 5.0]
    r = least_squares(f, y0, bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, DZ_MAX, np.inf]))
    return r


def fit_free_gap(k, Phi, sigma, beta_prior, zt_pin=ZT_PIN, zt_sig=ZT_SIG,
                 beta_sig=BETA_SIG):
    """Falsification test: release the slope change the model fixes at 2."""
    def f(x):
        beta, zt, dz, C, gap = x
        with np.errstate(all="ignore"):
            m = S.free_gap(k, beta, zt, dz, C, gap)
        return _resid(m, Phi, sigma,
                      [(zt - zt_pin) / zt_sig, (beta - beta_prior) / beta_sig])

    r = least_squares(f, [beta_prior, zt_pin, 10.0, 5.0, 2.0],
                      bounds=([0, 0, 0, -np.inf, 0.2], [np.inf, np.inf, DZ_MAX, np.inf, 6.0]))
    return r


def fit_mixture(k, Phi, sigma, beta_prior, zt_pin=ZT_PIN, zt_sig=ZT_SIG,
                beta_sig=BETA_SIG):
    """Base depth varying within the window: fit its mean and spread."""
    def f(x):
        beta, zt, dzm, dzs, C = x
        with np.errstate(all="ignore"):
            m = S.mixture(k, beta, zt, dzm, dzs, C)
        return _resid(m, Phi, sigma,
                      [(zt - zt_pin) / zt_sig, (beta - beta_prior) / beta_sig])

    r = least_squares(f, [beta_prior, zt_pin, 10.0, 3.0, 5.0],
                      bounds=([0, 0, 0, 0, -np.inf],
                              [np.inf, np.inf, DZ_MAX, 60.0, np.inf]))
    return r


def covariance(res):
    """Parameter covariance from the least_squares Jacobian at the solution."""
    J = res.jac
    try:
        return np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return np.full((J.shape[1], J.shape[1]), np.nan)
