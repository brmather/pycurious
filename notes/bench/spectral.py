"""The master spectral equation, and the generalisations it opens up.

Derivation (verified in derive_check.py to 1e-11 against pycurious):

    Phi(k) = (k^2 / 2pi) INT P_M(sqrt(k^2 + t^2)) |h_k(t)|^2 dt
    h_k(t) = INT g(z) exp(-(k + i t) z) dz

  P_M  3D magnetisation PSD, |q|^-beta for a fractal
  g(z) magnetisation-vs-depth profile of the layer
  k^2  the anomaly is the gradient of the potential: one |k| in the field
       relation, two in the power spectrum

Everything about the layer enters through |h_k(t)|^2 -- the power spectrum of
the depth profile g(z) weighted by the upward-continuation kernel exp(-kz).
Bouligand (2009) is the single case g = box-car on [zt, zt+dz], for which the
t-integral is a standard Bessel pair and the result collapses to the closed
form pycurious implements.

Two consequences the closed form hides:

* At low k, h_k(0) -> (INT g) (1 - k zbar + ...): the leading behaviour is set
  by the layer's total magnetic MOMENT and its CENTROID, not by the shape of
  its base. Any g with the same moment and centroid gives the same low-k
  spectrum. That is the whole robustness argument for centroid methods.
* At high k the profile is irrelevant beyond its top edge, so beta and zt are
  what the short wavelengths see.
"""
import numpy as np
from scipy.special import gamma, kv


def bouligand(k, beta, zt, dz, C):
    """Box-car g. Identical to pycurious.bouligand2009; kept local so the
    generalisations below sit next to the thing they generalise."""
    W = k * dz
    nu = 0.5 * (1.0 + beta)
    A = np.sqrt(np.pi) / gamma(1.0 + 0.5 * beta) * (
        0.5 * np.cosh(W) * gamma(nu) - kv(nu, W) * np.power(0.5 * W, nu)
    )
    return C - 2.0 * k * zt - (beta - 1.0) * np.log(k) - W + np.log(A)


# ---------------------------------------------------------------- asymptotes

def asymptote_high(k, beta, zt, C):
    """k dz >> 1: the layer's base is invisible, a fractal half-space.
        ln Phi = C' + (1 - beta) ln k - 2 k zt
    """
    return C + (1.0 - beta) * np.log(k) - 2.0 * k * zt


def asymptote_low(k, beta, z0, C):
    """k dz << 1: a sheet at the CENTROID z0 = zt + dz/2.
        ln Phi = C'' + (3 - beta) ln k - 2 k z0
    The exponential rate is 2*z0, not 2*zt -- the -k*dz sitting outside the
    bracket in the closed form is precisely what turns exp(-2k zt) into
    exp(-2k z0). dz survives only inside C'' (as 2 ln dz), so at low k it is
    perfectly degenerate with the amplitude.
    """
    return C + (3.0 - beta) * np.log(k) - 2.0 * k * z0


# The slope change between the two asymptotes is exactly 2, independently of
# beta, zt and dz. It is the model's one hard, falsifiable prediction.
SLOPE_CHANGE = 2.0


def break_wavenumber(dz, beta):
    """Where the two asymptotes cross. Determined by dz alone (given beta)."""
    d = np.linspace(-3, 3, 601)
    return None  # see fit_break() in the diagnostics; kept explicit there


# ------------------------------------------------- generalisation 1: mixture

def mixture(k, beta, zt, dz_mean, dz_spread, C, nodes=9):
    """Base depth varying WITHIN the window: an incoherent mixture of layers.

    Power adds, so Phi_mix = INT p(dz) Phi(k; dz) d(dz) -- averaged in Phi, not
    in ln Phi, which is where the bias lives: the fit minimises residuals in
    ln Phi and so lands near a geometric mean, while the physics gives an
    arithmetic one. Gauss-Hermite over a Gaussian p(dz), truncated at dz > 0.
    """
    if dz_spread <= 1e-6:
        return bouligand(k, beta, zt, dz_mean, C)
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    dzs = dz_mean + dz_spread * x
    keep = dzs > 0.05
    if not keep.any():
        return bouligand(k, beta, zt, max(dz_mean, 0.05), C)
    dzs, w = dzs[keep], w[keep] / w[keep].sum()
    acc = np.zeros_like(np.asarray(k, dtype=float))
    for d, wi in zip(dzs, w):
        acc += wi * np.exp(bouligand(k, beta, zt, d, 0.0))
    return C + np.log(acc)


# ------------------------------------------- generalisation 2: free slope gap

def free_gap(k, beta, zt, dz, C, gap):
    """Bouligand with the 2-in-the-slope-change released.

    Built by bending the box-car answer: multiply by (k dz)^(gap-2) in the
    low-k limit only, via the model's own transfer function. Used purely as a
    falsification test -- if real spectra want gap != 2, the box-car fractal
    layer is misspecified and dz is absorbing the mismatch.
    """
    full = bouligand(k, beta, zt, dz, 0.0)
    hi = asymptote_high(k, beta, zt, 0.0)
    # T in [0,1]: 0 deep in the low-k limit, 1 deep in the high-k limit.
    lnT = full - hi                      # <= 0, -> 0 at high k, ~ 2 ln(k dz)
    return C + hi + lnT * (gap / SLOPE_CHANGE)
