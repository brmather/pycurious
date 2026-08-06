"""Verify the Bouligand (2009) closed form against a direct numerical evaluation
of the double depth integral over a 3D fractal magnetisation.

Chain being checked
-------------------
  P_M(q)        = |q|^-beta            3D isotropic fractal magnetisation PSD
  S_M(k, zeta)  = (1/2pi) INT P_M(sqrt(k^2+kz^2)) exp(i kz zeta) dkz
                                       vertical cross-spectrum at horizontal k
  Phi(k)        = INT INT S_M(k, z-z') exp(-k(z+z')) dz dz'   over [zt, zt+dz]^2

The (z+z') exponential is upward continuation of each source depth to z=0
(Blakely 1995 eq. 11.36); the direction-cosine factors Theta_f Theta_m depend
only on the azimuth of k, so they survive radial averaging as a constant and
are absorbed into C.

The |k|^2 prefactor is NOT optional: the anomaly is the gradient of the
potential, so the field-from-magnetisation relation carries one factor of |k|
and the power spectrum carries two. Drop it and every Phi comes out k^-2 too
steep -- checked, that is exactly the discrepancy it produces. It is also what
makes the uniform-magnetisation limit reduce to Spector & Grant's
exp(-2k zt)(1 - exp(-k dz))^2 rather than that over k^2.

Reduce the double integral on the diagonal s = z+z', zeta = z-z':
  Phi(k) = k^2 (exp(-2 k zt)/k) INT_0^dz S_M(k,zeta)
                                [exp(-k zeta) - exp(-2 k dz + k zeta)] dzeta
"""
import numpy as np
from scipy.integrate import quad
import sys

sys.path.insert(0, "/home/unimelb.edu.au/matherb/git/pycurious")
from pycurious import bouligand2009


def S_M(k, zeta, beta):
    """Vertical cross-spectrum of the fractal magnetisation at horizontal k."""
    f = lambda t: (k * k + t * t) ** (-0.5 * beta)
    if zeta == 0.0:
        val, _ = quad(f, 0, np.inf, limit=400)
    else:
        # oscillatory: hand the cosine to QUADPACK as a weight
        val, _ = quad(f, 0, np.inf, weight="cos", wvar=zeta, limit=400)
    return val / np.pi


def phi_numeric(k, beta, zt, dz):
    """Phi(k) by direct quadrature of the reduced double integral."""
    def integrand(zeta):
        return S_M(k, zeta, beta) * (
            np.exp(-k * zeta) - np.exp(-2.0 * k * dz + k * zeta)
        )
    val, _ = quad(integrand, 0.0, dz, limit=300)
    return k * np.exp(-2.0 * k * zt) * val


if __name__ == "__main__":
    ks = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
    print(f"{'beta':>5} {'zt':>5} {'dz':>6}   max|ln ratio - mean| over k")
    print("-" * 58)
    worst = 0.0
    for beta in (2.0, 3.0, 4.0):
        for zt, dz in ((1.0, 10.0), (5.0, 30.0), (2.0, 3.0), (10.0, 60.0)):
            num = np.array([phi_numeric(k, beta, zt, dz) for k in ks])
            ana = bouligand2009(ks, beta, zt, dz, 0.0)   # returns ln Phi
            d = np.log(num) - ana
            spread = np.max(np.abs(d - d.mean()))
            worst = max(worst, spread)
            print(f"{beta:5.1f} {zt:5.1f} {dz:6.1f}   {spread:.3e}"
                  f"   (offset ln C = {d.mean():+.6f})")
    print("-" * 58)
    print(f"worst deviation from a constant offset: {worst:.3e}")
