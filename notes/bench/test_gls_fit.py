"""
Should the *fit* use the between-bin correlation, not just the covariance?

pycurious currently splits them: `_fit` minimises a diagonal sum of squares,
then `_covariance` solves `(J^T R^-1 J)^-1` with `R` from
`_banded_correlation`. So the estimator assumes independent bins and its error
bar assumes correlated ones. Generalised least squares would use `R` in both:

    minimise  r^T R^-1 r      instead of      r^T r

`R` is measured here rather than inferred from the residuals in hand -- it is a
property of the taper and the binning, and `measure_correlation.py` shows it is
stationary in `k` (hanning: rho_1 = 0.364 in every band), so a Toeplitz band
describes it exactly.

What GLS does to a fit is worth stating before measuring it. For positive
banded `R`, `R^-1` acts as a high-pass on the residual sequence: it discounts
the *smooth* part of the misfit and sharpens the fit's sensitivity to
curvature. `C` is a pure level and `beta` close to a pure slope, so those lose
weight; `dz` is a curvature feature -- the rollover -- so it should gain. That
is the hypothesis.

The prior rows stay at unit weight and never enter the whitening, exactly as
`_gls_covariance` treats them.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spectral_variants as sv  # noqa: E402
from run_experiments import degrade  # noqa: E402

DX_KM = 5.0
#: measured lag profiles, `measure_correlation.py`
RHO = {"hanning": [1.0, 0.364, 0.039],
       "dpss": [1.0, 0.860, 0.660, 0.445, 0.245, 0.089, 0.020]}


def toeplitz_chol(nbin, rho):
    """Cholesky factor of the banded stationary correlation, once per size."""
    R = np.eye(nbin)
    for lag, r in enumerate(rho[1:], 1):
        if lag < nbin:
            i = np.arange(nbin - lag)
            R[i, i + lag] = R[i + lag, i] = r
    return cholesky(R, lower=True)


def fit_gls(spectrum, g, L=None, x0=(3.0, 1.0, 10.0, 5.0)):
    """Least squares with the spectral residuals whitened by `L`."""
    k, Phi, sigma = spectrum
    lo = np.array([0.0, 0.0, 0.0, -np.inf])
    hi = np.array([np.inf, np.inf, g._max_thickness(), np.inf])

    def res(x):
        r = g.residuals(x, k, Phi, sigma)
        if L is None:
            return r
        # whiten the spectrum block only; prior rows keep unit weight
        w = solve_triangular(L, r[:k.size], lower=True)
        return np.concatenate([w, r[k.size:]])

    out = least_squares(res, np.array(x0), bounds=(lo, hi))
    J = out.jac
    try:
        cov = np.linalg.inv(J.T @ J)
        sd = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        sd = np.full(4, np.nan)
    return out.x, sd


def main():
    warnings.simplefilter("ignore")
    n, nseed = 801, 120
    g = sv.fitter()
    L = None

    print(f"4000 km window ({n} cells), hanning, {nseed} seeds per dz level.")
    print("OLS is what pycurious fits today; GLS whitens by the measured "
          "rho = [1, 0.364, 0.039].\n")

    for sigma_r, label in ((0.0, "clean"), (4.2, "degraded 4.2 km")):
        print(f"  {label}")
        print(f"  {'dz true':>8s} | {'OLS dz':>16s} {'sigma':>6s} | "
              f"{'GLS dz':>16s} {'sigma':>6s} | {'shift':>6s}")
        for dz_true in (10.0, 20.0, 30.0):
            o, gl, so, sg = [], [], [], []
            for seed in range(nseed):
                field = sv.synth(n + 10, seed, beta=3.0, zt=1.0, dz=dz_true, C=5.0)
                if sigma_r:
                    field = degrade(field, sigma_r)
                sub = sv.detrend(sv.centred(field, n))
                sp = sv.binned(sub, kmax=0.25)
                if L is None or L.shape[0] != sp[0].size:
                    L = toeplitz_chol(sp[0].size, RHO["hanning"])
                a, sa = fit_gls(sp, g, None)
                b, sb = fit_gls(sp, g, L)
                o.append(a[2]); gl.append(b[2])
                so.append(sa[2]); sg.append(sb[2])
            o, gl = np.array(o), np.array(gl)
            print(f"  {dz_true:8.0f} | {np.mean(o):9.2f} +/-{np.std(o, ddof=1):5.2f} "
                  f"{np.median(so):6.2f} | {np.mean(gl):9.2f} +/-{np.std(gl, ddof=1):5.2f} "
                  f"{np.median(sg):6.2f} | {np.mean(gl - o):+6.2f}")
        print()

    print("  sigma columns are (J^T J)^-1 in each metric, so they are directly")
    print("  comparable: the GLS one already carries the correlation.")


if __name__ == "__main__":
    main()
