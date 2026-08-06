"""
What breaks if `R^-1` goes into the objective?

The intervals under-cover because `_gls_covariance` corrects for correlation
between neighbouring bins and `min_func` does not
(`notes/collapsed-posterior.md`). Putting `R^-1` in the objective would make the
two consistent -- but the objective is used for six other things, and this asks
what each of them does about it *before* anything is changed.

`R` is tabulated per taper rather than estimated from the residuals in hand, and
that is not a detail. `_banded_correlation` reads smooth model mismatch as
correlation, which is exactly what you want for a covariance evaluated *at the
solution* and fatal in an objective: measured over one spectrum, `rho_1` runs
0.38 at the fitted point, 0.24 at `beta = 2`, 0.42 at `dz + 50%`. An objective
`r^T R(r)^-1 r` is then not a fixed function of the parameters, its Jacobian is
wrong, and the fit can lower it by making its own residuals look correlated.

Run with the BLAS pinned::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/check_gls_objective.py
"""

import argparse
import warnings

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares

import pycurious
from pycurious.grid import _banded_correlation, _gls_covariance

#: Measured lag profiles (`notes/bench/measure_correlation.py`), stationary in
#: `k`. Only hanning and untapered have been measured; `hamming` is in
#: `_TAPER_DOF` and would need the same treatment before this could ship.
RHO = {"hanning": [1.0, 0.364, 0.039], None: [1.0, 0.003, 0.0]}


def whitener(nbin, rho):
    """Lower Cholesky factor of the stationary banded correlation."""
    R = np.eye(nbin)
    for lag, r in enumerate(rho[1:], 1):
        if lag < nbin and r != 0.0:
            i = np.arange(nbin - lag)
            R[i, i + lag] = R[i + lag, i] = r
    return cholesky(R, lower=True)


def fit(grid, spectrum, L=None):
    """`_fit`, optionally with the spectral block whitened. Prior rows are left
    at unit weight, exactly as `_gls_covariance` treats them."""
    k = spectrum[0]
    lower, upper = grid._bound_arrays()

    def residual(x):
        r = grid.residuals(x, *spectrum)
        if L is None:
            return r
        return np.concatenate(
            [solve_triangular(L, r[: k.size], lower=True), r[k.size:]]
        )

    x0 = np.clip(grid._initial_guess(spectrum), lower, upper)
    out = least_squares(residual, x0, bounds=(lower, upper))
    return out.x, out


def numeric_jacobian(grid, x, spectrum, L):
    """Central differences of the (possibly whitened) residual."""
    k = spectrum[0]

    def residual(y):
        r = grid.residuals(y, *spectrum)
        if L is None:
            return r
        return np.concatenate(
            [solve_triangular(L, r[: k.size], lower=True), r[k.size:]]
        )

    base = residual(x)
    J = np.empty((base.size, x.size))
    for i in range(x.size):
        h = 1e-6 * max(abs(x[i]), 1.0)
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        J[:, i] = (residual(up) - residual(down)) / (2.0 * h)
    return J, base


def main(nseed, window, n, dx, dz_true):
    warnings.simplefilter("ignore")

    shifts, chi2, sigmas, identity = [], [], [], []
    L = None

    for seed in range(nseed):
        data, extent = pycurious.fractal_anomaly(
            n=n, dx=dx, beta=3.0, zt=1.0, dz=dz_true, C=5.0, seed=seed
        )
        grid = pycurious.CurieOptimiseBouligand(data, *extent)
        grid.reset_priors()
        xc, yc = grid.xcoords.mean(), grid.ycoords.mean()
        spectrum = grid.window_spectrum(window, xc, yc, taper=np.hanning, power=2.0)
        nbin = spectrum[0].size
        if L is None or L.shape[0] != nbin:
            L = whitener(nbin, RHO["hanning"])

        ols, _ = fit(grid, spectrum, None)
        gls, _ = fit(grid, spectrum, L)
        shifts.append(gls - ols)

        # chi-squared, in each metric, at that metric's own solution
        r_ols = grid.residuals(ols, *spectrum)
        r_gls = solve_triangular(L, grid.residuals(gls, *spectrum)[:nbin], lower=True)
        chi2.append([
            float(np.sum(r_ols ** 2) / (nbin - 4)),
            float(np.sum(r_gls ** 2) / (nbin - 4)),
        ])

        # three sigmas that ought to be the same thing:
        #   what pycurious reports now  -- GLS covariance on an OLS fit
        #   naive on the whitened fit   -- what it would report after the change
        #   naive on the OLS fit        -- the uncorrected one
        J_ols, res_ols = numeric_jacobian(grid, ols, spectrum, None)
        J_gls, _ = numeric_jacobian(grid, gls, spectrum, L)
        now = np.sqrt(np.diag(_gls_covariance(J_ols, res_ols, nbin)))
        after = np.sqrt(np.diag(np.linalg.inv(J_gls.T @ J_gls)))
        naive = np.sqrt(np.diag(np.linalg.inv(J_ols.T @ J_ols)))
        sigmas.append([now, after, naive])

        # does the variable-projection identity survive? `A = G^T R^-1 G` must
        # still not depend on (beta, dz), or `posterior()` and `_initial_guess`
        # lose the fact they are built on
        weight = 1.0 / spectrum[2]
        G = np.column_stack([weight, -2.0 * spectrum[0] * weight])
        Gw = solve_triangular(L, G, lower=True)
        identity.append(Gw.T @ Gw)

    shifts = np.array(shifts)
    chi2 = np.array(chi2)
    sigmas = np.array(sigmas)

    print("Putting R^-1 in the objective: what moves. {} seeds, {:.0f} km "
          "window,\ndz {:.0f}, hanning, rho = {}.\n".format(
              nseed, window / 1e3, dz_true, RHO["hanning"]))

    print("1. The point estimate moves.")
    print("   {:>6s} {:>12s} {:>12s}".format("", "mean shift", "max |shift|"))
    for i, name in enumerate(("beta", "zt", "dz", "C")):
        print("   {:>6s} {:12.4f} {:12.4f}".format(
            name, shifts[:, i].mean(), np.abs(shifts[:, i]).max()))

    print("\n2. Reduced chi-squared changes, so every stored value does.")
    print("   OLS metric {:.4f} +/- {:.4f}   GLS metric {:.4f} +/- {:.4f}".format(
        chi2[:, 0].mean(), chi2[:, 0].std(ddof=1),
        chi2[:, 1].mean(), chi2[:, 1].std(ddof=1)))
    print("   (`test_reduced_chi_squared_is_about_one` bounds this at 0.7 to 1.6;")
    print("    ~/Global_CPD stores it as qc/chi2_reduced)")

    print("\n3. `_covariance` would double-count and has to change with it.")
    print("   {:>6s} {:>14s} {:>14s} {:>14s} {:>10s}".format(
        "", "now (GLS/OLS)", "after (naive/GLS)", "naive/OLS", "now/after"))
    for i, name in enumerate(("beta", "zt", "dz", "C")):
        now, after, naive = (sigmas[:, 0, i].mean(), sigmas[:, 1, i].mean(),
                             sigmas[:, 2, i].mean())
        print("   {:>6s} {:14.4f} {:14.4f} {:14.4f} {:10.3f}".format(
            name, now, after, naive, now / after))
    print("   If they agree, the change is consistent: the same uncertainty,")
    print("   reached by whitening the fit instead of correcting the covariance.")

    A = np.array(identity)
    spread = np.abs(A - A[0]).max()
    print("\n4. The variable-projection identity survives.")
    print("   G^T R^-1 G varies across realisations (different spectra) but is")
    print("   a constant of (beta, dz) within one, which is what `posterior()`")
    print("   and `_initial_guess` need. Off-diagonal/diagonal: {}".format(
        np.array2string(A[0], precision=3)))
    print("   Across seeds it changes by {:.3e}, which is the spectrum "
          "changing, not the parameters.".format(spread))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--window", type=float, default=1000e3)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--dx", type=float, default=2.0)
    parser.add_argument("--dz", type=float, default=20.0)
    args = parser.parse_args()
    main(args.seeds, args.window, args.n, args.dx, args.dz)
