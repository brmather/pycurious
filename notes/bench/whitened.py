"""
An experimental fitter with the between-bin correlation in the objective.

Not in the library. This is the change `notes/collapsed-posterior.md` measures
the cost of before deciding whether to make it: `_gls_covariance` corrects the
covariance for correlation between neighbouring radial bins and `min_func` does
not, so every interval read off the deviance or the posterior is too narrow by
about 1.3 while `optimise`'s sigma is right.

`R` is **tabulated per taper**, not estimated from the residuals in hand.
`_banded_correlation` reads smooth model mismatch as correlation -- correct for a
covariance at the solution, fatal in an objective, where it makes `r^T R(r)^-1 r`
a function the fit can lower by making its own residuals look correlated.
Measured, `rho_1` runs 0.24 to 0.42 depending on where in parameter space it is
asked.

The table is `notes/bench/calibrate_correlation.py`, measured against the mean
over independent realisations so no deterministic term survives.
"""

import contextlib

import numpy as np
from scipy.linalg import cholesky, solve_triangular

import pycurious
from pycurious import optimise_bouligand as ob


@contextlib.contextmanager
def numeric_jacobian():
    """Make `_fit` difference every column instead of using the exact two.

    `_ANALYTIC_COLUMNS` supplies `dr/dzt = -2k/sigma` and `dr/dC = 1/sigma`,
    which are exact for the **diagonal** residual and wrong for a whitened one:
    the true columns are `L^-1` times those. Handing `least_squares` a wrong
    Jacobian for half its parameters is not harmless here -- it sent a `dz = 10`
    layer at a 4000 km window to 60.9.

    A real implementation would whiten the columns. For an experiment, dropping
    them is contained and, applied to both arms, keeps the comparison fair: the
    columns are exact, so differencing them changes nothing for the diagonal
    fit beyond costing two more evaluations per Jacobian.
    """
    original = ob._ANALYTIC_COLUMNS
    ob._ANALYTIC_COLUMNS = {}
    try:
        yield
    finally:
        ob._ANALYTIC_COLUMNS = original

#: Lag correlations by taper name. Dead by lag 3 in every case, which is what
#: `pycurious.grid._CORRELATION_BANDS = 2` already assumes.
TAPER_RHO = {
    None: (1.0, 0.0, 0.0),
    "hanning": (1.0, 0.355, 0.032),
    "hamming": (1.0, 0.310, 0.020),
}


def whitener(nbin, rho):
    """Lower Cholesky factor of the stationary banded correlation."""
    R = np.eye(nbin)
    for lag, value in enumerate(rho[1:], 1):
        if lag < nbin and value != 0.0:
            i = np.arange(nbin - lag)
            R[i, i + lag] = R[i + lag, i] = value
    return cholesky(R, lower=True)


class WhitenedBouligand(pycurious.CurieOptimiseBouligand):
    """`CurieOptimiseBouligand` with the spectral block of the residual whitened.

    The prior rows are left at unit weight, exactly as `_gls_covariance` treats
    them. `taper_name` stands in for the argument `residuals` does not receive
    -- the real change would have to put it somewhere, and that is one of the
    open questions.
    """

    taper_name = "hanning"

    def __init__(self, *args, **kwargs):
        self.taper_name = kwargs.pop("taper_name", "hanning")
        super().__init__(*args, **kwargs)
        self._cache = {}

    def _whitener(self, nbin):
        if nbin not in self._cache:
            rho = TAPER_RHO.get(self.taper_name, TAPER_RHO[None])
            self._cache[nbin] = whitener(nbin, rho)
        return self._cache[nbin]

    def residuals(self, x, kh, Phi, sigma_Phi, prior=None):
        r = super().residuals(x, kh, Phi, sigma_Phi, prior)
        nbin = np.size(kh)
        if nbin < 2:
            return r
        L = self._whitener(nbin)
        return np.concatenate(
            [solve_triangular(L, r[:nbin], lower=True), r[nbin:]]
        )

    def _covariance(self, x, kh, Phi, sigma_Phi):
        """`(J^T J)^-1`, because the residual is already whitened.

        Leaving `_gls_covariance` here would apply the correction a second time.
        Measured, the two arrangements agree to 1.3-5.3%: the same uncertainty,
        reached by whitening the fit rather than by correcting the covariance.
        """
        args = (kh, Phi, sigma_Phi)
        r = self.residuals(x, *args)
        J = self._jacobian(x, r, args)
        try:
            return np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            return np.full((np.size(x), np.size(x)), np.nan)
