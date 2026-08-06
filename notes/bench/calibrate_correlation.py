"""
Correlation between neighbouring radial bins, per taper, from known truth.

`_gls_covariance` estimates this from the residuals of one fit, which is right
for a covariance evaluated at the solution and wrong for an objective: the same
estimator reads smooth model mismatch as correlation, so `rho_1` runs 0.24 to
0.42 depending on where in parameter space it is asked. An objective needs a
number that does not move, which means a table -- the same arrangement
`_TAPER_DOF` already uses for the per-bin deflation.

Measured the honest way: against the **true** model rather than a fitted one, so
a fit cannot absorb part of the correlation into its parameters, and across
independent realisations rather than along one spectrum.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/calibrate_correlation.py
"""

import argparse
import warnings

import numpy as np

import pycurious

TAPERS = (None, np.hanning, np.hamming)
TRUTH = dict(beta=3.0, zt=1.0, dz=20.0, C=5.0)


def lag_correlation(residuals, lags):
    """Correlation at each lag, from the scatter *across* realisations.

    Each row is one realisation. The reference is the mean over rows, per bin,
    not an analytic model: the log-periodogram carries a deterministic bias
    that no fit recovers -- the Euler-Mascheroni constant from log-averaging
    and the taper's power loss, both absorbed into `C` -- and `sigma` falls as
    `1/sqrt(k)`, so that bias divided by `sigma` is a strong smooth trend in
    `k`. Removing only each row's *mean* leaves the trend behind, and a
    lag correlation then reads it as 0.9 at every lag, untapered included.
    Subtracting the across-realisation mean removes anything deterministic,
    whatever its shape, and leaves the stochastic part this is about.

    Each bin is then standardised by its own across-realisation spread, so a
    bin count that grows with `k` does not weight the high-`k` end.
    """
    r = np.asarray(residuals, dtype=float)
    r = r - r.mean(axis=0, keepdims=True)
    spread = r.std(axis=0, keepdims=True, ddof=1)
    r = np.divide(r, spread, out=np.zeros_like(r), where=spread > 0)

    out = [1.0]
    for lag in range(1, lags + 1):
        out.append(float(np.mean(r[:, :-lag] * r[:, lag:])))
    return out


def main(nseed, n, dx, window, lags):
    warnings.simplefilter("ignore")

    print("Bin-to-bin correlation of the log spectrum about the TRUE model.")
    print("{} realisations, n={}, dx={:.0f} km, {:.0f} km window.\n".format(
        nseed, n, dx, window / 1e3))
    print("  {:>10s} {:>5s} | {}".format(
        "taper", "bins", "  ".join("rho_{}".format(i) for i in range(lags + 1))))

    for taper in TAPERS:
        rows = []
        for seed in range(nseed):
            data, extent = pycurious.fractal_anomaly(
                n=n, dx=dx, seed=seed, **TRUTH
            )
            grid = pycurious.CurieOptimiseBouligand(data, *extent)
            xc, yc = grid.xcoords.mean(), grid.ycoords.mean()
            k, Phi, sigma = grid.window_spectrum(
                window, xc, yc, taper=taper, power=2.0
            )
            # `Phi` itself: `lag_correlation` references it against the mean
            # over realisations, which removes every deterministic term without
            # needing to know what they are. Fitting first would let the fit
            # absorb part of the correlation into `beta` and `C`.
            good = np.isfinite(Phi) & (sigma > 0)
            rows.append(Phi[good])

        width = min(len(r) for r in rows)
        rho = lag_correlation([r[:width] for r in rows], lags)
        name = getattr(taper, "__name__", str(taper))
        print("  {:>10s} {:5d} | {}".format(
            name, width, "  ".join("{:6.3f}".format(v) for v in rho)))

    print("\n  A lag whose correlation is below ~0.02 is noise at this sample")
    print("  size; the band that matters is where it is clearly positive.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--dx", type=float, default=2.0)
    parser.add_argument("--window", type=float, default=1000e3)
    parser.add_argument("--lags", type=int, default=4)
    args = parser.parse_args()
    main(args.seeds, args.n, args.dx, args.window, args.lags)
