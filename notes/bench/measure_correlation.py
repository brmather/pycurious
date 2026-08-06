"""
The bin-to-bin correlation of the binned log spectrum, measured rather than
estimated from one realisation.

``_banded_correlation`` infers this from the residuals of the fit in hand: one
realisation, ~158 numbers, two off-diagonals. It has to, because it must work
for a taper nobody calibrated. But the quantity it is inferring is a *property
of the taper and the binning*, not of the data -- so it can be measured once,
to any precision, and looked up.

Two things worth knowing before building anything on it:

* how far the correlation actually reaches (`_CORRELATION_BANDS = 2` assumes
  about 1.4 bins), and whether it is stationary in ``k`` -- the inner bins are
  sparse and a taper's main lobe spans several of them, so they should be far
  more correlated than the outer ones, which a single banded ``rho`` cannot
  express;
* whether a single realisation's residuals recover it, since that is what the
  library actually uses.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spectral_variants as sv  # noqa: E402
from calibrate_dof import bin_index  # noqa: E402
from pycurious.grid import _banded_correlation, _CORRELATION_BANDS  # noqa: E402

DX_KM = 5.0
TRUTH = dict(beta=3.0, zt=1.0, dz=20.0, C=5.0)


def spectra(n, M, taper, NW=3.0, K=None, seed0=0):
    keep, idx, w, counts, kbar, nb = bin_index(n)
    out = np.empty((M, nb))
    for m in range(M):
        sub = sv.detrend(sv.synth(n, seed0 + m, **TRUTH))
        P, _ = sv.cell_power(sub, taper, NW, K)
        with np.errstate(divide="ignore"):
            rr = np.log(P).ravel()[keep]
        out[m] = np.bincount(idx, weights=w * rr, minlength=nb) / counts
    return out, kbar, counts


def main():
    n, M = 401, 600
    for taper, NW, K in (("hanning", None, None), ("dpss", 3.0, 5)):
        S, kbar, counts = spectra(n, M, taper, NW or 3.0, K)
        label = taper if NW is None else f"dpss NW={NW} K={K}"
        # correlation of the deviation of each bin from its own mean across
        # realisations: the noise correlation, with the model divided out
        D = S - S.mean(axis=0)
        R = np.corrcoef(D.T)

        print(f"\n=== {label}, n={n}, {M} realisations ===")
        print("mean correlation at lag L, by band -- a single banded rho "
              "assumes these are one number")
        print(f"  {'band':>14s} " + " ".join(f"{f'lag {L}':>8s}"
                                             for L in range(1, 7)))
        for lo, hi, name in ((0, 5, "bins 0-4"), (5, 15, "bins 5-14"),
                             (15, 40, "bins 15-39"), (40, 100, "bins 40-99"),
                             (100, 199, "bins 100+")):
            vals = []
            for L in range(1, 7):
                d = np.diagonal(R, offset=L)[lo:min(hi, len(R) - L)]
                vals.append(np.mean(d) if d.size else np.nan)
            print(f"  {name:>14s} " + " ".join(f"{v:8.3f}" for v in vals))

        # how much total off-diagonal weight is being dropped by truncating
        tot = np.array([np.mean(np.abs(np.diagonal(R, offset=L)))
                        for L in range(1, 12)])
        kept = tot[:_CORRELATION_BANDS].sum()
        print(f"  off-diagonal weight kept by _CORRELATION_BANDS="
              f"{_CORRELATION_BANDS}: {kept / tot.sum():.0%}")

        # what a single realisation's residuals recover, for comparison
        est = []
        for m in range(min(M, 200)):
            r = D[m] / D.std(axis=0)
            ab = _banded_correlation(r)
            est.append(ab[1:, 0])
        est = np.array(est)
        true_lag = [np.mean(np.diagonal(R, offset=L))
                    for L in range(1, _CORRELATION_BANDS + 1)]
        print(f"  _banded_correlation from one realisation: "
              + ", ".join(f"lag {L} {est[:, L - 1].mean():.3f} "
                          f"+/- {est[:, L - 1].std():.3f} (true {t:.3f})"
                          for L, t in enumerate(true_lag, 1)))

        np.save(HERE / "wdmam_L2" / f"corr_n{n}_{label.replace(' ', '_')}.npy", R)


if __name__ == "__main__":
    main()
