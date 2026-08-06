"""
How well is `dz` recovered, as a function of layer thickness and window size?

`dz` is the parameter that decides the Curie depth and the one the method
struggles with. This maps where it is recoverable at all, against the two things
that control it.

**The dimensionless number is `k_min * dz`.** The layer rolloff sits at
`|k| dz ~ 1`, and the longest wavelength a window measures is `k_min = 2 pi /
window`. So a layer is visible when its rolloff is inside the band, i.e. when

    k_min * dz  =  2 pi dz / window  <  1

and everything to the right of that is a window looking at a layer whose
thickness it cannot see. If recovery is really controlled by that ratio rather
than by thickness and window separately, the rows below should collapse onto it.

Also compares the diagonal objective against the whitened one
(`notes/bench/whitened.py`), because whitening's measured cost falls on `dz`
specifically -- and if that cost is concentrated where `dz` is unrecoverable
anyway, it is a different decision from one that falls everywhere.

Run with the BLAS pinned::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/score_dz_recovery.py --seeds 40
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pycurious  # noqa: E402
from whitened import WhitenedBouligand, numeric_jacobian  # noqa: E402

#: One grid per (thickness, seed) at production cell size, wide enough that
#: every window below is a subgrid of it -- so the synthetic is generated once
#: and the window sweep costs only the sub-spectra.
N, DX = 811, 5.0

WINDOWS = (500e3, 1000e3, 2000e3, 4000e3)
THICKNESS = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0)


def main(nseed, thicknesses, windows, cells):
    warnings.simplefilter("ignore")
    # both arms difference the Jacobian, so neither gets the exact columns the
    # other cannot have -- see `whitened.numeric_jacobian`
    with numeric_jacobian():
        for dx in cells:
            _sweep(nseed, thicknesses, windows, dx)


def _sweep(nseed, thicknesses, windows, dx):

    print("Recovering dz against thickness and window size. {} realisations.".format(
        nseed))
    print("k_min*dz = 2 pi dz / window is the rolloff against the longest")
    print("wavelength measured: below 1 the layer is inside the band, above it")
    print("the window cannot see the thickness at all.\n")
    print("  {:>7s} {:>4s} {:>6s} {:>8s} {:>8s} | {:>17s} {:>6s} | {:>17s} {:>6s}".format(
        "window", "dx", "dz", "kmin*dz", "kmax*dz", "diagonal", "|err|",
        "whitened", "|err|"))

    n = int(N * DX / dx)
    for dz_true in thicknesses:
        recovered = {(w, m): [] for w in windows
                     for m in ("diagonal", "whitened")}
        for seed in range(nseed):
            # one synthetic per (thickness, realisation); every window below is
            # a subgrid of it, so the sweep costs sub-spectra rather than a
            # fresh n^2 transform per cell of the table
            data, extent = pycurious.fractal_anomaly(
                n=n, dx=dx, beta=3.0, zt=1.0, dz=dz_true, C=5.0, seed=seed
            )
            for name, cls in (("diagonal", pycurious.CurieOptimiseBouligand),
                              ("whitened", WhitenedBouligand)):
                grid = cls(data, *extent)
                grid.reset_priors()
                xc, yc = grid.xcoords.mean(), grid.ycoords.mean()
                for window in windows:
                    try:
                        out = grid.optimise(window, xc, yc, taper=np.hanning)
                    except Exception:
                        continue
                    recovered[(window, name)].append(out[2])

        for window in windows:

            row = "  {:7.0f} {:4.0f} {:6.0f} {:8.3f} {:8.1f} |".format(
                window / 1e3, dx, dz_true,
                2.0 * np.pi * dz_true / (window / 1e3),
                np.pi * dz_true / dx,
            )
            for name in ("diagonal", "whitened"):
                values = np.array(recovered[(window, name)])
                if values.size < 2:
                    row += " {:>17s} {:>6s} |".format("-", "-")
                    continue
                row += " {:7.2f} +/-{:6.2f} {:6.1f} |".format(
                    values.mean(), values.std(ddof=1),
                    np.median(np.abs(values - dz_true)),
                )
            print(row)
        print()

    print("  `|err|` is the median absolute error, the honest summary for a")
    print("  quantity whose per-realisation distribution has a long tail.")
    print("  kmax*dz = pi dz / dx is the other end: the high-k asymptote needs")
    print("  the rolloff well inside the band from *above* too, which is a")
    print("  statement about cell size rather than about window size.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--dz", type=float, nargs="+", default=list(THICKNESS))
    parser.add_argument("--windows", type=float, nargs="+",
                        default=[w / 1e3 for w in WINDOWS])
    parser.add_argument("--cells", type=float, nargs="+", default=[5.0])
    args = parser.parse_args()
    main(args.seeds, args.dz, [w * 1e3 for w in args.windows], args.cells)
