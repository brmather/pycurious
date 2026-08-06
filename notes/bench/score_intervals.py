"""
Do the intervals mean what they say? Deviance scan against posterior mesh.

An interval is calibrated when a nominal 68.27% interval contains the truth on
68.27% of independent realisations. That is the only question that can decide
between the two constructions, because they are genuinely different objects and
neither is an approximation to the other:

* `profile(method="scan")` returns a **profile-deviance** interval -- the set
  where the likelihood ratio stays within a chi-squared threshold. It is
  centred on the mode.
* `profile(method="mesh")` returns an **equal-tailed credible** interval from
  the marginal posterior, with `C` and `z_t` integrated out exactly. It is
  centred on the median.

On a Gaussian the two coincide. On `dz`, whose posterior has a long upper tail,
the credible interval sits higher and is wider -- so one of them is
under-covering and the other is not, and this says which.

There is a documented number to replace here. `optimise`'s docstring records
that `sigma_dz` "understates the true spread by about 40%", measured over 200
synthetics. That is the *Gaussian* sigma; this asks the same question of both
interval constructions.

Run with the BLAS pinned::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/score_intervals.py --seeds 200
"""

import argparse
import warnings

import numpy as np
from scipy import stats

import pycurious

#: (label, n, dx_km, window_m, beta, zt, dz). Two window sizes at one layer, so
#: the comparison spans a well-determined `dz` and a poorly determined one.
CASES = (
    ("1000 km, dz 20", 512, 2.0, 1000e3, 3.0, 1.0, 20.0),
    ("4000 km, dz 30", 811, 5.0, 4000e3, 3.0, 1.0, 30.0),
    ("400 km, dz 20", 512, 2.0, 400e3, 3.0, 1.0, 20.0),
)

LEVELS = (0.6827, 0.95)
TARGETS = ("beta", "dz", "CPD")


def truth_of(target, beta, zt, dz):
    return {"beta": beta, "zt": zt, "dz": dz, "CPD": zt + dz}[target]


def wilson(hits, n, level=0.95):
    """Binomial confidence interval on a coverage rate.

    A coverage of 0.63 from 200 trials is 0.56 to 0.69, so quoting the point
    estimate alone would make a calibrated interval look broken and a broken one
    look calibrated. Wilson rather than normal because the rates here run close
    to 1.
    """
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(0.5 * (1.0 + level))
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


METHODS = ("scan", "mesh")


def main(nseed, cases, calibrate=True):
    warnings.simplefilter("ignore")

    print("Interval coverage: how often a nominal interval contains the truth.")
    print("`scan` is the profile-deviance interval, `mesh` the equal-tailed")
    print("credible interval from the posterior. {} realisations per cell,".format(nseed))
    print("likelihood correction {}.".format("ON" if calibrate else "OFF"))
    print("95% Wilson bounds in brackets.\n")

    last_error = [None]

    for label, n, dx, window, beta, zt, dz in cases:
        counts = {(t, l, m): 0 for t in TARGETS for l in LEVELS
                  for m in METHODS}
        widths = {(t, l, m): [] for t in TARGETS for l in LEVELS
                  for m in METHODS}
        unbounded = {(t, l, m): 0 for t in TARGETS for l in LEVELS
                     for m in METHODS}
        failures = {(t, l, m): 0 for t in TARGETS for l in LEVELS
                    for m in METHODS}
        used = 0

        for seed in range(nseed):
            data, extent = pycurious.fractal_anomaly(
                n=n, dx=dx, beta=beta, zt=zt, dz=dz, C=5.0, seed=seed
            )
            grid = pycurious.CurieOptimiseBouligand(data, *extent)
            xc, yc = grid.xcoords.mean(), grid.ycoords.mean()
            spectrum = grid.window_spectrum(
                window, xc, yc, taper=np.hanning, power=2.0
            )

            # one density serves every target and every level
            shared = grid.posterior(
                window, xc, yc, calibrate=calibrate, spectrum=spectrum
            )

            for level in LEVELS:
                for target in TARGETS:
                    true = truth_of(target, beta, zt, dz)
                    for method in METHODS:
                        # A bare `except Exception: continue` here used to
                        # swallow the whole arm. When `profile` lost its
                        # `method=` keyword every mesh call raised TypeError,
                        # every mesh row silently emptied, and the table came
                        # out looking like a coverage result with one method
                        # quietly absent. Count what failed and say so.
                        try:
                            if method == "scan":
                                out = grid.profile(
                                    window, xc, yc, target, level=level,
                                    calibrate=calibrate, spectrum=spectrum,
                                )
                                lo, hi = out[2], out[3]
                            else:
                                lo, hi = shared.interval(target, level=level)
                        except Exception as error:
                            failures[(target, level, method)] += 1
                            last_error[0] = "{}: {}".format(
                                type(error).__name__, error
                            )
                            continue
                        key = (target, level, method)
                        if not np.isfinite(hi) or not np.isfinite(lo):
                            unbounded[key] += 1
                        else:
                            widths[key].append(hi - lo)
                        if lo <= true <= hi:
                            counts[key] += 1
            used += 1

        print("  {}  (truth: beta {:.1f}, dz {:.0f}, CPD {:.0f})".format(
            label, beta, dz, zt + dz))
        print("  {:>6s} {:>8s} | {:>26s} | {:>26s}".format(
            "target", "nominal", "scan", "mesh"))
        for target in TARGETS:
            for level in LEVELS:
                row = "  {:>6s} {:8.4f} |".format(target, level)
                for method in METHODS:
                    key = (target, level, method)
                    hits = counts[key]
                    lo, hi = wilson(hits, used)
                    width = (np.median(widths[key]) if widths[key] else np.nan)
                    row += " {:6.3f} [{:.2f},{:.2f}] w{:6.2f} {:2d}inf |".format(
                        hits / used, lo, hi, width, unbounded[key]
                    )
                print(row)

        broken = sum(failures.values())
        if broken:
            print("  {} of {} interval calls raised and were not counted; "
                  "last: {}".format(
                      broken, len(failures) * used, last_error[0]))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--case", type=int, default=None,
                        help="index into CASES, default all")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="read both intervals off the uncorrected "
                             "likelihood, as pre-v2 did")
    args = parser.parse_args()
    chosen = CASES if args.case is None else (CASES[args.case],)
    main(args.seeds, chosen, calibrate=not args.no_calibrate)
