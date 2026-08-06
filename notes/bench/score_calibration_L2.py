"""
What the likelihood correction does to a real archive, per dataset and rung.

Refits the cached L2 spectra of `~/Global_CPD` twice -- `calibrate=False`, which
is what every archive before v2 was written with, and `calibrate=True`, the new
default -- holding everything else fixed. Same spectra, same priors, same `zt`
pins, same optimiser. The only difference is whether the likelihood is corrected
for correlation between neighbouring bins before an interval is read off it.

That isolation is the point. Comparing against a stored archive instead cannot
separate this change from whatever else moved since it was written: the WDMAM L2
archive was produced with `zt` pinned at a constant 1.0 km, where the current
default pins per vertex at the water depth, and `d(CPD)/d(zt)` runs -1.7 to -5.8.
Two settings apart is 16 km of Curie depth, which would swamp anything measured
here.

Two claims to check, and they are different in kind:

* **the point estimates must be bit-identical.** Tempering divides part of the
  misfit where the likelihood is *read*; nothing that is minimised changes. If
  `beta`, `zt`, `dz` or `C` moves at all, that is a bug, not a result.
* **the intervals must widen by about `sqrt(t2)`**, and more of them become
  unbounded at the short rungs where the data never constrained `dz` in the
  first place.

Run with the BLAS pinned::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/score_calibration_L2.py --workdir /path/to/scratch
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np


def load(workdir, dataset, refinement):
    """The archive, its cached spectra and the fit configuration behind them."""
    sys.path.insert(0, str(Path.home() / "Global_CPD"))
    import zarr
    from zarr.storage import LocalStore
    import curie_config as cfg

    data = Path(workdir) / "data"
    root = zarr.open_group(
        store=LocalStore(str(data / "curie_{}_L{}.zarr".format(dataset, refinement))),
        mode="r",
    )
    spectra = zarr.open_group(
        store=LocalStore(str(data / "spectra_{}_L{}.zarr".format(dataset, refinement))),
        mode="r",
    )
    return cfg, root, spectra


def main(workdir, datasets, refinement, limit, rungs):
    warnings.simplefilter("ignore")

    print("The likelihood correction on real spectra: same fit, same priors,")
    print("same pins -- only `calibrate` differs.\n")

    for dataset in datasets:
        try:
            cfg, root, spectra = load(workdir, dataset, refinement)
        except Exception as error:
            print("  {}: no archive ({})".format(dataset, error))
            continue

        fit = cfg.FitConfig.from_dict(dict(root.attrs["fit"]))
        dx = float(root.attrs["dx_m"])
        windows = np.asarray(root["window_km"][:])
        beta_prior = np.asarray(root["qc/beta_prior"][:])
        zt_pin = cfg.read_zt_pin(root, fit)
        npoints = min(int(root.attrs["npoints"]), limit or 10 ** 9)
        cls = cfg.band_limited_bouligand()

        report = [w for w in rungs if w in windows]
        print("  == {} L{}, {} vertices ==".format(dataset, refinement, npoints))
        print("  {:>6s} | {:>10s} {:>10s} | {:>8s} {:>7s} | {:>6s} {:>6s} | {:>6s}".format(
            "window", "max |dbeta|", "max |ddz|", "width x", "median", "inf off", "inf on",
            "med t"))

        for window_km in report:
            iw = int(np.argmin(np.abs(windows - window_km)))
            group = spectra["w{:d}".format(int(round(window_km)))]

            moved = {"beta": 0.0, "zt": 0.0, "dz": 0.0, "C": 0.0}
            ratios, tempers = [], []
            unbounded = {False: 0, True: 0}
            counted = 0

            for j in range(npoints):
                k = np.asarray(group["k"][:, j], dtype=float)
                Phi = np.asarray(group["Phi"][:, j], dtype=float)
                sigma = np.asarray(group["sigma"][:, j], dtype=float)
                good = np.isfinite(k) & np.isfinite(Phi) & np.isfinite(sigma)
                if good.sum() < 8:
                    continue

                bp = float(beta_prior[iw, j])
                grid = cls(
                    np.zeros((2, 2)), 0.0, dx, 0.0, dx, fit=fit,
                    beta_prior=None if not np.isfinite(bp) else bp,
                    zt_pin=None if zt_pin is None else float(zt_pin[iw, j]),
                )
                spectrum = (k[good], Phi[good], sigma[good])

                try:
                    x0 = grid._x0()
                    fitted = grid.optimise(
                        0.0, 0.0, 0.0, beta=x0[0], zt=x0[1],
                        spectrum=spectrum,
                    )
                except Exception:
                    continue

                width = {}
                for calibrate in (False, True):
                    try:
                        _, _, lo, hi = grid.profile(
                            0.0, 0.0, 0.0, "dz",
                            level=cfg.PROFILE_LEVEL,
                            npoints=cfg.PROFILE_NPOINTS,
                            beta=fitted[0], zt=fitted[1],
                            dz=fitted[2], C=fitted[3],
                            calibrate=calibrate,
                            spectrum=spectrum,
                        )
                    except Exception:
                        lo = hi = np.nan
                    if np.isfinite(lo) and np.isfinite(hi):
                        width[calibrate] = hi - lo
                    else:
                        unbounded[calibrate] += 1

                # the point estimate is the same object in both arms by
                # construction; refit once more to be sure nothing about the
                # `calibrate` path leaked into it
                second = grid.optimise(
                    0.0, 0.0, 0.0, beta=x0[0], zt=x0[1], spectrum=spectrum,
                )
                for name, a, b in zip(("beta", "zt", "dz", "C"), fitted, second):
                    moved[name] = max(moved[name], abs(float(a) - float(b)))

                if width.get(False, 0.0) > 0.0 and True in width:
                    ratios.append(width[True] / width[False])
                tempers.append(grid._temperature(np.asarray(fitted[:4]),
                                                 spectrum))
                counted += 1

            if not counted:
                continue
            print("  {:6.0f} | {:10.2e} {:10.2e} | {:8.3f} {:7.2f} | {:6d} {:6d} | {:6.3f}".format(
                window_km, moved["beta"], moved["dz"],
                float(np.median(ratios)) if ratios else np.nan,
                float(np.median(ratios)) if ratios else np.nan,
                unbounded[False], unbounded[True],
                float(np.median(tempers)) if tempers else np.nan,
            ))
        print()

    print("  `max |dbeta|` and `max |ddz|` are between two identical fits, so")
    print("  they measure the optimiser's own determinism and nothing else.")
    print("  Anything above 0 there would mean `calibrate` had reached the fit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["emag2", "wdmam"])
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rungs", type=float, nargs="+",
                        default=[10000, 6000, 4000, 2500, 1500, 1000, 500, 250])
    args = parser.parse_args()
    main(args.workdir, args.datasets, args.refinement, args.limit, args.rungs)
