"""
How `dz` behaves against window size on real data, split deep against shallow.

The synthetic sweep (`score_dz_recovery.py`) says recovery is controlled by two
dimensionless numbers, not by thickness or window alone:

    k_min * dz = 2 pi dz / window     the rolloff against the longest
                                      wavelength measured -- a window statement
    k_max * dz = pi dz / dx           the rolloff against the shortest --
                                      a *cell size* statement

A thick layer needs a wide window; a thin one needs fine cells. Both ends have
to hold, and on WDMAM the cell size is fixed at 5 km, so the thin end is fixed
too.

This asks the same question of the archive, where there is no truth to compare
against. What stands in for it is **stability**: a `dz` that is being measured
settles as the window grows, and one that is not keeps drifting. Vertices are
split into terciles by their `dz` at a reference window, so "deep" and "shallow"
mean what the data say rather than what a map says.

Refits from the cached spectra, so no FFT runs.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/score_dz_L2.py
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "Global_CPD"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import zarr  # noqa: E402
from zarr.storage import LocalStore  # noqa: E402

import curie_config as cfg  # noqa: E402
from whitened import WhitenedBouligand, numeric_jacobian  # noqa: E402

ARCHIVE = Path.home() / "Global_CPD" / "data" / "curie_wdmam_L2.zarr"
SPECTRA = Path.home() / "Global_CPD" / "data" / "spectra_wdmam_L2.zarr"

#: Window whose `dz` defines deep, middle and shallow. Wide enough to resolve a
#: thick layer, narrow enough that most vertices have one.
REFERENCE_KM = 4000.0

REPORT = (10000, 6000, 4000, 2500, 1500, 1000, 750, 500, 250)


def load(spectra, window_km, vertex):
    group = spectra["w{:d}".format(int(round(window_km)))]
    k = np.asarray(group["k"][:, vertex], dtype=float)
    Phi = np.asarray(group["Phi"][:, vertex], dtype=float)
    sigma = np.asarray(group["sigma"][:, vertex], dtype=float)
    good = np.isfinite(k) & np.isfinite(Phi) & np.isfinite(sigma)
    return (k[good], Phi[good], sigma[good]) if good.sum() >= 8 else None


def refit(root, spectra, fit, dx, beta_prior, zt_pin, windows, cls, limit):
    """`dz` at every rung for every vertex, as an array (nwin, nvertex)."""
    npoints = int(root.attrs["npoints"])
    if limit:
        npoints = min(npoints, limit)
    out = np.full((windows.size, npoints), np.nan)

    for iw, window_km in enumerate(windows):
        for j in range(npoints):
            spectrum = load(spectra, window_km, j)
            if spectrum is None:
                continue
            pin = None if zt_pin is None else float(zt_pin[iw, j])
            bp = float(beta_prior[iw, j])
            grid = cls(np.zeros((2, 2)), 0.0, dx, 0.0, dx,
                       fit=fit, beta_prior=None if not np.isfinite(bp) else bp,
                       zt_pin=pin)
            b0, z0 = grid._x0()[0], grid._x0()[1]
            try:
                out[iw, j] = grid.optimise(
                    0.0, 0.0, 0.0, beta=b0, zt=z0, taper=fit.taper_fn,
                    spectrum=spectrum,
                )[2]
            except Exception:
                continue
    return out


def report(label, windows, dz, terciles, dx_km):
    print("\n  {}".format(label))
    print("  {:>7s} {:>8s} | {:>22s} | {:>22s} | {:>22s}".format(
        "window", "kmin*dz", "shallow third", "middle third", "deep third"))
    print("  {:>7s} {:>8s} | {:>7s} {:>7s} {:>6s} | {:>7s} {:>7s} {:>6s} | "
          "{:>7s} {:>7s} {:>6s}".format(
              "km", "(deep)", "med", "IQR", "n", "med", "IQR", "n",
              "med", "IQR", "n"))

    deep_med = np.nanmedian(dz[:, terciles == 2])
    for want in REPORT:
        iw = int(np.argmin(np.abs(windows - want)))
        row = "  {:7.0f} {:8.3f} |".format(
            windows[iw], 2.0 * np.pi * deep_med / windows[iw]
        )
        for band in (0, 1, 2):
            values = dz[iw, terciles == band]
            values = values[np.isfinite(values)]
            if values.size < 3:
                row += " {:>7s} {:>7s} {:6d} |".format("-", "-", values.size)
                continue
            q1, q3 = np.percentile(values, [25, 75])
            row += " {:7.2f} {:7.2f} {:6d} |".format(
                np.median(values), q3 - q1, values.size
            )
        print(row)


def drift(windows, dz, terciles):
    """How much the median `dz` still moves between adjacent rungs.

    A thickness that is being measured settles as the window grows; one that is
    not keeps moving. Reported over the long half of the ladder, where a window
    large enough to resolve anything should have stopped changing its mind.
    """
    print("\n  Median |change in dz| between adjacent rungs, by band:")
    print("  {:>16s} | {:>12s} | {:>12s}".format(
        "rungs", "shallow", "deep"))
    for label, mask in (("10000-4000 km", windows >= 4000.0),
                        ("4000-1500 km", (windows < 4000.0) & (windows >= 1500.0)),
                        ("1500-250 km", windows < 1500.0)):
        row = "  {:>16s} |".format(label)
        for band in (0, 2):
            series = np.nanmedian(dz[:, terciles == band], axis=1)[mask]
            step = np.abs(np.diff(series))
            row += " {:12.3f} |".format(np.nanmedian(step) if step.size else np.nan)
        print(row)


def main(limit, whitened):
    warnings.simplefilter("ignore")

    root = zarr.open_group(store=LocalStore(str(ARCHIVE)), mode="r")
    fit = cfg.FitConfig.from_dict(dict(root.attrs["fit"]))
    dx = float(root.attrs["dx_m"])
    beta_prior = np.asarray(root["qc/beta_prior"][:])
    zt_pin = cfg.read_zt_pin(root, fit)
    spectra = zarr.open_group(store=LocalStore(str(SPECTRA)), mode="r")
    windows = np.asarray(root["window_km"][:])

    print("WDMAM L2, dz against window size, vertices split by depth.")
    print("Cell size is fixed at {:.0f} km, so kmax*dz = {:.1f} for a 30 km "
          "layer --".format(dx / 1e3, np.pi * 30.0 / (dx / 1e3)))
    print("the thin end of the band is the same at every rung; only the wide")
    print("end moves.")

    base = cfg.band_limited_bouligand()
    arms = [("diagonal (as archived)", base)]
    if whitened:
        arms.append(("whitened objective", _whitened_band_limited(base)))

    reference = None
    for label, cls in arms:
        with numeric_jacobian():
            dz = refit(root, spectra, fit, dx, beta_prior, zt_pin, windows,
                       cls, limit)
        if reference is None:
            iw = int(np.argmin(np.abs(windows - REFERENCE_KM)))
            anchor = dz[iw]
            finite = np.isfinite(anchor)
            terciles = np.full(anchor.size, -1)
            if finite.sum() >= 6:
                cuts = np.percentile(anchor[finite], [33.3, 66.7])
                terciles[finite] = np.digitize(anchor[finite], cuts)
            reference = terciles
            print("\n  split at dz = {:.2f} and {:.2f} km (the {:.0f} km rung)".format(
                cuts[0], cuts[1], REFERENCE_KM))
        report(label, windows, dz, reference, dx / 1e3)
        drift(windows, dz, reference)


def _whitened_band_limited(base):
    """`~/Global_CPD`'s band-limited subclass with the whitened objective."""

    class Both(WhitenedBouligand, base):
        pass

    return Both


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--whitened", action="store_true")
    args = parser.parse_args()
    main(args.limit, args.whitened)
