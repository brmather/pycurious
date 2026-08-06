"""
Two experiments testing one diagnosis: WDMAM's `dz` tracks the window because
its high-`k` band cannot locate the layer rolloff.

`notes/dz-recoverability.md` found that on WDMAM L2 the median `dz` grows
monotonically with the window from 1500 km up -- 2.2x to 2.8x across the ladder,
in every depth band -- while on clean synthetics it converges. The proposed
cause is the `k_max` end of the band, not the `k_min` end:

    k_max * dz = pi dz / dx  >>  1     needed to reach the fractal half-space
                                       asymptote that locates the rolloff

WDMAM's effective `k_max` is reduced twice over: by an unmodelled ~4.2 km
resolution rolloff in the compilation, and by `SRC_KMAX = 0.25` deliberately
cutting the band there. On synthetics a layer whose `k_max*dz` is too small is
not merely noisy -- it is *worse the wider the window gets*, which is the
signature to look for.

**Experiment 1 (`--degrade`).** Take a synthetic of known `dz`, degrade it by
`exp(-k^2 sigma_r^2)`, and sweep the window. If the diagnosis holds, the
degraded synthetic reproduces the monotone growth and the undegraded one does
not. This is the controlled version of the L2 observation.

**Experiment 2 (`--kmax`).** Refit the cached WDMAM L2 spectra with the band cut
relaxed, and watch the drift. If the window dependence comes from the missing
high-`k` band, admitting more of it should reduce it -- at the cost of admitting
the artefact the cut exists to exclude, which is then a measurable trade rather
than an assumed one.

Run with the BLAS pinned::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/test_wdmam_diagnosis.py --degrade --kmax
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pycurious  # noqa: E402

WINDOWS = (1500e3, 2000e3, 2500e3, 3000e3, 4000e3, 6000e3, 8000e3, 10000e3)


def degrade(field, sigma_r_km, dx_km):
    """Blur by `exp(-k^2 sigma_r^2)`, the resolution loss of a compilation.

    The same operation `notes/bench/run_experiments.py` applies, written out
    here so this file stands alone. A real compilation's loss is patchy and
    direction dependent; this is the isotropic version of it, which is the one
    the model would have the best chance against.
    """
    n = field.shape[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx_km)
    ky = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx_km)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    return np.fft.irfft2(np.fft.rfft2(field) * np.exp(-k2 * sigma_r_km ** 2),
                         s=field.shape)


def production_fit():
    """`~/Global_CPD`'s own configuration, so the synthetic is fitted the way
    the map is.

    Reproducing "WDMAM" means more than a band cut. The cut at
    `kmax = 0.25` leaves the fit under-determined on its own -- with `zt` free
    and no `beta` prior a clean synthetic comes back at 350 km -- which is
    exactly why the production configuration carries a `zt` pin of 0.05 km, a
    `beta` prior of 0.15 and the `sigma` inflation at high `k`. All four
    together are the regime; any one of them alone is a different experiment.
    """
    sys.path.insert(0, str(Path.home() / "Global_CPD"))
    import curie_config as cfg

    return cfg, cfg.FitConfig(
        kmax=cfg.SRC_KMAX,
        sigma_weight_km=cfg.SIGMA_WEIGHT_KM,
        zt_fixed_km=1.0,
        zt_fix_sigma=cfg.ZT_FIX_SIGMA,
        zt_source="constant",
        beta_prior_sigma=cfg.BETA_PRIOR_SIGMA,
        taper="hanning",
        detrend=True,
        dataset="wdmam",
    )


def sweep(dz_true, sigma_r, nseed, n, dx, production, zt_true=1.0):
    """Median `dz` at each window, over independent realisations."""
    out = {w: [] for w in WINDOWS}
    cfg, fit = production
    for seed in range(nseed):
        field, extent = pycurious.fractal_anomaly(
            n=n, dx=dx, beta=3.0, zt=zt_true, dz=dz_true, C=5.0,
            seed=seed
        )
        if sigma_r:
            field = degrade(field, sigma_r, dx)
        grid = cfg.make_fitter(field, dx * 1e3, fit, beta_prior=3.0)
        for window in WINDOWS:
            try:
                out[window].append(grid.fit_window(window)[2])
            except Exception:
                continue
    return out


def show(label, dz_true, series):
    windows = [w for w in WINDOWS if series[w]]
    medians = [float(np.median(series[w])) for w in windows]
    print("  {:28s} |".format(label), end="")
    for value in medians:
        print(" {:7.2f}".format(value), end="")
    if len(medians) > 1:
        growth = medians[-1] / medians[0] if medians[0] > 0 else float("nan")
        rising = int(np.sum(np.diff(medians) > 0))
        print("  | x{:5.2f}  {}/{} rising".format(
            growth, rising, len(medians) - 1))
    else:
        print()
    return medians


def experiment_degrade(nseed, n, dx, thicknesses, sigmas, zt_truths):
    print("\n" + "=" * 78)
    print("1. Does a resolution rolloff make dz track the window?")
    print("=" * 78)
    production = production_fit()
    print("  {} realisations, n={}, dx={:.0f} km, fitted with ~/Global_CPD's"
          " own\n  configuration (kmax {:.2f}, zt pinned +/-{:.2f}, beta prior"
          " {:.2f}).".format(
              nseed, n, dx, production[1].kmax, production[1].zt_fix_sigma,
              production[1].beta_prior_sigma))
    print("  Median dz at each window; `x` is the widest over the narrowest.\n")
    header = "  {:28s} |".format("")
    for window in WINDOWS:
        header += " {:7.0f}".format(window / 1e3)
    print(header + "  | growth")

    for dz_true in thicknesses:
        for zt_true in zt_truths:
            print("  -- truth dz = {:.0f} km, truth zt = {:.1f} km (pinned at "
                  "1.0)".format(dz_true, zt_true))
            for sigma_r in sigmas:
                label = ("clean" if not sigma_r
                         else "degraded {:.1f} km".format(sigma_r))
                show(label, dz_true, sweep(dz_true, sigma_r, nseed, n, dx,
                                           production, zt_true))
            print()

    print("  A layer whose rolloff the band can reach should give a flat row.")
    print("  The L2 observation is a rise of 2.2x to 2.8x across this range.")


def experiment_kmax(cuts, limit):
    print("\n" + "=" * 78)
    print("2. Does relaxing the band cut reduce the drift on real data?")
    print("=" * 78)

    sys.path.insert(0, str(Path.home() / "Global_CPD"))
    import zarr
    from zarr.storage import LocalStore
    import curie_config as cfg

    root = zarr.open_group(
        store=LocalStore(str(Path.home() / "Global_CPD/data/curie_wdmam_L2.zarr")),
        mode="r",
    )
    fit = cfg.FitConfig.from_dict(dict(root.attrs["fit"]))
    dx = float(root.attrs["dx_m"])
    beta_prior = np.asarray(root["qc/beta_prior"][:])
    zt_pin = cfg.read_zt_pin(root, fit)
    spectra = zarr.open_group(
        store=LocalStore(
            str(Path.home() / "Global_CPD/data/spectra_wdmam_L2.zarr")),
        mode="r",
    )
    windows = np.asarray(root["window_km"][:])
    npoints = min(int(root.attrs["npoints"]), limit or 10 ** 9)

    print("  The cached spectra are already cut at kmax = {:.2f}, so this can "
          "only".format(fit.kmax))
    print("  *lower* it further -- raising it would need the spectra "
          "recomputed.")
    print("  Median dz over {} vertices at each window.\n".format(npoints))
    header = "  {:28s} |".format("kmax (rad/km)")
    report = [w for w in (10000, 6000, 4000, 2500, 1500) if w in windows]
    for window in report:
        header += " {:7.0f}".format(window)
    print(header + "  | growth")

    cls = cfg.band_limited_bouligand()
    for cut in cuts:
        medians = []
        for window_km in report:
            iw = int(np.argmin(np.abs(windows - window_km)))
            group = spectra["w{:d}".format(int(round(window_km)))]
            values = []
            for j in range(npoints):
                k = np.asarray(group["k"][:, j], dtype=float)
                Phi = np.asarray(group["Phi"][:, j], dtype=float)
                sigma = np.asarray(group["sigma"][:, j], dtype=float)
                good = np.isfinite(k) & np.isfinite(Phi) & np.isfinite(sigma)
                if cut is not None:
                    good &= k <= cut
                if good.sum() < 8:
                    continue
                bp = float(beta_prior[iw, j])
                grid = cls(np.zeros((2, 2)), 0.0, dx, 0.0, dx, fit=fit,
                           beta_prior=None if not np.isfinite(bp) else bp,
                           zt_pin=None if zt_pin is None
                           else float(zt_pin[iw, j]))
                try:
                    values.append(grid.optimise(
                        0.0, 0.0, 0.0, beta=grid._x0()[0], zt=grid._x0()[1],
                        spectrum=(k[good], Phi[good], sigma[good]),
                    )[2])
                except Exception:
                    continue
            medians.append(float(np.median(values)) if values else np.nan)

        label = "as archived ({:.2f})".format(fit.kmax) if cut is None \
            else "cut to {:.3f}".format(cut)
        print("  {:28s} |".format(label), end="")
        for value in medians:
            print(" {:7.2f}".format(value), end="")
        finite = [v for v in medians if np.isfinite(v)]
        if len(finite) > 1:
            print("  | x{:5.2f}".format(finite[0] / finite[-1]))
        else:
            print()

    print("\n  Growth is the 10,000 km median over the 1500 km one, so a value")
    print("  near 1 would mean dz had stopped depending on the window.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrade", action="store_true")
    parser.add_argument("--kmax", action="store_true")
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--n", type=int, default=2011)
    parser.add_argument("--dx", type=float, default=5.0)
    parser.add_argument("--dz", type=float, nargs="+", default=[20.0])
    parser.add_argument("--sigma-r", type=float, nargs="+",
                        default=[0.0, 4.2, 8.0])
    parser.add_argument("--zt-truth", type=float, nargs="+", default=[1.0],
                        help="true zt of the synthetic; the pin stays at 1.0, "
                             "so a value above it is a mis-specified pin")
    parser.add_argument("--cuts", type=float, nargs="+",
                        default=[0.10, 0.15, 0.25])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    if args.degrade:
        experiment_degrade(args.seeds, args.n, args.dx, args.dz,
                           args.sigma_r, args.zt_truth)
    if args.kmax:
        experiment_kmax([None] + list(args.cuts), args.limit)
