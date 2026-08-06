"""
The derived start against the WDMAM L2 archive: cost, and geological realism.

`~/Global_CPD/compare_archives.py` already diffs two archives rung by rung.
This adds the two things it does not measure:

* **Cost.** Refit the cached L2 spectra from the constant start and from the
  derived one, counting evaluations of `bouligand2009`. The spectra are read
  from zarr, so no FFT runs and the number is the fit alone.
* **Geological realism, in absolute terms** rather than as a difference:
  continent against ocean, craton against ridge, and the seafloor-age
  correlation both raw and controlling for `zt`. The partial is the one to
  read -- age and water depth correlate at rho = +0.72, and water depth enters
  CPD through the pinned `zt`, so a raw age correlation moves when nothing
  geological has.

Run with the BLAS pinned, from anywhere::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/score_wdmam_L2.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "Global_CPD"))

import zarr  # noqa: E402
from zarr.storage import LocalStore  # noqa: E402

import curie_config as cfg  # noqa: E402
from pycurious import optimise_bouligand as ob  # noqa: E402

ARCHIVE = Path.home() / "Global_CPD" / "data" / "curie_wdmam_L2.zarr"
SPECTRA = Path.home() / "Global_CPD" / "data" / "spectra_wdmam_L2.zarr"

#: The start `optimise` used before it derived one.
CONSTANT = dict(beta=3.0, zt=None, dz=10.0, C=5.0)

WINDOWS = (10000, 4000, 2000, 1000, 500)


class Counter:
    def __enter__(self):
        self.calls = 0
        self._real = ob.bouligand2009

        def counted(*args):
            self.calls += 1
            return self._real(*args)

        ob.bouligand2009 = counted
        return self

    def __exit__(self, *exc):
        ob.bouligand2009 = self._real
        return False

    def take(self):
        calls, self.calls = self.calls, 0
        return calls


def cost_report(root, spectra, fit, dx, beta_prior, zt_pin):
    print("Refitting the cached L2 spectra: evaluations of bouligand2009 per fit,")
    print("and how often each start reaches the lower misfit.\n")
    print("  {:>7s} {:>5s} {:>6s} | {:>9s} {:>9s} | {:>9s} {:>9s} {:>7s}".format(
        "window", "bins", "n", "const fit", "deriv fit", "deriv seed",
        "deriv tot", "lower"))

    windows = np.asarray(root["window_km"][:])

    for want in WINDOWS:
        iw = int(np.argmin(np.abs(windows - want)))
        group = spectra["w{:d}".format(int(round(float(windows[iw]))))]
        k_all = np.asarray(group["k"][:])
        P_all = np.asarray(group["Phi"][:])
        S_all = np.asarray(group["sigma"][:])

        totals = {"const": [], "seed": [], "fit": []}
        lower = 0
        used = 0

        for j in range(k_all.shape[1]):
            k, P, S = k_all[:, j], P_all[:, j], S_all[:, j]
            good = np.isfinite(k) & np.isfinite(P) & np.isfinite(S)
            if good.sum() < 8:
                continue
            spectrum = (k[good].astype(float), P[good].astype(float),
                        S[good].astype(float))

            pin = None if zt_pin is None else float(zt_pin[iw, j])
            bp = float(beta_prior[iw, j])
            g = cfg.refit_fitter(fit, dx, None if not np.isfinite(bp) else bp,
                                 zt_pin=pin)

            # exactly what `fit_window` passes: the pin for `zt`, the prior
            # centre for `beta` in pass B, and None for the two that are
            # derived. Calling `_initial_guess` without them measures a path
            # production never takes.
            start_beta, start_zt = g._x0()[0], g._x0()[1]

            with Counter() as counter:
                a = g._fit(np.array([
                    CONSTANT["beta"] if start_beta is None else start_beta,
                    1.0 if start_zt is None else start_zt,
                    CONSTANT["dz"], CONSTANT["C"]]), spectrum)
                totals["const"].append(counter.take())

                x0 = g._initial_guess(spectrum, beta=start_beta, zt=start_zt)
                totals["seed"].append(counter.take())
                b = g._fit(x0, spectrum)
                totals["fit"].append(counter.take())

            lower += b.cost < a.cost * (1.0 - 1e-9)
            used += 1

        if not used:
            continue
        print("  {:7.0f} {:5d} {:6d} | {:9.1f} {:9.1f} | {:9.1f} {:9.1f} "
              "{:6.0f}%".format(
                  windows[iw], int(good.sum()), used,
                  np.mean(totals["const"]), np.mean(totals["fit"]),
                  np.mean(totals["seed"]),
                  np.mean(totals["seed"]) + np.mean(totals["fit"]),
                  100.0 * lower / used))


def geology_report(paths, label, root):
    """Continent vs ocean, craton vs ridge, and the age correlation."""
    from scipy import stats

    lon = np.asarray(root["mesh/lon"][:])
    lat = np.asarray(root["mesh/lat"][:])
    windows = np.asarray(root["window_km"][:])
    land = np.asarray(root["qc/land_fraction"][:])

    age = cfg.sample_age(paths, lon, lat) if cfg.paths_age_available(paths) else None

    print("\n{}".format(label))
    print("  {:>7s} | {:>8s} {:>8s} {:>7s} | {:>7s} {:>7s} {:>6s} | "
          "{:>8s} {:>8s}".format(
              "window", "ocean", "cont", "c - o", "craton", "ridge", "ratio",
              "rho age", "rho|zt"))

    for want in (10000, 6000, 4000, 2000, 1000):
        iw = int(np.argmin(np.abs(windows - want)))
        cpd = np.asarray(root["params/cpd"][iw])
        zt = cfg.read_param(root, "zt")[iw]

        ocean = land[iw] < cfg.OCEAN_MAX_LAND
        cont = land[iw] > cfg.CONT_MIN_LAND
        finite = np.isfinite(cpd)

        med_o = np.nanmedian(cpd[ocean & finite]) if (ocean & finite).any() else np.nan
        med_c = np.nanmedian(cpd[cont & finite]) if (cont & finite).any() else np.nan

        def site_median(kind):
            values = [
                cpd[cfg.nearest_vertex(lon, lat, site_lon, site_lat)]
                for _, site_lon, site_lat, k in cfg.TECTONIC_SITES if k == kind
            ]
            values = [v for v in values if np.isfinite(v)]
            return float(np.median(values)) if values else np.nan

        craton, ridge = site_median("craton"), site_median("ridge")

        rho = rho_zt = np.nan
        if age is not None:
            m = ocean & finite & np.isfinite(age)
            if m.sum() > 8:
                rho = stats.spearmanr(age[m], cpd[m]).statistic
                rho_zt = cfg.partial_spearman(age[m], cpd[m], zt[m])[0]

        print("  {:7.0f} | {:8.2f} {:8.2f} {:7.2f} | {:7.2f} {:7.2f} {:6.2f} | "
              "{:8.3f} {:8.3f}".format(
                  windows[iw], med_o, med_c, med_c - med_o, craton, ridge,
                  craton / ridge if ridge else np.nan, rho, rho_zt))


def main(new_archive):
    warnings.simplefilter("ignore")

    root = zarr.open_group(store=LocalStore(str(ARCHIVE)), mode="r")
    fit = cfg.FitConfig.from_dict(dict(root.attrs["fit"]))
    dx = float(root.attrs["dx_m"])
    beta_prior = np.asarray(root["qc/beta_prior"][:])
    zt_pin = cfg.read_zt_pin(root, fit)
    spectra = zarr.open_group(store=LocalStore(str(SPECTRA)), mode="r")

    cost_report(root, spectra, fit, dx, beta_prior, zt_pin)

    paths = cfg.Paths(Path.home() / "Global_CPD", "wdmam")
    print("\n" + "=" * 78)
    print("Geological realism, absolute. Continents and cratons should be deeper;")
    print("CPD should rise with seafloor age. Read `rho|zt`, not `rho age`: age and")
    print("water depth correlate at +0.72 and water depth enters CPD through zt.")
    print("=" * 78)

    geology_report(paths, "constant start (archived)", root)
    if new_archive:
        other = zarr.open_group(store=LocalStore(new_archive), mode="r")
        geology_report(paths, "derived start (refit)", other)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
