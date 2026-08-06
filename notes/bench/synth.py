"""Known truth. Two questions the real-data tests cannot settle.

Q1  Does pinning zt at 1.0 km cost anything, when the truth is 4.5 km (the
    median CRUST1.0 value under an oceanic window -- mostly water column)?

Q2  The model predicts the log-log slope changes by exactly 2 across the layer
    rolloff. On the real archive the fitted gap is 2.04 at a 10,000 km window
    and 2.73 at 1,000 km. Is that breakdown a WINDOW ARTEFACT (taper, detrend,
    too few bins below the rolloff) or is it the fractal-layer model failing?
    Synthetics obey the model exactly, so if they reproduce the drift it is the
    window; and then fitting the gap free should de-bias dz.

Everything is held at the workflow's own settings by going through
curie_config's fitter, so the only thing that varies is what is asked.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import numpy as np
from dataclasses import replace
from multiprocessing import Pool

sys.path.insert(0, os.path.expanduser("~/Global_CPD"))
sys.path.insert(0, os.path.expanduser("~/git/pycurious"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curie_config as cfg              # noqa: E402
from pycurious import fractal_anomaly   # noqa: E402
import fitters as F                     # noqa: E402

DX = 5.0
BASE = cfg.FitConfig(dataset="wdmam", beta_prior_sigma=cfg.BETA_PRIOR_SIGMA)
BETA = 3.2


def one(job):
    window_km, zt_true, dz_true, seed = job
    n = int(round(window_km / DX)) | 1
    data, extent = fractal_anomaly(n=n, dx=DX, beta=BETA, zt=zt_true,
                                   dz=dz_true, C=5.0, seed=seed)
    out = {}
    spec = None
    for name, pin in (("pin@1.0", 1.0), ("pin@truth", zt_true)):
        fit = replace(BASE, zt_fixed_km=pin)
        g = cfg.make_fitter(data, DX * 1e3, fit, beta_prior=BETA)
        try:
            res = g.fit_window((n - 1) * DX * 1e3)
            out[name] = res[1] + res[2]
            if spec is None:
                spec = g._last_spectrum
        except Exception:
            out[name] = np.nan

    # the gap test, at the workflow's own pin
    out["gap"] = np.nan
    out["cpd_freegap"] = np.nan
    if spec is not None:
        k, Phi, sig = spec
        try:
            r = F.fit_free_gap(k, Phi, sig, BETA, zt_pin=1.0)
            out["gap"] = r.x[4]
            out["cpd_freegap"] = r.x[1] + r.x[2]
        except Exception:
            pass
    return window_km, zt_true, dz_true, seed, out


if __name__ == "__main__":
    jobs = [(w, zt, dz, s)
            for w in (4000, 2000, 1000)
            for zt in (1.0, 4.5)
            for dz in (10.0, 25.0, 40.0)
            for s in range(8)]

    with Pool(15) as p:
        rows = p.map(one, jobs)

    print("Q1. CPD recovery, median over 8 seeds (error in brackets)\n")
    print(f"{'window':>7} {'zt_true':>8} {'dz_true':>8} {'truth':>7} "
          f"{'pin@1.0':>17} {'pin@truth':>17}")
    agg = {}
    for w, ztt, dzt, s, out in rows:
        agg.setdefault((w, ztt, dzt), []).append(out)
    for (w, ztt, dzt), lst in sorted(agg.items()):
        truth = ztt + dzt
        cells = []
        for kk in ("pin@1.0", "pin@truth"):
            v = np.array([o[kk] for o in lst], dtype=float)
            v = v[np.isfinite(v)]
            cells.append(f"{np.median(v):6.1f} ({np.median(v)-truth:+6.1f})")
        print(f"{w:7d} {ztt:8.1f} {dzt:8.1f} {truth:7.1f} "
              f"{cells[0]:>17} {cells[1]:>17}")

    print("\n  median |error|, all cases:")
    for kk in ("pin@1.0", "pin@truth"):
        e = [abs(o[kk] - (ztt + dzt)) for w, ztt, dzt, s, o in rows
             if np.isfinite(o[kk])]
        print(f"    {kk:10s} {np.median(e):6.2f} km")
    print("  median |error|, realistic zt_true = 4.5 km only:")
    for kk in ("pin@1.0", "pin@truth"):
        e = [abs(o[kk] - (ztt + dzt)) for w, ztt, dzt, s, o in rows
             if np.isfinite(o[kk]) and ztt == 4.5]
        print(f"    {kk:10s} {np.median(e):6.2f} km")

    print("\n\nQ2. the fitted slope gap on synthetics that obey the model exactly")
    print("    (truth is 2.00 by construction; the real archive gives")
    print("     2.04 at 10,000 km, 2.24 at 4,000, 2.73 at 1,000)\n")
    print(f"{'window':>7} {'gap median':>11} {'IQR':>16}   "
          f"{'CPD gap=2':>10} {'CPD free':>9} {'truth':>7}")
    for w in (4000, 2000, 1000):
        sel = [(o, ztt + dzt) for ww, ztt, dzt, s, o in rows if ww == w]
        g = np.array([o["gap"] for o, _ in sel], dtype=float)
        g = g[np.isfinite(g)]
        e2 = np.array([o["pin@1.0"] - t for o, t in sel], dtype=float)
        ef = np.array([o["cpd_freegap"] - t for o, t in sel], dtype=float)
        q1, q3 = np.percentile(g, [25, 75])
        print(f"{w:7d} {np.median(g):11.3f}   [{q1:.2f}, {q3:.2f}]   "
              f"{np.nanmedian(np.abs(e2)):10.2f} {np.nanmedian(np.abs(ef)):9.2f}"
              f"   (median |error| km)")
