"""Pinning zt at the truth flatters the method. CRUST1.0 is not the truth.

Sweep the error in the pinned zt and find where the correction stops paying,
so the claim is "better if your zt is good to X km" rather than "better if you
already know the answer".
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
import curie_config as cfg
from pycurious import fractal_anomaly

DX = 5.0
BASE = cfg.FitConfig(dataset="wdmam", beta_prior_sigma=cfg.BETA_PRIOR_SIGMA)
ZT_TRUE = 4.5
ERRS = [-3.5, -2.0, -1.0, 0.0, 1.0, 2.0]     # -3.5 is the workflow's zt = 1.0


def one(job):
    window_km, dz_true, seed = job
    n = int(round(window_km / DX)) | 1
    data, _ = fractal_anomaly(n=n, dx=DX, beta=3.2, zt=ZT_TRUE, dz=dz_true,
                              C=5.0, seed=seed)
    out = []
    for e in ERRS:
        fit = replace(BASE, zt_fixed_km=max(ZT_TRUE + e, 0.1))
        g = cfg.make_fitter(data, DX * 1e3, fit, beta_prior=3.2)
        try:
            r = g.fit_window((n - 1) * DX * 1e3)
            out.append(r[1] + r[2] - (ZT_TRUE + dz_true))
        except Exception:
            out.append(np.nan)
    return out


if __name__ == "__main__":
    jobs = [(w, dz, s) for w in (4000, 2000, 1000)
            for dz in (10.0, 25.0, 40.0) for s in range(8)]
    with Pool(15) as p:
        rows = np.array(p.map(one, jobs), dtype=float)
    print(f"truth zt = {ZT_TRUE} km; CPD error vs the error in the pinned zt\n")
    print(f"{'zt pinned at':>13} {'zt error':>9} {'median |CPD error|':>19} {'median bias':>12}")
    for i, e in enumerate(ERRS):
        v = rows[:, i]; v = v[np.isfinite(v)]
        tag = "  <- the workflow" if e == -3.5 else ""
        print(f"{ZT_TRUE+e:13.1f} {e:+9.1f} {np.median(np.abs(v)):19.2f} "
              f"{np.median(v):+12.2f}{tag}")
