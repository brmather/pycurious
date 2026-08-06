"""Refit every cached WDMAM spectrum with zt taken from data, and score it.

Variants:
  base   zt pinned at 1.0 km              -- reproduces the published archive
  data   zt pinned at the CRUST1.0 window mean
  soft   zt centred there, sigma 1 km, so the spectrum can still object

beta keeps its published pass-A prior in every variant. That is defensible
rather than lazy: zt_scan.py measured beta moving by <0.13 over a zt pin
sweep of 1 -> 5 km, so the smoothed beta field is very nearly invariant to
this change and re-running pass A would move it inside its own prior width.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import numpy as np
from multiprocessing import Pool

import harness as H
import fitters as F
import score as SC
import zt_field

ZT_FLOOR = 0.1          # sources above the observation plane are not modellable
_G = {}


def _init(k, P, s, bp, ztd):
    _G.update(k=k, P=P, s=s, bp=bp, ztd=ztd)


def _one(j):
    kk, pp, ss = H.column(_G["k"], _G["P"], _G["s"], j)
    if kk.size < 8:
        return j, np.nan, np.nan, np.nan
    bp = _G["bp"][j]
    out = []
    for pin, sig in ((1.0, 0.05), (_G["ztd"][j], 0.05), (_G["ztd"][j], 1.0)):
        try:
            r = F.fit_bouligand(kk, pp, ss, bp, zt_pin=pin, zt_sig=sig)
            out.append(r.x[1] + r.x[2])
        except Exception:
            out.append(np.nan)
    return (j, *out)


def run_window(w, ztd, nproc=15):
    k, P, s, bp, valid = H.load_window(w)
    ztd = np.clip(ztd, ZT_FLOOR, None)
    idx = np.where(valid)[0]
    cpd = {n: np.full(bp.size, np.nan) for n in ("base", "data", "soft")}
    with Pool(nproc, initializer=_init, initargs=(k, P, s, bp, ztd)) as pool:
        for j, a, b, c in pool.imap_unordered(_one, idx, chunksize=32):
            cpd["base"][j], cpd["data"][j], cpd["soft"][j] = a, b, c
    return cpd


if __name__ == "__main__":
    windows = [int(x) for x in sys.argv[1:]] or H.WINDOWS
    zt_all = zt_field.load()
    results = {n: [] for n in ("base", "data", "soft")}
    store = {}
    for w in windows:
        iw = H.WINDOWS.index(w)
        cpd = run_window(w, zt_all[iw])
        store[w] = cpd
        for n in results:
            results[n].append((w, SC.score(cpd[n], w)))
        print(f"done {w} km", flush=True)

    np.savez_compressed("cpd_variants.npz",
                        **{f"{n}_{w}": store[w][n] for w in windows for n in store[w]})

    SC.table(results["base"], "A. zt pinned at 1.0 km  (the published method)")
    SC.table(results["data"], "B. zt pinned at the CRUST1.0 window mean")
    SC.table(results["soft"], "C. zt centred there, sigma 1 km")
