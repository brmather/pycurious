"""
Per-bin effective counts for each (window size, taper), by Monte Carlo.

``_TAPER_DOF`` parameterises this as ``(counts - lost) / dof_inf``, which fits
``hanning`` well and a multitaper badly -- the measured ``n_eff / counts`` for a
DPSS stack is not monotone in ``counts``, because what decides it is how the
annulus width compares with the concentration bandwidth ``NW dk``, not how many
cells the annulus holds. So the table is stored per bin rather than fitted.

Measured on synthetics drawn from the model being fitted, which is the same
assumption ``_TAPER_DOF`` makes. Leakage means the correlation between cells
depends slightly on the spectral slope, so this is calibrated for a red
spectrum (``beta = 3``) and would want re-measuring for a very different one.
"""

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spectral_variants as sv  # noqa: E402
from calibrate_dof import bin_index  # noqa: E402

OUT = HERE / "wdmam_L2" / "neff"
NSEED = 400
TRUTH = dict(beta=3.0, zt=1.0, dz=20.0, C=5.0)

CONFIGS = {
    "hanning": ("hanning", None, None),
    "dpss_NW2_K3": ("dpss", 2.0, 3),
    "dpss_NW3_K5": ("dpss", 3.0, 5),
}


def _one(job):
    n, taper, NW, K, seed = job
    keep, idx, w, counts, kbar, nb = bin_index(n)
    sub = sv.detrend(sv.synth(n, seed, **TRUTH))
    P, _ = sv.cell_power(sub, taper, NW or 3.0, K)
    with np.errstate(divide="ignore"):
        rr = np.log(P).ravel()[keep]
    return np.bincount(idx, weights=w * rr, minlength=nb) / counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    for n in (201, 401, 801):
        _, _, _, counts, kbar, nb = bin_index(n)
        for name, (taper, NW, K) in CONFIGS.items():
            path = OUT / f"n{n}_{name}.npy"
            if path.exists():
                continue
            t0 = time.perf_counter()
            jobs = [(n, taper, NW, K, s) for s in range(NSEED)]
            with ctx.Pool(16) as pool:
                means = np.array(pool.map(_one, jobs, chunksize=2))
            n_eff = sv.SIGMA_LOG ** 2 / means.var(axis=0, ddof=1)
            np.save(path, np.vstack([kbar, counts, n_eff]))
            band = kbar < 0.05
            print(f"n={n:4d} {name:12s} {nb:4d} bins  "
                  f"n_eff/counts: dz band {np.mean(n_eff[band] / counts[band]):5.2f}  "
                  f"outer {np.mean(n_eff[~band] / counts[~band]):5.2f}  "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
