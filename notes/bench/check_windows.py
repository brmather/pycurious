"""
Does the cached window reproduce the spectrum the L2 archive was fitted from?

``spectra_wdmam_L2.zarr`` holds pass A's own ``(k, Phi, sigma)`` per vertex per
window, stored at float32. If the spectrum computed from ``windows_L2.npy``
matches it, then the whole chain behind the cache -- upsample, projection,
sampling, detrend, taper, binning, band cut, sigma inflation -- is the one the
published archive used, and every experiment downstream is a like-for-like
comparison rather than a different workflow that happens to use the same data.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import LocalStore

CPD = Path.home() / "Global_CPD"
sys.path.insert(0, str(CPD))
import curie_config as cfg  # noqa: E402

HERE = Path(__file__).resolve().parent / "wdmam_L2"
DX_M = 5000.0

FIT = cfg.FitConfig(kmax=0.25, sigma_weight_km=8.0, zt_fixed_km=1.0, zt_fix_sigma=0.05,
                    zt_source="constant", taper="hanning", detrend=True,
                    projection="laea", field="iso", upsample=4, dataset="wdmam")


def centred(arr, cells):
    c = (arr.shape[-1] - 1) // 2
    h = cells // 2
    return arr[..., c - h:c + h + 1, c - h:c + h + 1]


def main():
    win = np.load(HERE / "windows_L2.npy", mmap_mode="r")
    sp = zarr.open_group(store=LocalStore(str(CPD / "data/spectra_wdmam_L2.zarr")), mode="r")

    for w_km in (1000, 2000, 4000):
        cells = int(w_km * 1e3 / DX_M)
        K = np.asarray(sp[f"w{w_km}/k"][:])
        P = np.asarray(sp[f"w{w_km}/Phi"][:])
        S = np.asarray(sp[f"w{w_km}/sigma"][:])
        dk = dP = dS = 0.0
        for j in (0, 37, 80, 161):
            sub = np.asarray(centred(win[j], cells), dtype=np.float64)
            g = cfg.make_fitter(sub, DX_M, FIT)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                k, Phi, sig = g._spectrum(w_km * 1e3, 0.0, 0.0, FIT.taper_fn,
                                          g.remove_trend_linear, None)
            n = k.size
            dk = max(dk, np.abs(k - K[:n, j]).max())
            dP = max(dP, np.abs(Phi - P[:n, j]).max())
            dS = max(dS, np.abs(sig - S[:n, j]).max())
        print(f"{w_km:5d} km  {n:4d} bins   max |dk| {dk:.2e}   "
              f"max |dPhi| {dP:.2e}   max |dsigma| {dS:.2e}")


if __name__ == "__main__":
    main()
