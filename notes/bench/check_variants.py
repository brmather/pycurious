"""
Three assertions the whole comparison rests on.

1. ``binned(taper="hanning", sigma="empirical", sigma_weight_km=8)`` is the
   spectrum the WDMAM L2 archive was fitted from. If it is not, nothing below
   is a comparison against the published result.

2. The empirical within-annulus scatter really does estimate ``pi/sqrt(6)``,
   the log-periodogram's own sd -- so ``sigma="theory"`` is not a different
   assumption, it is the same quantity known rather than measured.

3. ``binned(sigma="theory")`` and ``unbinned`` give the same fit. This is the
   sufficiency claim in ``spectral_variants``: binning a log-periodogram
   discards no information, provided the weight is ``s/sqrt(n)``.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spectral_variants as sv  # noqa: E402

sys.path.insert(0, str(Path.home() / "Global_CPD"))
import curie_config as cfg  # noqa: E402

DATA = HERE / "wdmam_L2"
DX_M = 5000.0
FIT_ARCHIVE = cfg.FitConfig(kmax=0.25, sigma_weight_km=8.0, zt_fixed_km=1.0,
                            zt_fix_sigma=0.05, zt_source="constant", taper="hanning",
                            detrend=True, projection="laea", field="iso",
                            upsample=4, dataset="wdmam")


def check_matches_archive(win):
    print("1. does `binned` reproduce the archive's own spectrum?")
    worst = np.zeros(3)
    for w_km in (1000, 2000, 4000):
        cells = int(w_km * 1e3 / DX_M)
        for j in (0, 37, 80, 161):
            sub = sv.centred(win[j], cells)
            g = cfg.make_fitter(sub, DX_M, FIT_ARCHIVE)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = g._spectrum(w_km * 1e3, 0.0, 0.0, np.hanning,
                                  g.remove_trend_linear, None)
            mine = sv.binned(sv.detrend(sub), taper="hanning", kmax=0.25,
                             sigma="empirical", sigma_weight_km=8.0)
            assert mine[0].size == ref[0].size, (mine[0].size, ref[0].size)
            worst = np.maximum(worst, [np.abs(a - b).max() for a, b in zip(mine, ref)])
    print(f"   max |dk| {worst[0]:.2e}   max |dPhi| {worst[1]:.2e}   "
          f"max |dsigma| {worst[2]:.2e}")
    assert worst.max() < 1e-9, "the binned variant is not the archive's spectrum"


def check_scatter_is_theory(n=401, nseed=60):
    print("\n2. is the within-annulus scatter pi/sqrt(6)?")
    print(f"   {nseed} synthetics, {n} cells; ratio of measured scatter to "
          f"{sv.SIGMA_LOG:.4f}")
    for taper in ("none", "hanning"):
        ratios = []
        for seed in range(nseed):
            sub = sv.detrend(sv.synth(n, seed))
            P, _ = sv.cell_power(sub, taper)
            dk, kbins = sv.bin_edges(n, DX_M / 1e3)
            kk = sv.cell_wavenumbers(n, n, dk).ravel()
            rr = np.log(P).ravel()
            idx = np.digitize(kk, kbins) - 1
            nb = kbins.size - 1
            keep = (idx >= 0) & (idx < nb)
            counts = np.bincount(idx[keep], minlength=nb).astype(float)
            S = np.bincount(idx[keep], weights=rr[keep], minlength=nb) / counts
            dev = rr[keep] - S[idx[keep]]
            sc = np.sqrt(np.bincount(idx[keep], weights=dev * dev, minlength=nb) / counts)
            # inner bins only have a handful of cells; report where it is
            # measurable, and separately in the band that carries dz
            ratios.append(sc / sv.SIGMA_LOG)
        r = np.array(ratios)
        kb = 0.5 * (kbins[:-1] + kbins[1:])
        lo = (kb < 0.05)
        print(f"   {taper:8s} all bins {np.nanmean(r):.3f}   "
              f"k < 0.05 (the dz band) {np.nanmean(r[:, lo]):.3f}   "
              f"innermost bin {np.nanmean(r[:, 0]):.3f}")


def check_sufficiency(n=401, nseed=8):
    print("\n3. binned(theory) vs unbinned -- same estimate?")
    print("       seed |    binned(theory)          |        unbinned            | "
          "bins  cells   t_bin   t_cell")
    for seed in range(nseed):
        sub = sv.detrend(sv.synth(n, seed, dz=20.0))
        g = sv.fitter()
        sb = sv.binned(sub, sigma="theory", kmax=0.25)
        su = sv.unbinned(sub, kmax=0.25)
        t0 = time.perf_counter()
        xb, _, eb = sv.fit(sb, g)
        t1 = time.perf_counter()
        xu, _, eu = sv.fit(su, g)
        t2 = time.perf_counter()
        print(f"   {seed:8d} | " + " ".join(f"{v:6.2f}" for v in xb) + " | "
              + " ".join(f"{v:6.2f}" for v in xu)
              + f" | {eb['nres']:4d} {eu['nres']:6d} "
              f" {t1 - t0:6.2f}s {t2 - t1:6.2f}s")


def main():
    win = np.load(DATA / "windows_L2.npy", mmap_mode="r")
    check_matches_archive(win)
    check_scatter_is_theory()
    check_sufficiency()


if __name__ == "__main__":
    main()
