"""
Score the variants: accuracy where truth is known, consistency where it is not.

On synthetics the questions are the usual two -- is the estimate right on
average, and how far does one realisation sit from it -- plus a third that
matters more here than it usually does: **is the reported uncertainty honest**.
``sigma_dz`` is what a downstream map is contoured against, and a variant that
buys a tighter ``dz`` by understating its error has bought nothing.

On WDMAM there is no truth, so the proxy is **nested-window disagreement**: the
same vertex fitted at 1000, 2000 and 4000 km should return the same Curie
depth, and the spread between rungs is a lower bound on systematic error that
no fit reports for itself.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "wdmam_L2"
ORDER = ["default", "archive", "kmax0.10", "tilt0.5", "tilt1.0", "unbinned",
         "hann-dof", "mt-NW2K3", "mt-NW3K5", "mt-naive", "mt+sw"]
CELLS = {201: 1000, 401: 2000, 801: 4000}


def load(source):
    d = np.load(DATA / f"results_{source}.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def synth_report():
    c = load("synth")
    nseed = len(set(c["seed"]))
    sigma_r = sorted(set(c["sigma_r"])) if "sigma_r" in c else [0.0]
    print("=" * 94)
    print(f"SYNTHETIC: {nseed} seeds per (window, dz), beta=3 zt=1 C=5, "
          f"zt pinned at 1.0 +/- 0.05")
    print("=" * 94)

    for sr in sigma_r:
        head = ("clean field" if not sr else
                f"degraded to {sr} km effective resolution -- WDMAM-like, and "
                f"a defect the model has no term for")
        print(f"\n\n### {head}")
        for n in sorted(set(c["n"])):
            print(f"\n  window {CELLS[int(n)]} km ({int(n)} cells)")
            print(f"  {'variant':10s} {'bias':>7s} {'median':>7s} {'MAD':>6s} "
                  f"{'RMS':>6s} {'|e|>25%':>8s} {'sigma_dz':>9s} {'true sd':>8s} "
                  f"{'ratio':>6s} {'cover':>6s} {'chi2':>5s} {'ms':>6s}")
            for v in ORDER:
                _row(c, v, (c["n"] == n) & (c["sigma_r"] == sr))
    print("\n  bias/median/MAD/RMS in km of dz; |e|>25% is the fraction of "
          "realisations that far out;\n  ratio is reported sigma over the true "
          "scatter (1.0 is honest); cover is the fraction\n  inside +/-1 sigma "
          "(0.68 is honest).")


def consistency_report():
    """Does nested-window agreement track accuracy? On WDMAM it is the only
    check available; here truth is known, so the proxy can be audited."""
    c = load("synth")
    print("\n" + "=" * 94)
    print("IS NESTED-WINDOW AGREEMENT A GOOD PROXY FOR ACCURACY?")
    print("=" * 94)
    print("  The same field fitted at 1000/2000/4000 km. On WDMAM the spread "
          "between rungs is all\n  we have; here it can be set beside the "
          "error it is standing in for.\n")
    for sr in sorted(set(c["sigma_r"])):
        print(f"  {'clean field' if not sr else f'degraded {sr} km'}")
        print(f"  {'variant':10s} {'window spread':>14s} {'|bias| 4000km':>14s} "
              f"{'RMS 4000km':>11s}")
        for v in ORDER:
            cols = []
            for n in (201, 401, 801):
                m = (c["variant"] == v) & (c["n"] == n) & (c["sigma_r"] == sr)
                o = np.lexsort((c["seed"][m], c["dz_true"][m]))
                cols.append((c["dz"][m] + c["zt"][m])[o])
            A = np.vstack(cols)
            m = (c["variant"] == v) & (c["n"] == 801) & (c["sigma_r"] == sr)
            err = c["dz"][m] - c["dz_true"][m]
            print(f"  {v:10s} {np.nanmedian(np.nanstd(A, axis=0, ddof=1)):14.2f} "
                  f"{abs(np.mean(err)):14.2f} {np.sqrt(np.mean(err ** 2)):11.2f}")
        print()


def _row(c, v, sel):
    m = (c["variant"] == v) & sel
    if not m.any():
        return
    if True:
        err = c["dz"][m] - c["dz_true"][m]
        rel = np.abs(err) / c["dz_true"][m]
        # the spread of the estimate about its own mean, per dz level, so a
        # bias does not inflate what is meant to be a scatter
        sd = np.mean([np.std(err[c["dz_true"][m] == t], ddof=1)
                      for t in sorted(set(c["dz_true"][m]))])
        sig = np.nanmedian(c["sigma_dz"][m])
        cover = np.mean(np.abs(err) < c["sigma_dz"][m])
        print(f"  {v:10s} {np.mean(err):+7.2f} {np.median(err):+7.2f} "
              f"{np.median(np.abs(err - np.median(err))):6.2f} "
              f"{np.sqrt(np.mean(err ** 2)):6.2f} {np.mean(rel > 0.25):8.2f} "
              f"{sig:9.2f} {sd:8.2f} {sig / sd:6.2f} {cover:6.2f} "
              f"{np.nanmedian(c['chi2_red'][m]):5.2f} "
              f"{1e3 * np.median(c['seconds'][m]):6.1f}")


def wdmam_report():
    c = load("wdmam")
    mesh = np.load(DATA / "mesh_L2.npz")
    print("\n" + "=" * 94)
    print("WDMAM L2: 162 mesh vertices, the windows the published archive used")
    print("=" * 94)

    windows = sorted(set(c["window_km"]))
    base = {}
    for w in windows:
        m = (c["variant"] == "default") & (c["window_km"] == w)
        o = np.argsort(c["vertex"][m])
        base[w] = c["dz"][m][o]

    print(f"\n  {'variant':10s} {'window':>7s} {'median dz':>10s} "
          f"{'vs default':>11s} {'|shift|>2km':>12s} {'median sigma':>13s} "
          f"{'chi2':>6s} {'onbound':>8s} {'ms':>7s}")
    for v in ORDER:
        for w in windows:
            m = (c["variant"] == v) & (c["window_km"] == w)
            o = np.argsort(c["vertex"][m])
            dz = c["dz"][m][o]
            d = dz - base[w]
            print(f"  {v:10s} {int(w):7d} {np.nanmedian(dz):10.2f} "
                  f"{np.nanmedian(d):+11.2f} {np.nanmean(np.abs(d) > 2.0):12.2f} "
                  f"{np.nanmedian(c['sigma_dz'][m]):13.2f} "
                  f"{np.nanmedian(c['chi2_red'][m]):6.2f} "
                  f"{np.nanmean(c['onbound'][m] > 0):8.2f} "
                  f"{1e3 * np.median(c['seconds'][m]):7.1f}")
        print()

    print("  nested-window disagreement -- the same vertex at 1000/2000/4000 km.")
    print("  CPD = zt + dz; the archive pins zt, so this is dz's own "
          "inconsistency.")
    print(f"\n  {'variant':10s} {'mean sd across windows':>24s} "
          f"{'median':>8s} {'|1000-4000|':>12s}")
    for v in ORDER:
        cols = []
        for w in windows:
            m = (c["variant"] == v) & (c["window_km"] == w)
            o = np.argsort(c["vertex"][m])
            cols.append(c["dz"][m][o] + c["zt"][m][o])
        A = np.vstack(cols)
        sd = np.nanstd(A, axis=0, ddof=1)
        gap = np.abs(A[0] - A[-1])
        print(f"  {v:10s} {np.nanmean(sd):24.2f} {np.nanmedian(sd):8.2f} "
              f"{np.nanmedian(gap):12.2f}")

    print("\n  contrast across the mesh -- the real-data analogue of the "
          "synthetic pathology.")
    print("  On a field degraded to WDMAM's resolution the default returns "
          "14.4/16.1/16.9 km for\n  truths of 10/20/30: a 3x range collapsed "
          "into 1.2x. A map that flat is the symptom.")
    print(f"\n  {'variant':10s} {'window':>7s} {'p10':>7s} {'median':>7s} "
          f"{'p90':>7s} {'p90-p10':>8s}")
    for v in ORDER:
        for w in windows:
            m = (c["variant"] == v) & (c["window_km"] == w)
            dz = c["dz"][m]
            p10, p50, p90 = np.nanpercentile(dz, [10, 50, 90])
            print(f"  {v:10s} {int(w):7d} {p10:7.2f} {p50:7.2f} {p90:7.2f} "
                  f"{p90 - p10:8.2f}")
        print()

    ok = mesh["status"][np.where(mesh["window_km"] == 2000)[0][0]] == 0
    print(f"  ({ok.sum()}/{ok.size} vertices are clean at 2000 km in the "
          f"published archive)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("synth", "both"):
        synth_report()
    if which in ("consistency", "both"):
        consistency_report()
    if which in ("wdmam", "both"):
        wdmam_report()
