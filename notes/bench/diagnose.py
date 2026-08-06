"""Three questions the re-derivation raises, asked of the real WDMAM spectra.

A. The derivation says the low-k spectrum is set by the CENTROID z0, so with
   z0 the determined quantity CPD = 2 z0 - zt and d(CPD)/d(zt) = -1. zt is
   pinned at a global 1.0 km. How much CPD does that pin buy or cost?

B. The model's one falsifiable prediction is that the log-log slope changes by
   exactly 2 across the layer rolloff, independently of every parameter. Fit
   that gap free and see what the data want.

C. Where in k does the dz information come from? Per-bin contribution to the
   Fisher information for dz.
"""
import numpy as np
import harness as H
import fitters as F
import spectral as S

rng = np.random.default_rng(7)
NV = 120


def sample(w):
    k, P, s, bp, valid = H.load_window(w)
    ap = H.archive_params(w)
    pool = np.where(valid & np.isfinite(ap["land_fraction"]))[0]
    return k, P, s, bp, ap, rng.choice(pool, min(NV, pool.size), replace=False)


def part_A(windows=(10000, 4000, 1000)):
    print("A. sensitivity of CPD to the pinned zt")
    print("   d(CPD)/d(zt_pin): -1 = the fit is anchored on the centroid,")
    print("                      0 = zt and dz are independently determined\n")
    print(f"   {'window':>7} {'zt=1':>8} {'zt=3':>8} {'zt=5':>8}   {'slope':>7}")
    for w in windows:
        k, P, s, bp, ap, idx = sample(w)
        pins = (1.0, 3.0, 5.0)
        cpd = {p: [] for p in pins}
        for j in idx:
            kk, pp, ss = H.column(k, P, s, j)
            for p in pins:
                r = F.fit_bouligand(kk, pp, ss, bp[j], zt_pin=p)
                cpd[p].append(r.x[1] + r.x[2])
        med = {p: np.median(cpd[p]) for p in pins}
        # least-squares slope of median CPD against the pin
        pa = np.array(pins)
        ma = np.array([med[p] for p in pins])
        slope = np.polyfit(pa, ma, 1)[0]
        print(f"   {w:7d} {med[1.0]:8.2f} {med[3.0]:8.2f} {med[5.0]:8.2f}   {slope:+7.3f}")
    print()


def part_B(windows=(10000, 4000, 1000)):
    print("B. the slope change the model fixes at 2, fitted free")
    print(f"   {'window':>7} {'gap median':>11} {'IQR':>16} {'frac>2':>8} "
          f"{'dz(gap=2)':>10} {'dz(free)':>9}")
    for w in windows:
        k, P, s, bp, ap, idx = sample(w)
        gaps, dz0, dz1 = [], [], []
        for j in idx:
            kk, pp, ss = H.column(k, P, s, j)
            a = F.fit_bouligand(kk, pp, ss, bp[j])
            b = F.fit_free_gap(kk, pp, ss, bp[j])
            gaps.append(b.x[4]); dz0.append(a.x[2]); dz1.append(b.x[2])
        g = np.array(gaps)
        q1, q3 = np.percentile(g, [25, 75])
        print(f"   {w:7d} {np.median(g):11.3f}   [{q1:.2f}, {q3:.2f}]"
              f"{(g > 2).mean():>10.2f} {np.median(dz0):10.2f} {np.median(dz1):9.2f}")
    print()


def part_C(windows=(10000, 4000, 1000)):
    print("C. where the dz information comes from (Fisher info per bin,")
    print("   cumulated; k_break = 1/dz marks the rolloff)\n")
    print(f"   {'window':>7} {'dz':>6} {'k_break':>8}   "
          f"{'% of info below k_break':>24}   {'% above 0.1':>11}")
    for w in windows:
        k, P, s, bp, ap, idx = sample(w)
        below, above = [], []
        for j in idx:
            kk, pp, ss = H.column(k, P, s, j)
            r = F.fit_bouligand(kk, pp, ss, bp[j])
            beta, zt, dz, C = r.x
            if not (0.5 < dz < 300):
                continue
            h = 1e-3 * max(dz, 1.0)
            d = (S.bouligand(kk, beta, zt, dz + h, C)
                 - S.bouligand(kk, beta, zt, dz - h, C)) / (2 * h)
            info = (d / ss) ** 2
            tot = info.sum()
            if tot <= 0:
                continue
            below.append(info[kk < 1.0 / dz].sum() / tot)
            above.append(info[kk > 0.1].sum() / tot)
        r0 = F.fit_bouligand(*H.column(k, P, s, idx[0]), bp[idx[0]])
        print(f"   {w:7d} {np.median([r0.x[2]]):6.1f} {1/max(r0.x[2],1e-9):8.4f}   "
              f"{100*np.median(below):>22.1f}%   {100*np.median(above):>10.1f}%")
    print()


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
