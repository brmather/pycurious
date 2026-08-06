"""Does the spectrum know where the top of the magnetic layer is?

Global_CPD pins zt at a global 1.0 km with sigma 0.05 -- effectively fixed --
and diagnose.py part A found CPD moves by -5.8 km per km of that pin at the
10,000 km window. Either the pin is carrying the depth scale of the entire
global map, or the data prefer 1.0 km strongly enough that the pin is only
stating what the fit would have found anyway.

The discriminator is the misfit: scan the pin, refit everything else, and look
at chi-squared over the SPECTRAL residuals only (the prior terms are excluded,
since a prior always penalises departure from itself and would hide a flat
likelihood).
"""
import numpy as np
import harness as H
import fitters as F
import spectral as S

rng = np.random.default_rng(11)
PINS = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])


def spectral_chi2(k, Phi, sigma, x):
    m = S.bouligand(k, *x)
    r = (m - Phi) / sigma
    return float(np.sum(r ** 2)), r.size


def run(w, nv=100):
    k, P, s, bp, valid = H.load_window(w)
    ap = H.archive_params(w)
    idx = rng.choice(np.where(valid)[0], nv, replace=False)

    chi = np.full((len(PINS), nv), np.nan)
    cpd = np.full((len(PINS), nv), np.nan)
    beta = np.full((len(PINS), nv), np.nan)
    for c, j in enumerate(idx):
        kk, pp, ss = H.column(k, P, s, j)
        for i, pin in enumerate(PINS):
            r = F.fit_bouligand(kk, pp, ss, bp[j], zt_pin=pin)
            x2, n = spectral_chi2(kk, pp, ss, r.x)
            chi[i, c] = x2 / n
            cpd[i, c] = r.x[1] + r.x[2]
            beta[i, c] = r.x[0]
    return chi, cpd, beta, idx


if __name__ == "__main__":
    for w in (10000, 4000, 1000):
        chi, cpd, beta, idx = run(w)
        # per-vertex, which pin fits best, and how much worse is the worst
        best = PINS[np.nanargmin(chi, axis=0)]
        rel = chi / chi[PINS == 1.0]
        print(f"\n=== window {w} km ===")
        print(f"{'zt pin':>7} {'median chi2/n':>14} {'vs pin=1':>9} "
              f"{'median CPD':>11} {'median beta':>12}")
        for i, pin in enumerate(PINS):
            print(f"{pin:7.1f} {np.nanmedian(chi[i]):14.4f} "
                  f"{np.nanmedian(rel[i]):9.3f} {np.nanmedian(cpd[i]):11.2f} "
                  f"{np.nanmedian(beta[i]):12.3f}")
        print(f"  best-fitting pin per vertex: median {np.median(best):.1f} km, "
              f"IQR [{np.percentile(best,25):.1f}, {np.percentile(best,75):.1f}]")
        spread = np.nanmax(chi, axis=0) / np.nanmin(chi, axis=0)
        print(f"  chi2 range across the whole pin scan: median x{np.median(spread):.3f}")
