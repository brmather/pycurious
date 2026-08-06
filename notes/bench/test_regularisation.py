"""
Two regularisation ideas that the correlation measurement points at.

**1. Bin a multitaper at its resolution bandwidth.** The measured bin-to-bin
correlation for ``NW=3, K=5`` is 0.86 at lag 1 and still 0.25 at lag 4: the
estimate is smoothed over ``NW dk``, so binning at ``dk`` produces ~5 bins
carrying one bin's worth of information. That is why ``_banded_correlation``
saturates its positive-definiteness cap there (it returns 0.270/0.205, whose
sum is exactly the 0.475 the ``_CORRELATION_LIMIT`` allows) and why the
sufficiency argument for binning stops applying. Widening the bins to match the
bandwidth should restore near-independence, cut the bin count, and cost
nothing -- there was no information between those bins to lose.

**2. Model the resolution rolloff instead of hiding from it.** WDMAM's problem
is an unmodelled ``exp(-k**2 sigma_r**2)``. Today it is handled by refusing to
look at the band it lives in (``kmax``) and discounting what is left
(``sigma_weight_km``). The alternative is a fifth parameter. ``curie_config``
records that ``sigma_r`` is degenerate with ``zt`` -- but that was measured with
``zt`` free, and the production workflow *pins* ``zt``. Pinned, the two have
different shapes in the exponent (``k**2`` against ``k``), so the degeneracy may
break. If it does, the rolloff becomes a fitted nuisance with an error bar
rather than a bias absorbed by ``dz``.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spectral_variants as sv  # noqa: E402
from calibrate_dof import bin_index  # noqa: E402
from run_experiments import degrade  # noqa: E402
from pycurious.grid import bouligand2009, _dof_factor  # noqa: E402

DX_KM = 5.0


# ---------------------------------------------------------------- idea 1

def binned_wide(sub, width, taper="dpss", NW=3.0, K=5, kmax=0.25):
    """Binned spectrum with annuli ``width`` times the DFT fundamental."""
    n = sub.shape[0]
    dk = 2.0 * np.pi / n / DX_KM
    kbins = np.arange(dk, dk * n / 2, dk * width)
    P, ntap = sv.cell_power(sub, taper, NW, K)
    rr = np.log(P).ravel()
    kk = sv.cell_wavenumbers(n, n, dk).ravel()
    w = sv.hermitian_weight(n, n)
    nb = kbins.size - 1
    idx = np.digitize(kk, kbins) - 1
    keep = (idx >= 0) & (idx < nb)
    idx, rr, kk, w = idx[keep], rr[keep], kk[keep], w[keep]
    counts = np.bincount(idx, weights=w, minlength=nb)
    S = np.bincount(idx, weights=w * rr, minlength=nb) / counts
    kbar = np.bincount(idx, weights=w * kk, minlength=nb) / counts
    return kbar, S, counts


def correlation_at_width(n=401, M=400, width=1, NW=3.0, K=5):
    """Lag-1..4 correlation of the wide-binned multitaper, across realisations."""
    rows = []
    for m in range(M):
        sub = sv.detrend(sv.synth(n, m, beta=3.0, zt=1.0, dz=20.0, C=5.0))
        kbar, S, counts = binned_wide(sub, width, NW=NW, K=K)
        rows.append(S)
    A = np.array(rows)
    R = np.corrcoef((A - A.mean(axis=0)).T)
    return kbar.size, [float(np.mean(np.diagonal(R, offset=L))) for L in (1, 2, 3, 4)]


# ---------------------------------------------------------------- idea 2

def fit_with_rolloff(spectrum, zt_pin=1.0, zt_sigma=0.05, sr_prior=None,
                     free_sr=True):
    """Bouligand plus ``-k**2 sigma_r**2``, with ``zt`` pinned as production does.

    ``sr_prior`` is ``(loc, scale)`` for a Gaussian prior on ``sigma_r``; with
    ``None`` it is free. ``free_sr=False`` reduces to the ordinary four
    parameter fit, for reference in the same code path.
    """
    k, Phi, sigma = spectrum
    x0 = np.array([3.0, zt_pin, 10.0, 5.0, 2.0 if free_sr else 0.0])
    lo = np.array([0.0, 0.0, 0.0, -np.inf, 0.0])
    hi = np.array([np.inf, np.inf, 200.0, np.inf, 20.0 if free_sr else 1e-9])

    def res(x):
        beta, zt, dz, C, sr = x
        with np.errstate(all="ignore"):
            m = bouligand2009(k, beta, zt, dz, C) - (k * sr) ** 2
        r = (m - Phi) / sigma
        r = np.where(np.isfinite(r), r, 1e3)
        rows = [r, [(zt - zt_pin) / zt_sigma]]
        if sr_prior is not None:
            rows.append([(sr - sr_prior[0]) / sr_prior[1]])
        return np.concatenate([np.asarray(a, dtype=float) for a in rows])

    out = least_squares(res, x0, bounds=(lo, hi))
    try:
        sd = np.sqrt(np.diag(np.linalg.inv(out.jac.T @ out.jac)))
    except np.linalg.LinAlgError:
        sd = np.full(5, np.nan)
    return out.x, sd


def main():
    warnings.simplefilter("ignore")

    print("=" * 78)
    print("1. binning a multitaper at its resolution bandwidth (NW=3, K=5)")
    print("=" * 78)
    print(f"  {'bin width':>10s} {'bins':>6s} {'lag1':>7s} {'lag2':>7s} "
          f"{'lag3':>7s} {'lag4':>7s}")
    for width in (1, 2, 3, 4, 6):
        nb, lags = correlation_at_width(width=width)
        print(f"  {width:8d}dk {nb:6d} " + " ".join(f"{v:7.3f}" for v in lags))
    print("\n  A multitaper binned at dk is ~5 copies of the same number. "
          "Widening to NW dk\n  leaves neighbours near-independent, which is "
          "what the fit's weighting assumes.")

    print("\n" + "=" * 78)
    print("2. fitting the resolution rolloff as a nuisance parameter")
    print("=" * 78)
    n, nseed = 801, 60
    print(f"  4000 km window, {nseed} seeds, degraded at sigma_r = 4.2 km, "
          f"zt pinned at 1.0\n")
    print(f"  {'dz true':>8s} {'scheme':>22s} {'dz recovered':>16s} "
          f"{'sigma_r recovered':>19s}")
    for dz_true in (10.0, 20.0, 30.0):
        rows = {"4-parameter (today)": ([], []),
                "+ free sigma_r": ([], []),
                "+ sigma_r prior 4+/-2": ([], [])}
        for seed in range(nseed):
            field = degrade(sv.synth(n + 10, seed, beta=3.0, zt=1.0,
                                     dz=dz_true, C=5.0), 4.2)
            sp = sv.binned(sv.detrend(sv.centred(field, n)), kmax=0.25)
            for label, kw in (("4-parameter (today)", dict(free_sr=False)),
                              ("+ free sigma_r", dict()),
                              ("+ sigma_r prior 4+/-2", dict(sr_prior=(4.0, 2.0)))):
                x, _ = fit_with_rolloff(sp, **kw)
                rows[label][0].append(x[2])
                rows[label][1].append(x[4])
        for label, (dz, sr) in rows.items():
            dz, sr = np.array(dz), np.array(sr)
            print(f"  {dz_true:8.0f} {label:>22s} "
                  f"{np.mean(dz):9.1f} +/-{np.std(dz, ddof=1):4.1f} "
                  f"{np.mean(sr):12.2f} +/-{np.std(sr, ddof=1):4.2f}")
        print()


if __name__ == "__main__":
    main()


def wdmam_rolloff(w_km=4000, kmax=0.25):
    """Fit the rolloff at every WDMAM L2 vertex, and on a clean synthetic.

    The clean case is the safety check: a fifth parameter that is not needed
    must return zero and leave `dz` where it was, or it is not a nuisance
    parameter, it is a second way to fit the same curve.
    """
    warnings.simplefilter("ignore")
    n = int(w_km * 1e3 / 5000.0)
    print("\n" + "=" * 78)
    print(f"3. the rolloff parameter on real WDMAM, {w_km} km window")
    print("=" * 78)

    print("\n  safety check on a CLEAN synthetic (sigma_r should come back 0)")
    print(f"  {'dz true':>8s} {'dz 4-param':>14s} {'dz + sigma_r':>16s} "
          f"{'sigma_r':>14s}")
    for dz_true in (10.0, 20.0, 30.0):
        a, b, s = [], [], []
        for seed in range(40):
            field = sv.synth(n + 10, seed, beta=3.0, zt=1.0, dz=dz_true, C=5.0)
            sp = sv.binned(sv.detrend(sv.centred(field, n)), kmax=kmax)
            a.append(fit_with_rolloff(sp, free_sr=False)[0][2])
            x = fit_with_rolloff(sp)[0]
            b.append(x[2]); s.append(x[4])
        print(f"  {dz_true:8.0f} {np.mean(a):8.1f} +/-{np.std(a, ddof=1):4.1f} "
              f"{np.mean(b):10.1f} +/-{np.std(b, ddof=1):4.1f} "
              f"{np.mean(s):8.2f} +/-{np.std(s, ddof=1):4.2f}")

    win = np.load(HERE / "wdmam_L2" / "windows_L2.npy", mmap_mode="r")
    rows = {"4-parameter": [], "+ free sigma_r": []}
    srs = []
    for j in range(win.shape[0]):
        sp = sv.binned(sv.detrend(sv.centred(win[j], n)), kmax=kmax)
        rows["4-parameter"].append(fit_with_rolloff(sp, free_sr=False)[0][2])
        x = fit_with_rolloff(sp)[0]
        rows["+ free sigma_r"].append(x[2])
        srs.append(x[4])
    print(f"\n  162 WDMAM vertices")
    print(f"  {'scheme':>16s} {'p10':>7s} {'median':>8s} {'p90':>7s} "
          f"{'p90-p10':>8s}")
    for label, v in rows.items():
        p = np.nanpercentile(v, [10, 50, 90])
        print(f"  {label:>16s} {p[0]:7.1f} {p[1]:8.1f} {p[2]:7.1f} "
              f"{p[2] - p[0]:8.1f}")
    srs = np.array(srs)
    print(f"\n  fitted sigma_r across the mesh: median {np.median(srs):.2f} km, "
          f"p10-p90 {np.percentile(srs, 10):.2f}-{np.percentile(srs, 90):.2f} km")
    print(f"  (WDMAM's grid is 5.56 km; curie_config's assumed degradation is "
          f"4.2 km)")


if __name__ == "__main__" and "--wdmam" in sys.argv:
    wdmam_rolloff()
