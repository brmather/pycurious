"""
Can a good ``x0`` be derived from the spectrum instead of guessed?

Three of the four parameters need not be guessed at all.

**``C`` and ``zt`` are linear.** The forward model is

    Phi = C - 2|k| zt - (beta - 1) ln|k| - |k| dz + ln A(beta, dz, |k|)

and ``A`` depends on neither, so for any ``(beta, dz)`` the best ``(C, zt)`` is
a two-column weighted least-squares solve -- exact, non-iterative, and correct
at every step rather than merely a good start. That is variable projection: the
four-parameter search collapses to two. A Gaussian prior on ``zt`` -- which is
what production uses -- stays inside the linear solve as one more row, so
pinning is not lost.

**``dz`` can be seeded from Tanaka.** The centroid method is two straight lines,
and it reads the same spectrum: ``power=1`` is exactly half of ``power=2`` in
the log, so ``lnPhi/2`` converts one to the other with no second FFT. Slope of
``ln Phi^0.5`` against ``k`` at high wavenumber gives ``zt``; slope of
``ln(Phi^0.5 / k)`` at low wavenumber gives the centroid ``z0``; and
``dz = 2 (z0 - zt)``. Rough is fine -- it only has to land in the right basin.

**``beta`` is left at 3.0.** It is not the multimodal direction, and the fitted
values here and on WDMAM sit within a few tenths of 3 anyway.

Scored against a dense 8-point multi-start, which stands in for the global
minimum.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spectral_variants as sv  # noqa: E402
from pycurious.grid import bouligand2009  # noqa: E402

DENSE = (2.5, 5.0, 10.0, 20.0, 30.0, 45.0, 65.0, 90.0)


def shape(k, beta, dz):
    """The model with ``C`` and ``zt`` removed -- the part they add onto."""
    with np.errstate(all="ignore"):
        return bouligand2009(k, beta, 0.0, dz, 0.0)


def solve_linear(k, Phi, sigma, beta, dz, zt_pin=None, zt_sigma=None):
    """Exact weighted (C, zt) at fixed (beta, dz), with the zt prior included."""
    h = shape(k, beta, dz)
    G = np.column_stack([np.ones_like(k), -2.0 * k]) / sigma[:, None]
    y = (Phi - h) / sigma
    if zt_pin is not None:
        G = np.vstack([G, [0.0, 1.0 / zt_sigma]])
        y = np.append(y, zt_pin / zt_sigma)
    sol, *_ = np.linalg.lstsq(G, y, rcond=None)
    return sol  # (C, zt)


def tanaka_seed(k, Phi, sigma, split=0.35):
    """(zt, dz) from two straight lines on the same spectrum.

    ``Phi`` is the ``power=2`` log spectrum; Tanaka wants ``power=1``, which is
    exactly half of it. The band split is taken as a fraction of the retained
    range rather than fixed wavenumbers, because this must work at every window
    size without being told anything.
    """
    lnA = 0.5 * Phi                      # ln Phi^0.5
    kcut = k[0] + split * (k[-1] - k[0])
    hi, lo = k > kcut, k <= kcut
    if hi.sum() < 3 or lo.sum() < 3:
        return 1.0, 10.0
    # high band: ln Phi^0.5 = b - zt k
    zt = -np.polyfit(k[hi], lnA[hi], 1)[0]
    # low band: ln(Phi^0.5 / k) = b - z0 k
    z0 = -np.polyfit(k[lo], lnA[lo] - np.log(k[lo]), 1)[0]
    dz = 2.0 * (z0 - zt)
    return (max(zt, 0.1), float(np.clip(dz, 1.0, 120.0)))


def fit_varpro(k, Phi, sigma, zt_pin=1.0, zt_sigma=0.05, dz0=10.0, beta0=3.0):
    """Search (beta, dz) only; (C, zt) solved exactly at every evaluation."""
    def res(p):
        beta, dz = p
        C, zt = solve_linear(k, Phi, sigma, beta, dz, zt_pin, zt_sigma)
        with np.errstate(all="ignore"):
            r = (shape(k, beta, dz) + C - 2.0 * k * zt - Phi) / sigma
        r = np.where(np.isfinite(r), r, 1e3)
        return np.append(r, (zt - zt_pin) / zt_sigma)

    out = least_squares(res, [beta0, dz0], bounds=([0.0, 0.1], [8.0, 200.0]))
    C, zt = solve_linear(k, Phi, sigma, out.x[0], out.x[1], zt_pin, zt_sigma)
    return np.array([out.x[0], zt, out.x[1], C]), out.cost, out.nfev


def main():
    warnings.simplefilter("ignore")
    g = sv.fitter()

    print("Reaching the global minimum: % of seeds within 0.1% of the best cost")
    print("found by a dense 8-point multi-start. 4000 km and 1000 km windows.\n")
    print(f"  {'window':>7s} {'dz':>4s} | {'default x0':>11s} {'Tanaka x0':>10s} "
          f"{'4-start':>8s} | {'varpro':>8s} {'varpro+Tanaka':>14s} | "
          f"{'fits':>5s}")

    for n, L in ((801, 4000), (401, 2000), (201, 1000)):
        for dz_true in (10.0, 20.0, 30.0):
            hit = {k: 0 for k in ("default", "tanaka", "multi", "vp", "vpt")}
            dzs = {k: [] for k in hit}
            nseed = 60
            for seed in range(nseed):
                f = sv.synth(n + 10, seed, beta=3.0, zt=1.0, dz=dz_true, C=5.0)
                sp = sv.binned(sv.detrend(sv.centred(f, n)), kmax=0.25)
                k, Phi, sigma = sp

                dense = [g._fit(np.array([3., 1., d, 5.]), sp) for d in DENSE]
                best = min(r.cost for r in dense)

                zt_t, dz_t = tanaka_seed(k, Phi, sigma)
                runs = {
                    "default": g._fit(np.array([3., 1., 10., 5.]), sp),
                    "tanaka": g._fit(np.array([3., 1., dz_t, 5.]), sp),
                    "multi": min((g._fit(np.array([3., 1., d, 5.]), sp)
                                  for d in (5., 15., 30., 60.)),
                                 key=lambda r: r.cost),
                }
                for name, r in runs.items():
                    hit[name] += r.cost <= best * 1.001
                    dzs[name].append(r.x[2])
                for name, d0 in (("vp", 10.0), ("vpt", dz_t)):
                    x, cost, _ = fit_varpro(k, Phi, sigma, dz0=d0)
                    hit[name] += cost <= best * 1.001
                    dzs[name].append(x[2])

            row = (f"  {L:7d} {dz_true:4.0f} | "
                   f"{100 * hit['default'] / nseed:10.0f}% "
                   f"{100 * hit['tanaka'] / nseed:9.0f}% "
                   f"{100 * hit['multi'] / nseed:7.0f}% | "
                   f"{100 * hit['vp'] / nseed:7.0f}% "
                   f"{100 * hit['vpt'] / nseed:13.0f}% |")
            print(row)
        print()

    print("  dz recovered (mean +/- sd), 4000 km, truth 30:")
    n, dz_true = 801, 30.0
    acc = {k: [] for k in ("default", "tanaka", "multi", "vpt")}
    for seed in range(60):
        f = sv.synth(n + 10, seed, beta=3.0, zt=1.0, dz=dz_true, C=5.0)
        sp = sv.binned(sv.detrend(sv.centred(f, n)), kmax=0.25)
        zt_t, dz_t = tanaka_seed(*sp)
        acc["default"].append(g._fit(np.array([3., 1., 10., 5.]), sp).x[2])
        acc["tanaka"].append(g._fit(np.array([3., 1., dz_t, 5.]), sp).x[2])
        acc["multi"].append(min((g._fit(np.array([3., 1., d, 5.]), sp)
                                 for d in (5., 15., 30., 60.)),
                                key=lambda r: r.cost).x[2])
        acc["vpt"].append(fit_varpro(*sp, dz0=dz_t)[0][2])
    for name, v in acc.items():
        v = np.array(v)
        print(f"    {name:10s} {v.mean():7.2f} +/- {v.std(ddof=1):5.2f}")


if __name__ == "__main__":
    main()
