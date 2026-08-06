"""
Measure the effective degrees of freedom per radial bin, per taper, directly.

``window_spectrum`` reports ``sigma = scatter / sqrt(counts / dof_factor)``,
i.e. it claims the bin mean has variance ``sigma_cell**2 / n_eff`` with
``n_eff = (counts - lost) / dof_inf`` from ``_TAPER_DOF``. That claim is
checkable without any fitting: generate many independent realisations of a
field whose spectrum is known, and look at how much each bin mean actually
moves. The ratio of the two is what the fit is really weighting by.

Two things are measured, and they are different:

* **n_eff** -- from the variance of the bin mean *across realisations*. This
  is the honest one. It is what a weight should be built from.
* **the within-annulus scatter** -- what pycurious actually uses, measured
  inside a single realisation. It estimates the per-cell sd ``pi/sqrt(6)``,
  and on the inner bins it estimates it badly.

``lost`` is the term that makes the deflation differential: an outer bin with
600 cells does not notice losing 4.9 of them, an inner bin with 8 loses more
than half its weight. So whether ``lost`` is right decides how much of the fit
the ``dz``-bearing wavenumbers pay for.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spectral_variants as sv  # noqa: E402
from pycurious.grid import _dof_factor  # noqa: E402

DX_KM = 5.0


def bin_index(n):
    dk, kbins = sv.bin_edges(n, DX_KM)
    kk = sv.cell_wavenumbers(n, n, dk).ravel()
    weight = sv.hermitian_weight(n, n)
    nb = kbins.size - 1
    idx = np.digitize(kk, kbins) - 1
    idx[(idx == nb) & (kk <= kbins[-1])] = nb - 1
    keep = (idx >= 0) & (idx < nb)
    counts = np.bincount(idx[keep], weights=weight[keep], minlength=nb)
    kbar = np.bincount(idx[keep], weights=weight[keep] * kk[keep], minlength=nb) / counts
    return keep, idx[keep], weight[keep], counts, kbar, nb


def measure(n=401, M=400, taper="hanning", NW=3.0, K=None, detrend=True,
            spectrum=None, seed0=0):
    """Per-bin (n_eff measured, mean within-bin scatter, counts, kbar)."""
    keep, idx, w, counts, kbar, nb = bin_index(n)
    kw = spectrum or dict(beta=3.0, zt=1.0, dz=20.0, C=5.0)

    means = np.empty((M, nb))
    scat = np.empty((M, nb))
    for m in range(M):
        sub = sv.synth(n, seed0 + m, **kw)
        if detrend:
            sub = sv.detrend(sub)
        P, _ = sv.cell_power(sub, taper, NW, K)
        with np.errstate(divide="ignore"):
            rr = np.log(P).ravel()[keep]
        mu = np.bincount(idx, weights=w * rr, minlength=nb) / counts
        dev = rr - mu[idx]
        means[m] = mu
        scat[m] = np.sqrt(np.bincount(idx, weights=w * dev * dev, minlength=nb) / counts)

    n_eff = sv.SIGMA_LOG ** 2 / means.var(axis=0, ddof=1)
    return n_eff, scat.mean(axis=0), counts, kbar, means.mean(axis=0)


def bias(kbar, mean_lnPhi, spectrum=None, band=(0.05, 0.2)):
    """Shape distortion of the estimated spectrum, in log units.

    The estimator sits below ``ln Phi`` by the log-averaging constant and by
    whatever power the taper removes, both of which are constant in ``k`` and
    are absorbed by ``C``. Only the part that *varies* with ``k`` can bias a
    depth, so the level is referred to the mid band, where every taper is well
    behaved, and what is left is leakage and resolution smoothing.
    """
    from pycurious.grid import bouligand2009
    kw = spectrum or dict(beta=3.0, zt=1.0, dz=20.0, C=5.0)
    truth = bouligand2009(kbar, **kw)
    d = mean_lnPhi - truth
    ref = (kbar >= band[0]) & (kbar <= band[1])
    return d - np.median(d[ref])


def main():
    n, M = 401, 500
    print(f"{M} independent synthetics at n={n} (2000 km), beta=3 zt=1 dz=20, "
          f"detrended\n")

    print("A. is the per-cell log-periodogram sd really pi/sqrt(6) = "
          f"{sv.SIGMA_LOG:.4f}?")
    ne, sc, counts, kbar, _ = measure(n, M, "none", detrend=False)
    # with no taper and no detrend, cells are independent: n_eff should equal
    # the number of *unique* cells, which is counts/2 by Hermitian symmetry
    sel = counts > 20
    print(f"   no taper, no detrend: n_eff / (counts/2) = "
          f"{np.mean(ne[sel] / (counts[sel] / 2)):.3f} over {sel.sum()} bins "
          f"-> cells are independent and the sd is as stated\n")

    print("B. n_eff measured, against what `_TAPER_DOF` claims; and the bias "
          "each taper leaves in the spectrum")
    print("   sigma_net is what pycurious ends up reporting, over the truth: "
          "(scatter/theory) x sqrt(n_eff meas / n_eff lib). 1.0 is honest.")
    for label, taper, NW, K in [("none", "none", None, None),
                                ("hanning", "hanning", None, None),
                                ("dpss NW=2 K=3", "dpss", 2.0, 3),
                                ("dpss NW=3 K=5", "dpss", 3.0, 5),
                                ("dpss NW=4 K=7", "dpss", 4.0, 7)]:
        ne, sc, counts, kbar, mu = measure(n, M, taper, NW or 3.0, K)
        lib = counts / _dof_factor(np.hanning if taper == "hanning" else None, counts)
        bs = bias(kbar, mu)
        print(f"\n   {label}")
        print(f"   {'bin':>4s} {'k':>7s} {'counts':>7s} {'n_eff meas':>11s} "
              f"{'n_eff lib':>10s} {'meas/lib':>9s} {'scat/thy':>9s} "
              f"{'sigma_net':>10s} {'bias':>7s}")
        for b in [0, 1, 2, 3, 5, 10, 20, 40, 78]:
            net = (sc[b] / sv.SIGMA_LOG) * np.sqrt(ne[b] / lib[b])
            print(f"   {b:4d} {kbar[b]:7.4f} {counts[b]:7.0f} {ne[b]:11.2f} "
                  f"{lib[b]:10.2f} {ne[b] / lib[b]:9.2f} "
                  f"{sc[b] / sv.SIGMA_LOG:9.3f} {net:10.3f} {bs[b]:+7.3f}")
        inner = kbar < 0.05
        net = (sc / sv.SIGMA_LOG) * np.sqrt(ne / lib)
        print(f"   -> dz band (k < 0.05, {inner.sum()} bins): "
              f"n_eff meas/lib {np.mean(ne[inner] / lib[inner]):.2f}, "
              f"scatter/theory {np.mean(sc[inner] / sv.SIGMA_LOG):.3f}, "
              f"sigma_net {np.mean(net[inner]):.3f}, "
              f"max |bias| {np.abs(bs[inner]).max():.3f}")
        np.save(HERE / "wdmam_L2" / f"neff_{label.replace(' ', '_')}.npy",
                np.vstack([kbar, counts, ne, sc, bs]))


if __name__ == "__main__":
    main()
