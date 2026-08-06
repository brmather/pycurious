"""
Where the binned and unbinned fits still disagree, and why.

The sufficiency argument in ``spectral_variants`` has a proviso that is easy to
miss: it needs the model to be *constant across the annulus*. The noise is,
the model is not. Binning replaces ``mean over the annulus of ln Phi(k)`` by
``ln Phi(kbar)``, and those differ wherever ``ln Phi`` curves -- which is worst
in the innermost bins, where the annulus is widest in relative terms
(``[dk, 2dk]`` spans a factor of two in ``k``) and where the spectrum turns
over. That is also precisely where ``dz`` is determined.

At low ``k`` the model is close to ``const - beta ln k - 2 zt k``, so the
average of ``ln Phi`` over an annulus is close to
``const - beta <ln k> - 2 zt <k>``. A single representative wavenumber cannot
satisfy both terms, but ``<ln k>`` -- the geometric mean -- serves the ``beta``
term, which dominates at low k, where the arithmetic mean serves neither.

So: three ways of writing the same bin, and an unbinned fit for reference.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spectral_variants as sv  # noqa: E402
from pycurious.grid import _dof_factor, bouligand2009  # noqa: E402

DX_KM = 5.0


def binned_keff(sub, n, kmax=0.25, mode="arithmetic"):
    """Binned spectrum with theoretical sigma, at a choice of representative k.

    ``mode="exact"`` returns the per-bin cell lists as well, so a caller can
    average the *model* over the annulus instead of evaluating it at one k.
    """
    dk, kbins = sv.bin_edges(n, DX_KM)
    P, _ = sv.cell_power(sub, "hanning")
    rr = np.log(P).ravel()
    kk = sv.cell_wavenumbers(n, n, dk).ravel()
    weight = sv.hermitian_weight(n, n)
    nb = kbins.size - 1
    idx = np.digitize(kk, kbins) - 1
    keep = (idx >= 0) & (idx < nb)
    idx, rr, kk, weight = idx[keep], rr[keep], kk[keep], weight[keep]

    counts = np.bincount(idx, weights=weight, minlength=nb)
    S = np.bincount(idx, weights=weight * rr, minlength=nb) / counts
    kar = np.bincount(idx, weights=weight * kk, minlength=nb) / counts
    kgeo = np.exp(np.bincount(idx, weights=weight * np.log(kk), minlength=nb) / counts)
    n_eff = counts / _dof_factor(np.hanning, counts)
    sigma = sv.SIGMA_LOG / np.sqrt(n_eff)

    ok = (counts > 0) & (kar <= kmax)
    k = {"arithmetic": kar, "geometric": kgeo}[mode][ok] if mode != "exact" else kar[ok]
    out = (k, S[ok], sigma[ok])
    if mode == "exact":
        cells = [kk[idx == b] for b in np.flatnonzero(ok)]
        wts = [weight[idx == b] for b in np.flatnonzero(ok)]
        return out, cells, wts
    return out


class ExactModel:
    """Residuals that average the model over each annulus, as the data are."""

    def __init__(self, g, cells, wts):
        self.g, self.cells, self.wts = g, cells, wts
        self.kbar = np.array([np.average(c, weights=w) for c, w in zip(cells, wts)])
        self.flat = np.concatenate(cells)
        self.fw = np.concatenate(wts)
        self.owner = np.concatenate([np.full(c.size, i) for i, c in enumerate(cells)])
        self.norm = np.bincount(self.owner, weights=self.fw)

    def model(self, x):
        m = bouligand2009(self.flat, *x)
        return np.bincount(self.owner, weights=self.fw * m) / self.norm


def fit_exact(spectrum, em, g):
    """Least squares with the annulus-averaged forward model."""
    from scipy.optimize import least_squares
    k, Phi, sigma = spectrum
    lo = np.array([0.0, 0.0, 0.0, -np.inf])
    hi = np.array([np.inf, np.inf, g._max_thickness(), np.inf])

    def res(x):
        with np.errstate(all="ignore"):
            r = (em.model(x) - Phi) / sigma
        r = np.where(np.isfinite(r), r, 1e3)
        return np.concatenate([r, [(x[1] - 1.0) / 0.05]])

    out = least_squares(res, np.array([3.0, 1.0, 10.0, 5.0]), bounds=(lo, hi))
    return out.x


def main():
    print("dz recovered from one synthetic window, four ways of writing the "
          "same spectrum")
    print("truth: beta 3.0  zt 1.0  dz 20.0  C 5.0;  zt pinned at 1.0 +/- 0.05\n")
    print("  n   seed |  unbinned  bin<k>  bin<lnk>  bin-exact |  <k> err  "
          "<lnk> err  exact err")
    g = sv.fitter()
    for n, w in ((201, 1000), (401, 2000), (801, 4000)):
        rows = []
        for seed in range(12):
            sub = sv.detrend(sv.synth(n, seed, dz=20.0))
            du = sv.fit(sv.unbinned(sub), g)[0][2]
            da = sv.fit(binned_keff(sub, n, mode="arithmetic"), g)[0][2]
            dg = sv.fit(binned_keff(sub, n, mode="geometric"), g)[0][2]
            sp, cells, wts = binned_keff(sub, n, mode="exact")
            de = fit_exact(sp, ExactModel(g, cells, wts), g)[2]
            rows.append((du, da, dg, de))
            print(f"{n:5d} {seed:4d} | {du:9.2f} {da:7.2f} {dg:8.2f} {de:9.2f} "
                  f"| {da - du:8.2f} {dg - du:10.2f} {de - du:10.2f}")
        r = np.array(rows)
        print(f"  -> {w} km: mean offset from unbinned  <k> {np.mean(r[:, 1] - r[:, 0]):+.2f}"
              f"   <lnk> {np.mean(r[:, 2] - r[:, 0]):+.2f}"
              f"   exact {np.mean(r[:, 3] - r[:, 0]):+.2f} km\n")


if __name__ == "__main__":
    main()
