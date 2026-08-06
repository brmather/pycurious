"""
Spectral estimators to fit a Bouligand model against: binned (the default),
unbinned, and multitaper -- plus the weighting schemes that decide how much of
the fit each wavenumber pays for.

Three questions live here.

**Is the binning needed at all?** For a log-periodogram it is a sufficient
reduction, and the argument is short. Each retained FFT cell carries
``ln|FFT|**2 = ln Phi(|k|) + ln E``, ``E ~ Exp(1)``: the noise is *additive in
the log and identically distributed*, mean ``-gamma``, variance ``pi**2/6``,
whatever the spectrum does. The model depends on the cell only through ``|k|``,
so within an annulus every cell has the same mean and the same variance, and

    sum_cell (m - y_cell)**2 / s**2  ==  n (m - ybar)**2 / s**2 + const

The constant does not move the minimum. So fitting ``n`` cells at ``sigma = s``
and fitting their mean at ``sigma = s/sqrt(n)`` are the *same* least-squares
problem -- same estimate, same curvature, same information. Binning throws
nothing away, provided the weight is the one the identity implies.

That proviso is the whole experiment. pycurious does not use ``s/sqrt(n)`` with
``s`` known; it uses the *empirical* scatter within each annulus, which is a
noisy estimate of ``s`` on the inner bins (they hold a handful of cells) and is
inflated by anisotropy and by model error. So there are three estimators to
compare, not two:

  ``binned``        empirical within-annulus scatter -- the default
  ``binned_theory`` the same bins, weighted by ``pi/sqrt(6*n_eff)``
  ``unbinned``      every unique cell, weighted by ``pi/sqrt(6)``

``binned_theory`` and ``unbinned`` must agree to the optimiser's tolerance if
the argument above is right. Any gap between them and ``binned`` is the cost of
estimating the weights from the same data being fitted.

**Multitaper.** ``hanning`` is one taper: one look at the field, ``pi**2/6`` of
log-variance per cell, and a main lobe that spreads each wavenumber over
several bins. ``K`` orthogonal Slepian tapers give ``K`` looks whose estimates
are near-independent, so averaging the *power* over them (before the log, which
is the only order that reduces variance rather than just smoothing) cuts the
per-cell variance by roughly ``K`` and cuts the log-bias with it. The price is
resolution: the estimate is smoothed over the concentration bandwidth
``NW*dk``, which is exactly the band where the Bouligand spectrum turns over
and ``dz`` is determined.

**Weighting.** ``sigma`` enters ``residuals`` as the uncertainty on ``Phi``,
and with the binned default it is ``s/sqrt(n_b)`` with ``n_b`` the cell count
in the annulus. ``n_b`` grows linearly with ``k``, so the default weight per
bin already rises as ``sqrt(k)``: the fit leans on the *short* wavelengths,
which is the opposite end of the spectrum from the one that carries ``dz``.
``sigma_tilt`` re-tilts it.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal.windows import dpss

sys.path.insert(0, str(Path.home() / "Global_CPD"))
import curie_config as cfg  # noqa: E402
import pycurious  # noqa: E402
from pycurious.grid import _dof_factor  # noqa: E402

#: sd of ``ln E``, ``E ~ Exp(1)``: the log-periodogram's own scatter, per cell,
#: independent of the spectrum. ``pi/sqrt(6) = 1.2825``.
SIGMA_LOG = np.pi / np.sqrt(6.0)

#: mean of ``ln E`` -- the Euler-Mascheroni constant, with a minus sign. Every
#: log-averaged spectrum sits this far below ``ln Phi``; ``C`` absorbs it.
LOG_BIAS = -np.euler_gamma

DX_M = 5000.0


# --------------------------------------------------------------------------
# geometry shared with pycurious
# --------------------------------------------------------------------------

def bin_edges(n, dx_km):
    """``kbins`` exactly as ``CurieGrid._taper_spectrum`` builds them."""
    dk = 2.0 * np.pi / n / dx_km
    return dk, np.arange(dk, dk * n / 2, dk)


def unique_cells(nr, nc):
    """Mask over the ``rfft2`` half plane keeping each Fourier cell once.

    ``rfft2`` already drops the conjugate half, but column 0 (and, for even
    ``nc``, the Nyquist column) is its own mirror: within it, row ``r`` and row
    ``-r`` are conjugates of one another and carry one measurement between
    them. Keeping both would count those cells twice -- harmless in the bulk,
    not harmless in the innermost annuli, which is where ``dz`` is decided.

    ``pycurious`` keeps them and compensates with the Hermitian factor of two
    baked into ``_TAPER_DOF``; an unbinned fit has no bin to compensate in, so
    it must drop them outright.
    """
    ncol = nc // 2 + 1
    keep = np.ones((nr, ncol), dtype=bool)
    half = nr // 2
    self_conj = [0] + ([ncol - 1] if nc % 2 == 0 else [])
    for col in self_conj:
        keep[half + 1:, col] = False
    return keep


def cell_wavenumbers(nr, nc, dk):
    """``|k|`` at every ``rfft2`` cell, in rad/km."""
    row = np.arange(nr)
    row[row > (nr - 1) // 2] -= nr
    return np.hypot((row * dk)[:, None], (np.arange(nc // 2 + 1) * dk)[None, :])


# --------------------------------------------------------------------------
# tapers
# --------------------------------------------------------------------------

def slepian(n, NW, K):
    """``K`` DPSS tapers of length ``n``, scaled to unit mean square.

    ``scipy`` normalises to unit *sum* of squares. Rescaling to unit mean
    square puts every taper here on the same footing as an untapered window, so
    ``C`` is comparable between variants instead of carrying each taper's power
    loss (``numpy.hanning`` alone costs ``ln(3/8)**2``).
    """
    return dpss(n, NW, K) * np.sqrt(n)


def taper_2d(name, n, NW=3.0, K=None):
    """``(K2, n, n)`` stack of separable 2-D tapers, and a label.

    A true 2-D Slepian is concentrated in a disk in wavenumber; this is the
    separable product ``h_i(x) h_j(y)``, which is what is usually done in
    practice and is what makes the cost ``K**2`` transforms rather than a
    2-D eigenproblem per window size. It concentrates in a square, so the
    corners of the ``(i, j)`` grid are the least well concentrated tapers in
    the stack.
    """
    if name == "hanning":
        h = np.hanning(n)[None, :]
    elif name == "none":
        h = np.ones((1, n))
    elif name == "dpss":
        K = int(2 * NW - 1) if K is None else K
        h = slepian(n, NW, K)
    else:
        raise ValueError(name)
    stack = np.einsum("ir,js->ijrs", h, h).reshape(-1, n, n)
    return stack


# --------------------------------------------------------------------------
# estimators: each returns (k, Phi, sigma) ready for `optimise(spectrum=...)`
# --------------------------------------------------------------------------

def cell_power(sub, taper="hanning", NW=3.0, K=None):
    """Taper-averaged power at every ``rfft2`` cell.

    One taper reproduces ``|rfft2(data * taper)|**2``; ``K**2`` DPSS tapers
    average the power over the stack *before* any log is taken.
    """
    n = sub.shape[0]
    stack = taper_2d(taper, n, NW, K)
    P = np.zeros((n, n // 2 + 1))
    for t in stack:
        P += np.abs(np.fft.rfft2(sub * t)) ** 2
    return P / len(stack), len(stack)


def hermitian_weight(nr, nc):
    """pycurious' cell weights: an interior ``rfft`` column stands for two
    full-plane cells, a self-mirrored one for a single cell."""
    ncol = nc // 2 + 1
    w = np.full(ncol, 2.0)
    w[0] = 1.0
    if nc % 2 == 0:
        w[-1] = 1.0
    return np.broadcast_to(w, (nr, ncol)).ravel()


def binned(sub, dx_km=DX_M / 1e3, taper="hanning", NW=3.0, K=None, kmax=0.25,
           sigma="empirical", tilt=0.0, sigma_weight_km=0.0, power=2.0,
           neff=None):
    """Radially binned log spectrum, binned exactly as pycurious bins it.

    ``taper="hanning"`` with ``sigma="empirical"``, ``tilt=0`` and
    ``sigma_weight_km=8`` reproduces the Global_CPD archive's ``_spectrum``
    to float64 (asserted in ``check_variants.py``).

    ``sigma``:
      ``"empirical"``  within-annulus scatter / sqrt(effective count) -- the
                       pycurious default
      ``"theory"``     ``pi/sqrt(6 n_eff)``, with ``n_eff`` per bin from
                       ``neff`` (measured by ``calibrate_dof.py``) or, failing
                       that, from ``_TAPER_DOF``

    A multitaper needs ``neff``. Its within-annulus scatter collapses --
    0.13 of the per-cell theory at ``NW=3, K=5`` -- because the estimate at
    neighbouring cells is built from overlapping bands, so the scatter no
    longer measures the bin mean's own variance and ``sigma="empirical"``
    understates it by 2.7x.
    """
    n = sub.shape[0]
    dk, kbins = bin_edges(n, dx_km)
    P, ntaper = cell_power(sub, taper, NW, K)
    # power=2 is ln|FFT|**2 -- take the log of the taper-averaged power, which
    # is the only order that buys the multitaper anything.
    rr = ((0.5 * power) * np.log(P)).ravel()

    kk = cell_wavenumbers(n, n, dk).ravel()
    weight = hermitian_weight(n, n)
    nb = kbins.size - 1
    idx = np.digitize(kk, kbins) - 1
    idx[(idx == nb) & (kk <= kbins[-1])] = nb - 1
    keep = (idx >= 0) & (idx < nb)
    idx, rr, kk, weight = idx[keep], rr[keep], kk[keep], weight[keep]

    counts = np.bincount(idx, weights=weight, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.bincount(idx, weights=weight * rr, minlength=nb) / counts
        kbar = np.bincount(idx, weights=weight * kk, minlength=nb) / counts
        dev = rr - S[idx]
        scatter = np.sqrt(
            np.bincount(idx, weights=weight * dev * dev, minlength=nb) / counts)

    # `_dof_factor` turns the full-plane count into an effective independent
    # count: it divides out the Hermitian factor of two and the fixed number of
    # cells a taper's main lobe costs.
    taper_fn = np.hanning if taper == "hanning" else None
    n_eff = counts / _dof_factor(taper_fn, counts)

    if sigma == "empirical":
        s = scatter / np.sqrt(n_eff)
    elif sigma == "theory":
        ne = n_eff if neff is None else np.asarray(neff, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = SIGMA_LOG * (0.5 * power) / np.sqrt(ne)
    else:
        raise ValueError(sigma)

    ok = counts > 0
    return _postprocess(kbar[ok], S[ok], s[ok], kmax, tilt, sigma_weight_km)


def unbinned(sub, dx_km=DX_M / 1e3, taper="hanning", NW=3.0, K=None, kmax=0.25,
             tilt=0.0, sigma_weight_km=0.0, power=2.0, thin=1):
    """Every unique Fourier cell, unaveraged, at the log-periodogram's own sigma.

    The band is the same one the bins cover -- ``[dk, kmax]`` -- so this is the
    same data as ``binned``, not more of it.

    ``sigma`` is ``pi/sqrt(6)`` per cell and constant: the log-periodogram's
    scatter does not depend on the spectrum, on ``k``, or on the taper (a taper
    correlates neighbouring cells, which changes the covariance of the fit, not
    the marginal variance of a cell). There is nothing to estimate.

    ``thin`` keeps every ``thin``-th cell, for the cost/precision trade-off.
    """
    n = sub.shape[0]
    dk, kbins = bin_edges(n, dx_km)
    P, ntaper = cell_power(sub, taper, NW, K)
    rr = (0.5 * power) * np.log(P)

    kk = cell_wavenumbers(n, n, dk)
    keep = unique_cells(n, n) & (kk >= kbins[0]) & (kk <= kbins[-1])
    k, Phi = kk[keep], rr[keep]

    order = np.argsort(k)
    k, Phi = k[order], Phi[order]
    if thin > 1:
        k, Phi = k[::thin], Phi[::thin]

    s = np.full(k.size, SIGMA_LOG * (0.5 * power))
    if ntaper > 1:
        raise ValueError("an unbinned multitaper fit has no calibrated sigma: "
                         "the per-cell variance depends on the taper overlap, "
                         "which is what the bins are there to average over")
    return _postprocess(k, Phi, s, kmax, tilt, sigma_weight_km)


def _postprocess(k, Phi, sigma, kmax, tilt, sigma_weight_km):
    """Band cut, then the weighting scheme.

    ``tilt`` multiplies ``sigma`` by ``(k/k[0])**tilt``, so a positive tilt
    down-weights short wavelengths and leans the fit on the long ones. The
    binned default already carries an implicit ``tilt = -0.5`` against the
    long wavelengths, because ``n_b`` grows linearly with ``k``; ``tilt = 0.5``
    cancels it and weights every bin equally.

    ``sigma_weight_km`` is Global_CPD's scheme, ``sigma**2 += (k**2 L**2)**2``,
    kept here so the archive's own choice is one of the things being compared.
    """
    good = np.isfinite(k) & np.isfinite(Phi) & np.isfinite(sigma)
    k, Phi, sigma = k[good], Phi[good], sigma[good]
    if kmax:
        cut = k <= kmax
        k, Phi, sigma = k[cut], Phi[cut], sigma[cut]
    if sigma_weight_km:
        sigma = np.sqrt(sigma ** 2 + ((k ** 2) * (sigma_weight_km ** 2)) ** 2)
    if tilt:
        # normalised to unit geometric mean over the retained band, so a tilt
        # only redistributes weight between wavenumbers. Without this it also
        # scales every sigma up, which leaves the estimate alone but makes the
        # reported uncertainty and the chi-squared incomparable -- and, with a
        # prior in the problem, quietly changes how much the prior is worth.
        f = (k / k[0]) ** tilt
        sigma = sigma * f / np.exp(np.mean(np.log(f)))
    return k, Phi, sigma


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def fitter(zt_pin=1.0, zt_sigma=0.05, beta_prior=None, beta_prior_sigma=None,
           dx_m=DX_M):
    """A pycurious fitter with no window behind it, for fitting a spectrum.

    Configured as the WDMAM L2 archive was: ``zt`` pinned at 1 km with a 0.05
    km prior, ``beta`` free unless a prior is given.
    """
    fit = cfg.FitConfig(kmax=None, sigma_weight_km=0.0, zt_fixed_km=zt_pin,
                        zt_fix_sigma=zt_sigma, zt_source="constant",
                        taper="hanning", detrend=True, projection="laea",
                        field="iso", upsample=4, dataset="wdmam",
                        beta_prior_sigma=beta_prior_sigma)
    return cfg.refit_fitter(fit, dx_m, beta_prior=beta_prior)


def fit(spectrum, g=None, x0=(3.0, 1.0, 10.0, 5.0), naive_cov=False):
    """Fit one ``(k, Phi, sigma)``. Returns ``(params, sigmas, extras)``.

    ``naive_cov`` returns ``(J^T J)^-1`` alongside, without the banded
    correction for correlation between neighbouring residuals. That correction
    is estimated from the residual *sequence*, which means something for bins
    ordered in ``k`` and nothing for a list of individual cells, so the naive
    covariance is the only one comparable between the two.
    """
    g = fitter() if g is None else g
    k, Phi, sigma = spectrum
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = g.optimise(1.0, 0.0, 0.0, beta=x0[0], zt=x0[1], dz=x0[2], C=x0[3],
                         spectrum=(k, Phi, sigma))
    x = np.array(out[:4])
    s = np.array(out[4:8])

    extras = {"nres": k.size, "warnings": [str(w.message)[:60] for w in caught]}
    r = g.residuals(x, k, Phi, sigma)
    extras["chi2_red"] = float(np.sum(r ** 2) / max(k.size - 4, 1))
    if naive_cov:
        J = g._jacobian(x, r, (k, Phi, sigma))
        try:
            cov = np.linalg.inv(J.T @ J)
            extras["sigma_naive"] = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            extras["sigma_naive"] = np.full(4, np.nan)
    return x, s, extras


def detrend(sub):
    """The separable plane fit Global_CPD uses; identical to pycurious' lstsq."""
    nr, nc = sub.shape
    i = np.arange(nr) - (nr - 1) / 2.0
    j = np.arange(nc) - (nc - 1) / 2.0
    mean = sub.mean()
    ci = (i * (sub.mean(axis=1) - mean)).sum() / (i * i).sum()
    cj = (j * (sub.mean(axis=0) - mean)).sum() / (j * j).sum()
    return sub - (mean + ci * i[:, None] + cj * j[None, :])


def centred(arr, cells):
    c = (arr.shape[-1] - 1) // 2
    h = cells // 2
    return np.asarray(arr[..., c - h:c + h + 1, c - h:c + h + 1], dtype=np.float64)


def synth(n, seed, beta=3.0, zt=1.0, dz=20.0, C=5.0, dx_km=DX_M / 1e3):
    """One synthetic window with a known answer, on the production cell size."""
    data, _ = pycurious.fractal_anomaly(n=n, dx=dx_km, beta=beta, zt=zt, dz=dz,
                                        C=C, seed=seed)
    return data
