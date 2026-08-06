"""
The posterior mesh against the WDMAM L2 archive, on every window rung.

162 mesh vertices x 40 rungs from 250 to 10,000 km, refitted from the cached
spectra so no FFT runs. Two things are compared against the archived
``dz_lo``/``dz_hi``, which were produced by ``profile``'s deviance scan:

* **Where the interval is finite, do the two agree?** They are different
  objects -- a deviance level set centred on the mode against an equal-tailed
  credible interval centred on the median -- so on the skewed ``dz`` posterior
  they should not agree exactly. The question is whether they agree in the way
  two constructions of the same thing should.
* **How often is the interval unbounded, rung by rung?** The short rungs are
  where ``dz`` is least constrained, and an unbounded interval is the honest
  answer there rather than a failure. A method that reports a confident number
  at 250 km where the data cannot support one is the worse method, so the
  ``inf`` count per rung is the headline rather than a footnote.

Disagreements are adjudicated by a 4-D chain -- **with a mixing diagnostic**.
An unmixed chain refereeing a disagreement is worth nothing, so it is run as
four chains from dispersed starts and reports the Gelman-Rubin R-hat and the
worst effective sample size; a verdict is only recorded where both pass.

Run with the BLAS pinned::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/score_mesh_L2.py
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "Global_CPD"))

import zarr  # noqa: E402
from zarr.storage import LocalStore  # noqa: E402

import curie_config as cfg  # noqa: E402

ARCHIVE = Path.home() / "Global_CPD" / "data" / "curie_wdmam_L2.zarr"
SPECTRA = Path.home() / "Global_CPD" / "data" / "spectra_wdmam_L2.zarr"

#: The archive's own profile level, so like is compared with like.
LEVEL = cfg.PROFILE_LEVEL

#: R-hat below this and ESS above it before a chain is allowed to referee.
RHAT_MAX = 1.05
ESS_MIN = 400.0


def load(root, spectra, window_km, vertex):
    group = spectra["w{:d}".format(int(round(window_km)))]
    k = np.asarray(group["k"][:, vertex], dtype=float)
    Phi = np.asarray(group["Phi"][:, vertex], dtype=float)
    sigma = np.asarray(group["sigma"][:, vertex], dtype=float)
    good = np.isfinite(k) & np.isfinite(Phi) & np.isfinite(sigma)
    return (k[good], Phi[good], sigma[good]) if good.sum() >= 8 else None


def ess(chain):
    """Effective sample size, by Geyer's initial positive sequence."""
    x = np.asarray(chain, dtype=float)
    n = x.size
    x = x - x.mean()
    variance = np.dot(x, x) / n
    if variance <= 0.0:
        return float(n)
    total, lag, rho = 0.0, 1, []
    while lag < n - 1:
        rho.append(np.dot(x[:-lag], x[lag:]) / (n * variance))
        lag += 1
        if len(rho) % 2 == 0:
            pair = rho[-2] + rho[-1]
            if pair < 0.0:
                break
            total += pair
    return float(n / (1.0 + 2.0 * total)) if total > -0.5 else float(n)


def rhat(chains):
    """Gelman-Rubin, over chains started from dispersed points.

    Within-chain variance against between-chain: a chain that has not found the
    rest of the posterior looks converged on its own and does not look
    converged next to three others.
    """
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    if m < 2 or n < 4:
        return np.inf
    means = chains.mean(axis=1)
    W = chains.var(axis=1, ddof=1).mean()
    B = n * means.var(ddof=1)
    if W <= 0.0:
        return np.inf
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def referee(fitter, spectrum, nsim, burnin):
    """A 4-D chain's dz interval, or None where it has not mixed."""
    x_hat = fitter._fit(fitter._initial_guess(spectrum), spectrum).x
    sigma = np.sqrt(np.abs(np.diag(fitter._covariance(x_hat, *spectrum))))
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)

    chains = []
    for c in range(4):
        offset = np.array([1.0, -1.0, 1.0, -1.0]) * ((-1) ** c) * (1 + c) * sigma
        start = np.clip(x_hat + offset, [0.1, 0.0, 0.5, -50.0], None)
        try:
            out = fitter.metropolis_hastings(
                0.0, 0.0, 0.0, nsim, burnin, seed=100 + c, spectrum=spectrum,
                beta=start[0], zt=start[1], dz=start[2], C=start[3],
            )
        except Exception:
            return None
        chains.append(np.asarray(out[2]))

    stacked = np.vstack(chains)
    if rhat(stacked) > RHAT_MAX or min(ess(c) for c in chains) < ESS_MIN:
        return None
    pooled = stacked.ravel()
    tail = 0.5 * (1.0 - LEVEL)
    return (float(np.quantile(pooled, tail)),
            float(np.quantile(pooled, 1.0 - tail)))


def main(limit, adjudicate, nsim, burnin):
    warnings.simplefilter("ignore")

    root = zarr.open_group(store=LocalStore(str(ARCHIVE)), mode="r")
    fit = cfg.FitConfig.from_dict(dict(root.attrs["fit"]))
    dx = float(root.attrs["dx_m"])
    beta_prior = np.asarray(root["qc/beta_prior"][:])
    zt_pin = cfg.read_zt_pin(root, fit)
    spectra = zarr.open_group(store=LocalStore(str(SPECTRA)), mode="r")

    windows = np.asarray(root["window_km"][:])
    dz_lo = cfg.read_param(root, "dz_lo")
    dz_hi = cfg.read_param(root, "dz_hi")
    npoints = dz_lo.shape[1]

    print("Posterior mesh against the archived deviance scan, WDMAM L2.")
    print("{} vertices x {} rungs, level {:.4f}. `inf` counts unbounded upper".format(
        npoints, windows.size, LEVEL))
    print("endpoints -- the honest answer where a window cannot bound dz.\n")
    print("  {:>7s} {:>5s} | {:>16s} | {:>16s} | {:>21s}".format(
        "window", "n", "scan (archived)", "mesh", "agreement"))
    print("  {:>7s} {:>5s} | {:>7s} {:>8s} | {:>7s} {:>8s} | {:>9s} {:>11s}".format(
        "km", "", "inf", "med width", "inf", "med width", "med |dlo|", "med |dhi|"))

    disagreements = []
    for iw, window_km in enumerate(windows):
        scan_inf = mesh_inf = used = 0
        scan_w, mesh_w, d_lo, d_hi = [], [], [], []

        for j in range(npoints if limit is None else min(limit, npoints)):
            spectrum = load(root, spectra, window_km, j)
            if spectrum is None:
                continue
            archived = (dz_lo[iw, j], dz_hi[iw, j])
            if not np.isfinite(archived[0]):
                continue

            pin = None if zt_pin is None else float(zt_pin[iw, j])
            bp = float(beta_prior[iw, j])
            g = cfg.refit_fitter(fit, dx, None if not np.isfinite(bp) else bp,
                                 zt_pin=pin)
            b0, z0 = g._x0()[0], g._x0()[1]
            try:
                _, _, lo, hi = g.profile(
                    0.0, 0.0, 0.0, "dz", level=LEVEL, method="mesh",
                    beta=b0, zt=z0, spectrum=spectrum,
                )
            except Exception:
                continue

            used += 1
            # the archive stores a non-finite bound as NaN, which is how an
            # unbounded scan endpoint survives int16 quantisation
            scan_unbounded = not np.isfinite(archived[1])
            scan_inf += scan_unbounded
            mesh_inf += not np.isfinite(hi)

            if not scan_unbounded:
                scan_w.append(archived[1] - archived[0])
            if np.isfinite(hi):
                mesh_w.append(hi - lo)
            if not scan_unbounded and np.isfinite(hi):
                d_lo.append(abs(lo - archived[0]))
                d_hi.append(abs(hi - archived[1]))
                width = archived[1] - archived[0]
                if width > 0 and (abs(lo - archived[0]) > 0.5 * width
                                  or abs(hi - archived[1]) > 0.5 * width):
                    disagreements.append((iw, j, window_km, archived, (lo, hi)))

        if not used:
            continue
        print("  {:7.0f} {:5d} | {:7d} {:8.2f} | {:7d} {:8.2f} | {:9.2f} {:11.2f}".format(
            window_km, used, scan_inf,
            np.median(scan_w) if scan_w else np.nan, mesh_inf,
            np.median(mesh_w) if mesh_w else np.nan,
            np.median(d_lo) if d_lo else np.nan,
            np.median(d_hi) if d_hi else np.nan))

    print("\n  {} vertex-rungs where the two differ by more than half the "
          "archived width".format(len(disagreements)))

    if adjudicate and disagreements:
        print("\n  Adjudicating with a 4-D chain: 4 dispersed starts, "
              "R-hat < {:.2f} and ESS > {:.0f} required.".format(RHAT_MAX, ESS_MIN))
        print("  {:>7s} {:>5s} | {:>16s} | {:>16s} | {:>16s} | {}".format(
            "window", "vtx", "scan", "mesh", "chain", "closer"))
        rng = np.random.default_rng(0)
        picks = rng.choice(len(disagreements),
                           size=min(adjudicate, len(disagreements)),
                           replace=False)
        verdict = {"scan": 0, "mesh": 0, "unmixed": 0}
        for pick in picks:
            iw, j, window_km, archived, mesh = disagreements[int(pick)]
            spectrum = load(root, spectra, window_km, j)
            pin = None if zt_pin is None else float(zt_pin[iw, j])
            bp = float(beta_prior[iw, j])
            g = cfg.refit_fitter(fit, dx, None if not np.isfinite(bp) else bp,
                                 zt_pin=pin)
            truth = referee(g, spectrum, nsim, burnin)
            if truth is None:
                verdict["unmixed"] += 1
                print("  {:7.0f} {:5d} | ({:6.2f},{:7.2f}) | ({:6.2f},{:7.2f}) "
                      "| {:>16s} | -".format(window_km, j, *archived, *mesh,
                                             "did not mix"))
                continue
            scan_err = abs(archived[0] - truth[0]) + abs(archived[1] - truth[1])
            mesh_err = abs(mesh[0] - truth[0]) + abs(mesh[1] - truth[1])
            closer = "mesh" if mesh_err < scan_err else "scan"
            verdict[closer] += 1
            print("  {:7.0f} {:5d} | ({:6.2f},{:7.2f}) | ({:6.2f},{:7.2f}) "
                  "| ({:6.2f},{:7.2f}) | {}".format(
                      window_km, j, *archived, *mesh, *truth, closer))
        print("\n  closer to the chain: mesh {}, scan {}, unmixed {}".format(
            verdict["mesh"], verdict["scan"], verdict["unmixed"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="vertices per rung, default all 162")
    parser.add_argument("--adjudicate", type=int, default=0,
                        help="how many disagreements to referee with a chain")
    parser.add_argument("--nsim", type=int, default=8000)
    parser.add_argument("--burnin", type=int, default=2000)
    args = parser.parse_args()
    main(args.limit, args.adjudicate, args.nsim, args.burnin)
