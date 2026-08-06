"""
Collapsed (Rao-Blackwellised) sampling of the Bouligand posterior.

``metropolis_hastings`` walks all four parameters. Two of them need not be
walked at all. Writing the forward model out,

    Phi = C*1 + zt*(-2k) + h(k; beta, dz)

``C`` and ``zt`` are linear coefficients on basis vectors that do not depend on
``(beta, dz)``, so at fixed ``(beta, dz)`` their conditional posterior is an
exact 2-D Gaussian and can be integrated out in closed form.

**The marginal is the reduced misfit, with nothing left over.** Writing
``p = (C, zt)``, the whitened residual is ``G p - y`` with

    G = [1/sigma, -2k/sigma]        y = (Phi - h(beta, dz))/sigma

plus one row per Gaussian prior on ``C`` or ``zt``. Then

    0.5||G p - y||^2 = 0.5 (p - p_hat)^T A (p - p_hat) + F(beta, dz)

with ``A = G^T G`` and ``p_hat = A^-1 G^T y``, and integrating ``p`` out gives

    -log p(beta, dz | data) = F(beta, dz) + 0.5 log det A + const.

``G`` is built from ``k``, ``sigma`` and the prior widths **only**, so ``A`` --
and hence ``log det A`` -- does not depend on ``(beta, dz)``. The whole
correction is a constant, and the collapsed target is exactly the reduced misfit
``CurieOptimiseBouligand._solve_linear`` already returns. This is the same
identity the derived starting point rests on, used for sampling instead of for
one argmin.

Why bother: the four parameters lie on a ridge -- ``beta``-``zt`` at about
-0.92, ``zt``-``C`` at about +0.87 -- and the 4-D chain needs a
covariance-shaped proposal to crawl along it. Integrating out ``C`` and ``zt``
does not shape the ridge better; it removes it.

**And then the chain is worth removing too.** Collapsing buys about 1.9x the
effective sample per step, which is real but modest -- upstream's proposal
already follows the ridge, so there was less left on the table than the ridge
suggests. The larger result is that two dimensions is small enough to
*integrate* rather than sample: a 24x24 mesh over ``(beta, dz)`` reproduces the
chain's moments to four decimal places for **576** evaluations of the forward
model against the chain's 24,000, in 0.06 s against 2 to 4 s, with no
autocorrelation, no burn-in and no seed. `quadrature` does that; the sampler
below is kept because it is what shows the mesh is right.

**One wrinkle, handled exactly.** ``zt >= 0`` is a real bound, so the conditional
is a Gaussian truncated to a half plane and its mass
``Phi_cdf(p_hat_zt / s_zt)`` *does* depend on ``(beta, dz)``. That term is in the
target rather than assumed away, and the run reports how far from negligible it
got.

This is a prototype. It is not in the library and should not be until the
marginals below are shown to match the 4-D chain on more than the cases here --
in particular on a band-limited spectrum with a `zt` pin, which is the regime
that matters and which is not covered.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/collapsed_mcmc.py
"""

import argparse
import time
import warnings

import numpy as np
from scipy import stats

import pycurious
from pycurious import optimise_bouligand as ob

#: Acceptance the burn-in tunes towards, as upstream.
TARGET_ACCEPTANCE = 0.234

#: Beyond this many conditional standard deviations from the `zt` bound the
#: truncated mass is 1 to double precision, and the CDF is pure overhead.
NEGLIGIBLE = 8.0

PARAMETERS = ("beta", "zt", "dz", "C")


class Counter:
    """Evaluations of the forward model, which is what both chains spend."""

    def __enter__(self):
        self.calls = 0
        self._real = ob.bouligand2009

        def counted(*args):
            self.calls += 1
            return self._real(*args)

        ob.bouligand2009 = counted
        return self

    def __exit__(self, *exc):
        ob.bouligand2009 = self._real
        return False


def make_target(grid, spectrum):
    """The collapsed target, and the constants the sampler needs alongside it.

    Returns ``(target, conditional_cov, conditional_chol, state)``. ``target``
    maps ``(beta, dz)`` to ``(log posterior, conditional mean of (C, zt))``.

    This does not call `_solve_linear`, although it computes the same thing:
    that method clamps ``zt`` onto its bound, and the collapsed target wants the
    *unclamped* conditional mean, with the bound carried by the truncation term
    instead. `check_target_matches_library` holds the two together where the
    clamp does not fire.
    """
    kh, Phi, sigma = spectrum
    prior = grid.prior

    column_C = 1.0 / sigma
    column_zt = -2.0 * kh / sigma

    A = np.array([
        [np.dot(column_C, column_C), np.dot(column_C, column_zt)],
        [np.dot(column_zt, column_C), np.dot(column_zt, column_zt)],
    ])
    linear_prior = []
    for index, name in ((0, "C"), (1, "zt")):
        args = prior.get(name)
        if args is not None:
            A[index, index] += 1.0 / args[1] ** 2
            linear_prior.append((index, args[0], args[1]))

    conditional_cov = np.linalg.inv(A)
    conditional_chol = np.linalg.cholesky(conditional_cov)
    sd_zt = np.sqrt(conditional_cov[1, 1])

    # priors on the two coordinates the chain walks are constants of the linear
    # solve, but not of the target
    walked_prior = [
        (index, prior[name]) for index, name in ((0, "beta"), (1, "dz"))
        if prior.get(name) is not None
    ]

    lower, upper = grid._bound_arrays()
    lo = np.array([lower[0], lower[2]])
    hi = np.array([upper[0], upper[2]])

    state = {"min_mass": 1.0, "zt_floor": lower[1]}

    def target(theta):
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf, None
        beta, dz = float(theta[0]), float(theta[1])

        with np.errstate(all="ignore"):
            shape = ob.bouligand2009(kh, beta, 0.0, dz, 0.0)
        y = (Phi - shape) / sigma
        if not np.all(np.isfinite(y)):
            return -np.inf, None

        rhs = np.array([np.dot(column_C, y), np.dot(column_zt, y)])
        for index, loc, scale in linear_prior:
            rhs[index] += loc / scale ** 2
        mean = conditional_cov @ rhs

        residual = mean[0] * column_C + mean[1] * column_zt - y
        cost = 0.5 * np.dot(residual, residual)
        for index, loc, scale in linear_prior:
            cost += 0.5 * ((mean[index] - loc) / scale) ** 2
        for index, (loc, scale) in walked_prior:
            cost += 0.5 * ((theta[index] - loc) / scale) ** 2

        reduced = (mean[1] - state["zt_floor"]) / sd_zt
        if reduced < NEGLIGIBLE:
            mass = stats.norm.cdf(reduced)
            state["min_mass"] = min(state["min_mass"], float(mass))
            if mass <= 0.0:
                return -np.inf, None
            cost -= np.log(mass)

        return -cost, mean

    return target, conditional_cov, conditional_chol, state


def check_target_matches_library(grid, spectrum):
    """The collapsed target must be the reduced misfit `optimise` minimises.

    Otherwise this samples a different posterior from the one the rest of the
    package reports, and every comparison below is meaningless. Checked away
    from the `zt` bound, where `_solve_linear`'s clamp does not fire and the
    truncation term is exactly 1, so the two are the same quantity.
    """
    kh, Phi, sigma = spectrum
    target, cov, _, _ = make_target(grid, spectrum)
    sd_zt = np.sqrt(cov[1, 1])

    worst, checked = 0.0, 0
    for beta, dz in ((3.0, 20.0), (2.0, 8.0), (4.0, 60.0), (2.5, 35.0)):
        _, zt, cost = grid._solve_linear(kh, Phi, sigma, beta, dz, grid.prior)
        if zt < NEGLIGIBLE * sd_zt:
            continue
        value, _ = target(np.array([beta, dz]))
        worst = max(worst, abs(-value - cost) / max(abs(cost), 1.0))
        checked += 1
    return worst, checked


def collapsed_chain(grid, spectrum, nsim, burnin, seed, x_hat, proposal):
    """Metropolis in (beta, dz); (C, zt) drawn from their exact conditional.

    Returns samples in the same ``(beta, zt, dz, C)`` order as
    `metropolis_hastings`, so the two are directly comparable.
    """
    rng = np.random.default_rng(seed)
    target, _, conditional_chol, state = make_target(grid, spectrum)

    theta = np.array([x_hat[0], x_hat[2]], dtype=float)
    F, mean = target(theta)
    chol = np.linalg.cholesky(proposal) * (2.38 / np.sqrt(2))

    scale = 1.0
    accepted_burn = 0
    for i in range(int(burnin)):
        proposed = theta + scale * chol @ rng.normal(size=2)
        F1, mean1 = target(proposed)
        take = bool(np.isfinite(F1) and np.log(rng.random()) < F1 - F)
        if take:
            theta, F, mean = proposed, F1, mean1
        accepted_burn += take
        scale *= np.exp((take - TARGET_ACCEPTANCE) / (i + 1.0) ** 0.6)

    samples = np.empty((int(nsim), 4))
    accepted = 0
    rejected_draws = 0
    for i in range(int(nsim)):
        proposed = theta + scale * chol @ rng.normal(size=2)
        F1, mean1 = target(proposed)
        take = bool(np.isfinite(F1) and np.log(rng.random()) < F1 - F)
        if take:
            theta, F, mean = proposed, F1, mean1
        accepted += take

        # (C, zt) from the conditional, rejecting the half plane below the `zt`
        # bound. The target already carries the mass of that rejection, so the
        # pair is an exact draw from the truncated conditional rather than an
        # approximation to it.
        while True:
            draw = mean + conditional_chol @ rng.normal(size=2)
            if draw[1] >= state["zt_floor"]:
                break
            rejected_draws += 1
        samples[i] = (theta[0], draw[1], theta[1], draw[0])

    return samples, {
        "acceptance": accepted / max(int(nsim), 1),
        "burnin_acceptance": accepted_burn / max(int(burnin), 1),
        "min_truncation_mass": state["min_mass"],
        "conditional_rejections": rejected_draws,
    }


def quadrature(grid, spectrum, x_hat, proposal, nodes, span):
    """The 2-D posterior on a mesh, with no chain at all.

    Once ``(C, zt)`` are integrated out the posterior lives in two dimensions,
    and two dimensions is small enough to evaluate rather than sample. A
    ``nodes x nodes`` mesh over ``(beta, dz)`` gives the joint density directly;
    the moments below are sums against it, with no autocorrelation, no burn-in
    and no seed. ``(C, zt)`` follow from the same conditional the sampler draws
    from -- their mean is the density-weighted mean of ``p_hat``, and their
    variance adds the conditional variance to the spread of ``p_hat`` by the
    law of total variance.

    The mesh is centred on the mode and stretched by ``span`` marginal standard
    deviations, further above in ``dz`` because that tail is the long one. The
    mass left outside is reported: a quadrature that has clipped the tail is
    wrong in exactly the way a chain is not, so it has to be checked rather
    than assumed.
    """
    target, conditional_cov, _, state = make_target(grid, spectrum)

    sd = np.sqrt(np.diag(proposal))
    beta_axis = np.linspace(
        x_hat[0] - span * sd[0], x_hat[0] + span * sd[0], nodes
    )
    lower, upper = grid._bound_arrays()
    dz_axis = np.linspace(
        max(x_hat[2] - span * sd[1], lower[2]),
        min(x_hat[2] + 2.0 * span * sd[1], upper[2]),
        nodes,
    )

    log_density = np.full((nodes, nodes), -np.inf)
    means = np.zeros((nodes, nodes, 2))
    for i, beta in enumerate(beta_axis):
        for j, dz in enumerate(dz_axis):
            value, mean = target(np.array([beta, dz]))
            log_density[i, j] = value
            if mean is not None:
                means[i, j] = mean

    weight = np.exp(log_density - np.max(log_density))
    weight /= weight.sum()

    edge = (weight[0].sum() + weight[-1].sum()
            + weight[:, 0].sum() + weight[:, -1].sum())

    B, D = np.meshgrid(beta_axis, dz_axis, indexing="ij")
    moments = {}
    for name, values in (("beta", B), ("dz", D),
                         ("C", means[..., 0]), ("zt", means[..., 1])):
        mean = float((weight * values).sum())
        variance = float((weight * (values - mean) ** 2).sum())
        if name in ("C", "zt"):
            # law of total variance: the spread of the conditional mean plus
            # the conditional variance, which the mesh does not otherwise see
            variance += conditional_cov[0 if name == "C" else 1,
                                        0 if name == "C" else 1]
        moments[name] = (mean, np.sqrt(variance))

    return moments, {
        "evaluations": nodes * nodes,
        "edge_mass": float(edge),
        "min_truncation_mass": state["min_mass"],
    }


def ess(chain):
    """Effective sample size, by Geyer's initial positive sequence."""
    x = np.asarray(chain, dtype=float)
    n = x.size
    x = x - x.mean()
    variance = np.dot(x, x) / n
    if variance <= 0.0:
        return float(n)

    rho, total, lag = [], 0.0, 1
    while lag < n - 1:
        rho.append(np.dot(x[:-lag], x[lag:]) / (n * variance))
        lag += 1
        if len(rho) % 2 == 0:
            pair = rho[-2] + rho[-1]
            if pair < 0.0:
                break
            total += pair
    return float(n / (1.0 + 2.0 * total)) if total > -0.5 else float(n)


def report(label, samples, elapsed, calls, extra=""):
    print("  {:11s} {:7.2f} s {:8d} evals  {}".format(
        label, elapsed, calls, extra))
    print("    {:>6s} {:>10s} {:>9s} {:>8s} {:>9s} {:>9s}".format(
        "", "mean", "sd", "ESS", "ESS/1000", "ESS/s"))
    out = {}
    for j, name in enumerate(PARAMETERS):
        n_eff = ess(samples[:, j])
        out[name] = (n_eff, n_eff / elapsed)
        print("    {:>6s} {:10.4f} {:9.4f} {:8.0f} {:9.0f} {:9.1f}".format(
            name, samples[:, j].mean(), samples[:, j].std(ddof=1),
            n_eff, 1000.0 * n_eff / samples.shape[0], n_eff / elapsed))
    return out


def compare(full, coll, full_ess, coll_ess):
    """Do the two chains sample the same distribution?

    KS assumes independent draws and an MCMC chain has none: at 8000 correlated
    steps with an effective size near 500, KS on the raw samples calls a
    statistic of 0.03 significant when the two chains are indistinguishable.
    Each chain is thinned to its own ESS first, which is the number of
    independent draws it actually carries.
    """
    print("\n  Same distribution? KS on chains thinned to their own ESS --")
    print("  on the raw correlated draws it rejects everything.")
    print("    {:>6s} {:>11s} {:>11s} {:>10s} {:>9s} {:>8s} {:>7s} {:>7s}".format(
        "", "full mean", "collapsed", "d mean/sd", "sd ratio", "ESS gain",
        "KS", "p"))
    for j, name in enumerate(PARAMETERS):
        a, b = full[:, j], coll[:, j]
        thin_a = a[:: max(int(round(a.size / full_ess[name][0])), 1)]
        thin_b = b[:: max(int(round(b.size / coll_ess[name][0])), 1)]
        result = stats.ks_2samp(thin_a, thin_b)
        pooled = 0.5 * (a.std(ddof=1) + b.std(ddof=1))
        print("    {:>6s} {:11.4f} {:11.4f} {:10.3f} {:9.3f} {:8.2f}x "
              "{:7.4f} {:7.4f}".format(
                  name, a.mean(), b.mean(), (b.mean() - a.mean()) / pooled,
                  b.std(ddof=1) / a.std(ddof=1),
                  coll_ess[name][0] / full_ess[name][0],
                  result.statistic, result.pvalue))


def case_synthetic(n, dx, dz_true, window, seed, prior=None):
    data, extent = pycurious.fractal_anomaly(
        n=n, dx=dx, beta=3.0, zt=1.0, dz=dz_true, C=5.0, seed=seed
    )
    grid = pycurious.CurieOptimiseBouligand(data, *extent)
    if prior:
        grid.add_prior(**prior)
    xc, yc = grid.xcoords.mean(), grid.ycoords.mean()
    spectrum = grid.window_spectrum(window, xc, yc, taper=np.hanning, power=2.0)
    return grid, spectrum


CASES = (
    ("1000 km, dz 20", dict(n=512, dx=2.0, dz_true=20.0, window=1000e3, seed=1)),
    ("4000 km, dz 45", dict(n=811, dx=5.0, dz_true=45.0, window=4000e3, seed=1)),
    ("4000 km, dz 20, zt pinned",
     dict(n=811, dx=5.0, dz_true=20.0, window=4000e3, seed=1,
          prior=dict(zt=(1.0, 0.05)))),
)


def main(nsim, burnin, seed, nodes):
    warnings.simplefilter("ignore")

    for label, kwargs in CASES:
        grid, spectrum = case_synthetic(**kwargs)
        kh, Phi, sigma = spectrum

        x_hat = grid._fit(grid._initial_guess(spectrum), spectrum).x
        covariance = grid._covariance(x_hat, kh, Phi, sigma)
        proposal = covariance[np.ix_([0, 2], [0, 2])]

        print("\n{}\n{}".format(label, "=" * len(label)))
        print("  mode  beta {:.3f}  zt {:.3f}  dz {:.3f}  C {:.3f}".format(*x_hat))
        worst, checked = check_target_matches_library(grid, spectrum)
        print("  collapsed target vs _solve_linear: max relative difference "
              "{:.2e} over {} points".format(worst, checked))

        with Counter() as counter:
            start = time.perf_counter()
            full = np.array(grid.metropolis_hastings(
                0.0, 0.0, 0.0, nsim, burnin, seed=seed, spectrum=spectrum
            )).T
            full_time = time.perf_counter() - start
            full_calls = counter.calls

        with Counter() as counter:
            start = time.perf_counter()
            coll, diagnostics = collapsed_chain(
                grid, spectrum, nsim, burnin, seed, x_hat, proposal
            )
            coll_time = time.perf_counter() - start
            coll_calls = counter.calls

        full_ess = report("full 4-D", full, full_time, full_calls)
        coll_ess = report(
            "collapsed", coll, coll_time, coll_calls,
            "acc {:.2f}, min P(zt>=bound) {:.4f}, rejected draws {}".format(
                diagnostics["acceptance"],
                diagnostics["min_truncation_mass"],
                diagnostics["conditional_rejections"]))
        compare(full, coll, full_ess, coll_ess)

        start = time.perf_counter()
        moments, mesh = quadrature(grid, spectrum, x_hat, proposal, nodes, 6.0)
        mesh_time = time.perf_counter() - start

        print("\n  No chain at all: the 2-D posterior on a {0}x{0} mesh, "
              "{1:.3f} s, {2} evaluations".format(
                  nodes, mesh_time, mesh["evaluations"]))
        print("  mass outside the mesh {:.2e} (a clipped tail is how this goes "
              "wrong, so it is reported rather than assumed)".format(
                  mesh["edge_mass"]))
        print("    {:>6s} {:>11s} {:>11s} {:>10s} {:>9s}".format(
            "", "chain mean", "mesh mean", "d mean/sd", "sd ratio"))
        for name in PARAMETERS:
            j = PARAMETERS.index(name)
            mean, sd = moments[name]
            chain_mean = coll[:, j].mean()
            chain_sd = coll[:, j].std(ddof=1)
            print("    {:>6s} {:11.4f} {:11.4f} {:10.3f} {:9.3f}".format(
                name, chain_mean, mean, (mean - chain_mean) / chain_sd,
                sd / chain_sd))

        # A mesh that has not converged is wrong in a way no comparison against
        # a noisy chain would show, so check it against a finer one directly.
        fine, _ = quadrature(grid, spectrum, x_hat, proposal, 4 * nodes, 6.0)
        drift = max(
            abs(moments[name][0] - fine[name][0]) / fine[name][1]
            for name in PARAMETERS
        )
        ratio = max(abs(moments[name][1] / fine[name][1] - 1.0)
                    for name in PARAMETERS)
        print("  against a {0}x{0} mesh: means agree to {1:.1e} sd, sds to "
              "{2:.1e}".format(4 * nodes, drift, ratio))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsim", type=int, default=20000)
    parser.add_argument("--burnin", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1)
    # 24 is where the moments stop moving; 32 is that with a margin
    parser.add_argument("--nodes", type=int, default=32)
    args = parser.parse_args()
    main(args.nsim, args.burnin, args.seed, args.nodes)
