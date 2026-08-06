"""
Is a derived starting point better than the constant, and what does it cost?

This drives **pycurious itself**, not the parallel implementation in
``spectral_variants``. ``test_starting_values.py`` next door is the earlier
prototype, written before any of this was in the library; it answered "would
variable projection help", and it scored a standalone ``fit_varpro``. The
question now is narrower and has to be asked of the real code: does
``CurieOptimiseBouligand._initial_guess`` land in the basin the ``dz = 10``
constant missed, and what does the ladder cost against the fit it seeds.

Two measurements, per ``CLAUDE.md``:

* **Forward-model evaluations.** Counted by wrapping ``bouligand2009``, which is
  the one number a loaded machine cannot distort. The ladder and the fit are
  counted separately, because the whole argument for seeding rather than
  multi-starting is that a node is cheap next to a fit -- so the ratio is the
  claim under test, not a detail.
* **Wall clock**, with the BLAS pinned. Reported because the evaluation count
  misses scipy's own overhead, which at a small window is most of a fit.

The reference minimum is a dense multi-start over ``DENSE``, standing in for the
global minimum. A variant "misses" when its cost exceeds that by 0.1%.

Run with the BLAS pinned::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/score_starting_values.py
"""

import argparse
import time
import warnings

import numpy as np

import pycurious
from pycurious import optimise_bouligand as ob

#: Starting thicknesses of the reference multi-start. Wide and dense enough that
#: reaching its best cost is evidence of having found the global minimum, and
#: far too expensive to be a strategy anyone would ship.
DENSE = (2.5, 5.0, 10.0, 20.0, 30.0, 45.0, 65.0, 90.0, 130.0, 200.0)

#: The value `optimise` used to start every fit at, before this change.
CONSTANT = np.array([3.0, 1.0, 10.0, 5.0])

#: (label, n, dx_km, window_m, zt_pin). The first three stand for a window that
#: cannot see the layer, one that just can, and the production regime -- 5 km
#: cells at a few thousand km, which is what `~/Global_CPD` fits.
REGIMES = (
    ("200 km", 256, 2.0, 200e3, None),
    ("1000 km", 512, 2.0, 1000e3, None),
    ("4000 km", 811, 5.0, 4000e3, None),
    ("4000 km pinned", 811, 5.0, 4000e3, (1.0, 0.05)),
)


class Counter:
    """Count evaluations of the forward model, by wrapping it where it is used.

    ``optimise_bouligand`` imports ``bouligand2009`` by name, so rebinding it in
    that module catches every call the fit and the ladder make and none of the
    ones the synthetic generator makes.
    """

    def __init__(self):
        self.calls = 0
        self._real = ob.bouligand2009

    def __enter__(self):
        def counted(kh, beta, zt, dz, C):
            self.calls += 1
            return self._real(kh, beta, zt, dz, C)

        ob.bouligand2009 = counted
        return self

    def __exit__(self, *exc):
        ob.bouligand2009 = self._real
        return False

    def take(self):
        calls, self.calls = self.calls, 0
        return calls


def grid_for(n, dx, dz_true, seed, zt_pin):
    data, extent = pycurious.fractal_anomaly(
        n=n, dx=dx, beta=3.0, zt=1.0, dz=dz_true, C=5.0, seed=seed
    )
    g = pycurious.CurieOptimiseBouligand(data, *extent)
    if zt_pin is not None:
        g.add_prior(zt=zt_pin)
    return g


def run_case(g, spectrum, x0_of):
    """One strategy on one spectrum: seed, fit, and account for both."""
    with Counter() as counter:
        start = time.perf_counter()
        x0 = x0_of()
        seed_calls = counter.take()
        result = g._fit(x0, spectrum)
        elapsed = time.perf_counter() - start
        fit_calls = counter.take()
    return x0, result, seed_calls, fit_calls, elapsed


def main(nseed, thicknesses):
    warnings.simplefilter("ignore")

    print("Reaching the global minimum, and what it costs to get there.")
    print("`miss` counts seeds whose cost exceeds a dense {}-point multi-start "
          "by >0.1%.".format(len(DENSE)))
    print("`seed`/`fit` are evaluations of bouligand2009, per fit.\n")

    header = ("  {:15s} {:>4s} {:>4s} | {:>8s} {:>5s} {:>15s} {:>6s} {:>5s} "
              "{:>5s} {:>7s}".format("regime", "dz", "bins", "start", "miss",
                                     "dz recovered", "|err|", "seed", "fit",
                                     "ms"))

    for label, n, dx, window, zt_pin in REGIMES:
        print(header)
        for dz_true in thicknesses:
            rows = {name: dict(miss=0, dz=[], seed=[], fit=[], ms=0.0)
                    for name in ("constant", "derived")}
            nbins = 0
            better = 0

            for seed in range(nseed):
                g = grid_for(n, dx, dz_true, seed, zt_pin)
                xc, yc = g.xcoords.mean(), g.ycoords.mean()
                spectrum = g.window_spectrum(
                    window, xc, yc, taper=np.hanning, power=2.0
                )
                nbins = spectrum[0].size

                best = min(
                    g._fit(np.array([3.0, 1.0, node, 5.0]), spectrum).cost
                    for node in DENSE
                )

                costs = {}
                for name, x0_of in (
                    ("constant", lambda: CONSTANT.copy()),
                    ("derived", lambda: g._initial_guess(spectrum)),
                ):
                    _, result, seed_calls, fit_calls, elapsed = run_case(
                        g, spectrum, x0_of
                    )
                    row = rows[name]
                    row["miss"] += result.cost > best * 1.001
                    row["dz"].append(result.x[2])
                    row["seed"].append(seed_calls)
                    row["fit"].append(fit_calls)
                    row["ms"] += 1e3 * elapsed
                    costs[name] = result.cost

                better += costs["derived"] < costs["constant"] * (1.0 - 1e-9)

            for name, row in rows.items():
                dz = np.array(row["dz"])
                print("  {:15s} {:4.0f} {:4d} | {:>8s} {:4d}/{:d} "
                      "{:7.2f} +/- {:5.2f} {:6.2f} {:5.1f} {:5.1f} {:7.2f}".format(
                          label, dz_true, nbins, name, row["miss"], nseed,
                          dz.mean(), dz.std(ddof=1),
                          np.median(np.abs(dz - dz_true)),
                          np.mean(row["seed"]), np.mean(row["fit"]),
                          row["ms"] / nseed))
            print("  {:15s} {:4.0f} {:4d} | {:>8s} {:4d}/{:d}".format(
                "", dz_true, nbins, "lower cost", better, nseed))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--dz", type=float, nargs="+",
                        default=[10.0, 20.0, 30.0, 45.0])
    args = parser.parse_args()
    main(args.seeds, args.dz)
