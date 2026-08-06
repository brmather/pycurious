"""
Does re-seeding `sensitivity` per realisation report an honest spread?

`sensitivity` used to fit the unresampled spectrum once and start every Monte
Carlo realisation from that solution -- "every resampled spectrum lands in the
same basin". Where the misfit is bimodal in ``dz`` that assumption fails, and
the ensemble reports a spread conditional on a basin it never revisits.

The only thing that calibrates a reported uncertainty is the spread over
**independent realisations of the field**, so that is the reference here. Three
numbers per case:

* ``ensemble``  -- sd of the fitted ``dz`` over independent synthetics. Truth.
* ``locked``    -- sd from `sensitivity` with the start supplied, which holds it
                   fixed across realisations and so reproduces the old
                   behaviour exactly.
* ``reseeded``  -- sd from `sensitivity` as it now is, deriving a start from
                   each resampled spectrum.

A ratio of 1 means the routine reports what an independent repeat would find.
Below 1 means it understates.

Run with the BLAS pinned::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notes/bench/score_sensitivity.py
"""

import argparse
import warnings

import numpy as np

import pycurious

#: (label, n, dx_km, window_m). The thick layer at a large window is where the
#: misfit is bimodal in dz and the locked start has something to be locked into.
CASES = (
    ("4000 km, dz 45", 811, 5.0, 4000e3, 45.0),
    ("4000 km, dz 30", 811, 5.0, 4000e3, 30.0),
    ("1000 km, dz 20", 512, 2.0, 1000e3, 20.0),
)


def fitted(n, dx, window, dz_true, seed):
    data, extent = pycurious.fractal_anomaly(
        n=n, dx=dx, beta=3.0, zt=1.0, dz=dz_true, C=5.0, seed=seed
    )
    g = pycurious.CurieOptimiseBouligand(data, *extent)
    xc, yc = g.xcoords.mean(), g.ycoords.mean()
    out = g.optimise(window, xc, yc, taper=np.hanning)
    return g, xc, yc, out


def main(nfield, nsim):
    warnings.simplefilter("ignore")

    print("Reported spread in dz against the spread over independent fields.")
    print("`locked` supplies the start, which is what the routine used to do "
          "for every realisation.\n")
    print("  {:16s} {:>9s} | {:>9s} {:>6s} | {:>9s} {:>6s}".format(
        "case", "ensemble", "locked", "ratio", "reseeded", "ratio"))

    for label, n, dx, window, dz_true in CASES:
        ensemble, locked, reseeded = [], [], []

        for seed in range(nfield):
            g, xc, yc, out = fitted(n, dx, window, dz_true, seed)
            ensemble.append(out[2])

            spectrum = g.last_spectrum
            held = g.sensitivity(
                window, xc, yc, nsim, beta=out[0], zt=out[1], dz=out[2],
                C=out[3], taper=np.hanning, seed=100 + seed, spectrum=spectrum,
            )[2]
            free = g.sensitivity(
                window, xc, yc, nsim, taper=np.hanning, seed=100 + seed,
                spectrum=spectrum,
            )[2]
            locked.append(np.std(held, ddof=1))
            reseeded.append(np.std(free, ddof=1))

        truth = float(np.std(ensemble, ddof=1))
        med_locked = float(np.median(locked))
        med_reseeded = float(np.median(reseeded))
        print("  {:16s} {:9.2f} | {:9.2f} {:6.2f} | {:9.2f} {:6.2f}".format(
            label, truth, med_locked, med_locked / truth,
            med_reseeded, med_reseeded / truth))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=int, default=16)
    parser.add_argument("--nsim", type=int, default=40)
    args = parser.parse_args()
    main(args.fields, args.nsim)
