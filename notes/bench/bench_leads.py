"""
Three remaining efficiency leads, measured.

1. `warnings.catch_warnings()` in `residuals`, entered on every single residual
   evaluation. The warnings it suppresses are numpy floating-point warnings,
   which `np.errstate` handles far more cheaply.
2. `scipy.fft.rfft2(workers=...)` in place of `np.fft.rfft2` -- already a
   dependency, bit-identical, no planning.
3. Spectrum reuse when `optimise` and `profile` are called at the same
   centroid, which recomputes the whole spectrum the second time.
"""

import time
import warnings
import numpy as np
import scipy.fft
import pycurious
from pycurious.grid import bouligand2009


def timeit(fn, min_time=1.5, min_reps=5):
    fn()
    reps, el = 0, 0.0
    t0 = time.perf_counter()
    while el < min_time or reps < min_reps:
        fn()
        reps += 1
        el = time.perf_counter() - t0
    return el / reps


def lead1_warnings():
    print("=== 1. catch_warnings overhead per residual evaluation ===")
    kh = np.linspace(0.01, 1.5, 64)

    def with_catch():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bouligand2009(kh, 3.0, 1.0, 20.0, 5.0)

    def with_errstate():
        with np.errstate(all="ignore"):
            return bouligand2009(kh, 3.0, 1.0, 20.0, 5.0)

    def bare():
        return bouligand2009(kh, 3.0, 1.0, 20.0, 5.0)

    t_c = timeit(with_catch)
    t_e = timeit(with_errstate)
    t_b = timeit(bare)
    print("  bouligand2009 bare              : %8.2f us" % (t_b * 1e6))
    print("  + np.errstate                   : %8.2f us  (+%.2f us)"
          % (t_e * 1e6, (t_e - t_b) * 1e6))
    print("  + warnings.catch_warnings       : %8.2f us  (+%.2f us)"
          % (t_c * 1e6, (t_c - t_b) * 1e6))
    print("  -> catch_warnings costs %.2f us per call, %.0f%% of the bare model"
          % ((t_c - t_e) * 1e6, 100 * (t_c - t_e) / t_b))

    # the context managers alone, isolated
    print("  context manager alone: catch_warnings %6.2f us   errstate %6.2f us"
          % (timeit(_enter_catch) * 1e6, timeit(_enter_errstate) * 1e6))


def _enter_catch():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 0


def _enter_errstate():
    with np.errstate(all="ignore"):
        return 0


def lead2_scipy_fft(grid, xc, yc, windows):
    print("\n=== 2. scipy.fft(workers) in _FFT_spectrum, end to end ===")
    import pycurious.grid as G

    orig = np.fft.rfft2
    print("%-10s %6s %12s %12s %12s %9s"
          % ("window/km", "cells", "numpy/s", "scipy w=4/s", "scipy w=8/s", "best"))
    print("-" * 68)
    for w in windows:
        n = grid.subgrid(w, xc, yc).shape[0]
        ts = {}
        for label, fn in (
            ("numpy", orig),
            ("w4", lambda a: scipy.fft.rfft2(a, workers=4)),
            ("w8", lambda a: scipy.fft.rfft2(a, workers=8)),
        ):
            np.fft.rfft2 = fn
            ts[label] = timeit(lambda: grid.optimise(w, xc, yc), min_time=2.0)
        np.fft.rfft2 = orig
        best = ts["numpy"] / min(ts["w4"], ts["w8"])
        print("%-10.0f %6d %12.4f %12.4f %12.4f %8.2fx"
              % (w / 1e3, n, ts["numpy"], ts["w4"], ts["w8"], best), flush=True)


def lead3_spectrum_reuse(grid, xc, yc, windows):
    print("\n=== 3. optimise + profile at one centroid: spectrum computed twice ===")
    print("%-10s %6s %12s %12s %12s %9s"
          % ("window/km", "cells", "opt+prof/s", "spectrum/s", "if shared", "saving"))
    print("-" * 70)
    for w in windows:
        n = grid.subgrid(w, xc, yc).shape[0]
        t_both = timeit(
            lambda: (grid.optimise(w, xc, yc),
                     grid.profile(w, xc, yc, "CPD", npoints=11)),
            min_time=3.0, min_reps=3)
        t_spec = timeit(lambda: grid.window_spectrum(w, xc, yc, power=2.0),
                        min_time=1.5)
        print("%-10.0f %6d %12.4f %12.4f %12.4f %8.1f%%"
              % (w / 1e3, n, t_both, t_spec, t_both - t_spec,
                 100 * t_spec / t_both), flush=True)


def main():
    lead1_warnings()
    data, extent = pycurious.fractal_anomaly(2100, 1.0, beta=3.0, zt=1.0,
                                             dz=20.0, C=5.0, seed=1)
    grid = pycurious.CurieOptimiseBouligand(data, *extent)
    xc = 0.5 * (grid.xmin + grid.xmax)
    yc = 0.5 * (grid.ymin + grid.ymax)
    windows = (128e3, 256e3, 512e3, 1024e3, 2048e3)
    lead2_scipy_fft(grid, xc, yc, windows)
    lead3_spectrum_reuse(grid, xc, yc, (256e3, 1024e3, 2048e3))


if __name__ == "__main__":
    main()
