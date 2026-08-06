"""
Breakdown inside `_FFT_spectrum`: the transform versus the radial binning.

Even a free FFT only helps `window_spectrum` by whatever share the transform
holds here -- the digitize/bincount pass over every cell of the half plane is
the other half of the routine.
"""

import time
import numpy as np
import pycurious

DX = 1.0
NGRID = 2100
WINDOWS = [64e3, 128e3, 256e3, 512e3, 1024e3, 2048e3]


def timeit(fn, min_time=0.6, min_reps=3):
    fn()
    reps, elapsed = 0, 0.0
    t0 = time.perf_counter()
    while elapsed < min_time or reps < min_reps:
        fn()
        reps += 1
        elapsed = time.perf_counter() - t0
    return elapsed / reps


def main():
    data, extent = pycurious.fractal_anomaly(
        NGRID, DX, beta=3.0, zt=1.0, dz=20.0, C=5.0, seed=1
    )
    grid = pycurious.CurieOptimiseBouligand(data, *extent)
    xc = 0.5 * (grid.xmin + grid.xmax)
    yc = 0.5 * (grid.ymin + grid.ymax)

    print("%-10s %7s %12s %12s %12s %10s %10s"
          % ("window/km", "cells", "_FFT_spec/s", "rfft2/s", "binning/s",
             "fft %", "bin %"))
    print("-" * 80)

    for w in WINDOWS:
        sub = grid.subgrid(w, xc, yc)
        n = sub.shape[0]
        vtaper, dk, kbins = grid._taper_spectrum(sub, np.hanning) \
            if hasattr(grid, "_taper_spectrum") else (None, None, None)
        if vtaper is None:
            t = np.hanning(n)
            vtaper = np.outer(t, t)
            dk = 2.0 * np.pi / n / (grid.dx * 1.0e-3)
            kbins = np.arange(dk, dk * n / 2, dk)

        t_all = timeit(lambda: grid._FFT_spectrum(sub, vtaper, dk, kbins, 2.0))
        t_fft = timeit(lambda: np.abs(np.fft.rfft2(sub * vtaper)))
        t_bin = t_all - t_fft

        print("%-10.0f %7d %12.5f %12.5f %12.5f %9.1f%% %9.1f%%"
              % (w / 1e3, n, t_all, t_fft, t_bin,
                 100 * t_fft / t_all, 100 * t_bin / t_all), flush=True)


if __name__ == "__main__":
    main()
