"""
Where the time actually goes in `optimise()` and `profile()`, and what an
FFTW swap could win.

`CurieGrid.subgrid` returns `nw//2*2 + 1` cells, so every transform pycurious
performs is on an ODD square. Those are the sizes benchmarked here, not the
powers of two an FFT library is usually shown off on.
"""

import cProfile
import pstats
import time
import numpy as np
import scipy.fft
import pyfftw

import pycurious

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(600)

DX = 1.0  # cell size in km (fractal_anomaly takes km; extent comes back in m)
NGRID = 2100  # full synthetic grid, cells
WINDOWS = [64e3, 128e3, 256e3, 512e3, 1024e3, 2048e3]


def timeit(fn, min_time=1.0, min_reps=3):
    fn()
    reps, elapsed = 0, 0.0
    t0 = time.perf_counter()
    while elapsed < min_time or reps < min_reps:
        fn()
        reps += 1
        elapsed = time.perf_counter() - t0
    return elapsed / reps


def build_grid():
    print("generating a %d x %d synthetic ..." % (NGRID, NGRID), flush=True)
    data, extent = pycurious.fractal_anomaly(
        NGRID, DX, beta=3.0, zt=1.0, dz=20.0, C=5.0, seed=1
    )
    return pycurious.CurieOptimiseBouligand(data, *extent)


def profile_call(fn):
    """cProfile one call, return (total, {label: cumulative seconds})."""
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    st = pstats.Stats(pr)

    wanted = {
        "rfft2": ("rfft2", "_raw_fft", "pocketfft"),
        "_FFT_spectrum": ("_FFT_spectrum",),
        "window_spectrum": ("window_spectrum",),
        "radial_spectrum": ("radial_spectrum",),
        "bouligand2009": ("bouligand2009",),
        "scipy.special.kv": ("kv",),
        "least_squares": ("least_squares",),
    }
    out = {}
    for label, needles in wanted.items():
        best = 0.0
        for (fname, _lineno, func), (_cc, _nc, _tt, ct, _cal) in st.stats.items():
            if any(nd in func for nd in needles) or (
                label == "rfft2" and "rfft2" in fname
            ):
                best = max(best, ct)
        out[label] = best
    total = max(ct for (_cc, _nc, _tt, ct, _cal) in st.stats.values())
    return total, out


def main():
    grid = build_grid()
    xc = 0.5 * (grid.xmin + grid.xmax)
    yc = 0.5 * (grid.ymin + grid.ymax)

    print("\n%-10s %8s %12s %12s %12s %12s %8s"
          % ("window/km", "cells", "optimise/s", "spectrum/s", "rfft2/s",
             "rfft2 %", "n_fev"))
    print("-" * 82)

    results = {}
    for w in WINDOWS:
        n = grid.subgrid(w, xc, yc).shape[0]

        t_opt = timeit(lambda: grid.optimise(w, xc, yc), min_time=2.0)
        t_spec = timeit(
            lambda: grid.window_spectrum(w, xc, yc, power=2.0), min_time=2.0
        )

        # the transform in isolation, exactly as _FFT_spectrum performs it
        sub = grid.subgrid(w, xc, yc)
        t = np.hanning(n)
        vtaper = np.outer(t, t)
        t_fft = timeit(lambda: np.abs(np.fft.rfft2(sub * vtaper)), min_time=1.0)

        # how many forward-model evaluations the fit needed
        res = grid._fit(
            np.array([3.0, 1.0, 10.0, 5.0]),
            grid.window_spectrum(w, xc, yc, power=2.0),
        )

        results[w] = dict(n=n, opt=t_opt, spec=t_spec, fft=t_fft, nfev=res.nfev)
        print("%-10.0f %8d %12.4f %12.4f %12.5f %11.2f%% %8d"
              % (w / 1e3, n, t_opt, t_spec, t_fft, 100 * t_fft / t_opt, res.nfev),
              flush=True)

    # profile() is much slower, so only two window sizes
    print("\n%-10s %8s %12s %12s %12s"
          % ("window/km", "cells", "profile/s", "rfft2/s", "rfft2 %"))
    print("-" * 60)
    for w in [256e3, 1024e3]:
        n = results[w]["n"]
        t_prof = timeit(
            lambda: grid.profile(w, xc, yc, "CPD", npoints=11), min_time=5.0, min_reps=2
        )
        t_fft = results[w]["fft"]
        print("%-10.0f %8d %12.4f %12.5f %11.3f%%"
              % (w / 1e3, n, t_prof, t_fft, 100 * t_fft / t_prof), flush=True)

    # cProfile breakdown at a representative window
    for w in [256e3, 1024e3]:
        print("\n\n=== cProfile, optimise() at %.0f km (%d cells) ==="
              % (w / 1e3, results[w]["n"]))
        total, parts = profile_call(lambda: grid.optimise(w, xc, yc))
        for label, ct in sorted(parts.items(), key=lambda kv: -kv[1]):
            print("  %-20s %8.4f s  %6.2f%%" % (label, ct, 100 * ct / total))

    print("\n\n=== cProfile, profile() at 1024 km ===")
    total, parts = profile_call(
        lambda: grid.profile(1024e3, xc, yc, "CPD", npoints=11)
    )
    for label, ct in sorted(parts.items(), key=lambda kv: -kv[1]):
        print("  %-20s %8.4f s  %6.2f%%" % (label, ct, 100 * ct / total))

    # ---- what the very best FFTW swap could win, end to end ----
    print("\n\n=== best case if the transform itself were free ===")
    print("%-10s %8s %14s %14s %14s"
          % ("window/km", "cells", "optimise/s", "-rfft2", "saving"))
    print("-" * 66)
    for w in WINDOWS:
        r = results[w]
        print("%-10.0f %8d %14.4f %14.4f %13.2f%%"
              % (w / 1e3, r["n"], r["opt"], r["opt"] - r["fft"],
                 100 * r["fft"] / r["opt"]))


if __name__ == "__main__":
    main()
