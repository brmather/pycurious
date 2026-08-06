"""
The cheap alternative: `scipy.fft.rfft2`, which is already a dependency, at the
odd sizes pycurious transforms -- single-threaded and multithreaded -- against
the best pyfftw can manage (aligned buffers, FFTW_MEASURE, reused plan).
"""

import time
import numpy as np
import scipy.fft
import pyfftw

SIZES = [65, 129, 257, 513, 1025, 2049]
THREADS = [1, 2, 4, 8]


def timeit(fn, min_time=0.6, min_reps=5):
    fn()
    reps, elapsed = 0, 0.0
    t0 = time.perf_counter()
    while elapsed < min_time or reps < min_reps:
        fn()
        reps += 1
        elapsed = time.perf_counter() - t0
    return elapsed / reps


def main():
    hdr = ["n", "numpy"] + ["scipy w=%d" % w for w in THREADS] \
        + ["FFTW t=%d" % t for t in (1, 4)]
    print("per-call wall time, ms (taper multiply and abs included)")
    print("".join("%-12s" % h for h in hdr))
    print("-" * (12 * len(hdr)))

    rows = {}
    for n in SIZES:
        rng = np.random.default_rng(7)
        data = rng.normal(size=(n, n))
        t = np.hanning(n)
        vtaper = np.outer(t, t)

        r = {"numpy": timeit(lambda: np.abs(np.fft.rfft2(data * vtaper)))}
        for w in THREADS:
            r["scipy w=%d" % w] = timeit(
                lambda w=w: np.abs(scipy.fft.rfft2(data * vtaper, workers=w))
            )
        for th in (1, 4):
            a = pyfftw.empty_aligned((n, n), dtype="float64")
            b = pyfftw.empty_aligned((n, n // 2 + 1), dtype="complex128")
            plan = pyfftw.FFTW(a, b, axes=(0, 1), flags=("FFTW_MEASURE",),
                               threads=th)

            def run(a=a, plan=plan):
                np.multiply(data, vtaper, out=a)
                return np.abs(plan())

            r["FFTW t=%d" % th] = timeit(run)

        rows[n] = r
        print("%-12d" % n + "".join("%-12.3f" % (r[h] * 1e3) for h in hdr[1:]),
              flush=True)

    print("\nspeed-up vs numpy.fft.rfft2")
    print("".join("%-12s" % h for h in hdr))
    print("-" * (12 * len(hdr)))
    for n in SIZES:
        r = rows[n]
        print("%-12d" % n
              + "".join("%-12.2f" % (r["numpy"] / r[h]) for h in hdr[1:]))


if __name__ == "__main__":
    main()
