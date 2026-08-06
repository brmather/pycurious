"""
The sizes pycurious actually transforms.

`subgrid` returns `nw//2*2 + 1` cells, so every window is an odd square:
65, 129, 257 (prime), 513, 1025, 2049 (= 3 x 683, and 683 is prime). Awkward
lengths are where FFT implementations diverge most, and where FFTW_MEASURE
planning gets expensive.
"""

import time
import numpy as np
import scipy.fft
import pyfftw

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(300)

SIZES = [65, 129, 257, 513, 1025, 2049]


def factorise(n):
    f, d = [], 2
    while d * d <= n:
        while n % d == 0:
            f.append(d)
            n //= d
        d += 1
    if n > 1:
        f.append(n)
    return "x".join(str(x) for x in f)


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
    print("%-8s %-12s %10s %10s %10s %10s %10s %12s"
          % ("n", "factors", "numpy/ms", "scipy/ms", "pyfftw/ms", "FFTWest/ms",
             "FFTWmeas", "plan(meas)/s"))
    print("-" * 92)

    for n in SIZES:
        rng = np.random.default_rng(7)
        data = rng.normal(size=(n, n))
        t = np.hanning(n)
        vtaper = np.outer(t, t)

        t_np = timeit(lambda: np.abs(np.fft.rfft2(data * vtaper)))
        t_sp = timeit(lambda: np.abs(scipy.fft.rfft2(data * vtaper, workers=1)))
        fftw_np = pyfftw.interfaces.numpy_fft
        t_pf = timeit(lambda: np.abs(fftw_np.rfft2(data * vtaper, threads=1)))

        row = {}
        for flag in ("FFTW_ESTIMATE", "FFTW_MEASURE"):
            a = pyfftw.empty_aligned((n, n), dtype="float64")
            b = pyfftw.empty_aligned((n, n // 2 + 1), dtype="complex128")
            t0 = time.perf_counter()
            plan = pyfftw.FFTW(a, b, axes=(0, 1), flags=(flag,), threads=1)
            row[flag + "_plan"] = time.perf_counter() - t0

            def run(a=a, plan=plan):
                np.multiply(data, vtaper, out=a)
                return np.abs(plan())

            row[flag] = timeit(run)

        print("%-8d %-12s %10.3f %10.3f %10.3f %10.3f %10.3f %12.2f"
              % (n, factorise(n), t_np * 1e3, t_sp * 1e3, t_pf * 1e3,
                 row["FFTW_ESTIMATE"] * 1e3, row["FFTW_MEASURE"] * 1e3,
                 row["FFTW_MEASURE_plan"]),
              flush=True)


if __name__ == "__main__":
    main()
