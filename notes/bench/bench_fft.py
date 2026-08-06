"""
Microbenchmark: rfft2 backends at the sizes pycurious actually transforms.

Everything here transforms a real (n, n) float64 array exactly as
`CurieGrid._FFT_spectrum` does -- `np.abs(rfft2(data * vtaper))` -- so the
taper multiply and the abs are inside the timed region for every backend, which
is what a drop-in replacement would have to beat.
"""

import time
import numpy as np
import scipy.fft
import pyfftw

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(300)

SIZES = [64, 128, 256, 512, 1000, 1024, 2001, 2048]
NTHREADS = 8


def timeit(fn, min_time=0.5, min_reps=5):
    """Wall-clock seconds per call, after one warm-up."""
    fn()
    reps, elapsed = 0, 0.0
    t0 = time.perf_counter()
    while elapsed < min_time or reps < min_reps:
        fn()
        reps += 1
        elapsed = time.perf_counter() - t0
    return elapsed / reps


def make(n, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, n))
    t = np.hanning(n)
    vtaper = np.outer(t, t)
    return data, vtaper


def bench_size(n):
    data, vtaper = make(n)
    out = {}

    out["numpy"] = timeit(lambda: np.abs(np.fft.rfft2(data * vtaper)))
    out["scipy(w=1)"] = timeit(lambda: np.abs(scipy.fft.rfft2(data * vtaper, workers=1)))
    out["scipy(w=%d)" % NTHREADS] = timeit(
        lambda: np.abs(scipy.fft.rfft2(data * vtaper, workers=NTHREADS))
    )

    # pyfftw drop-in interface, cache warm (plans reused across calls)
    fftw_np = pyfftw.interfaces.numpy_fft
    out["pyfftw.interfaces"] = timeit(
        lambda: np.abs(fftw_np.rfft2(data * vtaper, threads=1))
    )
    out["pyfftw.interfaces(t=%d)" % NTHREADS] = timeit(
        lambda: np.abs(fftw_np.rfft2(data * vtaper, threads=NTHREADS))
    )

    # planned FFTW object, the fastest pyfftw can go: aligned buffers, a
    # MEASURE plan built once and reused. The copy into the aligned input is
    # timed too, because `data * vtaper` cannot be written in place.
    for threads in (1, NTHREADS):
        a = pyfftw.empty_aligned((n, n), dtype="float64")
        b = pyfftw.empty_aligned((n, n // 2 + 1), dtype="complex128")
        t_plan = time.perf_counter()
        plan = pyfftw.FFTW(
            a, b, axes=(0, 1), flags=("FFTW_MEASURE",), threads=threads
        )
        t_plan = time.perf_counter() - t_plan

        def run(a=a, plan=plan):
            np.multiply(data, vtaper, out=a)
            return np.abs(plan())

        key = "pyfftw.FFTW" + ("" if threads == 1 else "(t=%d)" % threads)
        out[key] = timeit(run)
        out[key + " plan"] = t_plan

    # the multiply and the abs alone, i.e. the floor no backend can go below
    FT = np.fft.rfft2(data * vtaper)
    out["taper+abs only"] = timeit(lambda: (np.multiply(data, vtaper), np.abs(FT)))

    return out


def main():
    print("threads for the multithreaded rows: %d\n" % NTHREADS)
    rows = {}
    for n in SIZES:
        rows[n] = bench_size(n)
        print("n = %d done" % n, flush=True)

    keys = [k for k in rows[SIZES[0]] if not k.endswith(" plan")]
    print("\n\nper-call wall time, ms")
    print("%-26s" % "backend" + "".join("%10d" % n for n in SIZES))
    print("-" * (26 + 10 * len(SIZES)))
    for k in keys:
        print("%-26s" % k + "".join("%10.3f" % (rows[n][k] * 1e3) for n in SIZES))

    print("\nspeed-up vs numpy (>1 is faster)")
    print("%-26s" % "backend" + "".join("%10d" % n for n in SIZES))
    print("-" * (26 + 10 * len(SIZES)))
    for k in keys:
        if k == "numpy":
            continue
        print(
            "%-26s" % k
            + "".join("%10.2f" % (rows[n]["numpy"] / rows[n][k]) for n in SIZES)
        )

    print("\none-off FFTW_MEASURE planning cost, ms (paid per new window size)")
    for k in rows[SIZES[0]]:
        if k.endswith(" plan"):
            print("%-26s" % k + "".join("%10.1f" % (rows[n][k] * 1e3) for n in SIZES))


if __name__ == "__main__":
    main()
