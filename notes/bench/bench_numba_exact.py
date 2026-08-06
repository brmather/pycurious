"""
The numba binning kernel, this time reproducing `np.digitize` exactly.

`counts` is a partition of the full spectrum and the `_TAPER_DOF` calibration
is baked onto it, so a kernel that gets counts even slightly wrong is not
adoptable at any speed. The arithmetic bin index `int((kk-lo)/width)` does NOT
reproduce digitize at the edges; this uses the same binary search searchsorted
does.
"""

import time
import numpy as np
import numba
import pycurious


@numba.njit(cache=True, inline="always")
def _bin_of(kbins, kk):
    """Exactly np.digitize(kk, kbins) - 1, i.e. searchsorted(side='right') - 1."""
    lo, hi = 0, kbins.size
    while lo < hi:
        mid = (lo + hi) >> 1
        if kk < kbins[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo - 1


@numba.njit(cache=True)
def bin_numba(FT, dk, kbins, const, nr, ncol, nc):
    nbins = kbins.size - 1
    hi = kbins[-1]
    counts = np.zeros(nbins)
    Ssum = np.zeros(nbins)
    ksum = np.zeros(nbins)
    idx_of = np.empty((nr, ncol), dtype=np.int32)
    for i in range(nr):
        fr = i if i <= (nr - 1) // 2 else i - nr
        ix = fr * dk
        for j in range(ncol):
            iy = j * dk
            kk = np.hypot(ix, iy)
            b = _bin_of(kbins, kk)
            if b == nbins and kk <= hi:
                b = nbins - 1
            if b < 0 or b >= nbins:
                idx_of[i, j] = -1
                continue
            w = 1.0 if (j == 0 or (nc % 2 == 0 and j == ncol - 1)) else 2.0
            idx_of[i, j] = b
            counts[b] += w
            ksum[b] += w * kk
            Ssum[b] += w * const * np.log(FT[i, j])
    S = Ssum / counts
    k = ksum / counts
    var = np.zeros(nbins)
    for i in range(nr):
        for j in range(ncol):
            b = idx_of[i, j]
            if b < 0:
                continue
            w = 1.0 if (j == 0 or (nc % 2 == 0 and j == ncol - 1)) else 2.0
            d = const * np.log(FT[i, j]) - S[b]
            var[b] += w * d * d
    return k, S, np.sqrt(var / counts), counts


def timeit(fn, min_time=1.5, min_reps=5):
    fn()
    reps, el = 0, 0.0
    t0 = time.perf_counter()
    while el < min_time or reps < min_reps:
        fn()
        reps += 1
        el = time.perf_counter() - t0
    return el / reps


def main():
    data, extent = pycurious.fractal_anomaly(2100, 1.0, beta=3.0, zt=1.0,
                                             dz=20.0, C=5.0, seed=1)
    g = pycurious.CurieOptimiseBouligand(data, *extent)
    xc = 0.5 * (g.xmin + g.xmax)
    yc = 0.5 * (g.ymin + g.ymax)

    print("%-10s %6s %11s %11s %9s %11s %11s %10s"
          % ("window/km", "cells", "stock/ms", "numba/ms", "speed-up",
             "max|dS|", "max|dsig|", "counts"))
    print("-" * 88)
    for w in (256e3, 512e3, 1024e3, 2048e3):
        sub = g.subgrid(w, xc, yc)
        n = sub.shape[0]
        vtaper, dk, kbins = g._taper_spectrum(sub, np.hanning)
        FT = np.abs(np.fft.rfft2(sub * vtaper))
        ncol = n // 2 + 1

        ref = g._FFT_spectrum(sub, vtaper, dk, kbins, 2.0)
        got = bin_numba(FT, dk, kbins, 2.0, n, ncol, n)

        t_fft = timeit(lambda: np.abs(np.fft.rfft2(sub * vtaper)))
        t_stock = timeit(lambda: g._FFT_spectrum(sub, vtaper, dk, kbins, 2.0)) - t_fft
        t_nb = timeit(lambda: bin_numba(FT, dk, kbins, 2.0, n, ncol, n))

        dS = np.nanmax(np.abs(ref[1] - got[1]))
        dsig = np.nanmax(np.abs(ref[2] - got[2]))
        dc = np.nanmax(np.abs(ref[3] - got[3]))
        print("%-10.0f %6d %11.3f %11.3f %8.2fx %11.2e %11.2e %10s"
              % (w / 1e3, n, t_stock * 1e3, t_nb * 1e3, t_stock / t_nb,
                 dS, dsig, "EXACT" if dc == 0 else "OFF by %g" % dc), flush=True)


if __name__ == "__main__":
    main()
