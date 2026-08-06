"""
Is numba worth it for pycurious?

Two candidate targets, prototyped and measured against the stock numpy/scipy:

1. `bouligand2009` -- the forward model, the largest single self-time entry at
   small and mid windows. Needs `kv` at real order, which numba can reach
   through `scipy.special.cython_special` (unlike CuPy/JAX, which cannot).
2. the radial binning inside `_FFT_spectrum` -- digitize + masking + four
   bincounts over the whole half plane, with a lot of temporaries.
"""

import ctypes
import time
import numpy as np
from scipy.special import gamma, kv
import numba
from numba.extending import get_cython_function_address
import scipy.special.cython_special  # noqa: F401  (registers the symbols)

import pycurious
from pycurious.grid import bouligand2009

# ---- bind scipy's real-order kv so numba can call it scalar-wise ----
# signature: double kv(double v, double z)
_addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1kv")
_kv_scalar = ctypes.CFUNCTYPE(
    ctypes.c_double, ctypes.c_double, ctypes.c_double
)(_addr)


@numba.njit(cache=True, fastmath=False)
def _bouligand_numba(kh, beta, zt, dz, C, gam_a, gam_b):
    """One fused pass; gamma terms are scalars, passed in precomputed."""
    n = kh.size
    out = np.empty(n, dtype=np.float64)
    order = -0.5 * (1.0 + beta)
    expo = 0.5 * (1.0 + beta)
    pref = np.sqrt(np.pi) / gam_a
    for i in range(n):
        k = kh[i]
        khdz = k * dz
        A = pref * (
            0.5 * np.cosh(khdz) * gam_b
            - _kv_scalar(order, khdz) * (0.5 * khdz) ** expo
        )
        out[i] = C - 2.0 * k * zt - (beta - 1.0) * np.log(k) - khdz + np.log(A)
    return out


def bouligand_numba(kh, beta, zt, dz, C):
    return _bouligand_numba(
        kh, beta, zt, dz, C, gamma(1.0 + 0.5 * beta), gamma(0.5 * (1.0 + beta))
    )


# ---- fused radial binning, replacing digitize + masks + 4 bincounts ----
@numba.njit(cache=True)
def _bin_numba(FT, dk, kbins, const, nr, ncol, nc):
    nbins = kbins.size - 1
    lo, hi = kbins[0], kbins[-1]
    width = kbins[1] - kbins[0]
    counts = np.zeros(nbins)
    Ssum = np.zeros(nbins)
    ksum = np.zeros(nbins)
    idx_of = np.empty((nr, ncol), dtype=np.int64)
    for i in range(nr):
        fr = i if i <= (nr - 1) // 2 else i - nr
        ix = fr * dk
        for j in range(ncol):
            iy = j * dk
            kk = np.sqrt(ix * ix + iy * iy)
            b = int((kk - lo) / width)
            if kk == hi:
                b = nbins - 1
            if b < 0 or b >= nbins:
                idx_of[i, j] = -1
                continue
            w = 2.0
            if j == 0 or (nc % 2 == 0 and j == ncol - 1):
                w = 1.0
            idx_of[i, j] = b
            counts[b] += w
            ksum[b] += w * kk
            Ssum[b] += w * const * np.log(FT[i, j])
    S = Ssum / counts
    k = ksum / counts
    var = np.zeros(nbins)
    for i in range(nr):  # second pass for the two-pass variance
        for j in range(ncol):
            b = idx_of[i, j]
            if b < 0:
                continue
            w = 2.0
            if j == 0 or (nc % 2 == 0 and j == ncol - 1):
                w = 1.0
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
    # ---------------- target 1: bouligand2009 ----------------
    print("=== bouligand2009: stock vs numba ===")
    print("%-8s %14s %14s %10s %14s"
          % ("nbins", "stock/us", "numba/us", "speed-up", "max |dPhi|"))
    print("-" * 64)
    for nbins in (32, 64, 128, 512, 1024):
        kh = np.linspace(0.01, 1.5, nbins)
        a = bouligand2009(kh, 3.0, 1.0, 20.0, 5.0)
        b = bouligand_numba(kh, 3.0, 1.0, 20.0, 5.0)
        t_a = timeit(lambda: bouligand2009(kh, 3.0, 1.0, 20.0, 5.0))
        t_b = timeit(lambda: bouligand_numba(kh, 3.0, 1.0, 20.0, 5.0))
        print("%-8d %14.2f %14.2f %9.2fx %14.2e"
              % (nbins, t_a * 1e6, t_b * 1e6, t_a / t_b, np.max(np.abs(a - b))))

    # what fraction of bouligand2009 is kv itself -- the floor numba cannot beat
    print("\n=== how much of it is kv (the irreducible floor) ===")
    for nbins in (64, 512):
        kh = np.linspace(0.01, 1.5, nbins)
        khdz = kh * 20.0
        t_all = timeit(lambda: bouligand2009(kh, 3.0, 1.0, 20.0, 5.0))
        t_kv = timeit(lambda: kv(-2.0, khdz))
        print("  nbins=%4d  total %7.2f us   kv %7.2f us  (%4.1f%%)"
              % (nbins, t_all * 1e6, t_kv * 1e6, 100 * t_kv / t_all))

    # ---------------- target 2: the radial binning ----------------
    print("\n=== _FFT_spectrum radial binning: stock vs numba ===")
    data, extent = pycurious.fractal_anomaly(2100, 1.0, beta=3.0, zt=1.0,
                                             dz=20.0, C=5.0, seed=1)
    g = pycurious.CurieOptimiseBouligand(data, *extent)
    xc = 0.5 * (g.xmin + g.xmax)
    yc = 0.5 * (g.ymin + g.ymax)

    print("%-10s %7s %13s %13s %10s %12s"
          % ("window/km", "cells", "stock/ms", "numba/ms", "speed-up", "max |dS|"))
    print("-" * 70)
    for w in (256e3, 512e3, 1024e3, 2048e3):
        sub = g.subgrid(w, xc, yc)
        n = sub.shape[0]
        vtaper, dk, kbins = g._taper_spectrum(sub, np.hanning)
        FT = np.abs(np.fft.rfft2(sub * vtaper))
        ncol = n // 2 + 1

        ref = g._FFT_spectrum(sub, vtaper, dk, kbins, 2.0)
        got = _bin_numba(FT, dk, kbins, 2.0, n, ncol, n)

        # stock timing minus the FFT, so we compare binning against binning
        t_fft = timeit(lambda: np.abs(np.fft.rfft2(sub * vtaper)))
        t_stock = timeit(lambda: g._FFT_spectrum(sub, vtaper, dk, kbins, 2.0)) - t_fft
        t_numba = timeit(lambda: _bin_numba(FT, dk, kbins, 2.0, n, ncol, n))

        dS = np.nanmax(np.abs(ref[1] - got[1]))
        dc = np.nanmax(np.abs(ref[3] - got[3]))
        print("%-10.0f %7d %13.3f %13.3f %9.2fx %12.2e%s"
              % (w / 1e3, n, t_stock * 1e3, t_numba * 1e3, t_stock / t_numba,
                 dS, "" if dc == 0 else "  COUNTS DIFFER by %g" % dc), flush=True)


if __name__ == "__main__":
    main()
