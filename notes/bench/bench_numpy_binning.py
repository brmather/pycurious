"""
Does the binning need numba, or does it just need fewer temporaries?

The stock routine materialises kk, a broadcast `weight` (broadcast_to + ravel
forces a copy), then THREE more copies through boolean masking, before four
bincounts and a second pass for the variance. This version routes out-of-range
cells to a sentinel bin instead of masking, which removes the copies while
keeping np.hypot and searchsorted -- so counts stay exact by construction.
"""

import time
import numpy as np
import pycurious


def bin_numpy(FT, dk, kbins, const, nr, ncol, nc):
    nbins = kbins.size - 1

    row_freq = np.arange(nr)
    row_freq[row_freq > (nr - 1) // 2] -= nr
    ix = (row_freq * dk)[:, np.newaxis]
    iy = (np.arange(ncol) * dk)[np.newaxis, :]
    kk = np.hypot(ix, iy).ravel()

    # same as np.digitize(kk, kbins) - 1
    idx = np.searchsorted(kbins, kk, side="right") - 1
    idx[(idx == nbins) & (kk <= kbins[-1])] = nbins - 1
    # out-of-range -> sentinel bin nbins, dropped by the slice below. No masks,
    # so no copies of kk / idx / weight.
    np.clip(idx, -1, nbins, out=idx)
    idx[idx < 0] = nbins

    weight = np.full(ncol, 2.0)
    weight[0] = 1.0
    if nc % 2 == 0:
        weight[-1] = 1.0
    weight = np.broadcast_to(weight, (nr, ncol)).ravel()

    rr = np.log(FT.ravel())
    rr *= const

    ml = nbins + 1
    counts = np.bincount(idx, weights=weight, minlength=ml)[:nbins]
    S = np.bincount(idx, weights=weight * rr, minlength=ml)[:nbins]
    k = np.bincount(idx, weights=weight * kk, minlength=ml)[:nbins]
    with np.errstate(invalid="ignore", divide="ignore"):
        S /= counts
        k /= counts
        Sfull = np.concatenate((S, [0.0]))
        dev = rr - Sfull[idx]
        dev *= dev
        dev *= weight
        sigma = np.sqrt(np.bincount(idx, weights=dev, minlength=ml)[:nbins] / counts)

    empty = counts == 0
    S[empty] = k[empty] = sigma[empty] = np.nan
    return k, S, sigma, counts


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

    print("%-10s %6s %11s %11s %9s %11s %10s"
          % ("window/km", "cells", "stock/ms", "numpy2/ms", "speed-up",
             "max|dS|", "counts"))
    print("-" * 76)
    for w in (256e3, 512e3, 1024e3, 2048e3):
        sub = g.subgrid(w, xc, yc)
        n = sub.shape[0]
        vtaper, dk, kbins = g._taper_spectrum(sub, np.hanning)
        FT = np.abs(np.fft.rfft2(sub * vtaper))
        ncol = n // 2 + 1

        ref = g._FFT_spectrum(sub, vtaper, dk, kbins, 2.0)
        got = bin_numpy(FT, dk, kbins, 2.0, n, ncol, n)

        t_fft = timeit(lambda: np.abs(np.fft.rfft2(sub * vtaper)))
        t_stock = timeit(lambda: g._FFT_spectrum(sub, vtaper, dk, kbins, 2.0)) - t_fft
        t_new = timeit(lambda: bin_numpy(FT, dk, kbins, 2.0, n, ncol, n))

        dS = np.nanmax(np.abs(ref[1] - got[1]))
        dc = np.nanmax(np.abs(ref[3] - got[3]))
        print("%-10.0f %6d %11.3f %11.3f %8.2fx %11.2e %10s"
              % (w / 1e3, n, t_stock * 1e3, t_new * 1e3, t_stock / t_new,
                 dS, "EXACT" if dc == 0 else "OFF by %g" % dc), flush=True)


if __name__ == "__main__":
    main()
