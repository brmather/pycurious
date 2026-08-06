# When is `dz` recoverable? Two limits, not one — and WDMAM sits outside both

Harnesses: `notes/bench/score_dz_recovery.py` (synthetics, known truth) and
`notes/bench/score_dz_L2.py` (WDMAM L2, all 40 rungs, 162 vertices).

## The band has two ends and both bind

The layer rolloff sits at `|k| dz ~ 1`. For a window to see it, the rolloff has
to be **inside the measured band**, which is two conditions, not one:

```
k_min·dz = 2π dz / window  <  1     the rolloff is above the longest wavelength
                                    measured    -- a WINDOW statement
k_max·dz = π dz / dx       >>  1    the rolloff is below the shortest
                                    -- a CELL SIZE statement
```

The first is the one everybody thinks about. The second is the one that decides
whether a *thin* layer is recoverable at all, and it does not improve by
widening the window — only by shrinking the cell.

## Measured, 40 realisations per cell, 5 km cells

Median absolute error in `dz`, km:

| dz | k_max·dz | 500 km | 1000 km | 2000 km | 4000 km |
|---|---|---|---|---|---|
| 5 | 3.1 | 32.7 | 82.2 | 98.7 | **212.0** |
| 10 | 6.3 | 11.2 | 13.6 | 1.9 | **0.6** |
| 20 | 12.6 | 6.8 | 7.3 | 2.8 | **1.3** |
| 30 | 18.8 | 12.1 | 9.5 | 3.7 | **1.6** |
| 45 | 28.3 | 20.7 | 16.4 | 8.3 | **3.1** |
| 60 | 37.7 | 26.6 | 22.8 | 13.2 | **5.5** |

Three things fall out, and the first was not what I expected.

**1. `k_min·dz < 1` is not the criterion.** At `dz = 5` that number runs 0.008
to 0.063 — deep inside "resolvable" — and recovery is hopeless at every window,
with a mean of 171 to 363 km against a truth of 5. What binds there is
`k_max·dz = 3.1`: with 5 km cells the band reaches only three times past the
rolloff, which is not enough to establish the fractal half-space asymptote that
locates it from above.

**2. For a layer that thin, more window makes it worse.** 32.7 → 82.2 → 98.7 →
212.0 as the window goes 500 → 4000 km. Every bin added at low `k` says nothing
about a rolloff the band cannot reach, and dilutes the ones that do. This is the
one place in this package where a bigger window is actively harmful, and it is
the signature to look for.

**3. Above that floor, error grows with thickness and falls with window.** In
relative terms at 4000 km: `dz` 10 → 6%, 20 → 6.5%, 30 → 5.3%, 45 → 6.9%,
60 → 9.2%. So there is a broad sweet spot from about 10 to 45 km, thickening
slowly, with a hard floor below and a slow rise above.

**Whitening the objective is neutral here.** It costs about 15–20% on the
scatter where `dz` is well determined (`dz = 20` at 4000 km: ±2.15 → ±2.55) and
occasionally helps a lot where it is not (`dz = 20` at 500 km: ±164.5 → ±33.0).
Nothing in this table argues for or against it.

## WDMAM L2: `dz` does not converge, it tracks the window

162 vertices split into terciles by their `dz` at the 4000 km rung — so "deep"
and "shallow" mean what the data say, not what a map says. Median `dz`, km:

| window | shallow third | middle third | deep third |
|---|---|---|---|
| 10000 | 22.59 | 36.55 | 53.55 |
| 6000 | 17.05 | 26.04 | 45.82 |
| 4000 | 13.50 | 22.31 | 39.70 |
| 2500 | 10.42 | 20.20 | 33.56 |
| 1500 | **8.07** | **16.66** | **23.69** |
| 1000 | 9.60 | 20.41 | 43.03 |
| 750 | 7.75 | 18.20 | 41.94 |
| 500 | 7.97 | 17.72 | 36.85 |
| 250 | 7.07 | 11.89 | 40.89 |

**From 1500 km up, `dz` grows monotonically with the window in every band and
never settles** — by a factor of 2.8 (shallow), 2.2 (middle) and 2.3 (deep)
across the ladder. It drifts 0.5 to 0.8 km per 250 km of window at the long end
and does not slow down. Below 1500 km it stops being monotone and becomes
erratic, with the deep third's IQR reaching 355 km at 250 km.

On synthetics `dz` converges: at 4000 km, `dz = 30` comes back 30.91 ± 3.89 and
widening further changes nothing. On WDMAM it does not. **The difference is not
the estimator** — the whitened arm reproduces the same table to within 1–2 km at
every rung — **it is the data.**

And the signature matches finding 2 above exactly: a spectrum whose high-`k`
band cannot locate the rolloff returns a `dz` that grows with the window. WDMAM's
high-`k` band carries an unmodelled ~4.2 km resolution rolloff
(`notes/spectrum-binning-weighting-multitaper.md`), which is a reduction of the
*effective* `k_max`. Every layer on that map is therefore behaving like the
synthetic `dz = 5` case: nominally inside the band by the `k_min` criterion, and
outside it by the `k_max` one.

## What this means

- **A WDMAM `dz` is not a thickness, it is a thickness-at-a-window.** Quoting
  one without its window is quoting an arbitrary point on a monotone curve that
  spans a factor of two.
- **1500 km is the turning point**, not an optimum: it is where the monotone
  growth from above meets the erratic regime from below. Both sides of it are
  failure modes, of different kinds.
- **The band cut is doing more than protecting against gridding artefacts.**
  `SRC_KMAX = 0.25` discards the high-`k` band that would otherwise locate the
  rolloff. It is defensible because that band is corrupted — but it is also what
  puts every vertex on the wrong side of the `k_max·dz` condition, and the
  window-dependence above is the price.
- **This is upstream of the interval question.** Whitening the objective fixes
  under-coverage on synthetics and changes nothing here. A calibrated interval
  around a number that moves by a factor of two with an arbitrary choice of
  window is still not a measurement of anything.

## What would test it

The diagnosis says the `k_max` end is the problem. Two things follow that have
not been done:

1. **Degrade a synthetic by WDMAM's resolution rolloff and repeat the L2 table.**
   `notes/bench/run_experiments.py:degrade` already applies `exp(-k² σ_r²)`. If
   the prediction holds, a degraded synthetic of known `dz` should reproduce the
   monotone growth with window, and an undegraded one should not.
2. **Raise `kmax` and watch the drift.** If the window-dependence comes from the
   missing high-`k` band, relaxing the cut should reduce it — at the cost of
   admitting the artefact the cut exists to exclude. The trade would then be
   measurable rather than assumed.
