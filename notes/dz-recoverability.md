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

The signature looks like finding 2 above — a spectrum whose high-`k` band
cannot locate the rolloff returns a `dz` that grows with the window — and WDMAM's
high-`k` band does carry an unmodelled ~4.2 km resolution rolloff
(`notes/spectrum-binning-weighting-multitaper.md`), which reduces the *effective*
`k_max`. **That was the proposed cause, and it was tested below. It is half
right: the band cut modulates the drift but does not produce it.** Read the
"Tested" section before relying on the paragraph above.

## What this means

- **A WDMAM `dz` is not a thickness, it is a thickness-at-a-window.** Quoting
  one without its window is quoting an arbitrary point on a monotone curve that
  spans a factor of two.
- **1500 km is the turning point**, not an optimum: it is where the monotone
  growth from above meets the erratic regime from below. Both sides of it are
  failure modes, of different kinds.
- **The band cut is a real lever on the drift**, measured below: lowering
  `kmax` from 0.25 to 0.10 takes the growth from ×2.84 to ×4.20. It is not the
  cause, but it is the one knob that demonstrably moves it.
- **This is upstream of the interval question.** Whitening the objective fixes
  under-coverage on synthetics and changes nothing here. A calibrated interval
  around a number that moves by a factor of two with an arbitrary choice of
  window is still not a measurement of anything.

## Tested, and the diagnosis is half wrong

Both experiments below were run. The second supports the `k_max` story; the
first refutes it as a sufficient explanation, so the section above overstates
the case and this is the correction.

### Experiment 1: degrade a synthetic — **refuted**

A synthetic of known `dz`, blurred by `exp(-k² σ_r²)` and fitted with
`~/Global_CPD`'s own configuration (`kmax = 0.25`, `zt` pinned to ±0.05, a
`beta` prior of 0.15, the `sigma` inflation), swept over the same windows. 12
realisations. `growth` is the 10,000 km median over the 1500 km one, so the L2
observation is 2.2–2.8:

| truth | clean | degraded 4.2 km | degraded 8 km |
|---|---|---|---|
| dz 20, zt 1.0 | ×0.84 | ×0.90 | ×0.78 |
| dz 20, zt 4.4 (pin wrong) | ×0.87 | ×0.84 | ×0.73 |
| dz 40, zt 1.0 | ×0.85 | ×0.77 | ×0.69 |
| dz 40, zt 4.4 (pin wrong) | ×0.72 | ×0.69 | ×0.66 |

**Every combination is flat or slightly falling — none rises.** The rolloff and
a mis-specified `zt` pin both produce large *biases*: at `dz = 20` and 10,000 km,
degradation takes the answer from 18.5 to 25.7 (4.2 km) and 32.3 (8 km), and
pinning `zt` at 1.0 when the truth is 4.4 takes it from 18.5 to 28.5. Stacked,
+89% over the truth. But a bias is not a drift, and **no stationary synthetic
reproduced the window dependence.**

Worth being clear about what that rules out: the effect is not a band-limit
artefact of the forward model, because a synthetic that obeys the model exactly
and is band-limited exactly the same way does not show it.

### Experiment 2: lower the band cut — **supported, in direction**

Refitting the cached L2 spectra with `kmax` reduced (they are already cut at
0.25, so it can only be lowered without recomputing them). Median `dz`, 162
vertices:

| kmax | 10000 | 6000 | 4000 | 2500 | 1500 | growth |
|---|---|---|---|---|---|---|
| 0.25 (archived) | 42.76 | 28.15 | 22.31 | 19.67 | 15.04 | **×2.84** |
| 0.20 | 45.72 | 28.38 | 22.20 | 19.41 | 14.72 | ×3.11 |
| 0.15 | 48.39 | 28.42 | 21.82 | 19.10 | 14.14 | ×3.42 |
| 0.10 | 52.76 | 28.28 | 21.27 | 18.01 | 12.56 | ×4.20 |

**Taking the high-`k` band away makes the drift monotonically worse**, 2.84 →
4.20, and it acts almost entirely on the long-window end: the 10,000 km median
rises 23% while the 1500 km one falls 17%. So the band cut is a real lever on
how much `dz` depends on the window, exactly as the diagnosis predicts —
extrapolating, *raising* `kmax` above 0.25 should reduce the drift below 2.84.

### What the two together say

The high-`k` band **modulates** the drift but does not **cause** it. The cause
has to be something a stationary synthetic cannot have, and the obvious
candidate is the one the geometry forces: **a 10,000 km window averages over
genuinely different lithosphere.** `notes/bouligand-rederivation.md` already
records that a mixture of base depths produces a spectrum that still looks like
a single layer — so a window spanning thin oceanic and thick cratonic crust
returns one `dz`, and which one it returns need not be the mean, nor stable as
the mixture changes.

That reframes the whole thing. The window dependence would then not be an
artefact to remove but a statement that a single-layer `dz` is not a
well-defined property of a large window, which no improvement to the estimator
can fix.

### What would test *that*

1. **A synthetic with spatially varying `dz`.** Build a field whose thickness
   varies across the grid — say oceanic 10 km against cratonic 40 km in blocks
   — and sweep the window. If heterogeneity is the cause, the drift appears
   here and only here, and its size should scale with the contrast.
2. **Split the L2 drift by how heterogeneous each window is.** `qc/land_fraction`
   already measures how mixed a window is. If the drift is heterogeneity, the
   vertices whose windows stay within one province should drift least.
3. **Raise `kmax` above 0.25**, which needs pass A re-run to recompute the
   spectra (~2 hours at L2). Experiment 2 says this should help; it is the only
   one of the three that is a lever rather than a diagnosis.
