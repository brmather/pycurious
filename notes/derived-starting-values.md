# Deriving the Bouligand starting point instead of declaring it

`CurieOptimiseBouligand` started every fit at `beta=3.0, zt=1.0, dz=10.0,
C=5.0`. Three of those are questions the spectrum can answer; the fourth, `dz`,
is the one that matters, and the constant got it wrong often enough to change
published numbers. The four keyword arguments now default to `None`, meaning
*derive this from the spectrum*; a value still means *use it*.

Harnesses: `notes/bench/score_starting_values.py` (synthetics),
`notes/bench/score_wdmam_L2.py` (cost and geology on the WDMAM L2 archive),
`notes/bench/score_sensitivity.py` (is the reported spread honest).
All timings with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## What it does

`Phi(k) = C·1 + zt·(-2k) + h(k; beta, dz)` exactly — `C` additive, `zt` linear
in a known basis, and the Bessel term depends on neither. So for any
`(beta, dz)` the best `(C, zt)` is a two-column weighted linear solve
(`_solve_linear`), and the objective reduces to a surface over `(beta, dz)`
alone. `_initial_guess` scans an eight-node `dz` ladder on that reduced surface
and takes the argmin; `beta` comes from a prior centre if there is one, else
from the high-`k` asymptote, else 3.0.

Two properties, because they decide the design:

- The design matrix `G = [1, -2k]/sigma` does **not** depend on `(beta, dz)`,
  nor does a Gaussian prior row on `zt` or `C`. So `log det(GᵀG)` is constant
  and profiling out `(C, zt)` is exactly marginalising them. The reduced
  surface is the marginal posterior of `(beta, dz)`, not an approximation.
- Because the reduced objective *is* the full objective profiled over
  `(C, zt)`, a scan of it is a strictly better basin detector than
  multi-starting the full fit over the same nodes.

Variable projection is used only to choose where the 4-D fit starts. `_fit`,
`residuals`, `_jacobian` and `_covariance` are untouched, so nothing about the
uncertainty story had to be re-derived.

## Synthetics: it works where the window resolves the layer

20 seeds per cell, scored against a dense ten-point multi-start standing in for
the global minimum. `miss` counts realisations whose cost exceeds that by 0.1%.
`seed`/`fit` are evaluations of `bouligand2009` — the measurement a loaded
machine cannot distort.

| regime | dz | start | miss | dz recovered | seed | fit |
|---|---|---|---|---|---|---|
| 4000 km | 45 | constant | **7/20** | 32.61 ± 15.32 | 0 | 44.5 |
| 4000 km | 45 | derived | **0/20** | **46.08 ± 5.85** | 9.8 | 18.1 |
| 4000 km | 30 | constant | 1/20 | 28.93 ± 4.83 | 0 | 31.2 |
| 4000 km | 30 | derived | 0/20 | 30.42 ± 3.32 | 9.7 | 17.7 |
| 4000 km | 10 | constant | 0/20 | 10.00 ± 1.12 | 0 | **18.6** |
| 4000 km | 10 | derived | 0/20 | 10.00 ± 1.12 | 9.3 | **39.1** |
| 1000 km | any | either | — | identical | 9.9 | 16–24 vs 24–33 |
| 200 km | any | either | — | both unusable | — | — |

Three readings:

1. **The gain is concentrated at thick layers on large windows** — exactly
   where the `dz = 10` constant is furthest from the answer. At `dz = 45` it
   removes a 35% failure rate and cuts the scatter 2.6x, while costing *less*
   in total (27.9 evaluations against 44.5).
2. **At `dz = 10` on a large window it costs 2.6x for the same answer.** The
   constant was already almost exactly right there, and the ladder is pure
   premium. Two ways of avoiding this were measured and both made the average
   worse: a coarse `beta` axis on the scan, and scanning at the measured `beta`
   and 3.0 together, each buying identical accuracy for 1.4x to 2x the
   evaluations.
3. **At 200 km neither start is worth anything.** Both land tens of km from a
   truth of 30, because that window cannot see a layer that thick.

### A lower misfit is not always a better answer

Worth stating separately, because it cuts against the instinct that finding the
global minimum is always progress. At 200 km an earlier version of the seeder
reached *lower* misfits than the constant on 8 of 20 realisations and was
*further* from the truth: median |error| went from 19.6 to 27.4 km at
`dz = 30`. The likelihood there has a second minimum at high `beta` and low
`dz`, and the old constant avoided it by anchoring rather than by being right.

The fix was not to hobble the search. It was to stop the `beta` estimate being
noisy enough to fall into it — see below — after which the 200 km cells came
back to a wash. But the general point stands: where the window does not
constrain the fit, a better optimiser reports a confident worse number, and the
lever is `add_prior(beta=...)`.

## The band split has to be logarithmic

`beta` is read off the high-`k` asymptote, `ln Phi = C' + (1-beta) ln k - 2k zt`
— one three-column regression, no forward-model evaluation. The two regressors
`ln k` and `k` are nearly collinear over a narrow range in `k`, and a *linear*
split of the wavenumber range leaves the high band spanning a factor of ~2.8
whatever the window:

| split | high band | beta at 200 km (truth 3) | beta at 1000 km (truth 2) |
|---|---|---|---|
| linear 0.35 | 1.45 octaves | 3.25 ± 1.13 | 2.01 ± 0.20 |
| **log 0.5** | **2.7–6 octaves** | **3.12 ± 0.48** | **1.94 ± 0.05** |

The scatter falls 2.4x at 200 km and 4x at 1000 km. It was a `beta` of 4.69 on
one 200 km realisation — from the linear split — that sent the ladder to the
wrong surface and moved a pinned `profile` interval; the log split leaves all
four pinned intervals unchanged.

The estimate is still needed. Dropping it and scanning at `beta = 3.0` fails
`test_bouligand_dz_is_recovered_in_the_mean[2.0-5.0-15.0]`: the ladder ranks
thicknesses on the wrong surface and seed 8 lands 14% worse in misfit.

## A `beta` prior outranks the estimate

`C` and `zt` are linear, so their priors enter `_solve_linear` as extra rows and
shape the answer directly. `beta` is not, and the analogue is to start it at the
prior centre.

This is not a nicety. On WDMAM the asymptote estimate returns **1.78–2.03**
against a fitted `beta` of **2.86–3.47** — biased low by more than a unit,
because that dataset's high-`k` band carries an unmodelled ~4 km resolution
rolloff which steepens the spectrum, and the regression reads the steepening as
`beta`. Weighting the regression by `1/sigma` does not help (1.72–2.00); the
band placement is not the problem, the model error is. A `beta` prior of sigma
0.15 is exactly how that dataset says so, and starting there instead:

| window | fit evals, constant | fit evals, derived | before this change |
|---|---|---|---|
| 10000 km | 38.1 | **29.9** | 48.5 |
| 4000 km | 32.0 | **27.8** | 46.7 |
| 2000 km | 29.8 | **25.2** | 43.6 |

## WDMAM L2: no change in the answers, and the fit gets cheaper

162 vertices x 40 window rungs, refit from the cached spectra
(`03_compute_curie.py --stage b`), diffed with `compare_archives.py`.

- **Reduced chi²: identical to four decimal places at every window.** Effect
  size `|median d| / IQR` runs 0.000 to 0.004 across all 40 rungs.
- **CPD unchanged** at every rung ≥1500 km (median difference 0.00, correlation
  1.000). Below that it moves: median 0.3–1.4 km, IQR 1.5–2.7 km, max 3.3–5.6 km
  at 250–1250 km, correlation 0.979–0.988. For scale, the `minimize` →
  `least_squares` swap moved those same short rungs by **up to 41 km**.
- **Where the difference lives** at 4000 km: median 0.00 in ocean, mixed and
  land alike.
- **Cost**: the fit itself is 13–22% cheaper; including the ladder the total is
  +4% to +17%.

### Geology is unchanged, and right in both

| window | ocean | continent | c − o | craton | ridge | ratio | ρ(age) | ρ(age∣zt) |
|---|---|---|---|---|---|---|---|---|
| 6000 | 21.56 | 51.15 | 29.58 | 46.39 | 11.65 | 3.98 | 0.233 | 0.363 |
| 4000 | 18.73 | 41.09 | 22.36 | 36.81 | 8.17 | 4.51 | 0.441 | 0.469 |
| 2000 | 18.50 | 30.98 | 12.48 | 26.99 | 9.61 | 2.81 | 0.324 | 0.336 |

The derived start reproduces every one of these to within 0.01–0.02, except at
1000 km where the craton:ridge ratio moves 10.90 → 7.85 on a two-site median.
Continents are deeper than oceans, cratons deeper than ridges, and CPD rises
with seafloor age, in both.

**Read `ρ(age∣zt)`, not `ρ(age)`.** Seafloor age and water depth correlate at
+0.72 and water depth enters CPD through the pinned `zt`, so a raw age
correlation moves when nothing geological has — see
`notes/bouligand-rederivation.md` §3.

## `sensitivity`: the diagnosis was right, the cause was not

`sensitivity` fitted the unresampled spectrum once and started every Monte Carlo
realisation there. It did strand its ensemble: on a synthetic whose true `dz` is
45 km the ensemble came back with a **median of 10.3**.

But the cause was the constant, not the sharing. Warm starting every realisation
from a *correct* fit gives a spread of 13.28 against re-seeding's 13.28 —
indistinguishable. Resampling `Phi` within `sigma_Phi` does not move a
realisation across a basin boundary. Against the spread over independent
realisations of the field, both report 0.50–0.62 of it, which is the
bin-independence assumption and is untouched by any choice of start:

| case | ensemble sd | locked | ratio | re-seeded | ratio |
|---|---|---|---|---|---|
| 4000 km, dz 45 | 6.13 | 3.27 | 0.53 | 3.27 | 0.53 |
| 4000 km, dz 30 | 3.40 | 2.11 | 0.62 | 2.11 | 0.62 |
| 1000 km, dz 20 | 4.68 | 2.32 | 0.50 | 2.32 | 0.50 |

**The warm start therefore stays.** Re-deriving per realisation was built,
measured and reverted: 1.4x the forward-model evaluations for a spread that
agrees to two decimal places. What the ensemble depends on is the fit it starts
from, and that is now derived rather than declared.

## Loose ends

- **The `dz = 10` premium at thin layers on large windows** (2.6x evaluations
  for the same answer) is the one unambiguous regression. It is the case the
  constant was built for.
- **The same identity has a second use, on the `posterior-mesh` branch.** The
  reduced surface is the exact marginal posterior of `(beta, dz)`, so it can be
  integrated rather than sampled: `posterior()` and `profile(method="mesh")`
  live there, with `notes/collapsed-posterior.md` recording what they measured.
  Deliberately not merged here — that work raised its own questions about
  interval calibration which are unresolved.
- **`_asymptote_estimates` also returns a `dz`**, from the low-`k` band. It is
  badly biased — 4 km against a truth of 15 at a 1000 km window, because that
  band never reaches `k·dz ≪ 1` — and is used only as one more ladder node,
  where being wrong costs nothing.
