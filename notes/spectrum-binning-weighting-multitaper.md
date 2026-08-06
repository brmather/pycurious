# Binning, low-k weighting, and multitaper: measured

Three questions about how `optimise` is fed, answered on generated synthetics
(where truth is known) and on the WDMAM L2 mesh (162 vertices, the windows the
published archive used). Harnesses in `notes/bench/`:
`spectral_variants.py` (the estimators), `calibrate_dof.py`,
`run_experiments.py`, `score_spectra.py`. Everything below is measured, with
the BLAS pinned to one thread.

**Verdict in one line each.** Binning is a *sufficient statistic* — dropping it
costs 31x the CPU and buys nothing. Weighting low wavenumbers more is either
harmful or the single biggest improvement available, depending entirely on
whether the model is right at high k — and on WDMAM it is not. A multitaper is
a large, clean win on data that fits the model (`dz = 30` km recovers as
30.4 ± 1.5 where the default gives 23.3 ± 10.1) and is *actively dangerous*
on WDMAM, where it sharpens the error bars on an answer that is wrong.

The chain is verified: the spectra computed here reproduce
`spectra_wdmam_L2.zarr` — the archive's own cached spectra — to float32
(`check_windows.py`), and the binned variant reproduces the archive's
`_spectrum` to 7e-15 (`check_variants.py`).

---

## 0. What the current default actually weights by

Confirming the premise first, since it decides everything after it.
`residuals` (`optimise_bouligand.py:366`) is

    r = (Phi_syn - Phi) / sigma_Phi

with `Phi` the mean of `ln|FFT|**2` over an annulus and `sigma_Phi` the
uncertainty of that mean, both in log units. So yes — `sigma_Phi` is carried
through as the uncertainty *on* `Phi`, and it is the only thing deciding how
much of the fit each wavenumber pays for.

`window_spectrum` sets `sigma = scatter / sqrt(counts / dof_factor)`, and
`counts` grows linearly with `k`. **The default therefore weights each bin
in proportion to `k`**: at a 2000 km window the outermost retained bin counts
about 70x the innermost. The wavenumbers that determine `dz` are the ones the
fit listens to least. That is a consequence of the geometry, not a choice
anyone made.

## 1. Is the binning needed? No — but removing it gains nothing

For a log-periodogram, binning is *lossless*. Each cell carries
`ln|FFT|**2 = ln Phi(|k|) + ln E`, `E ~ Exp(1)`: the noise is additive in the
log and identically distributed, mean `-gamma`, variance `pi**2/6`, **whatever
the spectrum does**. The model depends on a cell only through `|k|`, so within
an annulus every cell has the same mean and variance, and

    sum_cell (m - y_cell)**2 / s**2  ==  n (m - ybar)**2 / s**2 + const

Fitting `n` cells at `sigma = s` and fitting their mean at `sigma = s/sqrt(n)`
are the same least-squares problem. Measured, both parts hold:

* the per-cell sd is `pi/sqrt(6)` to 0.4% (untapered, undetrended: measured
  `n_eff` over 196 bins is 1.004x the count of unique cells);
* the annulus-averaging of the *model* — the one term the identity neglects,
  since `ln Phi` is not flat across a bin — is worth **≤ 0.03 km of `dz`**.
  Evaluating the model at the bin's geometric mean `k`, or averaging it
  properly over the cells, moves nothing (`check_binning_bias.py`).

What it costs to skip binning, at the WDMAM windows:

| window | bins | unique cells | binned | unbinned | ratio |
|---|---|---|---|---|---|
| 1000 km | 39 | 2 506 | 6.7 ms | 59.9 ms | 8.9x |
| 2000 km | 79 | 9 994 | 11.9 ms | 209 ms | 17.5x |
| 4000 km | 158 | 39 894 | 27.4 ms | 851 ms | 31.1x |

The cost is `kv` evaluations, so it scales with the residual count.

And the accuracy is the same. On WDMAM the unbinned fit moves median `dz` by
**0.07–0.18 km**, and on clean synthetics at 4000 km it lands within 0.05 km
of a binned fit that simply uses a *known* weight instead of an estimated one:

| 4000 km, clean | bias | RMS | median sigma_dz | ms |
|---|---|---|---|---|
| `default` (binned, empirical sigma) | −3.12 | 8.02 | 2.39 | 24.7 |
| `unbinned` | −2.64 | 7.77 | 1.66 | 576 |
| `hann-dof` (binned, measured sigma) | −2.59 | 7.72 | 2.60 | 25.1 |

**`hann-dof` reproduces `unbinned` to 0.05 km at 1/23 of the cost.** So the
small edge unbinning has over the default is not from unbinning: it is from not
estimating the weights out of the same data being fitted. You can have that
inside the bins.

There is also a reason to prefer bins beyond cost. A taper correlates
neighbouring cells, and a bin is where that correction is applied
(`_dof_factor`). An unbinned fit with a diagonal weight cannot represent it at
all — it would need the full covariance of ~40 000 cells.

## 2. The degrees-of-freedom table, and why the default is honest

`_TAPER_DOF` deflates the count to an effective one, `(counts - lost)/dof_inf`.
Measuring both terms directly — 500 realisations, variance of each bin mean
across them — shows the two halves of the default are each individually wrong
and cancel (`calibrate_dof.py`, n=401, hanning):

| bin | k | counts | n_eff measured | n_eff from `_TAPER_DOF` | scatter/theory | **net sigma** |
|---|---|---|---|---|---|---|
| 0 | 0.0038 | 8 | 2.18 | 0.94 | 0.645 | **0.982** |
| 1 | 0.0073 | 16 | 4.56 | 3.36 | 0.816 | **0.951** |
| 5 | 0.0199 | 36 | 11.32 | 9.42 | 0.917 | **1.005** |
| 78 | 0.2489 | 496 | 158.69 | 148.82 | 0.993 | **1.025** |

`lost = 4.9` over-deflates the sparse inner bins by up to 2.3x — and the
*empirical within-annulus scatter*, which pycurious multiplies it by,
under-estimates the per-cell sd by exactly the reciprocal, because with 8 cells
under a taper there is barely any scatter to measure. The product is honest to
1–3% across the whole band. **Do not "fix" either half alone.** Anyone
replacing the empirical scatter with the theoretical `pi/sqrt(6)` must drop the
`lost` term with it, or every low-k bin gets 1.5x too much sigma.

The measured table is the reason `hann-dof` above is a fair variant rather than
a mis-weighted one.

## 3. Weighting low k more: right answer depends on the data

Two ways to lean on long wavelengths, both normalised so they only
*redistribute* weight: `tilt` multiplies sigma by `(k/k0)**alpha` (alpha = 0.5
exactly cancels the geometric `k` tilt, giving every bin equal weight), and
Global_CPD's own `sigma_weight_km = 8`, which adds `k**2 L**2` in quadrature.

On a **clean** field it can only hurt — there is no bias to trade variance
against. 4000 km, dz recovered (120 seeds per level):

| variant | dz=10 | dz=20 | dz=30 | mean abs bias |
|---|---|---|---|---|
| `default` | 9.6 ± 1.6 | 19.0 ± 4.1 | 23.3 ± 10.1 | 2.67 |
| `tilt0.5` | 10.9 ± 5.6 | 20.1 ± 6.6 | 29.1 ± 8.9 | 0.64 |
| `tilt1.0` | 23.6 ± 31.5 | 34.3 ± 29.5 | 39.9 ± 23.6 | 12.61 |
| `archive` (sigma_weight 8) | 9.9 ± 4.1 | 21.4 ± 9.2 | 32.0 ± 12.4 | 1.19 |

Even here a moderate tilt trades a real bias reduction for a real variance
increase; `tilt1.0` is a disaster.

On a field **degraded to WDMAM's effective resolution** — a Gaussian rolloff at
`sigma_r = 4.2 km`, which multiplies the power spectrum by
`exp(-k**2 sigma_r**2)` and which the Bouligand model has no term for — the
ranking inverts completely:

| variant | dz=10 | dz=20 | dz=30 | mean abs bias |
|---|---|---|---|---|
| `default` | **14.4 ± 0.3** | **16.1 ± 0.5** | **16.9 ± 0.7** | 7.14 |
| `archive` (sigma_weight 8) | 15.2 ± 5.6 | 24.2 ± 5.4 | 30.3 ± 5.8 | **3.24** |
| `kmax 0.10` (hard cut) | 13.5 ± 5.8 | 25.9 ± 8.9 | 33.5 ± 7.8 | 4.30 |
| `tilt0.5` | 16.8 ± 2.5 | 21.5 ± 2.2 | 24.2 ± 2.5 | 4.71 |
| `unbinned` | 14.1 ± 0.3 | 15.6 ± 0.3 | 16.3 ± 0.4 | 7.38 |
| `mt-NW3K5` | 13.2 ± 0.2 | 15.1 ± 0.2 | 16.1 ± 0.3 | 7.34 |

The default collapses a 3x range of truth into a 1.2x range of estimate, with a
scatter of ±0.3–0.7 km. It is precise and it is wrong, and nothing that
improves the *statistics* rescues it — unbinned and multitaper are the same or
slightly worse, with tighter error bars.

So: **the archive's `sigma_weight_km = 8` is doing real work**, and this
reproduces its documented calibration independently. The user-facing rule is
that down-weighting high k is not a statistical improvement, it is a defence
against model error, and it should be turned on exactly when there is model
error to defend against.

## 4. Multitaper

Separable DPSS (`h_i(x) h_j(y)`, `K**2` tapers, power averaged across the stack
*before* the log — the only order that reduces variance). Note this
concentrates in a square rather than the disk a true 2-D Slepian would use, so
the corner tapers are the least well concentrated in the stack.

**It must be recalibrated or it lies.** `_dof_factor` falls back to the
untapered `(2.0, 3.4)` for any taper it does not know, and a multitaper's
within-annulus scatter collapses to 0.13 of the per-cell theory (neighbouring
cells are built from overlapping bands), so the two together understate sigma
by **2.7x**. Fitted naively (`mt-naive`) the zt prior gets overwhelmed —
`zt` walks off its 1.0 ± 0.05 pin to 1.22 — and reported coverage falls to 1%.
Per-bin `n_eff` measured by Monte Carlo (`make_neff_tables.py`) fixes this.

Recalibrated, it is the largest clean-data gain measured here. 4000 km,
clean field:

| variant | dz=10 | dz=20 | dz=30 | RMS | ms |
|---|---|---|---|---|---|
| `default` | 9.6 ± 1.6 | 19.0 ± 4.1 | 23.3 ± 10.1 | 8.02 | 24.7 |
| `mt-NW2K3` (9 tapers) | 10.1 ± 0.8 | 20.5 ± 1.4 | 29.6 ± 4.1 | 2.75 | 141 |
| `mt-NW3K5` (25 tapers) | 10.1 ± 0.8 | 20.5 ± 1.4 | **30.4 ± 1.5** | **1.20** | 363 |

At 4000 km the fraction of realisations more than 25% out in `dz` goes from
**24% to 0%**. This is the direct answer to the standing "`dz` does not recover
— one realisation in five is tens of percent out" in `CLAUDE.md`: that is a
property of a *single-taper* estimate, not of the problem. The measured gain is
12.6x more effective looks in the innermost bin (`n_eff` 27.5 against 2.18).

The cost is 5.2x (`NW=2, K=3`) or 13.3x (`NW=3, K=5`) at a 4000 km window.
`NW=2, K=3` gets most of the benefit for a third of the cost, and its
resolution bandwidth is narrower, which matters because the smoothing is over
`NW dk` — right where the spectrum turns over. The measured leakage bias at the
innermost bin is about the same as hanning's (−0.21 against −0.22) but is
spread over ~NW bins instead of one.

**On WDMAM it does not help, and it is the most dangerous option on the list.**
It reports the *smallest* sigma_dz of any variant (0.82 km at 4000 km, against
the default's 2.97) while returning essentially the default's answer, and its
reduced chi-squared rises to 8.7 — the extra precision is not measuring the
depth, it is exposing that the model does not fit. If it is used there, sigma
must be inflated by `sqrt(chi2)` at minimum, which gives back 2.4 km.

## 5. WDMAM L2: what actually changes

162 vertices, `kmax = 0.25`, `zt` pinned at 1.0 ± 0.05, beta free, three window
sizes. Median `dz` and its spread across the mesh at 4000 km:

| variant | median dz | p10 | p90 | **p90−p10** | median sigma_dz | chi2 | ms |
|---|---|---|---|---|---|---|---|
| `default` | 16.07 | 12.43 | 18.63 | 6.20 | 2.97 | 1.84 | 27 |
| `unbinned` | 15.75 | 12.78 | 18.25 | 5.47 | 1.74 | 1.12 | 851 |
| `hann-dof` | 15.75 | 12.84 | 18.20 | 5.37 | 2.99 | 2.13 | 28 |
| `mt-NW3K5` | 14.96 | 12.53 | 17.66 | **5.13** | 0.82 | 8.72 | 364 |
| `tilt0.5` | 22.75 | 13.57 | 32.78 | 19.21 | 2.26 | 3.73 | 26 |
| `archive` (sigma_weight 8) | 25.16 | 10.18 | 51.72 | **41.54** | 9.90 | 0.24 | 25 |

The top four are the same map. Every variant that is statistically better —
unbinned, measured-dof, multitaper — converges on ~15.5 km everywhere with an
ever tighter error bar and an ever flatter map. That is the degraded-synthetic
pathology in the field: 12–19 km from p10 to p90 across the entire globe, no
craton/ocean contrast. The high-k band of WDMAM is not carrying depth
information, and fitting it more faithfully just gets the same wrong answer more
confidently.

### The nested-window consistency proxy is misleading here

Since WDMAM has no truth, the obvious proxy is whether the 1000/2000/4000 km
rungs agree. **They should not be trusted.** Fitting nested windows of the same
*synthetic* field, where the error is knowable:

| degraded 4.2 km | window spread | abs bias at 4000 km |
|---|---|---|
| `default` | **0.68** | 4.22 |
| `unbinned` | **0.48** | 4.63 |
| `mt-NW3K5` | **0.54** | 5.18 |
| `tilt0.5` | 2.31 | **0.85** |
| `archive` | 5.02 | 3.24 |

Window agreement is *anti-correlated* with accuracy once the model is wrong:
the default absorbs the rolloff the same way at every window, so it is
consistently wrong. On a clean field the proxy behaves (multitaper wins both).
The WDMAM consistency table in `score_spectra.py` — where `default` scores 1.20
km and `archive` 6.30 — is therefore **not** evidence that the default is
better.

## 6. What I would change in pycurious

1. **Give `_dof_factor` a loud failure for uncalibrated tapers.** It silently
   returns the untapered `(2.0, 3.4)` for anything not in `_TAPER_DOF`, which
   understated a multitaper's sigma by 2.7x here. A warning naming the taper
   would have saved that debugging.
2. **Offer the measured weight as an option.** `sigma = pi/sqrt(6 n_eff)`
   with the `_TAPER_DOF` `lost` term dropped removes the weight-estimation
   noise, matches an unbinned fit to 0.05 km, and costs nothing. It is also the
   only correct weighting for a multitaper.
3. **Do not bother with unbinned fitting.** 31x for nothing.
4. **A multitaper is worth adding** for well-resolved data — it is the only
   thing measured here that makes `dz = 30 km` recoverable from one realisation
   — but only together with a calibrated `_TAPER_DOF` entry and a note that it
   is a variance tool, not a bias tool.
5. The `CLAUDE.md` claim that `dz` recovery is hopeless in one realisation in
   five should be qualified: that is single-taper behaviour.

## Caveats

* The multitaper is separable DPSS, not a true 2-D Slepian.
* `n_eff` is calibrated on `beta = 3` synthetics; leakage makes it mildly
  spectrum-dependent.
* Synthetic accuracy is measured with `zt` pinned at the truth, as the WDMAM
  workflow pins it. With `zt` free the rolloff is absorbed into `zt` instead
  and every variant degrades — see `ZT_FIXED_KM` in `curie_config.py`.
* WDMAM fits here are pass A (beta free). The published archive is pass B,
  with a prior on beta at a 1500 km-smoothed value, which is why its median
  `dz` at 2000 km is 19.6 km against the 26.7 km the `archive` variant gives
  here. The comparison between variants is like-for-like; the comparison to the
  published number is not.

---

# Addendum: regularisation, correlation, and a starting-point defect

Follow-up to "should the fit be regularised, given the FFT carries
correlation". Harnesses: `measure_correlation.py`, `test_gls_fit.py`,
`test_regularisation.py`.

## The correlation is real, stationary, and already handled correctly

Measured across 600 realisations at n=401 (`measure_correlation.py`), the
bin-to-bin correlation of the binned log spectrum under `numpy.hanning`:

| bins | lag 1 | lag 2 | lag 3 | lag 4 |
|---|---|---|---|---|
| 0–4 | 0.348 | 0.021 | 0.018 | 0.038 |
| 15–39 | 0.367 | 0.051 | 0.008 | 0.009 |
| 100+ | 0.365 | 0.038 | −0.001 | 0.001 |

Three things follow. It is **stationary** — 0.36 in every band, so a Toeplitz
band describes it exactly and no `k`-dependence is being missed. It **dies at
lag 2**, so `_CORRELATION_BANDS = 2` is right. And `_banded_correlation`
recovers it from a *single* realisation's residuals to 0.346 ± 0.047 against a
true 0.364 — it is doing its job, and confirms the 0.363 in its own docstring.

**Using it in the fit as well as the covariance gains nothing.** GLS
(`r^T R^-1 r`, whitened by the measured band) against the current diagonal fit,
4000 km, 120 seeds:

| | clean dz=20 | clean dz=30 | degraded dz=20 |
|---|---|---|---|
| OLS (today) | 19.05 ± 4.13 | 23.31 ± 10.08 | 16.09 ± 0.54 |
| GLS | 18.65 ± 4.51 | 23.40 ± 10.43 | 16.31 ± 0.67 |

Shifts are ≤ 0.4 km and the scatter is marginally *worse*. This is the expected
answer, not a surprise: for stationary noise and smooth regressors — and all
four Jacobian columns are smooth in `k` — OLS is asymptotically as efficient as
GLS. The correlation is worth ~30% on `sigma` and nothing on the estimate,
which is exactly the split pycurious already implements. **The asymmetry
between `_fit` and `_covariance` is justified; leave it alone.**

## A multitaper should be binned at its resolution bandwidth

The one place the correlation is not benign. For `NW=3, K=5` it reaches five
bins (0.86, 0.66, 0.44, 0.24, 0.09), because the estimate is smoothed over
`NW dk` while the bins are `dk` wide — so ~5 bins carry one bin's information.
`_banded_correlation` cannot express that: it returns 0.270/0.205, whose sum is
**exactly** the 0.475 that `_CORRELATION_LIMIT = 0.95` permits. It is
saturating a clamp, silently. Widening the bins fixes it at the source:

| bin width | bins | lag 1 | lag 2 | lag 3 |
|---|---|---|---|---|
| 1 dk | 199 | 0.860 | 0.661 | 0.437 |
| 2 dk | 99 | 0.704 | 0.268 | 0.036 |
| 3 dk | 66 | 0.522 | 0.063 | −0.004 |
| **4 dk** | **49** | **0.376** | **0.008** | −0.012 |

At 4 dk a multitaper has the same correlation structure as a hanning spectrum
at 1 dk (0.376 against 0.364) — so the existing `_banded_correlation` and
`_TAPER_DOF` machinery applies unmodified, over a quarter as many bins. There
was no information between those bins to lose.

## The rolloff is better modelled than down-weighted

`curie_config` records that `sigma_r` is degenerate with `zt`. **That was
measured with `zt` free.** Pinned — which is what production does — the
degeneracy breaks: `exp(-2 k zt)` and `exp(-k**2 sigma_r**2)` have different
shapes in the exponent. Fitting a fifth parameter, 4000 km, 60 seeds, field
degraded at `sigma_r = 4.2` km:

| truth dz | 4-parameter (today) | + free sigma_r | sigma_r recovered |
|---|---|---|---|
| 10 | 14.4 ± 0.3 | 9.2 ± 2.3 | 4.29 ± 0.24 |
| 20 | 16.1 ± 0.6 | 19.3 ± 4.9 | 4.18 ± 0.26 |
| 30 | 16.9 ± 0.8 | 28.5 ± 8.2 | 4.07 ± 0.40 |

Mean absolute bias drops from **7.14 km to 1.0 km** — better than
`sigma_weight_km = 8` (3.24) or a hard cut (4.30) — and `sigma_r` comes back
within 0.1 km of truth. A prior on `sigma_r` changes nothing (4.29 against
4.29): once `zt` is pinned it is well identified and needs no regularisation.
**The answer to "should we regularise" is, here, "you do not need a penalty,
you need a parameter".**

On the 162 WDMAM L2 vertices at 4000 km the fitted `sigma_r` is **3.68 km
(p10–p90 2.17–4.58)** — derived from the data alone, against the 4.2 km
`curie_config` assumed on other grounds. `dz`'s p10–p90 spread goes from 6.2 km
(the flat map) to 37.7 km, comparable to what `sigma_weight_km` recovers.

Two cautions. The synthetic degradation is *exactly* the Gaussian being fitted;
WDMAM's real resolution loss is a patchy, direction-dependent compilation
artefact, so recovery in the field will be worse than this. And on a **clean**
field the extra parameter is not free — at dz=30 the scatter goes from ±3.3 to
±12.4 — so it should be enabled deliberately, not by default.

## The `dz = 10` starting value loses a basin — and it is ~1 in 5

Found while checking the above. On identical spectra, `optimise` and a plain
`least_squares` call disagree on **8 of 40** clean synthetics at `dz = 30`,
by 25–35 km. It is not the Jacobian — the analytic `zt` and `C` columns are
exact to 2e-9. The misfit is genuinely **bimodal in `dz`** and the default
start sits in the wrong basin:

| start | seed 24 | cost | seed 9 | cost |
|---|---|---|---|---|
| `dz = 10` (the default) | 9.45 | 99.10 | 7.64 | 97.23 |
| `dz = 20, 30, 40` | 43.33 | **73.71** | 39.98 | **65.91** |

The misfit is 25% lower in the other basin — this is not the documented
flat-likelihood tie where "both answers are equally good". Multi-starting over
`dz ∈ {5, 15, 30, 60}` and keeping the lowest cost, 120 seeds at 4000 km:

| truth dz | single start (today) | multi-start | mt-NW3K5 single | mt-NW3K5 multi |
|---|---|---|---|---|
| 10 | 9.6 ± 1.6 | 9.6 ± 1.6 | 10.1 ± 0.8 | 10.1 ± 0.8 |
| 20 | 19.0 ± 4.1 | 19.5 ± 4.1 | 20.5 ± 1.4 | 20.5 ± 1.4 |
| 30 | **23.3 ± 10.1** | **30.7 ± 4.9** | 30.4 ± 1.5 | 30.4 ± 1.5 |

**This corrects the multitaper claim in the main note.** Section 4 credited the
multitaper with taking `dz = 30` from 23.3 ± 10.1 to 30.4 ± 1.5. Roughly half
of that is the starting point: a properly started single-taper fit already
reaches 30.7 ± 4.9. The multitaper's real contribution is the remaining 3.3x on
variance, and its immunity to the basin problem (identical with and without
multi-start — the smoother spectrum has no second minimum). It is still the
better estimator; it is not worth 13x the CPU for something four extra starts
fix for free.

The known defect in `CLAUDE.md` — "`profile` reports one basin of a multimodal
deviance" — is therefore not confined to `profile`. `optimise` walks into the
same second basin, at 20% of thick-layer synthetics, and reports it without
complaint. Worth doing before anything else on this list.
