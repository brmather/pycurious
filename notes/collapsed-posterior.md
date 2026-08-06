# The Bouligand posterior is two-dimensional, and small enough to integrate

`posterior()` and `profile(..., method="mesh")` are **in the library**;
`notes/bench/collapsed_mcmc.py` is the prototype that established the identity
and measured what a chain is worth against it. Harnesses:
`notes/bench/score_intervals.py` (coverage) and `notes/bench/score_mesh_L2.py`
(WDMAM L2, all 40 rungs).

**Read the calibration section before using either interval.** Both the mesh and
the existing deviance scan under-cover, for a reason that is upstream of both
and is not fixed here.

`metropolis_hastings` walks four parameters. Two of them need not be walked, and
once they are gone the remaining two need not be walked either.

In one paragraph: `C` and `z_t` enter the forward model as linear coefficients
on basis vectors that do not depend on the other two parameters, so their
conditional posterior is an exact Gaussian and integrates out in closed form,
leaving a marginal posterior over `(beta, dz)` that is *exactly* the reduced
misfit `_solve_linear` already computes. Sampling that 2-D marginal instead of
the 4-D posterior is worth about **1.9x** in effective sample size. Evaluating it
on a mesh instead of sampling it at all is worth about **40x** against the chain,
and lands at **parity with two calls to `profile`** — while returning the entire
joint posterior rather than one interval.

## The identity

Write the forward model out (`pycurious/grid.py:946`):

```
Phi(k) = C·1  +  z_t·(-2k)  +  h(k; beta, dz)
```

`C` appears once, additively. `z_t` appears once, as `-2k z_t`. The Bessel term
`h` depends on `beta` and `dz` only. So with `p = (C, z_t)` the whitened
residual is affine in `p`:

```
r(p) = G p - y,    G = [1/sigma, -2k/sigma],    y = (Phi - h(beta, dz))/sigma
```

plus one row per Gaussian prior on `C` or `z_t`. Completing the square,

```
0.5‖G p - y‖² = 0.5 (p - p̂)ᵀ A (p - p̂) + F(beta, dz)
A = GᵀG,   p̂ = A⁻¹ Gᵀ y,   F = 0.5‖G p̂ - y‖²
```

and the Gaussian integral over `p` is

```
∫ exp(-0.5 (p-p̂)ᵀA(p-p̂)) d²p = 2π / √(det A)
```

so

```
-log p(beta, dz | data) = F(beta, dz) + 0.5 log det A + const.
```

**The whole result turns on `A` being constant.** `G` is built from `k`, `sigma`
and the prior widths — nothing in it depends on `(beta, dz)`. So `log det A` is
an additive constant, and the marginal posterior of `(beta, dz)` *is* the reduced
misfit `F`, which is exactly what
`CurieOptimiseBouligand._solve_linear` returns as its third value.

Verified against it to **2.5e-16 to 5.2e-16** relative, at three or four
`(beta, dz)` points per case, by `check_target_matches_library`. That check is
not decoration: without it the prototype could be sampling a subtly different
posterior from the one `optimise` minimises, and every comparison below would be
meaningless.

### What would break it

- **A `sigma` that depended on the parameters.** It does not — `sigma_Phi` comes
  out of `window_spectrum` and is fixed for the fit. But a subclass that
  reweighted `sigma` per iteration would destroy the constancy of `A`.
  `~/Global_CPD`'s inflation is applied once in `_spectrum`, so it is safe.
- **A non-Gaussian prior on `C` or `z_t`.** In principle fatal; in practice
  already ruled out. `add_prior` accepts any frozen `scipy.stats` distribution,
  but `_prior_loc_scale` reduces it to `(loc, scale)` and `residuals` uses only
  `(m - loc)/scale`. The objective the package minimises is Gaussian in its
  priors **by construction**, so this imposes no restriction that is not
  already there.
- **Priors on `beta` or `dz`.** No problem at all — they are constants of the
  linear solve and simply ride along in `F`.

### The same identity, three ways

Worth naming, because it keeps reappearing:

| use | where | what it gives |
|---|---|---|
| two exact Jacobian columns | `_ANALYTIC_COLUMNS` | `dr/dz_t = -2k/sigma`, `dr/dC = 1/sigma` |
| the derived starting point | `_solve_linear`, `_initial_guess` | one argmin of `F` over a `dz` ladder |
| the marginal posterior | this note | all of `F`, as a density |

The first two are in the library. The third is the same algebra read as a
probability rather than as an optimisation.

## Collapsing the chain is worth 1.9x, not an order of magnitude

The collapsed sampler walks `(beta, dz)` with target `-F`, and for each stored
step draws `(C, z_t)` from `N(p̂(beta, dz), A⁻¹)`. That is Rao-Blackwellisation:
the linear pair's marginals come out exactly right without the chain having to
explore them.

20,000 samples after 4,000 burn-in, three cases, BLAS pinned, both chains
spending exactly one forward-model evaluation per step (24,001 against 24,038 —
the difference is the burn-in bookkeeping).

| case | | beta | z_t | dz | C |
|---|---|---|---|---|---|
| **1000 km, dz 20** | ESS, 4-D | 1294 | 1311 | 866 | 1330 |
| | ESS, collapsed | 1944 | 2068 | 2080 | 2119 |
| | gain | 1.50x | 1.58x | **2.40x** | 1.59x |
| | KS *p* (thinned) | 0.81 | 0.32 | 0.83 | 0.13 |
| **4000 km, dz 45** | ESS, 4-D | 1207 | 1203 | 1163 | 1213 |
| | ESS, collapsed | 2172 | 2268 | 2197 | 2212 |
| | gain | 1.80x | 1.89x | 1.89x | 1.82x |
| | KS *p* (thinned) | 0.52 | 0.77 | 0.04 | 0.19 |
| **4000 km, dz 20, `z_t` pinned** | ESS, 4-D | 1024 | 981 | 1047 | 1006 |
| | ESS, collapsed | 1782 | 1863 | 1979 | 1801 |
| | gain | 1.74x | 1.90x | 1.89x | 1.79x |
| | KS *p* (thinned) | 0.66 | 0.48 | 0.25 | 0.38 |

Means agree to 0.002–0.050 of a standard deviation and sd ratios run 0.98–1.03.
The one KS at *p* = 0.04 is unremarkable across twelve comparisons — Bonferroni
would want 0.004.

In wall clock the gain is smaller than in ESS: the collapsed chain is ~3–6%
slower per step for the 2x2 solve and the conditional draw, so **ESS/s runs
1.4–1.8x**.

**Why only 1.9x.** `metropolis_hastings` already proposes along the fit
covariance (`_proposal_cholesky`), which follows the `beta`–`z_t` (−0.92) and
`z_t`–`C` (+0.87) ridge closely. Removing the ridge helps less than its
correlation suggests, because the existing proposal was already shaped to it.
An earlier claim in `notes/derived-starting-values.md` that the ridge "simply
stops existing" was true and beside the point — what matters is how much of the
sampler's difficulty the ridge was still causing, and the answer is: about half.

The one place it does more is `dz` at the 1000 km window, **2.40x**, which is
the case where `dz` is worst determined and the 4-D chain's `dz` ESS (866) falls
furthest behind the other three (~1300). Collapsing brings it level with them.

## Not sampling at all is worth ~40x

Two dimensions is small enough to evaluate rather than sample. `quadrature`
places a `nodes x nodes` grid over `(beta, dz)`, normalises `exp(-F)` on it, and
takes moments as sums against the density. `(C, z_t)` follow from the same
conditional: their mean is the density-weighted mean of `p̂`, and their variance
adds the conditional variance `A⁻¹` to the spread of `p̂` across the mesh — the
law of total variance, because the mesh sees the conditional *mean* moving but
not the conditional spread.

Convergence, against a 192x192 reference:

| nodes | evaluations | seconds | dz mean error | dz sd ratio | edge mass |
|---|---|---|---|---|---|
| 16 | 256 | 0.02–0.04 | 0.0008–0.020 sd | 0.985–1.000 | 7e-12 – 1e-7 |
| **24** | **576** | **0.05–0.08** | **≤0.0002 sd** | **1.0000** | 3e-12 – 9e-8 |
| 32 | 1024 | 0.09–0.15 | 0.0000 sd | 1.0000 | 2e-12 – 6e-8 |
| 128 | 16384 | 1.5–2.2 | — | — | 6e-13 – 2e-8 |

It converges at **24x24 = 576 evaluations**. At 32 nodes the moments agree with a
128x128 mesh to **3e-9** of a standard deviation and the standard deviations to
**1e-8** — which is to say the mesh is exact long before anything else in the
calculation is.

Against the 4-D chain's 24,000 evaluations and 2.1–3.6 s for an ESS of
1,000–1,200, that is **~40x fewer forward-model evaluations and ~40x less wall
clock, for an exact answer rather than a noisy one**: no autocorrelation, no
burn-in, no seed, no acceptance rate to tune, and no possibility of a chain that
has not mixed reporting a confident wrong interval.

Mass falling outside the mesh runs 2e-8 to 8e-13. That is the number to watch. A
quadrature that has clipped `dz`'s long upper tail is wrong in a way a chain is
not — silently, and in the direction of a *tighter* interval — so the prototype
reports it rather than assuming it.

## The comparison that matters is `profile`, not MCMC

Nobody runs `metropolis_hastings` at scale; `~/Global_CPD` runs `profile("dz")`
at 15 points, 26 million times at L8. That is the incumbent.

| case | `optimise` | `profile("dz")`, 15 pts | 24x24 mesh |
|---|---|---|---|
| 1000 km, dz 20 | 37 ev, 5.1 ms | 225 ev, 35.4 ms | 576 ev, 49.3 ms |
| 4000 km, dz 45 | 34 ev, 5.3 ms | 224 ev, 42.3 ms | 576 ev, 72.3 ms |
| 4000 km, dz 20, pinned | 37 ev, 6.5 ms | 243 ev, 50.7 ms | 576 ev, 80.5 ms |

So the mesh is **1.4–1.6x one `profile`, and roughly at parity with two** — and
one `profile` gives one interval on one target. The mesh gives the joint
posterior of all four parameters, from which an interval on `dz`, on `beta`, on
`C`, and on `CPD` are each a 1-D integral over a density already in hand.

That parity, not the 40x against MCMC, is the case for building this.

## The `z_t >= 0` bound, handled exactly

`log det A` drops out. The bound does not. `z_t` is bounded below at zero, so the
conditional is a Gaussian truncated to a half plane, and its mass

```
P(z_t >= 0 | beta, dz) = Phi_cdf(p̂_{z_t} / s_{z_t}),    s_{z_t} = √(A⁻¹)_{22}
```

depends on `(beta, dz)` through the conditional mean. It belongs in the target,
and it is there. The conditional draw then rejects into the same half plane, so
the pair `(beta, dz, C, z_t)` is an exact draw from the truncated posterior
rather than an approximation to it.

Measured: `min P(z_t >= bound) = 1.0000` and **zero** rejected draws in all three
cases — exact and entirely inert. That is what one should expect where the data
determine `z_t` well. It would not be inert on a window where `z_t` runs to its
bound, which is exactly where `_warn_on_bounds` fires and where the reported
covariance is already meaningless.

## Two measurement traps, both of which produced wrong numbers first

**KS on raw chain draws rejects everything.** The two-sample KS test assumes
independent draws; a chain of 20,000 steps with ESS ~1,000 has none. Un-thinned,
every one of twelve comparisons came back *p* < 0.006 on statistics of
0.02–0.05, for chains that are in fact indistinguishable — the test was
detecting autocorrelation, not disagreement. Thinning each chain to its own ESS
gives *p* = 0.13–0.83. Any comparison of two MCMC outputs needs this.

**A sd ratio of 0.96 against the chain looked like mesh error and was chain
noise.** With ESS ~2,000 the sampling sd of a sd estimate is ~1.6%, so a 4%
discrepancy is ~2.5 sigma and tempting to believe. It was not real: the mesh
agrees with a 4x finer mesh to 1e-8. **Compare a deterministic method against a
finer version of itself, never against a sampler** — the sampler is the noisy
one, and it is the one you would end up "fixing".

## Built: `posterior()` and `profile(method="mesh")`

The five points below were the shipping list, and four of them are now settled
in code. `posterior()` returns the density, its axes, the conditional mean of
`(C, z_t)` at each node, their constant conditional covariance, the truncated
mass, the mass on each edge, and `k_min`. `profile(..., method="mesh")`
integrates it down to the same 4-tuple the scan returns, so
`~/Global_CPD` switches with one keyword — verified end to end.

Three things bit during implementation, all now guarded by tests:

1. **The mesh is a quadrature rule, not a discrete distribution.** `z_t`'s
   conditional width is 0.007 km against a 2 km node spacing, so summing one
   narrow Gaussian per node gives a picket fence of 1024 spikes. Interpolating
   first costs no forward-model evaluations.
2. **`CPD = z_t + Δz` is a convolution along the mesh, not a marginal of it.**
   `E[CPD] = E[z_t] + E[Δz]` holds whatever the correlation, and is the identity
   that caught both wrong versions — the atoms version failed it by 0.5 km, a
   `mass` array shadowed by a loop variable by 0.53.
3. **An interval running past `1/k_min` is not a measurement.** Beyond that the
   rolloff sits below the longest wavelength the window measured and `Δz` is
   degenerate with `C`, so the endpoint is set by where `bouligand2009`
   overflows. At a 250 km window that produced `(190, 943)` km before the guard.

A highest-density interval was tried first and abandoned: it is read off a
derivative of an interpolated cumulative, and its endpoints moved 1.5 km between
a 192-node and a 768-node refinement while reporting spurious multimodality
below that. Inverting the CDF moves 0.05 km over the same range. A mode-counting
warning went the same way — on a flat posterior it reported 14 modes.

## Measured: neither interval is calibrated, and it is not the mesh's fault

200 synthetic realisations per cell, coverage of the known truth:

| regime | target | nominal | scan | mesh |
|---|---|---|---|---|
| 1000 km, dz 20 | beta | 0.6827 | 0.510 | 0.545 |
| | dz | 0.6827 | 0.430 | 0.420 |
| | dz | 0.95 | 0.805 | **0.835** |
| 4000 km, dz 30 | dz | 0.6827 | 0.550 | 0.555 |
| | dz | 0.95 | 0.835 | **0.850** |

Both under-cover, by about 0.25 at 68% and 0.11 at 95%. The mesh is marginally
better at 95% and indistinguishable at 68%. **The cause is shared and it is
upstream of both.**

`_gls_covariance` corrects the reported covariance for correlation between
neighbouring spectral bins. `min_func` does not — the likelihood still treats
them as independent. So the two disagree by exactly that factor:

| beta interval, 100 realisations | 68.27% | 95% |
|---|---|---|
| `optimise` sigma (GLS-corrected) | **0.650** | **0.920** |
| `profile(method="scan")` | 0.520 | 0.830 |
| `profile(method="mesh")` | 0.580 | 0.850 |
| predicted, likelihood too sharp by 1.298 | 0.559 | 0.869 |

The measured GLS inflation is **1.298 ± 0.047**, and it predicts the observed
under-coverage of both. Neither reading of the likelihood is at fault and no
change to how the interval is read can fix it. The split — OLS point estimate,
GLS uncertainty — was a deliberate and measured choice for the *fit*
(`notes/spectrum-binning-weighting-multitaper.md`: "OLS is already efficient for
smooth regressors"); what nobody noticed is that it leaves the *deviance*
uncorrected, so every interval inherits a likelihood that is too confident. It
is the same 0.5–0.6 that `sensitivity` reports against independent fields and
that `sigma_dz`'s "understates by about 40%" records.

The fix is to put `R^-1` in the objective, or to temper the deviance by the
ratio of GLS to naive curvature. **Neither is made here** — it changes every
published interval and is the user's call.

## Measured: WDMAM L2, all 40 rungs

162 vertices x 40 rungs from the cached spectra, mesh against the archived
`dz_lo`/`dz_hi`. `inf` counts unbounded upper endpoints.

| window | n | scan inf | mesh inf | scan width | mesh width | med \|Δlo\| | med \|Δhi\| |
|---|---|---|---|---|---|---|---|
| 10000 | 155 | 2 | 0 | 9.68 | 10.93 | 0.23 | 0.79 |
| 4000 | 162 | 0 | 0 | 8.2 | 8.9 | 0.35 | 0.95 |
| 2000 | 160 | 0 | 0 | 9.03 | 9.83 | 0.47 | 1.23 |
| 1250 | 155 | 0 | **13** | 10.78 | 11.43 | 0.63 | 1.71 |
| 1000 | 155 | 2 | **23** | 11.56 | 11.58 | 0.83 | 1.82 |
| 750 | 150 | 0 | **38** | 12.34 | 12.38 | 1.88 | 2.35 |
| 500 | 150 | 12 | **89** | 14.50 | 14.72 | 3.00 | 4.00 |
| 250 | 144 | 29 | **143** | 19.04 | 4.34 | 2.68 | 5.78 |

Two readings:

- **From 1500 km up the two track closely**, the mesh about 1 km wider on the
  upper end — the skew, which an equal-tailed interval carries and a level set
  does not.
- **Below that they part company, and the direction is the interesting one.**
  At 250 km the mesh declares `Δz` unbounded at **143 of 144** vertices; the
  archived scan does so at 29 and reports a confident ~19 km interval at the
  other 115. The mesh is not failing there — 250 km cannot resolve a layer whose
  rolloff sits below its longest wavelength, and saying so is the answer.

## The 4-D chain cannot referee, and that is a result

Disagreements were to be adjudicated by `metropolis_hastings`, run as four
dispersed chains with a mixing diagnostic because an unmixed chain refereeing
anything is worth nothing. **All 12 failed the diagnostic**, and the reason is
not the one that was guarded against:

| case | samples | R-hat | worst ESS |
|---|---|---|---|
| 9000 km, vertex 87 | 8,000 | 1.014 | **20** |
| | 40,000 | 1.002 | **33** |
| 2000 km, vertex 123 | 8,000 | 1.004 | **62** |
| | 40,000 | 1.001 | **396** |

R-hat is fine — the chains agree with each other. They simply do not move: an
effective sample size of 20 from 8,000 draws is an autocorrelation time of 400.
Reaching ESS 400 would take roughly half a million samples per vertex.

On the clean synthetics earlier in this note the same sampler reached ESS
1,200 from 20,000. The production regime — band limited, `sigma` inflated at
high `k`, `z_t` pinned to 0.05 km, a `beta` prior — is where it stops working.
So **MCMC is not an option there at all**, which is a stronger argument for
integrating the 2-D posterior than any of the speed numbers above.

It also means the adjudication question was the wrong one. Scan and mesh do not
disagree about which basin is right; they apply two different constructions to
the same too-sharp likelihood, and the coverage measurement above says what
that costs — for both.

## Putting `R^-1` in the objective: measured before doing it

The under-coverage above is the GLS correction missing from the likelihood. What
it would take, and what it costs, measured rather than assumed.

**It works.** A whitened objective, 200 realisations, 1000 km:

| target | nominal | diagonal (today) | whitened |
|---|---|---|---|
| beta | 0.6827 | 0.510 | **0.640** |
| beta | 0.95 | 0.850 | **0.945** |
| dz | 0.6827 | 0.430 | 0.515 |
| dz | 0.95 | 0.805 | 0.860 |

(An earlier version of this table read 0.670 and 0.545 for the whitened arm. It
was measured through `_fit`, which supplies exact Jacobian columns for `z_t` and
`C` that are only exact for the *diagonal* residual — so the whitened fit was
handed a wrong Jacobian for half its parameters. Both arms now difference every
column. The correction moves the numbers a little and the conclusion not at
all.)

`beta` becomes calibrated outright. `dz` improves by half the gap and does not
close it — the rest is its skew, which is what `profile` exists for.

**`R` cannot be estimated from the residuals in hand.** `_banded_correlation`
reads smooth model mismatch as correlation, which is correct for a covariance at
the solution and fatal in an objective. Measured over one spectrum, `rho_1` is
0.38 at the fitted point, 0.42 at `dz + 50%`, 0.24 at `beta = 2`. An objective
`r^T R(r)^-1 r` is not a fixed function of the parameters, its Jacobian is wrong,
and the fit can lower it by making its own residuals look correlated. It has to
be a table.

**The table, measured from known truth** (`notes/bench/calibrate_correlation.py`,
300 realisations, referenced against the across-realisation mean so no
deterministic term survives):

| taper | rho_1 | rho_2 | rho_3 |
|---|---|---|---|
| none | 0.003 | −0.005 | −0.002 |
| hanning | **0.355** | **0.032** | −0.007 |
| hamming | **0.310** | **0.020** | −0.006 |

Dead by lag 3 in every case, which is what `_CORRELATION_BANDS = 2` already
assumes. `hamming` had never been measured. An earlier attempt at this
referenced the residual against the *analytic* model and got 0.9 at every lag
including untapered — the log-periodogram's deterministic bias, divided by a
`sigma` that falls as `1/sqrt(k)`, is a strong smooth trend that a row-mean
subtraction leaves behind.

**What else moves:**

| consumer | effect |
|---|---|
| point estimate | `beta`, `zt`, `C` unchanged (≤0.02). `dz` mean shift +0.43 km, max 1.60 |
| `optimise`'s sigma | ≤5% — `now/after` is 0.987, 0.984, 0.947, 0.983 |
| `_covariance` | **must** drop `_gls_covariance` for `(J^T J)^-1`, or it corrects twice |
| reduced chi² | 0.983 → 0.997, inside the test's 0.7–1.6 |
| `dz` recovery in the mean | +6.4% → +5.7%, +6.8% → +6.9%; the 10% budget holds |
| variable projection | survives — `G^T R^-1 G` is still constant in `(beta, dz)` |
| `_ANALYTIC_COLUMNS` | must be whitened too, or the exact Jacobian columns are wrong — and this is not cosmetic: unwhitened, they sent a `dz = 10` layer at a 4000 km window to **60.9** |

**The cost, and it is not small.** Whitening degrades the `dz` *estimator*:
across 60 realisations its spread rises from 5.98 to 7.65 km, +28%, while its
sigma rises only 3.30 → 3.48. So `dz`'s own calibration ratio gets *worse*
(1.81 → 2.20) even as its interval coverage improves. This is the finding
`notes/spectrum-binning-weighting-multitaper.md` recorded as "whitening the fit
slightly worsens scatter"; for `dz` it is 28%, not slight. `R^-1` is a high-pass
on the residual sequence and `dz` is the curvature feature it was hoped would
gain — it loses.

That leaves a real choice, and it is not obvious:

- **Whiten everything.** One objective everywhere, internally consistent, and
  the intervals become honest. Costs 28% on the `dz` estimator and moves every
  published `dz` by up to 1.6 km.
- **Whiten only the uncertainty** — keep the OLS point estimate, which is the
  efficient one, and evaluate the deviance and the posterior with `R^-1`. Every
  published parameter is untouched and the intervals still widen. But the
  package then minimises one objective and reports intervals from another, and
  `profile`'s interval is no longer centred on `optimise`'s answer.

There is also an API question with no obvious answer: the whitening depends on
the **taper**, and `residuals(x, kh, Phi, sigma_Phi)` does not receive one. It
would have to become instance state (like `bounds` and `prior`), or ride on the
spectrum tuple — which would change `_SPECTRUM_RETURNS` and the `spectrum=`
contract that `~/Global_CPD`'s three-array zarr caches depend on.

## What is still needed

In rough order of how much each could change the answer:

1. **The likelihood, not the interval.** Both constructions under-cover by the
   GLS factor, and until `R^-1` reaches the objective (or the deviance is
   tempered by the ratio of GLS to naive curvature) every interval this package
   reports is a lower bound on the uncertainty. This is now the largest known
   error in the uncertainty machinery and it is not specific to the mesh.
2. **Decide what the short rungs should say.** The mesh calls `Δz` unbounded at
   143 of 144 vertices at 250 km where the archive reports a number. If that is
   right, a large part of the published short-window map is not a measurement;
   if it is too conservative, `_MESH_IDENTIFIABLE` is the knob. Nothing here
   settles it — coverage at 400 km cannot, because an unbounded interval covers
   the truth trivially and so flatters whichever method returns more of them.
3. **Coverage on the production regime.** The 200-realisation coverage above is
   on clean synthetics with a free `z_t`. Repeating it with the band cut, the
   `sigma` inflation and the `z_t` pin needs synthetics that carry WDMAM's
   unmodelled resolution rolloff, which `notes/bench/run_experiments.py` can
   already degrade for.
4. **A degenerate posterior still returns a lower endpoint.** When the upper is
   unbounded the lower is recomputed from the identifiable part, which is
   defensible and is what makes it comparable to the scan — but it is a
   quantile of a truncated posterior and should be labelled as such rather than
   read as a bound.
5. **Cost is parity, not a win.** 576 evaluations against `profile`'s 225, so
   about two profiles, returning the joint posterior of all four parameters
   instead of one interval on one target. That is a better product at the same
   price; it does not make the L8 run cheaper.
