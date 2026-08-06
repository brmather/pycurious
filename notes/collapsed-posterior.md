# The Bouligand posterior is two-dimensional, and small enough to integrate

`posterior()` and the `Posterior` it returns are **in the library**, alongside
`profile`, which keeps the deviance scan. `notes/bench/collapsed_mcmc.py` is the
prototype that established the identity and measured what a chain is worth
against it. Harnesses: `notes/bench/score_intervals.py` (coverage) and
`notes/bench/score_mesh_L2.py` (WDMAM L2, all 40 rungs).

`metropolis_hastings` walks four parameters. Two of them need not be walked, and
once they are gone the remaining two need not be walked either.

In one paragraph: `C` and `z_t` enter the forward model as linear coefficients
on basis vectors that do not depend on the other two parameters, so their
conditional posterior is an exact Gaussian and integrates out in closed form,
leaving a marginal posterior over `(beta, dz)` that is *exactly* the reduced
misfit `_solve_linear` already computes. Sampling that 2-D marginal instead of
the 4-D posterior is worth about **1.9x** in effective sample size. Evaluating it
on a mesh instead of sampling it at all is worth about **20x** against the chain,
and lands at a few times one call to `profile` — while returning the entire
joint posterior rather than one interval on one target.

**Two things changed after this note was first written, and both are worth
reading before the middle sections.** The under-coverage both interval
constructions showed is fixed, by correcting the likelihood where it is read
rather than by whitening the objective; and the mesh box is now measured from
the data rather than taken from the curvature at the mode, which is what made
the convergence numbers here honest. Sections are in the order the work
happened, so the early ones describe the prototype and the late ones describe
what shipped.

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

Convergence, measured **as shipped** — a coarse 6x20 sweep sizing the box, then
one fine mesh — against a 192-node reference at a 1000 km window:

| nodes | evaluations | dz mean error | dz sd ratio | edge mass |
|---|---|---|---|---|
| 16 | 120 + 256 | 0.15 km | 1.16 | 1.4e-5 |
| 24 | 120 + 576 | 0.013 km | 1.07 | 6.7e-6 |
| **32** | **120 + 1024** | **0.007 km** | **1.012** | 4.9e-6 |
| 64 | 120 + 4096 | 0.001 km | 1.003 | 2.4e-6 |

The default is 32. Note these are **not** the numbers an earlier version of this
note carried — it claimed exactness to 3e-9 of a standard deviation at 24 nodes,
on a box taken from the curvature at the mode. That box was both too small at
short windows (a fitted `dz` of 54 km against a `1/k_min` of 26.6) and 38 to 240
standard deviations across at long ones, and the apparent exactness was a
reference computed the same wrong way. Measuring the box from a coarse sweep
costs 120 evaluations and makes the convergence honest.

Against the 4-D chain's 24,000 evaluations for an ESS of 1,000–1,200, that is
still **~20x fewer forward-model evaluations for an exact answer rather than a
noisy one**: no autocorrelation, no burn-in, no seed, no acceptance rate to
tune, and no possibility of a chain that has not mixed reporting a confident
wrong interval.

Mass falling outside the mesh is the number to watch. A quadrature that has
clipped `dz`'s long upper tail is wrong in a way a chain is not — silently, and
in the direction of a *tighter* interval — so `edge_mass` is reported rather
than assumed, and `Posterior.interval` compares it against the mass the endpoint
in question is placed to cut off.

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


## Built: `posterior()` and the `Posterior` it returns

`posterior()` returns a `Posterior`; the readings live on it —
`interval(target, level)`, `marginal(target)`, `moments(target)`, for any of the
four parameters or the Curie depth. `profile` keeps the deviance scan and loses
its `method=` keyword.

That split was not the original plan. The mesh first arrived as
`profile(method="mesh")`, which returned a differently shaped tuple, redefined
`deviance` as `-2 log(p/p_max)` of a marginal rather than `2(F - F_min)` of a
profile, silently ignored `npoints`, and reinterpreted `bracket` from a range in
the target into a box in two other parameters. Four incoherences for one
keyword, and they were the symptom of two genuinely different objects sharing an
entry point.

Retiring `profile` altogether was considered and rejected on measurement. Two
things a credible interval cannot supply:

- **An interval invariant to where the box sits.** Every target except `dz` is a
  marginal *over* `dz`. Measured at a 200 km window across three defensible
  boxes, the `beta` credible interval moved by 0.14 — 44% of its own width —
  while the profile interval was identical for two of the three and unchanged to
  four figures. Where `dz` is unbounded, the mesh has to say so about every other
  target too, which it now does via `Interval.conditional`.
- **A cheap single-target path.** `~/Global_CPD` runs `profile("dz")` 26 million
  times at L8, at 225 evaluations against the mesh's ~1150.

So both ship. They are not rivals: minimising the reduced surface along a mesh
axis **is** `_profiled_misfit`, verified to 3e-4 in misfit — 6e-4 in deviance
against a 3.84 threshold — because `_solve_linear` has already profiled `C` and
`z_t` out exactly. One surface, two readings, and a test that says so.

Five things bit during implementation, all now guarded:

1. **The mesh is a quadrature rule, not a discrete distribution.** `z_t`'s
   conditional width is 0.007 km against a 2 km node spacing, so summing one
   narrow Gaussian per node gives a picket fence of spikes. Interpolate first.
2. **`CPD = z_t + dz` is a convolution along the mesh, not a marginal of it.**
   `E[CPD] = E[z_t] + E[dz]` holds whatever the correlation and is the identity
   that caught both wrong versions — the atoms version by 0.5 km, a `mass` array
   shadowed by a loop variable by 0.53.
3. **A CDF with a flat run cannot be inverted by `numpy.interp`.** It needs
   increasing `xp` and resolves ties by position, which read a `dz` of −0.71 km
   off a posterior with no support below zero. Invert only where it rises.
4. **A marginal's grid must not be sized by `mean ± 6·spread`.** On a window
   that bounds nothing, the spread is most of the mesh and that reached −537 km.
   The grid is the range the target takes, widened only by the conditional
   Gaussian's reach — which for `beta` and `dz` is exactly zero.
5. **An interval running past `1/k_min` is not a measurement.** Beyond it the
   rolloff sits below the longest wavelength measured and `dz` is degenerate
   with `C`, so the endpoint is set by where `bouligand2009` overflows: `(190,
   943)` km at a 250 km window. And when the identifiable part is a sliver its
   low quantile lands on the floor of the box — 0.05 km against a fitted 368 —
   which `~/Global_CPD` would archive as a real bound. That returns `nan`.

A highest-density interval was tried first and abandoned: read off a derivative
of an interpolated cumulative, its endpoints moved 1.5 km between a 192-node and
a 768-node refinement and reported spurious multimodality below that. Inverting
the CDF moves 0.05 km. A mode-counting warning went the same way — on a flat
posterior it reported 14 modes.

## The box is measured, and three ways of measuring it were wrong

The first box came from the curvature at the mode: the Gaussian approximation
`profile` exists because it distrusts. It was wrong in both directions at once —
at a 200 km window the fitted `dz` was 54 km against a `1/k_min` of 26.6, and at
1000 km and beyond a range reaching `1/k_min` was 38 to 240 standard deviations
across, where the moments need a couple of dozen nodes over about six.

It also had a dependence nobody had noticed until the likelihood correction
below went in: **correcting the likelihood widens the posterior by ~1.27**, so a
box scaled from the *uncorrected* curvature stopped containing it and the mesh
quietly stopped converging. That is how the box got looked at at all.

What ships is a coarse 6x20 logarithmic sweep across everything `dz` may be —
log because that range spans four orders of magnitude, and to the *bound* rather
than to `1/k_min`, so the mode is inside the box even where the data cannot pin
it — followed by one fine uniform mesh sized from that coarse density's spread.

Three attempts at "sized from the spread" failed first, and each failure is a
different lesson:

- **A weighted standard deviation over the raw coarse density: 118 sigma.** A
  log axis is not a quadrature rule until each node is weighted by how much of
  the axis it stands for. Without that, the sparse high-`dz` nodes count as
  heavily as the dense low ones and the mass lands where the *spacing* is.
- **A quantile range: still too wide, and unfixable at this resolution.** Twelve
  nodes over four orders of magnitude cannot place a 1e-3 quantile; the tail it
  was meant to exclude carried more than the threshold.
- **The threshold crossing taken directly as the box: too wide again.** On a log
  axis a slowly decaying upper tail crosses a thousandth of the peak a long way
  out, and padding it in value space pushed the *low* end through zero onto its
  floor, which then set the width.

What works is to use the threshold to **select** which nodes are the posterior
and then measure the spread over those alone. `_MESH_BOX_SPAN` is 5, measured
against a 192-node reference:

| span | box width | dz mean error | dz sd ratio |
|---|---|---|---|
| 3 | 5.4 sigma | −0.32 km | 0.839 |
| 4 | 11.5 sigma | −0.05 km | 0.972 |
| **5** | **18.6 sigma** | **−0.01 km** | **1.012** |
| 6 | 25.3 sigma | +0.00 km | 1.036 |

Below 4 the box clips the upper tail and the spread comes back a sixth too
small; above 5 the same 32 nodes resolve a wider box less well. There is no
plateau of safe values, which is why this is measured rather than chosen.

**The expansion loop is gone.** A window that cannot resolve the thickness
leaves ~1e-5 of the mass out at hundreds of km — the degenerate plateau — and a
loop widening any edge above a fixed tolerance walked the box out to the
parameter bound chasing it, taking the resolution of everything else with it.
That plateau does not decay, so no box contains it. `edge_mass` reports it and
`interval` compares it against the mass that endpoint is placed to cut off,
which is the question that actually matters and is a hundred times looser than
any accounting tolerance.

## Fixing the calibration: temper the likelihood, do not whiten the objective

Both interval constructions under-covered, by the factor `_gls_covariance`
applies and `min_func` does not: nominal 68.27% on `beta` covering **0.52
(scan)** and **0.58 (mesh)** where `optimise`'s corrected sigma covers **0.65**,
which an inflation of 1.298 predicts exactly.

**Whitening the objective was measured and rejected.** It works — `beta` goes to
0.640/0.945 — but it costs the `dz` estimator 28% of its scatter, needs `R`
tabulated per taper because `_banded_correlation` reads model mismatch as
correlation (`rho_1` from 0.24 to 0.42 depending where it is asked), forces
`_covariance` to drop `_gls_covariance` or correct twice, and needs
`_ANALYTIC_COLUMNS` whitened too — unwhitened, that sent a `dz = 10` layer at a
4000 km window to **60.9**. Four touch points and a 28% regression to fix an
interval.

What ships instead corrects the likelihood where it is *read*.
`_correlation_inflation` measures how far the GLS covariance exceeds what the
likelihood implies; `_temperature` divides the spectral misfit by it in
`profile`, in `posterior` and in `metropolis_hastings`. **Nothing that is
minimised changes, so no fitted value moves anywhere.**

Two properties make one scalar defensible, and both are measured rather than
assumed.

**The four per-parameter inflations agree.** Under `numpy.hanning` at 1000 km
they are 1.265 / 1.267 / 1.267 / 1.265 for `beta`, `z_t`, `dz`, `C` — 0.2%
apart. So the correction is a loss of effective degrees of freedom, not a
reshaping of the covariance, and a scalar is what it *is* rather than a
convenience. `_TEMPER_SPREAD_LIMIT` warns when they stop agreeing; it fires at
12% on the legacy 305 km fixture, which is a window too narrow for the model
fitted to it.

**Only the spectral block enters, on both sides.** This is the correction that
took two goes. `residuals` appends one row per prior, so dividing `F` whole
widens every prior by `sqrt(t2)` — a 30% loosening of the 0.05 km `z_t` pin that
carries the whole depth scale, applied silently in the name of calibration, and
exactly what `_gls_covariance` refuses when it keeps prior rows out of its
solve. Estimating `t2` from the whole Jacobian has the matching problem: the
per-parameter agreement above holds for a *free* fit and degrades to 6–18% under
production priors, because a prior-dominated direction has no correlated
information in it to inflate. Taking `t2` from the spectral block alone and
applying it to the spectral rows alone fixes both:

| `z_t` prior | spectral only | whole misfit | `sqrt(t2)` |
|---|---|---|---|
| 0.01 (hard pin) | **1.049** | 1.310 | 1.302 |
| 0.05 (production) | **1.210** | 1.299 | 1.302 |
| 0.20 | 1.302 | 1.309 | 1.302 |
| 1.00 (weak) | 1.309 | 1.309 | 1.302 |

Both limits are the point. Under a hard pin the prior sets the width and the
correction nearly vanishes; under a weak one there is nothing holding it and the
two agree, as they must. And the estimate itself becomes prior-independent —
0.6% to 3.9% between a free fit and a production-pinned one, against 14% when
taken over the whole Jacobian.

**What it does.** On windows that bound `dz`, the interval widens by `sqrt(t2)`
and nothing else moves: 1.285 measured against a predicted 1.269 at 1000 km,
1.305 against 1.298 at 2000 km, the small excess being the upper tail a wider
level set reaches further into. At 200 km, where a 30 km layer is not resolvable
at all, the corrected deviance stops crossing its threshold and reports
unbounded. That is the correction working rather than overreaching: it is
largest where the model cannot follow the data, because `_banded_correlation`
reads smooth model mismatch as correlation and a model that cannot follow the
data genuinely leaves its parameters less determined.

**One review finding did not reproduce, and it matters which way.** The estimator
was said to be biased *above* 1 on uncorrelated residuals (mean 1.025–1.038, max
1.35), which would make the untapered reading of 1.03 indistinguishable from
noise. Measured with the geometric mean of the diagonal ratios, 400 replicates
of residuals projected off a four-column Jacobian:

| bins | 20 | 49 | 120 | 249 |
|---|---|---|---|---|
| `t` | 0.974 ± 0.043 | 0.982 ± 0.028 | 0.992 ± 0.014 | **0.997 ± 0.006** |

Biased slightly *low*, and about five times tighter than claimed. So the
untapered 1.03 sits ~5 sigma clear of the floor and is signal. What survives of
the concern is the ±4% per-vertex scatter at 20 bins, which is why nothing
asserts `t2 == 1` — a test that cannot pass.

**`calibrate=False`** reproduces a pre-v2 interval exactly, on all three
routines. That is the audit hook, and it is what the four pinned `dz` intervals
in the test suite are now read through: they are an archive of an uncorrected
likelihood, and reading them through the new default would compare two different
quantities.

**The temper is blind to one thing.** It is a ratio of two covariances, so it is
invariant to the absolute scale of `sigma_Phi`. That makes it robust to a
mis-set `dof_factor` and equally unable to detect one: an uncalibrated taper
still gets `_TAPER_DOF`'s untapered fallback in the *within*-bin term, and
`np.blackman` returning 1.29 is not evidence that such a taper is fully handled.
