# Bouligand (2009) re-derived, and what the derivation exposes

Re-derivation of the forward model `pycurious.grid.bouligand2009` from fractal
magnetisation, the master equation it is a special case of, and three
measurements on the WDMAM archive in `~/Global_CPD` that the derivation
prompted.

Scripts that produced every number here are listed at the bottom.

## 1. The derivation

### Setup

Magnetisation `M(r)` is a zero-mean random field, statistically homogeneous and
isotropic **in three dimensions**, with power spectral density

```
P_M(q) = C_M |q|^(-beta),      q = (kx, ky, kz)
```

confined to a layer `zt <= z <= zt + dz` (z positive down). The observation
plane is `z = 0`.

### Step 1 — field from magnetisation

For magnetisation varying only horizontally within a thin slab at depth `z`,
Blakely (1995, eq. 11.36) gives the total-field anomaly in the horizontal
Fourier domain as

```
dT(k) = 2 pi Cm Theta_f Theta_m |k| INT M(k, z) exp(-|k| z) dz
```

Two factors matter and one does not:

* `exp(-|k| z)` is upward continuation from each source depth to the
  observation plane. This is the whole depth sensitivity of the method.
* **`|k|`** converts the magnetic *potential* to the anomalous *field* — the
  anomaly is a gradient. Omit it and every `Phi` comes out `k^-2` too steep;
  that is exactly the discrepancy `derive_check.py` measures if the factor is
  dropped. It is also what makes the uniform-magnetisation limit reduce to
  Spector & Grant's `exp(-2k zt) (1 - exp(-k dz))^2` rather than that over
  `k^2`.
* `Theta_f Theta_m`, the direction cosines of field and magnetisation, depend
  only on the **azimuth** of `k`, not on `|k|`. A full-annulus radial average
  therefore turns them into a constant, absorbed by `C`. **This is why the
  method does not need reduction to the pole** — an azimuthally complete radial
  average has already done the equivalent. (It is not a licence to ignore
  inclination if the annuli are incomplete, or if magnetisation direction
  varies within the window.)

### Step 2 — vertical cross-spectrum

At fixed horizontal `k`, the covariance of the magnetisation between two depths
is the 1-D inverse transform of the 3-D PSD over `kz`:

```
S_M(k, zeta) = (1/2pi) INT (k^2 + kz^2)^(-beta/2) exp(i kz zeta) dkz
```

which is a Matern covariance in `zeta = z - z'` with range `1/k`. This is the
step that makes the model *fractal* rather than a stack of independent sheets:
magnetisation at different depths is correlated, with correlation length set by
the horizontal wavenumber being probed.

### Step 3 — the master equation

Squaring step 1 and substituting step 2, then writing `g(z)` for the
magnetisation-versus-depth profile of the layer (`g = 1` inside it for
Bouligand), the double depth integral collapses to a single one:

```
Phi(k) = (k^2 / 2pi) INT P_M(sqrt(k^2 + t^2)) |h_k(t)|^2 dt

h_k(t) = INT g(z) exp(-(k + i t) z) dz
```

**Everything about the layer enters through `|h_k(t)|^2`** — the power spectrum
of the depth profile weighted by the upward-continuation kernel `exp(-kz)`.

For the box-car `g = 1` on `[zt, zt+dz]`,

```
|h_k(t)|^2 = exp(-2 k zt) [1 - 2 exp(-k dz) cos(t dz) + exp(-2 k dz)] / (k^2 + t^2)
```

and both resulting `t`-integrals are the standard Bessel pair
`INT (k^2+t^2)^(-nu-1/2) exp(i t zeta) dt`, giving

```
Phi = (2/pi) C k^(1-beta) exp(-2 k zt - k dz)
      * sqrt(pi)/Gamma(1+beta/2)
      * [ cosh(k dz) Gamma((1+beta)/2)/2 - K_((1+beta)/2)(k dz) (k dz/2)^((1+beta)/2) ]
```

which is `bouligand2009` exactly, with a constant `2/pi` absorbed into `C`.

**Verified numerically.** `derive_check.py` evaluates the double integral by
quadrature, with no closed form anywhere in the chain, and compares to
`bouligand2009` over `k = 0.005 .. 0.8` rad/km for twelve `(beta, zt, dz)`
combinations. The two agree to a constant offset to **1e-11**, and the offset is
`ln(2/pi) = -0.451583` in every case — the same constant, confirming it is a
normalisation and not a parameter-dependent discrepancy.

### The asymptotes, which is where the physics is

```
k dz >> 1:   ln Phi = C' + (1 - beta) ln k - 2 k zt        fractal half-space
k dz << 1:   ln Phi = C'' + (3 - beta) ln k - 2 k z0       sheet at the CENTROID
```

with `z0 = zt + dz/2`. Three consequences, none visible in the closed form:

1. **The low-`k` exponential rate is set by the centroid, not by `zt`.** The
   stray `- k dz` sitting outside the bracket is precisely what turns
   `exp(-2k zt)` into `exp(-2k z0)`. Confirmed numerically: at `beta=2.5,
   zt=5, dz=12` the measured low-`k` rate is 23.0 against `2 z0 = 22` and
   `2 zt = 10`.

2. **`dz` is degenerate with `C` in the low-`k` limit.** It survives there only
   as `2 ln dz` inside `C''`. All information about the layer's *base* lives in
   the curvature around `k ~ 1/dz` and in the centroid rate — never in the
   amplitude.

3. **The slope changes by exactly 2 across the rolloff, independently of
   `beta`, `zt` and `dz`.** This is the model's one hard, falsifiable
   prediction, and section 3 tests it.

### The assumptions, listed

In roughly decreasing order of how much they are likely to cost:

| # | assumption | relaxable? |
|---|---|---|
| A1 | `zt` is known, or is determined by the fit | **it is neither** — section 2 |
| A2 | box-car `g(z)`: magnetisation constant to `zt+dz`, zero below | yes, via the master equation |
| A3 | one `beta`, one `zt`, one `dz` for the whole window | only by a mixture model |
| A4 | statistical isotropy in 3D, single `beta` for horizontal and vertical | not from radial spectra alone |
| A5 | the base is a sharp horizontal interface | no — the Curie isotherm undulates |
| A6 | infinite horizontal extent (no window, no taper, no detrend) | section 3 measures the cost |
| A7 | `Theta` constant over each annulus | fine for complete annuli |

## 2. The model is not the problem

The derivation hands over one falsifiable structural prediction: **the log-log
slope changes by exactly 2 across the rolloff, whatever `beta`, `zt` and `dz`
are.** `free_gap` in `spectral.py` releases that 2 and fits it.

Measured on the WDMAM archive against synthetics that obey the model exactly,
so the window's own bias is controlled rather than assumed away:

| window | real archive | synthetic control (truth = 2.00) |
|---|---|---|
| 10,000 km | 2.04, IQR [1.89, 2.24] | — |
| 4,000 km | 2.24, IQR [1.99, 2.66] | 2.26, IQR [1.93, 2.60] |
| 2,000 km | — | 1.86, IQR [1.51, 2.52] |
| 1,000 km | 2.73, IQR [1.96, 3.40] | 2.36, IQR [1.93, 2.99] |

At 10,000 km the real data give 2.04 with a tight spread: the fractal-layer
model is **confirmed**, not merely unfalsified. At 4,000 km real and synthetic
are indistinguishable. The apparent drift to 2.7 at 1,000 km is mostly the
window itself — taper, detrend and too few bins below the rolloff — and the
residual real-minus-synthetic difference is modest.

**So do not reformulate the forward model.** Two further checks agree: fitting
the gap free *degrades* known-truth recovery (median |CPD error| 7.29 km against
4.27 km at a 4,000 km window), and a mixture model over a within-window spread
of base depths recovers what `Global_CPD`'s README already reports — the
spectrum of a mixture still looks like a single layer.

## 3. What the derivation does expose: `zt` is carrying the map

`Global_CPD` pins `zt` at a global **1.0 km** with sigma 0.05, i.e. fixed. Two
independent facts make that the dominant error:

**(a) The data cannot determine it.** Sweeping the pin and refitting everything
else (`zt_scan.py`, spectral residuals only, priors excluded so a prior cannot
disguise a flat likelihood):

| zt pin | chi2/n vs pin=1 | median CPD, 10,000 km |
|---|---|---|
| 0.5 | 1.016 | 39.2 |
| 1.0 | 1.000 | 37.6 |
| 3.0 | **0.983** | 26.4 |
| 5.0 | 0.999 | 18.9 |
| 8.0 | 1.074 | 12.9 |

The misfit is flat to 3% from 0.5 to 5 km while median CPD falls by **a factor
of two**. The best-fitting pin per vertex is 3–5 km, not 1. `d(CPD)/d(zt)` is
−1.7 at 1,000 km and −5.8 at 10,000 km — the derivation explains why it exceeds
the naive −1: the low-`k` band constrains the centroid `z0`, so `CPD = 2 z0 −
zt` to first order, and the residual `beta`–`dz` valley amplifies the rest.

**(b) It is known independently.** `zt` is the depth from the observation plane
to magnetic basement — water column plus sediment, minus elevation. Water is
non-magnetic, so the water column is *literally* the `exp(-2 k zt)`
continuation the model already contains. From CRUST1.0, taper-averaged over
each window (`zt_field.py`):

| | median `zt` | the workflow assumes |
|---|---|---|
| ocean windows | **4.40 km** | 1.00 km |
| continental windows | 0.71 km | 1.00 km |
| East Pacific Rise / Mid-Atlantic Ridge | 3.5 km | 1.00 km |
| NW Pacific abyssal | 5.8 km | 1.00 km |

## 4. What it costs, on known truth

Synthetics with `zt = 4.5 km` (the oceanic median), fitted through the
workflow's own settings so the *only* difference is the pin (`synth.py`):

| | median &#124;CPD error&#124; |
|---|---|
| `zt` pinned at 1.0 km — the workflow | **5.96 km** |
| `zt` pinned at truth | **3.29 km** |

A 45% reduction, and the error is a *bias*, not scatter: pinning `zt` too
shallow overestimates CPD every time.

CRUST1.0 is not truth either, so `synth_robust.py` sweeps the error in the pin:

| zt pinned at | error in zt | median &#124;CPD error&#124; | median bias |
|---|---|---|---|
| 1.0 | −3.5 | 5.96 | +5.86 &nbsp;&nbsp;*(the workflow)* |
| 2.5 | −2.0 | 4.25 | +3.46 |
| 3.5 | −1.0 | 3.60 | +1.67 |
| 4.5 | 0.0 | **3.29** | −0.18 |
| 5.5 | +1.0 | 3.41 | −1.98 |
| 6.5 | +2.0 | 4.38 | −3.76 |

**A `zt` good to about ±2 km beats the global constant.** Over the ocean —
where water depth is known to metres and dominates — CRUST1.0 is comfortably
inside that.

## 5. What it does to the global map

Full-mesh refit of every cached WDMAM spectrum (`run_zt.py`), scored with the
workflow's own battery (`score.py`). Baseline reproduces the published archive
exactly, including `rho(CPD, age) = +0.432` at 1,000 km.

| window | | `rho(CPD,age)` ocean | `rho(CPD,zt)` | median CPD | craton:ridge |
|---|---|---|---|---|---|
| 10,000 | published | +0.292 | — | 43.2 | 2.69 |
| 10,000 | zt from data | +0.243 | — | 27.6 | **4.64** |
| 4,000 | published | +0.377 | +0.228 | 23.5 | 4.12 |
| 4,000 | zt from data | +0.289 | **+0.084** | 17.5 | 4.82 |
| 1,000 | published | **+0.432** | +0.294 | 18.4 | 5.27 |
| 1,000 | zt from data | +0.390 | **+0.203** | 13.4 | 5.12 |

The headline age correlation **falls**. That looked at first like a failure,
and it is not — it is the artefact being removed:

| window | | `rho(CPD,age)` | partial `rho(CPD,age &#124; zt)` |
|---|---|---|---|
| 4,000 | published | +0.377 | +0.213 |
| 4,000 | zt from data | +0.289 | **+0.219** |
| 2,000 | published | +0.387 | +0.131 |
| 2,000 | zt from data | +0.295 | **+0.139** |
| 1,000 | published | +0.432 | +0.202 |
| 1,000 | zt from data | +0.390 | **+0.215** |

Seafloor age and water depth correlate at `rho = +0.72` (plate-cooling
subsidence). **Roughly half of the published +0.43 is water depth read as Curie
depth.** Once water depth is controlled for, the genuine age signal is +0.20,
and it is *unchanged to marginally stronger* after the correction, at every
window — while the spurious pathway `rho(CPD, zt)` drops by a third to two
thirds. The correction removes the artefact and leaves the signal.

Also: the craton:ridge contrast becomes much more stable across the window
ladder (4.6–5.1 against the published 2.7–5.3), which is what an estimator
measuring one physical quantity should do.

## 6. What did not work, and what remains broken

* **Releasing the slope gap** — worse on known truth. It is a diagnostic, not a
  parameter.
* **A mixture over within-window base depths** — the spectrum of a mixture
  still looks like a single layer, as `Global_CPD`'s README already found.
* **Surface heat flow** — `rho(CPD, q)` within regime is ~0 and slightly
  *positive* (wrong sign) in the published map, and stays ~0 under every
  variant tried here (ocean +0.01, continental +0.05 at 1,000 km). No change to
  the spectral model moved it. Either the comparison is underpowered — point
  heat flow at 223 km vertex spacing against a CPD smoothed over 1,000 km — or
  CPD at these window sizes is not measuring temperature. This is not resolved.
* **Window dependence** remains large: median CPD still falls by ~2x from the
  10,000 km to the 1,000 km window (2.06x after correction, 2.28x before). The
  correction helps slightly; it does not fix it.

## 7. Caveats on section 5

* `beta` keeps its published pass-A prior in every variant rather than being
  re-smoothed under the new `zt`. Justified by measurement — `beta` moves by
  <0.13 over a 1→5 km pin sweep, inside its own 0.15 prior width — but a full
  two-pass rerun is the correct version of this experiment.
* WDMAM's observation altitude is taken as sea level, inherited from
  `Global_CPD`. If the product is in fact at altitude, every `zt` here shifts
  by a constant. That changes absolute depths but not the ocean-continent
  contrast, which is what drives the result.
* CRUST1.0 sediment thickness is far less certain on continents than water
  depth is at sea, so the correction is on much firmer ground over the ocean —
  which is also where it is largest.
* The `zt` window average is taken in `zt`, not in `exp(-2 k zt)`. Jensen makes
  the true attenuation larger, so this understates the correction slightly.

## Scripts

In `notes/bench/`, alongside the existing benchmark scripts (that directory is
untracked). They read `~/Global_CPD` directly and are run from their own
directory:

| script | what it does |
|---|---|
| `derive_check.py` | double integral by quadrature vs `bouligand2009` |
| `spectral.py` | master equation, asymptotes, mixture and free-gap models |
| `harness.py` | loads the cached `Global_CPD` spectra and archive |
| `fitters.py` | standalone refit, checked against the archive to 0.003 km |
| `diagnose.py` | the three questions of sections 2-3 |
| `zt_scan.py` | misfit as the `zt` pin is swept |
| `zt_field.py` | `zt` from CRUST1.0, taper-averaged per window |
| `score.py` | the workflow's own test battery |
| `run_zt.py` | full-mesh refit under each `zt` treatment |
| `synth.py` | known-truth recovery |
