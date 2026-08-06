# PyCurious

Estimates Curie point depth — the depth at which rock loses its magnetisation —
from the radially averaged spectrum of a magnetic anomaly. Two methods, sharing
one grid/spectrum layer:

- **Bouligand *et al.* (2009)** — fits a four-parameter analytic spectrum
  (`beta, zt, dz, C`) by optimisation.
- **Tanaka *et al.* (1999)** — the centroid method: two straight lines fitted to
  separate wavenumber bands.

Both return **uncertainties**, which is the defining change in v2. Anything that
returns a bare number without one is a pre-v2 remnant.

## Commands

```bash
pytest                     # 142 tests, ~27 s
pytest -m "not slow"       # 139 tests, ~20 s -- skips the calibration tests that
                           # fit a few hundred realisations
pytest tests/test_tanaka.py -q
```

Notebooks need extras that are not runtime dependencies:
`pip install -e ".[examples,mapping,geotiff,download]"`. GDAL is deliberately
its own extra — its bindings need a matching system libgdal, and folding it into
`mapping` would break `pip install` for most users. `conda install gdal` is
usually easier than pip.

To execute notebooks headlessly:
`MPLBACKEND=Agg jupyter nbconvert --to notebook --execute --inplace <nb>`

CI runs on **GitHub Actions** (`.github/workflows/`): `tests.yml` runs the suite
on every push and PR across Python 3.9-3.13; `docs.yml` builds the Sphinx docs
(and deploys to GitHub Pages from `master`); `publish.yml` uploads to PyPI via
OIDC trusted publishing when a GitHub release is published. Travis is gone.

## Layout

```
pycurious/
  grid.py               CurieGrid + the shared spectrum and covariance machinery
  optimise_bouligand.py CurieOptimiseBouligand(CurieGrid)
  optimise_tanaka.py    CurieOptimiseTanaka(CurieGrid)
  parallel.py           CurieParallel mixin -- CurieGrid inherits it
  synthetic.py          fractal_anomaly(): synthetics with a known answer
  mapping.py            projections, netCDF, GeoTIFF (all lazily imported)
  download.py           cached downloads with md5 checks
```

`CurieGrid(CurieParallel)`, and both optimisers extend `CurieGrid`, so every
class has `subgrid`, `radial_spectrum`, `window_spectrum` and
`parallelise_routine`.

## Conventions that are easy to get wrong

These caused real bugs. Check them before changing anything spectral.

**Wavenumbers are rad/km, everywhere.** `radial_spectrum` returns `k` in rad/km,
and Tanaka's fitting bands are given in the same units. They used to be
cycles/km; a guard warns when a band looks like a leftover cycles/km value,
because such a call still runs and returns a plausible number.

**`dk = 2*pi/(N*dx)`**, the DFT fundamental — not `(N-1)`. Using `(N-1)`
understates every depth by `(N-1)/N`.

**`power` selects which spectrum you get.** `radial_spectrum` raises `|FFT|` to
`power` before averaging. Since `Phi = |FFT|**2`:

| method | `power` | quantity |
|---|---|---|
| Bouligand | `2.0` (default) | `ln Phi`, log power |
| Tanaka | `1.0` | `ln Phi**0.5`, log amplitude |

`power=0.5` appears in pre-v2 code and docs and is simply wrong; it halves every
depth.

**Depths are positive downwards.** `optimise` returns depths, not the negative
gradients the fits produce.

**Every Bouligand fit goes through `_fit`, which is `least_squares`, not
`minimize`.** Do not put L-BFGS-B back: it needs **2012** evaluations of the
forward model per vertex where trust-region reflective needs 359, about 3x the
CPU. `_fit` also supplies the two exact Jacobian columns (`dr/dzt = -2k/sigma`,
`dr/dC = 1/sigma`); `beta` enters through the *order* of a Bessel function and
`dz` costs the same analytically as by difference, so those two stay numerical.

**The starting point is derived, not declared.** `beta`, `zt`, `dz` and `C`
default to `None` in all five routines, meaning *read this off the spectrum*; a
value still means *use it*, and the four are independent, so `dz=30` starts the
other three from the best they can be at that thickness. `_initial_guess` scans
an eight-node `dz` ladder on the objective with `C` and `zt` profiled out
exactly — they are linear in `Phi = C·1 + zt·(-2k) + h(beta, dz)`, so
`_solve_linear` gets them in one step. That reduced surface *is* the full
objective profiled over the linear pair, which makes the scan a strictly better
basin detector than multi-starting the fit over the same nodes.

It fixes a real defect: the old `dz = 10` sat in the wrong basin on 7 of 20
thick-layer synthetics at a 4000 km window, at a misfit 25% higher, taking
`dz = 45` recovery from 32.6 ± 15.3 to 46.1 ± 5.9. Two things about it are
easy to get wrong and are measured in `notes/derived-starting-values.md`:

- **The asymptote band split is logarithmic.** `beta` comes from a regression
  of `ln Phi` on `[1, ln k, k]` over the high-`k` band, and those two regressors
  are nearly collinear unless the band spans several octaves. A linear split
  gave `beta = 3.25 ± 1.13` at 200 km and moved a pinned `profile` interval.
- **A `beta` prior outranks the estimate.** Where the high-`k` band carries
  something the model does not contain the regression reads it as `beta` — on
  WDMAM it returns 1.8 against a fitted 3.3, because of the unmodelled ~4 km
  resolution rolloff. Weighting by `1/sigma` does not help. The prior is how a
  dataset says so, and starting there cuts the fit cost by a third.

`last_x0` carries the derived start on the instance, alongside `last_spectrum`.
A fit that returns exactly its start never moved — `~/Global_CPD` raises
`STARTING_GUESS` on it — and with a derived start that is no longer something a
caller can reconstruct. A spectrum with fewer than two usable bins falls back to
the old constants, deliberately, so that signal stays recognisable.

**Pin the BLAS before timing anything.**

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python bench.py
```

L-BFGS-B calls a threaded BLAS whose workers spin-wait. Unpinned on a loaded
machine it looks 20x to 400x slower than TRF depending on contention, and
`time.process_time` makes it worse by charging every spinning thread — a
"3.1 s CPU" fit whose wall clock was 1.7 s. Both numbers were wrong; the truth
is 3x. Counting `residuals` calls is the measurement that does not lie.

**Tanaka bands have no defaults, deliberately.** Each straight-line limit holds
only over part of the spectrum: the `zt` band needs wavelengths shorter than
~4x the source thickness, the `z0` band needs `|k|d << 1`. Violating either
still yields a confident-looking fit through the wrong part of the curve. Use
`check_bands(k, zt_range, z0_range, thickness=...)`, which reports point counts,
wavelengths, `|k|d` and the bias in km.

Beware apparent accuracy on the legacy 305 km fixture: its `|k|d` violation
biases `z0` low and unmodelled fractal magnetisation biases both depths high,
and the two partly cancel. That cancellation is a property of that particular
synthetic, not evidence the method worked.

## Uncertainties

The interesting machinery, and where most of the subtlety lives.

`radial_spectrum` returns `sigma_Phi`, the **scatter of FFT cells within an
annulus**. That is not what a fit needs. `window_spectrum` converts it to the
**uncertainty of the annulus mean** and is what both optimisers call (through
their private `_spectrum`). Prefer it over `radial_spectrum` for anything fitted.

Computing that spectrum is most of what a fit costs — 96% of a Tanaka
`optimise` at a 1025-cell window — so every fitting routine takes `spectrum=`
and sets `last_spectrum`, and one spectrum serves a whole sweep at a centroid.
`CurieGrid._resolve_spectrum` is the single seam: it calls the subclass's
`_spectrum` or takes the caller's, never both. A subclass supplying `_spectrum`
also supplies `_SPECTRUM_ARGS`, `_SPECTRUM_PROVENANCE` and `_SPECTRUM_RETURNS`,
which is what lets one implementation serve Bouligand's 3-array return and
Tanaka's 4-array one. Two rules that look arbitrary and are not: **callables
never go in the provenance** (holding a `process_subgrid` on the instance makes
the bound routine unpicklable, which silently drops `parallelise_routine` to
serial), and **anything that changes the spectrum's values does** — including
Tanaka's `beta`, which subtracts the fractal contribution. `parallelise_routine`
rejects `spectrum` outright; one spectrum cannot describe a list of centroids.

Two corrections, both measured by Monte Carlo rather than assumed:

1. **Within a bin**, cells are not independent — a real field is Hermitian, so
   about half repeat (exactly a factor of 2 untapered), and a taper correlates
   neighbours further. `_dof_factor` / `_TAPER_DOF` encode this as
   `dof_inf * N / max(N - lost, 1)`; the `lost` term dominates in the sparse
   inner bins, which is exactly where a centroid depth is determined.
2. **Between bins**, neighbouring residuals correlate because a taper spreads
   each wavenumber over several bins. `_banded_correlation` estimates this from
   the residuals and `_gls_covariance` solves
   `(J^T R^-1 J)^-1`. Ignoring it understates every reported sigma by ~30%.

Both live in `grid.py` and are shared. Tanaka builds `J` analytically (a line);
Bouligand uses finite differences.

Beyond the covariance: `CurieOptimiseBouligand.profile()` gives profile-deviance
intervals, which matter because `dz` and `CPD` are genuinely asymmetric;
`posterior()` evaluates the posterior itself; `metropolis_hastings()` samples
it; `sensitivity()` resamples the spectrum. `CurieOptimiseTanaka.sensitivity()`
additionally jitters the band edges, since band placement usually dominates
spectral scatter. There is deliberately no `profile` on the Tanaka side — each
band is a straight-line fit, where profile and covariance intervals are provably
identical.

**The posterior is two-dimensional.** `Phi = C·1 + zt·(-2k) + h(beta, dz)`, so
`C` and `zt` are linear coefficients on basis vectors that do not involve the
other two. Their conditional posterior is an exact Gaussian, and its precision
`A = GᵀG` is built from `k`, `sigma` and the prior widths **only** — nothing in
it depends on `(beta, dz)`. So `½ log det A` is an additive constant, and the
marginal posterior of `(beta, dz)` *is* the reduced misfit `_solve_linear`
already returns. `posterior()` evaluates it on a mesh; two dimensions is small
enough to integrate rather than sample.

**`profile` and `posterior` read one surface two ways.** Minimising the reduced
misfit along a mesh axis *is* `_profiled_misfit` — verified to 3e-4, because
`_solve_linear` has already profiled `C` and `zt` out exactly — and integrating
along it is the marginal. So they are not rival constructions to be kept in step
by hand. They are different objects, though: a likelihood level set centred on
the mode against an equal-tailed credible interval centred on the median, which
agree on `beta`, `zt` and `C` and differ on the skewed `dz`. Keep both; `profile`
is the cheap single-target path (225 evaluations against ~1150), `posterior`
answers every target from one density.

`posterior()` returns a `Posterior`, and the readers live on it —
`p.interval(target, level)`, `p.marginal`, `p.moments`. They take no window, so
a density cannot be asked for an interval at a window it was not computed at;
the defect `_Spectrum.provenance` guards one layer down cannot be expressed
here. `parallelise_routine` rejects `posterior`, because `_collect` stacks by
`np.ndim` and a density is not an array — that is the obstacle, not the idea.

Four things about the mesh are easy to get wrong and are all guarded by tests:

- **The mesh is a quadrature rule, not a discrete distribution.** `zt`'s
  conditional width is 0.007 km against a 2 km node spacing, so summing one
  narrow Gaussian per node gives a picket fence. Interpolate first
  (`_MARGINAL_REFINE`); it costs no forward-model evaluations.
- **`CPD = zt + dz` is a convolution along the mesh, not a marginal of it.**
  Propagating it from `dz` drops `zt`'s conditional spread. `E[CPD] = E[zt] +
  E[dz]` holds whatever the correlation, and is the identity that catches both
  mistakes — the atoms version failed it by 0.5 km.
- **Every target except `dz` is an integral *over* `dz`.** So where the window
  cannot bound the thickness, an interval on `beta` is a statement about where
  the box was put — 44% of its own width across three defensible boxes. Those
  are integrated over the resolvable range only and come back with
  `.conditional` set. `moments` returns `nan` there rather than the box's number.
- **An interval running past `1/k_min` is not a measurement.** Beyond that the
  rolloff is below the longest wavelength the window measured and `dz` is
  degenerate with `C`; the endpoint is then set by where `bouligand2009`
  overflows. At a 250 km window that produced `(190, 943)` km before the guard.
  It returns `inf`, as the scan does by never crossing its threshold — and `nan`
  rather than a lower bound when the identifiable part is a sliver, because that
  quantile lands on the floor of the box (0.05 km against a fitted 368).

**The box is measured, not assumed.** A coarse log sweep across everything `dz`
may be locates the mass; the fine uniform mesh is sized from that density's own
spread (`_MESH_BOX_SPAN = 5`, measured — at 3 it clips the tail and the spread
comes back a sixth small, at 6 the same nodes resolve a wider box less well).
Three traps, all of which produced a wrong box first: a log axis needs its node
spacing as a quadrature weight (without it the spread read 118 sigma on a
posterior whose own is four); select nodes by density *then* measure, since one
spread over everything is ruined by the degenerate plateau; and there is no
expansion loop, because a loop chasing any edge above a fixed tolerance walks
the box out to the parameter bound to reach a plateau that never decays.

**The likelihood is corrected where it is read.** `_gls_covariance` corrects the
covariance for correlation between bins; `min_func` does not, so every interval
read off it was too sharp by exactly that factor — nominal 68.27% on `beta`
covering 0.52 (scan) and 0.58 (mesh) against `optimise`'s 0.65.
`_correlation_inflation` measures the factor and `_temperature` applies it, to
the deviance, the mesh density and the MCMC target alike. Nothing that is
minimised changes, so **no fitted value moves**. `calibrate=False` reproduces a
pre-v2 interval exactly, which is how an archive is audited.

Four things make it defensible, all measured:

- **One scalar is enough because the four agree** — the per-parameter inflations
  sit within 0.6–3.1% of each other, so this is a loss of degrees of freedom,
  not a reshaping. `_TEMPER_SPREAD_LIMIT` warns when they stop agreeing; it
  fires at 12% on the legacy 305 km fixture, which is a window too narrow for
  its model.
- **Only the spectral block enters, on both sides.** `residuals` appends a row
  per prior, so dividing `F` whole would widen a 0.05 km `zt` pin by 30% —
  loosening the constraint that carries the depth scale in the name of
  calibrating it. Measured: 1.049 under a hard pin where tempering everything
  gives 1.310, and agreement with the full factor where no prior holds it.
- **It is computed once, at the mode, and held.** `_banded_correlation` reads
  model mismatch as correlation, which is right for a covariance at the solution
  and fatal if re-estimated per node, where a fit could lower the objective by
  making its own residuals look correlated.
- **On windows that bound `dz` it widens by `sqrt(t2)` and does nothing else** —
  1.285 against 1.269 at 1000 km, 1.305 against 1.298 at 2000. At 200 km, where
  a 30 km layer is not resolvable, the corrected deviance stops crossing and
  reports unbounded. The correction is largest where the model cannot follow the
  data, which is the honest answer, not an overreach.

Whitening the objective was the other candidate and is **rejected**: 28% scatter
on the `dz` estimator, `R` tabulated per taper, `_covariance` dropping
`_gls_covariance` or correcting twice, and `_ANALYTIC_COLUMNS` whitened too —
unwhitened, that sent a `dz = 10` layer to 60.9.

Note the temper is a ratio of two covariances and so is blind to the absolute
scale of `sigma_Phi`: it cannot detect a wrong `dof_factor`, and an uncalibrated
taper still gets `_TAPER_DOF`'s untapered fallback in the *within*-bin term.

**`sensitivity` reports about half the spread an independent repeat would
find** — 0.50 to 0.62 of the scatter over independent realisations of the
field, measured across three regimes. That is the bin-independence assumption
above and no choice of starting point touches it.

**Its warm start is measured, not assumed.** Every realisation begins at one
fit to the unresampled spectrum. Re-deriving a start per realisation costs 1.4x
and moves the reported spread not at all — 13.28 against 13.28 — because
resampling `Phi` within `sigma_Phi` does not carry a realisation across a basin
boundary. What *did* strand the ensemble was the old `dz = 10` constant, which
put its median at 10.3 against a truth of 45; deriving the start fixed that.
Do not re-derive per realisation again without a case the warm start demonstrably
gets wrong.

## Synthetics

`pycurious.fractal_anomaly(n, dx, beta, zt, dz, C, seed)` filters white noise by
the square root of `bouligand2009`, so the expected spectrum *is* the model.
Returns `(data, extent)`, ready to splat: `CurieOptimiseBouligand(data, *extent)`.

- True Curie depth is `zt + dz`; true centroid depth (Tanaka) is `zt + dz/2`.
- `beta` and `zt` recover tightly. **`dz` does not** — roughly one realisation in
  five is tens of percent out, so assert on it across seeds or in the mean.
- **`dz` has two band limits, not one** (`notes/dz-recoverability.md`). The
  rolloff sits at `|k| dz ~ 1` and must be inside the band at *both* ends:
  `2*pi*dz/window < 1` **and** `pi*dz/dx >> 1`. The second is a statement about
  **cell size**, it is the one nobody checks, and it is the one that bites — at
  `dz = 5` on 5 km cells it is 3.1 and recovery is hopeless at every window,
  with **more window making it worse** (median error 32.7 km at 500 km, 212 km
  at 4000 km). That is the only place in this package where a wider window
  hurts, and it is the signature of a spectrum whose high-`k` band cannot locate
  the rolloff. Above that floor, relative error at 4000 km is 5–7% for `dz` of
  10–45 km and 9% at 60.
- **On WDMAM, `dz` never converges — it tracks the window.** Over 162 L2
  vertices the median `dz` grows monotonically from the 1500 km rung to 10,000
  km by 2.8x (shallow third), 2.2x (middle) and 2.3x (deep), drifting 0.5–0.8 km
  per 250 km of window without slowing. Synthetics converge; the difference is
  the data, and a whitened objective reproduces the table to within 1–2 km. The
  cause is the unmodelled ~4.2 km resolution rolloff reducing the *effective*
  `k_max`, which puts every vertex on the wrong side of the second condition
  above. **A WDMAM `dz` is a thickness-at-a-window, not a thickness.**
- **`C` is not recoverable**; treat it as a nuisance parameter. Log-averaging
  costs the Euler-Mascheroni constant and a `np.hanning` taper costs
  `ln(3/8)**2`, so a fit returns `C` about 2.5 low under hanning.

Prefer generated synthetics over `tests/test_mag_data.txt`. That fixture is only
305 km across for a 10 km layer — too narrow to constrain the long wavelengths,
so `test_optimise.py` asserts no accuracy against it. Accuracy claims belong in
`tests/test_recovery.py`.

## Parallelism

`parallelise_routine(window, xc_list, yc_list, func, *args, **kwargs)` consumes
two reserved keywords rather than forwarding them:

- `on_error` — `"raise"` (default) or `"ignore"`, which NaN-fills and warns.
- `seed` — gives each centroid an independent child seed, so results do not
  depend on processor count. Only routines marked `@stochastic` accept it;
  passing it elsewhere warns.

macOS and Windows default to the **spawn** start method, so a script calling this
at module level must guard with `if __name__ == "__main__":` or every worker
re-executes it. A `taper` or `process_subgrid` defined in a notebook cell or in
`__main__` cannot be reconstructed in a spawned child; this is detected up front
and falls back to serial with a warning rather than deadlocking.

## Notebooks

`Examples/Notebooks/{Bouligand,Tanaka}/Ex1..Ex5`, plus `Examples/0-StartHere.ipynb`.
All are committed **without outputs** — re-running one will dirty it, so strip
outputs before committing. They quote their own numbers in prose, so changing
anything numerical means re-running *and* re-reading the surrounding markdown.

Ex5 in both folders needs real data: EMAG2 v3 (497 MB) and Li et al. 2017
(124 MB), fetched by `pycurious.download` into `Examples/data/`, which is
gitignored apart from `test_mag_data.txt`. There is no "EMAG3" — EMAG2 version 3
is the current release.

Style: explanation belongs in markdown cells, with inline code comments kept to a
minimum.

## Style

LGPL header on every module, then a short module docstring. Google-style
docstrings with `Args:` / `Returns:` / `Notes:` / `References:`, rendered by
Sphinx (`sphinx.ext.napoleon`) into the API reference — see `docs/`. The heavy
method narrative and math live in the hand-written theory pages
(`docs/theory/`), not the module docstrings, which were trimmed to summaries in
the pdoc→Sphinx migration; keep new math there rather than re-growing the
docstrings. Calibration numbers in docstrings are measured — if you change the
method, re-measure rather than adjusting the prose.

## Packaging

`MANIFEST.in` lists the eleven notebooks individually rather than globbing them.
It does not consult `.gitignore`, so a `recursive-include` shipped whatever was
in the working tree — an sdist once carried 35 notebooks, 20 of them untracked.
Adding a notebook means adding a line there.

To check what an sdist would contain:

```bash
rm -rf pycurious.egg-info          # stale SOURCES.txt is reused otherwise,
python -m build --sdist            # silently reinstating files you excluded
tar tzf dist/pycurious-*.tar.gz | grep ipynb
```

The `rm` matters. `SOURCES.txt` is regenerated from the *old* manifest if it is
left in place, which makes a correct `MANIFEST.in` look broken.

## Known defects

- **`profile` reports one basin of a multimodal deviance.** The scan walks
  outward from the best node to the first threshold crossing, so where the
  misfit has two minima it covers the one around the best node and never sees
  the other, and the interval can then exclude the fitted value. Measured on
  synthetics at a 200 km window, 2 of 20 intervals did. It does **not** appear
  in the regime that matters: over cached EMAG2 spectra in `~/Global_CPD` —
  three window sizes from 1000 to 4000 km, three targets, twelve mesh vertices
  — 0 of 108 intervals excluded their estimate. Band limiting and the prior
  pinning `zt` between them seem to remove it. Worth knowing if you profile a
  small unconstrained synthetic; not worth guarding against.

  The sibling defect in `optimise` — walking into the same second basin and
  reporting it without complaint — **is** fixed, by deriving the starting point
  (`notes/derived-starting-values.md`). The interval construction here is not,
  and is a separate thing.

- **A better minimum is not always a better answer.** Where a window does not
  constrain the fit, the likelihood has a second minimum at high `beta` and low
  `dz`, and anything that searches harder finds it and reports a confident
  worse number. Measured at a 200 km window on a 30 km layer: reaching lower
  misfits on 8 of 20 realisations took median |error| from 19.6 to 27.4 km.
  The lever is `add_prior(beta=...)`, not a worse optimiser.

- **`install_documentation()` fails for an installed package.** It is still
  advertised in the README, but `[tool.setuptools] packages =
  ["pycurious"]` installs only the package directory, and `Examples/` sits at the
  repository root. `_find_examples` looks inside the package and one level above
  it, and in a wheel neither exists. Works from a source checkout only.
  Pre-dates the v2 restructure that moved `Examples/` out of the package.

`notes/bouligand-findings.md` records seven findings from the Tanaka work and how
each was resolved — including two whose prescribed fix turned out to be wrong on
measurement. Worth reading before trusting any of them.
`notes/derived-starting-values.md` records what deriving the starting point cost
and bought, including where it is a straight regression.
`notes/collapsed-posterior.md` records what `posterior()` measured: the 2-D
density needs a few hundred forward-model evaluations against a chain's 24,000,
the production regime is one where MCMC does not work at all (ESS 20 from 8000
draws, so it cannot referee anything), and what tempering the likelihood cost
and bought. `notes/dz-recoverability.md` maps where `dz` is recoverable at all,
and finds it never converges on WDMAM. `notes/bench/` holds the harness behind
every number in `notes/` — tracked, while what it caches is not.
Everything in `notes/` stays out of the Sphinx build, unlike `docs/`.
