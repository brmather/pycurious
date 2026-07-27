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
pytest                     # 108 tests, ~25 s
pytest -m "not slow"       # 105 tests, ~14 s -- skips the calibration tests that
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
`minimize`.** Do not put L-BFGS-B back. It spends seconds of CPU on this
four-parameter problem — 3.1 s for a fit and 16.5 s for a `dz` profile against
13 ms and 49 ms, measured in CPU time over synthetic windows — and its cost
does not track the number of function evaluations, so the time is going into
its own machinery rather than the forward model. `_fit` also supplies the two
exact Jacobian columns (`dr/dzt = -2k/sigma`, `dr/dC = 1/sigma`); `beta` enters
through the *order* of a Bessel function and `dz` costs the same analytically
as by difference, so those two stay numerical. Benchmark in
`time.process_time`, never wall clock — this is a shared machine.

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
`metropolis_hastings()` samples the posterior; `sensitivity()` resamples the
spectrum. `CurieOptimiseTanaka.sensitivity()` additionally jitters the band
edges, since band placement usually dominates spectral scatter. There is
deliberately no `profile` on the Tanaka side — each band is a straight-line fit,
where profile and covariance intervals are provably identical.

## Synthetics

`pycurious.fractal_anomaly(n, dx, beta, zt, dz, C, seed)` filters white noise by
the square root of `bouligand2009`, so the expected spectrum *is* the model.
Returns `(data, extent)`, ready to splat: `CurieOptimiseBouligand(data, *extent)`.

- True Curie depth is `zt + dz`; true centroid depth (Tanaka) is `zt + dz/2`.
- `beta` and `zt` recover tightly. **`dz` does not** — roughly one realisation in
  five is tens of percent out, so assert on it across seeds or in the mean.
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

- **`install_documentation()` fails for an installed package.** It is still
  advertised in the README, but `[tool.setuptools] packages =
  ["pycurious"]` installs only the package directory, and `Examples/` sits at the
  repository root. `_find_examples` looks inside the package and one level above
  it, and in a wheel neither exists. Works from a source checkout only.
  Pre-dates the v2 restructure that moved `Examples/` out of the package.

`notes/bouligand-findings.md` records seven findings from the Tanaka work and how
each was resolved — including two whose prescribed fix turned out to be wrong on
measurement. Worth reading before trusting any of them. (It lives in `notes/`,
not `docs/`, so it stays out of the Sphinx build.)
