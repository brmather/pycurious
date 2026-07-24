# Theory

Both methods estimate the depth and thickness of a buried magnetic source from
the slope of the radially averaged spectrum of the magnetic anomaly. They differ
in the model fitted to that spectrum and in how the fit is turned into a Curie
depth with an uncertainty.

```{toctree}
:maxdepth: 2

bouligand
tanaka
```

## Conventions

A few conventions run through the whole package and are easy to get wrong:

- **Wavenumbers are in rad/km, everywhere.** {py:meth}`~pycurious.CurieGrid.radial_spectrum`
  returns `k` in rad/km, and the Tanaka fitting bands are given in the same units.
- **The DFT fundamental is $dk = 2\pi/(N\,dx)$**, not $2\pi/((N-1)\,dx)$; using
  $N-1$ understates every depth by a factor $(N-1)/N$.
- **Depths are positive downwards.** The optimisers return depths, not the
  negative gradients the fits produce.
- **The `power` argument selects which spectrum you get.**
  {py:meth}`~pycurious.CurieGrid.radial_spectrum` raises $|\mathrm{FFT}|$ to
  `power` before averaging. Since $\Phi = |\mathrm{FFT}|^2$, Bouligand fits the
  log power spectrum (`power=2.0`, the default) while Tanaka fits the log
  amplitude spectrum (`power=1.0`).
