# PyCurious

**PyCurious** estimates the **Curie point depth** — the depth at which rock loses
its magnetisation — from the radially averaged spectrum of a magnetic anomaly.

Magnetic data is one of the most common geophysics datasets available at the
surface of the Earth. The most prevalent magnetic mineral is magnetite, whose
Curie point is 580 °C, so the Curie depth is often interpreted as the 580 °C
isotherm. PyCurious computes the (fast) Fourier transform over square windows of
a magnetic anomaly reduced to the pole and estimates the depth and thickness of
the magnetic source from the slope of the radial spectrum. It implements two
methods, sharing one grid and spectrum layer, and — the defining feature of v2 —
**both return uncertainties**:

- **Bouligand *et al.* (2009)** — fits a four-parameter analytic spectrum
  (`beta, zt, dz, C`) by optimisation, with a full Bayesian toolkit
  (profile-deviance intervals, Metropolis–Hastings sampling, sensitivity
  analysis).
- **Tanaka *et al.* (1999)** — the centroid method: two straight lines fitted to
  separate wavenumber bands, each weighted by the measured spectral scatter.

PyCurious ingests maps of the magnetic anomaly and distributes the computation of
Curie depth across multiple CPUs.

## Citation

> Mather, B. and Delhaye, R. (2019). PyCurious: A Python module for computing the
> Curie depth from the magnetic anomaly. *Journal of Open Source Software*,
> 4(39), 1544, <https://doi.org/10.21105/joss.01544>

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started
dependencies
tutorials/index
theory/index
api/index
contributing
```

## References

1. Bouligand, C., Glen, J. M. G., & Blakely, R. J. (2009). Mapping Curie
   temperature depth in the western United States with a fractal model for
   crustal magnetization. *Journal of Geophysical Research*, 114(B11104), 1–25.
   <https://doi.org/10.1029/2009JB006494>
2. Tanaka, A., Okubo, Y., & Matsubayashi, O. (1999). Curie point depth based on
   spectrum analysis of the magnetic anomaly data in East and Southeast Asia.
   *Tectonophysics*, 306(3–4), 461–470.
   <https://doi.org/10.1016/S0040-1951(99)00072-4>
