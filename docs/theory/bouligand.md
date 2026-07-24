# The Bouligand fractal model

Bouligand *et al.* (2009) model the crust as a magnetic layer with **fractal**
(self-similar) magnetisation, rather than the spatially random magnetisation
Tanaka assumes. Their equation (4) gives the radially averaged power spectrum
$\Phi$ as a function of four parameters, and PyCurious fits all four at once.

## The analytic spectrum

{py:func}`pycurious.bouligand2009` returns the **log power spectrum**

$$ \ln \Phi(|k|) = C - 2|k|Z_t - (\beta - 1)\ln|k| - |k|\,\Delta z + \ln A $$

with

$$ A = \frac{\sqrt{\pi}}{\Gamma\!\left(1 + \tfrac{\beta}{2}\right)}
   \left[ \tfrac{1}{2}\cosh(|k|\,\Delta z)\,\Gamma\!\left(\tfrac{1+\beta}{2}\right)
   - K_{-\frac{1+\beta}{2}}(|k|\,\Delta z)
     \left(\tfrac{|k|\,\Delta z}{2}\right)^{\frac{1+\beta}{2}} \right] $$

where $K_\nu$ is the modified Bessel function of the second kind. The four
parameters are

| parameter | meaning |
|---|---|
| $\beta$ | fractal parameter (spectral exponent of the magnetisation) |
| $Z_t$ | depth to the top of the magnetic source |
| $\Delta z$ | thickness of the magnetic source |
| $C$ | field constant (Maus *et al.*, 1997) |

The Curie point depth is $Z_t + \Delta z$. {py:func}`pycurious.maus1995` is a
simplified version of the same model without the higher-order Bessel integration.

Because the spectrum is fitted in log power, `power=2.0` (the default of
{py:meth}`~pycurious.CurieGrid.radial_spectrum`) is the right choice for this
method.

## The inverse problem

{py:class}`~pycurious.CurieOptimiseBouligand` poses the recovery of the four
parameters as a Bayesian inverse problem. The posterior is

$$ P(\mathbf{m} \mid \mathbf{d}) = P(\beta, Z_t, \Delta z, C \mid \Phi_d) $$

where $\Phi_d$ is the radial power spectrum measured from the FFT over square
windows of the magnetic anomaly. The class provides a flexible objective function
that accepts *a priori* and likelihood terms, so priors can be placed on any
parameter, and it decomposes the computation across CPUs to map Curie depth over a
whole grid.

## Uncertainties

The likelihood is weighted by the **uncertainty of the annulus mean** returned by
{py:meth}`~pycurious.CurieGrid.window_spectrum`, not the raw within-annulus
scatter. Two corrections, both measured by Monte Carlo rather than assumed, turn
that scatter into an honest weight:

1. **Within a bin**, the FFT cells are not independent — a real field is
   Hermitian, so about half of them repeat (exactly a factor of two untapered),
   and a taper correlates neighbours further. The lost degrees of freedom
   dominate the sparse inner bins, which is exactly where the depth is
   determined.
2. **Between bins**, neighbouring residuals correlate because a taper spreads each
   wavenumber over several bins. Estimating this from the residuals and solving a
   generalised least-squares covariance matters: ignoring it understates every
   reported $\sigma$ by about 30 %.

Beyond the covariance,
{py:meth}`~pycurious.CurieOptimiseBouligand.profile` gives profile-deviance
intervals — which matter because $\Delta z$ and the Curie depth are genuinely
asymmetric — {py:meth}`~pycurious.CurieOptimiseBouligand.metropolis_hastings`
samples the posterior, and
{py:meth}`~pycurious.CurieOptimiseBouligand.sensitivity` resamples the spectrum.

```{note}
`C` is not recoverable in practice and is best treated as a nuisance parameter:
log-averaging costs the Euler–Mascheroni constant and a Hanning taper costs a
further fixed offset. `beta` and `Z_t` recover tightly, but `dz` is noisy — assert
on it across seeds or in the mean, not on a single realisation.
```

## References

Bouligand, C., Glen, J. M. G., & Blakely, R. J. (2009). Mapping Curie temperature
depth in the western United States with a fractal model for crustal
magnetization. *Journal of Geophysical Research*, 114(B11104), 1–25.
[doi:10.1029/2009JB006494](https://doi.org/10.1029/2009JB006494)

Maus, S., Gordon, D., & Fairhead, D. (1997). Curie temperature depth estimation
using a self-similar magnetization model. *Geophysical Journal International*,
129, 163–168.
[doi:10.1111/j.1365-246X.1997.tb00945.x](https://doi.org/10.1111/j.1365-246X.1997.tb00945.x)
