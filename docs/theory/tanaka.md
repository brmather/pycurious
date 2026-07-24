# The Tanaka centroid method

The centroid method of Tanaka *et al.* (1999) assumes a randomly magnetised layer
between depths $Z_t$ and $Z_b$, for which the radially averaged power spectrum is

$$ \Phi_{\Delta T}(|k|) = A\, e^{-2 |k| Z_t} \left( 1 - e^{-|k|(Z_b - Z_t)} \right)^2 $$

Two limits of this expression each give a straight line, fitted over a different
band of wavenumbers.

## The two bands

At wavelengths **short** compared with the source thickness the layer looks like
a half space, and the log amplitude spectrum has slope $-Z_t$:

$$ \ln \Phi_{\Delta T}^{1/2} = \ln B - |k| Z_t $$

At **long** wavelengths the expression is rewritten about the centroid depth
$Z_0$, with $d$ for the *half* thickness. Since
$\Phi^{1/2} \propto e^{-|k|Z_t} - e^{-|k|Z_b}$, and $Z_t - Z_0 = -d$ while
$Z_b - Z_0 = +d$, factoring out $e^{-|k|Z_0}$ turns the difference into a
hyperbolic sine:

$$ \Phi_{\Delta T}^{1/2} = C\, e^{-|k| Z_0} \left( e^{+|k|d} - e^{-|k|d} \right)
   = 2 C\, e^{-|k| Z_0} \sinh(|k| d) $$

For $|k| d \ll 1$ the sinh is approximately $|k| d$, so dividing through by $|k|$
leaves a line of slope $-Z_0$:

$$ \ln \left( \Phi_{\Delta T}^{1/2} / |k| \right) = \ln D - |k| Z_0 $$

Note the factor $e^{-|k|Z_0}$ has to be taken out of the exponential prefactor for
this to work: it is not the bracket of the first equation that becomes a sinh, but
the whole right-hand side once it is re-centred.

The base of the magnetic source, taken to be the Curie point depth, then follows
from the two:

$$ Z_b = 2 Z_0 - Z_t $$

## What PyCurious adds

The original method requires the user to pick both bands by eye and read the
gradients off a plot, yielding a single number with no error bar. In
{py:class}`~pycurious.CurieOptimiseTanaka` both bands are fitted with
{py:func}`scipy.optimize.curve_fit`, weighted by the measured scatter of the
spectrum, so the fit covariance propagates into an uncertainty on the Curie
depth.

## Choosing the bands

There are no default bands, deliberately. Each fit is only valid over the range of
wavenumbers where its approximation holds, and that depends on the data:

- the $Z_t$ band needs wavelengths shorter than twice the source thickness;
- the $Z_0$ band needs $|k| d \ll 1$.

Violating the second is easy to do and biases $Z_0$ low, which is not obvious from
the plot because the fit still looks straight. Use
{py:meth}`~pycurious.CurieOptimiseTanaka.check_bands` to test a choice before
relying on it — it reports point counts, wavelengths, $|k|d$, and the bias in km.

Bands are given in **rad/km**, the same units
{py:meth}`~pycurious.CurieGrid.radial_spectrum` returns. (They used to be
cycles/km; a guard warns when a band looks like a leftover cycles/km value,
because such a call still runs and returns a plausible number.)

## Fractal magnetisation

Tanaka assumes the magnetisation is spatially random. Real crust is better
described as fractal, and a fractal source adds a term
$-\tfrac{1}{2}(\beta - 1)\ln|k|$ to the log amplitude spectrum, which biases $Z_t$
high by $(\beta - 1) / 2\bar{k}$. Pass `beta` to remove it — see
{py:func}`pycurious.bouligand2009`, which fits $\beta$ directly.

**The correction is verified for $Z_t$ only.** Removing the $\ln|k|$ term recovers
the top of the source almost exactly, but it does not make the two models agree:
`bouligand2009` also carries $\beta$ inside its $\cosh$/Bessel factor, and what is
left over falls on the centroid. Fitting an exact, noiseless `bouligand2009`
spectrum with $Z_t = 1$, $\Delta z = 20$ and the correction applied:

| $\beta$ | $Z_t$ | $Z_b$ | error in $Z_b$ |
|---|---|---|---|
| 1 | 0.999 | 37.6 | +16.6 |
| 2 | 0.999 | 25.2 | +4.2 |
| 3 | 0.998 | 21.6 | +0.6 |
| 4 | 0.997 | 20.5 | −0.5 |

$Z_t$ comes back to three decimal places throughout; the Curie depth does not.
Near $\beta = 3$ the residual happens to cancel the opposing $|k|d$ bias, which is
a coincidence of that one value and not a validation. For a fractal source, fit
$\beta$ directly with {py:class}`~pycurious.CurieOptimiseBouligand` rather than
correcting for it here.

## References

Tanaka, A., Okubo, Y., & Matsubayashi, O. (1999). Curie point depth based on
spectrum analysis of the magnetic anomaly data in East and Southeast Asia.
*Tectonophysics*, 306(3–4), 461–470.
[doi:10.1016/S0040-1951(99)00072-4](https://doi.org/10.1016/S0040-1951(99)00072-4)
