# API reference

The public API of PyCurious. Everything below is importable directly from the
top-level `pycurious` namespace.

## Grid and spectra

The base class shared by both optimisers — subgrid decomposition, the radial and
window spectra, and the covariance machinery behind the uncertainties.

```{eval-rst}
.. autoclass:: pycurious.CurieGrid
   :members:
```

## Bouligand optimiser

```{eval-rst}
.. autoclass:: pycurious.CurieOptimiseBouligand
   :members:
```

### The posterior

What `CurieOptimiseBouligand.posterior` returns. `C` and `z_t` integrate out
exactly, so the joint posterior of all four parameters is a density over
`beta` and `dz` alone — small enough to evaluate rather than sample, and read
for an interval on any target without evaluating the forward model again.

```{eval-rst}
.. autoclass:: pycurious.Posterior
   :members: interval, marginal, moments, identifiable_dz

.. autoclass:: pycurious.Interval
```

## Tanaka optimiser

```{eval-rst}
.. autoclass:: pycurious.CurieOptimiseTanaka
   :members:
```

## Synthetic and analytic spectra

```{eval-rst}
.. autofunction:: pycurious.fractal_anomaly

.. autofunction:: pycurious.bouligand2009

.. autofunction:: pycurious.maus1995
```

## Mapping

Projections, netCDF, and GeoTIFF I/O (all lazily imported — see the
[`mapping` and `geotiff` extras](../dependencies.md)).

```{eval-rst}
.. automodule:: pycurious.mapping
   :members:
```

## Downloads

```{eval-rst}
.. automodule:: pycurious.download
   :members:
```

## Deprecated

`tanaka1999` and `ComputeTanaka` implement the centroid method without
uncertainties and are superseded by
{py:class}`~pycurious.CurieOptimiseTanaka`. They are retained for backward
compatibility.

```{eval-rst}
.. autofunction:: pycurious.tanaka1999

.. autofunction:: pycurious.ComputeTanaka
```
