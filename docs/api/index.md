# API reference

The public API of PyCurious. Everything below is importable directly from the
top-level `pycurious` namespace.

## Grid and spectra

The base class shared by both optimisers — subgrid decomposition, the radial and
window spectra, and the covariance machinery behind the uncertainties.

```{autoclass} pycurious.CurieGrid
:members:
```

## Bouligand optimiser

```{autoclass} pycurious.CurieOptimiseBouligand
:members:
```

## Tanaka optimiser

```{autoclass} pycurious.CurieOptimiseTanaka
:members:
```

## Synthetic and analytic spectra

```{autofunction} pycurious.fractal_anomaly
```

```{autofunction} pycurious.bouligand2009
```

```{autofunction} pycurious.maus1995
```

## Mapping

Projections, netCDF, and GeoTIFF I/O (all lazily imported — see the
[`mapping` and `geotiff` extras](../dependencies.md)).

```{automodule} pycurious.mapping
:members:
```

## Downloads

```{automodule} pycurious.download
:members:
```

## Deprecated

`tanaka1999` and `ComputeTanaka` implement the centroid method without
uncertainties and are superseded by
{py:class}`~pycurious.CurieOptimiseTanaka`. They are retained for backward
compatibility.

```{autofunction} pycurious.tanaka1999
```

```{autoclass} pycurious.ComputeTanaka
:members:
```
