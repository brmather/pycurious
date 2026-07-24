# Software dependencies

PyCurious keeps a deliberately small import-time footprint: only **numpy** and
**scipy** are imported when you `import pycurious`. Everything else is imported
lazily inside the function that needs it, so it lives in an optional *extra* and
is pulled in only when you ask for it.

## Core dependencies

| Package | Purpose |
|---|---|
| [`numpy`](https://numpy.org) (≥ 1.20) | arrays, FFTs, the grid and spectrum layer |
| [`scipy`](https://scipy.org) (≥ 1.5) | optimisation, special functions, banded linear algebra |

Requires **Python 3.9 or newer**.

## Optional extras

Install an extra with `pip install "pycurious[<name>]"`. They compose, e.g.
`pip install "pycurious[mapping,download]"`.

### `download`

- [`requests`](https://requests.readthedocs.io/)

Backs {py:mod}`pycurious.download` — cached downloads with md5 checks, used by the
mapping tutorials to fetch the EMAG2 and reference datasets.

### `mapping`

- [`pyproj`](https://pyproj4.github.io/pyproj/) — coordinate reference system transforms
- [`netCDF4`](https://unidata.github.io/netcdf4-python/) — reading and writing gridded data

Backs the projection and netCDF parts of {py:mod}`pycurious.mapping`.

### `geotiff`

- [`gdal`](https://gdal.org/)

Backs the GeoTIFF import/export in {py:mod}`pycurious.mapping`. It is kept **out
of `mapping`** on purpose: the GDAL Python bindings need a matching system
`libgdal` already installed, so folding it into `mapping` would make
`pip install "pycurious[mapping]"` fail for most users. `conda install gdal` is
usually easier than installing it from pip.

### `examples`

- [`matplotlib`](https://matplotlib.org/), [`jupyter`](https://jupyter.org/),
  [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/),
  [`pyproj`](https://pyproj4.github.io/pyproj/)

Everything needed to run the bundled [Tutorials](tutorials/index.md).

### `test`

- [`pytest`](https://pytest.org/)

The test suite imports only `pytest`, `numpy`, `scipy`, and `pycurious`, so
`pip install -e ".[test]" && pytest` runs the whole suite.

### `docs`

- [`sphinx`](https://www.sphinx-doc.org/), [`furo`](https://pradyunsg.me/furo/),
  [`myst-nb`](https://myst-nb.readthedocs.io/),
  [`sphinx-copybutton`](https://sphinx-copybutton.readthedocs.io/)

Builds this documentation. Building the tutorials additionally needs the
`examples` and `mapping` extras so the notebooks execute.
