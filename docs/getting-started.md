# Getting started

## Installation

You will need **Python 3.9 or newer**. Only `numpy` and `scipy` are required at
import time; everything else is optional and grouped into
[extras](dependencies.md).

### Installing with pip

Install `pycurious` from [PyPI](https://pypi.org/project/pycurious/) with the
[pip package manager](https://pypi.org/project/pip/):

```bash
python3 -m pip install pycurious
```

The optional dependencies are grouped into extras, so you install only what you
need:

```bash
python3 -m pip install "pycurious[download]"   # requests
python3 -m pip install "pycurious[mapping]"    # pyproj, netCDF4
python3 -m pip install "pycurious[examples]"   # matplotlib, jupyter, cartopy
```

### Installing with conda

The required dependencies install cleanly from conda-forge:

```bash
conda install numpy scipy
```

and the full set for the notebooks with:

```bash
conda install numpy scipy matplotlib pyproj cartopy netcdf4 requests
```

then `pycurious` itself with pip:

```bash
pip install pycurious
```

Alternatively, create a dedicated environment from the bundled `environment.yml`:

```bash
git clone https://github.com/brmather/pycurious
cd pycurious
conda env create -f environment.yml
conda activate pycurious
pip install pycurious
```

```{note}
If installation fails due to
[an issue with `gcc` and Anaconda](https://github.com/Anaconda-Platform/anaconda-project/issues/183),
install `gxx_linux-64` with conda (`conda install gxx_linux-64`) and try again.
```

See [Software dependencies](dependencies.md) for what each extra provides and the
GDAL caveat behind the `geotiff` extra.

## A first spectrum

PyCurious exposes three classes: `CurieGrid` (the shared grid and spectrum
layer), `CurieOptimiseBouligand`, and `CurieOptimiseTanaka`. A minimal workflow
to compute the radial power spectrum of a window:

```python
import pycurious

# initialise a CurieOptimiseBouligand object with a 2D magnetic anomaly
grid = pycurious.CurieOptimiseBouligand(mag_anomaly, xmin, xmax, ymin, ymax)

# extract a square window of the magnetic anomaly
subgrid = grid.subgrid(window_size, x, y)

# compute the radial power spectrum
k, Phi, sigma_Phi = grid.radial_spectrum(subgrid)
```

Here `k` is the wavenumber in **rad/km**, `Phi` the radially averaged spectrum,
and `sigma_Phi` the scatter of the FFT cells within each annulus. To fit a Curie
depth, work through the [Tutorials](tutorials/index.md), which build a synthetic
anomaly with a known answer and recover it with both methods, complete with
uncertainties.

## Running the tests

The test suite runs with [`pytest`](https://pypi.org/project/pytest/) once the
`test` extra is installed:

```bash
git clone https://github.com/brmather/pycurious
cd pycurious
python3 -m pip install -e ".[test]"
pytest              # the full suite
pytest -m "not slow"  # skip the calibration tests that fit hundreds of realisations
```
