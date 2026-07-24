![PyCurious](https://github.com/brmather/pycurious/blob/master/Examples/Images/pycurious-logo.png?raw=true)

[![Tests](https://github.com/brmather/pycurious/actions/workflows/tests.yml/badge.svg)](https://github.com/brmather/pycurious/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://brmather.github.io/pycurious/)
[![PyPI](https://img.shields.io/pypi/v/pycurious.svg)](https://pypi.org/project/pycurious/)
[![DOI](https://zenodo.org/badge/123281222.svg)](https://zenodo.org/badge/latestdoi/123281222)

Magnetic data is one of the most common geophysics datasets available on the surface of the Earth. Curie depth is the depth at which rocks lose their magnetism. The most prevalent magnetic mineral is magnetite, which has a Curie point of 580°C, thus the Curie depth is often interpreted as the 580°C isotherm.

Current methods to derive Curie depth first compute the (fast) Fourier transform over a square window of a magnetic anomaly that has been reduced to the pole. The depth and thickness of magnetic sources is estimated from the slope of the radial power spectrum. `pycurious` implements the Tanaka *et al.* (1999) and Bouligand *et al.* (2009) methods for computing the thickness of a buried magnetic source. `pycurious` ingests maps of the magnetic anomaly and distributes the computation of Curie depth across multiple CPUs. Common computational workflows and geospatial manipulation of magnetic data are covered in the Jupyter notebooks bundled with this package.

#### Citation

[![DOI](http://joss.theoj.org/papers/10.21105/joss.01544/status.svg)](https://doi.org/10.21105/joss.01544)

Mather, B. and Delhaye, R. (2019). PyCurious: A Python module for computing the Curie depth from the magnetic anomaly. _Journal of Open Source Software_, 4(39), 1544, https://doi.org/10.21105/joss.01544

## Navigation / Notebooks

There are two matching sets of Jupyter notebooks - one set for the [Tanaka](#Tanaka) and one for [Bouligand](#Bouligand) implementations. The Bouligand set of noteboks are a natural choice for Bayesian inference applications.

Note, these examples can be installed from the package itself by running:

```python
import pycurious
pycurious.install_documentation(path="Notebooks")
```

### Tanaka

- [Ex1-Plot-amplitude-spectrum.ipynb](Examples/Notebooks/Tanaka/Ex1-Plot-amplitude-spectrum.ipynb)
- [Ex2-Compute-Curie-depth.ipynb](Examples/Notebooks/Tanaka/Ex2-Compute-Curie-depth.ipynb)
- [Ex3-Parameter-exploration.ipynb](Examples/Notebooks/Tanaka/Ex3-Parameter-exploration.ipynb)
- [Ex4-Spatial-variation-of-Curie-depth.ipynb](Examples/Notebooks/Tanaka/Ex4-Spatial-variation-of-Curie-depth.ipynb)
- [Ex5-Mapping-Curie-depth-EMAG2.ipynb](Examples/Notebooks/Tanaka/Ex5-Mapping-Curie-depth-EMAG2.ipynb)

### Bouligand

- [Ex1-Plot-power-spectrum.ipynb](Examples/Notebooks/Bouligand/Ex1-Plot-power-spectrum.ipynb)
- [Ex2-Compute-Curie-depth.ipynb](Examples/Notebooks/Bouligand/Ex2-Compute-Curie-depth.ipynb)
- [Ex3-Posing-the-inverse-problem.ipynb](Examples/Notebooks/Bouligand/Ex3-Posing-the-inverse-problem.ipynb)
- [Ex4-Spatial-variation-of-Curie-depth.ipynb](Examples/Notebooks/Bouligand/Ex4-Spatial-variation-of-Curie-depth.ipynb)
- [Ex5-Mapping-Curie-depth-EMAG2.ipynb](Examples/Notebooks/Bouligand/Ex5-Mapping-Curie-depth-EMAG2.ipynb)


## Installation

### Dependencies

You will need **Python 3.9 or newer**.
Also, the following packages are required:

- [`numpy`](http://numpy.org)
- [`scipy`](https://scipy.org)

__Optional dependencies__ for mapping module and running the Notebooks:

- [`jupyter`](https://jupyter.org/)
- [`matplotlib`](https://matplotlib.org/)
- [`pyproj`](https://github.com/jswhit/pyproj)
- [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/)
- [`netCDF4`](https://unidata.github.io/netcdf4-python/)
- [`requests`](https://requests.readthedocs.io/)

### Installing using pip

You can install `pycurious` using the
[`pip package manager`](https://pypi.org/project/pip/):

```bash
python3 -m pip install pycurious
```
All the required dependencies will be automatically installed by `pip`.

The optional dependencies are grouped into extras, so you can install only
what you need:

```bash
python3 -m pip install pycurious[download]   # requests
python3 -m pip install pycurious[mapping]    # pyproj, netCDF4
python3 -m pip install pycurious[examples]   # matplotlib, jupyter, cartopy
```

### Installing with conda

You can install `pycurious` using the [conda package manager](https://conda.io).
Its required dependencies can be easily installed with:

```bash
conda install numpy scipy
```

And the full set of dependencies with:

```bash
conda install numpy scipy matplotlib pyproj cartopy netcdf4 requests
```

Then `pycurious` can be installed with `pip`:

```bash
pip install pycurious
```

#### Conda environment

Alternatively, you can create a custom
[conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
where `pycurious` can be installed along with its dependencies.

Clone the repository:
```bash
git clone https://github.com/brmather/pycurious
cd pycurious
```

Create the environment from the `environment.yml` file:
```bash
conda env create -f environment.yml
```

Activate the newly created environment:
```bash
conda activate pycurious
```

And install `pycurious` with `pip`:
```bash
pip install pycurious
```

#### Issue with gcc

If the `pycurious` installation fails due to [an issue with `gcc` and
Anaconda](https://github.com/Anaconda-Platform/anaconda-project/issues/183), you just
need to install `gxx_linux-64` with conda:

```bash
conda install gxx_linux-64
```

And then install `pycurious` normally.

## Usage

PyCurious consists of 3 classes:

- `CurieGrid`: base class that computes radial power spectrum, centroids for processing, decomposition of subgrids.
- `CurieOptimiseBouligand`: optimisation module for fitting the synthetic power spectrum of Bouligand *et al.* (2009) (inherits CurieGrid).
- `CurieOptimiseTanaka`: optimisation module for the centroid method of Tanaka *et al.* (1999) (inherits CurieGrid).

Also included is a `mapping` module for gridding scattered data points, and converting between coordinate reference systems (CRS).

Below is a simple workflow to calculate the radial power spectrum:

```python
import pycurious

# initialise CurieOptimiseBouligand object with 2D magnetic anomaly
grid = pycurious.CurieOptimiseBouligand(mag_anomaly, xmin, xmax, ymin, ymax)

# extract a square window of the magnetic anomaly
subgrid = grid.subgrid(window_size, x, y)

# compute the radial power spectrum
k, Phi, sigma_Phi = grid.radial_spectrum(subgrid)
```

A series of tests are located in the *tests* subdirectory.
In order to perform these tests, clone the repository and run [`pytest`](https://pypi.org/project/pytest/):

```bash
git checkout https://github.com/brmather/pycurious.git
cd pycurious
pytest -v
```

### API Documentation

The API for all functions and classes in `pycurious` can be accessed from [https://brmather.github.io/pycurious/](https://brmather.github.io/pycurious/).


## References

1. Bouligand, C., Glen, J. M. G., & Blakely, R. J. (2009). Mapping Curie temperature depth in the western United States with a fractal model for crustal magnetization. Journal of Geophysical Research, 114(B11104), 1–25. https://doi.org/10.1029/2009JB006494
2. Tanaka, A., Okubo, Y., & Matsubayashi, O. (1999). Curie point depth based on spectrum analysis of the magnetic anomaly data in East and Southeast Asia. Tectonophysics, 306(3–4), 461–470. https://doi.org/10.1016/S0040-1951(99)00072-4
