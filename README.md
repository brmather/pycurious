![PyCurious](https://github.com/brmather/pycurious/blob/master/pycurious/Examples/Images/pycurious-logo.png?raw=true)

[![Docker Cloud Automated build](https://img.shields.io/docker/cloud/automated/brmather/pycurious.svg)](https://hub.docker.com/r/brmather/pycurious)
[![PyPI](https://img.shields.io/pypi/v/pycurious.svg)](https://pypi.org/project/pycurious/)
[![DOI](https://zenodo.org/badge/123281222.svg)](https://zenodo.org/badge/latestdoi/123281222)
[![Build Status](https://travis-ci.org/brmather/pycurious.svg?branch=master)](https://travis-ci.org/brmather/pycurious)

Magnetic data is one of the most common geophysics datasets available on the surface of the Earth. Curie depth is the depth at which rocks lose their magnetism. The most prevalent magnetic mineral is magnetite, which has a Curie point of 580°C, thus the Curie depth is often interpreted as the 580°C isotherm.

Current methods to derive Curie depth first compute the (fast) Fourier transform over a square window of a magnetic anomaly that has been reduced to the pole. The depth and thickness of magnetic sources is estimated from the slope of the radial power spectrum. `pycurious` implements the Tanaka *et al.* (1999) and Bouligand *et al.* (2009) methods for computing the thickness of a buried magnetic source. `pycurious` ingests maps of the magnetic anomaly and distributes the computation of Curie depth across multiple CPUs. Common computational workflows and geospatial manipulation of magnetic data are covered in the Jupyter notebooks bundled with this package.

#### Binder

Launch the demonstration at [mybinder.org](https://mybinder.org/v2/gh/brmather/pycurious/binder?filepath=Notebooks%2F0-StartHere.ipynb)

[![badge](https://img.shields.io/badge/launch-pycurious-E66581.svg?logo=data:image/png;base64,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)](https://mybinder.org/v2/gh/brmather/pycurious/binder?filepath=Notebooks%2F0-StartHere.ipynb)

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

- [Ex1-Plot-power-spectrum.ipynb](pycurious/Examples/Notebooks/Tanaka/Ex1-Plot-power-spectrum.ipynb)
- [Ex2-Compute-Curie-depth.ipynb](pycurious/Examples/Notebooks/Tanaka/Ex2-Compute-Curie-depth.ipynb)
- [Ex3-Parameter-exploration.ipynb](pycurious/Examples/Notebooks/Tanaka/Ex3-Parameter-exploration.ipynb)

### Bouligand

- [Ex1-Plot-power-spectrum.ipynb](pycurious/Examples/Notebooks/Bouligand/Ex1-Plot-power-spectrum.ipynb)
- [Ex2-Compute-Curie-depth.ipynb](pycurious/Examples/Notebooks/Bouligand/Ex2-Compute-Curie-depth.ipynb)
- [Ex3-Posing-the-inverse-problem.ipynb](pycurious/Examples/Notebooks/Bouligand/Ex3-Posing-the-inverse-problem.ipynb)
- [Ex4-Spatial-variation-of-Curie-depth.ipynb](pycurious/Examples/Notebooks/Bouligand/Ex4-Spatial-variation-of-Curie-depth.ipynb)
- [Ex5-Mapping-Curie-depth-EMAG2.ipynb](pycurious/Examples/Notebooks/Bouligand/Ex5-Mapping-Curie-depth-EMAG2.ipynb)


## Installation

### Dependencies

You will need **Python 2.7 or 3.5+**.
Also, the following packages are required:

- [`numpy`](http://numpy.org)
- [`scipy`](https://scipy.org)
- [`cython`](https://cython.org/)

__Optional dependencies__ for mapping module and running the Notebooks:

- [`jupyter`](https://jupyter.org/)
- [`matplotlib`](https://matplotlib.org/)
- [`pyproj`](https://github.com/jswhit/pyproj)
- [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/)

### Installing using pip

You can install `pycurious` using the
[`pip package manager`](https://pypi.org/project/pip/) with either version of Python:

```bash
python2 -m pip install pycurious
python3 -m pip install pycurious
```
All the dependencies will be automatically installed by `pip`.

### Installing with conda

You can install `pycurious` using the [conda package manager](https://conda.io).
Its required dependencies can be easily installed with:

```bash
conda install numpy scipy cython
```

And the full set of dependencies with:

```bash
conda install numpy scipy cython matplotlib pyproj cartopy
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


### Installing using Docker

A more straightforward installation for `pycurious` and all of its dependencies may be deployed with [Docker](https://www.docker.com).
To install the docker image and start the Jupyter notebook examples:

```bash
docker run --name pycurious -p 127.0.0.1:8888:8888 brmather/pycurious:latest
```

## Usage

PyCurious consists of 2 classes:

- `CurieGrid`: base class that computes radial power spectrum, centroids for processing, decomposition of subgrids.
- `CurieOptimise`: optimisation module for fitting the synthetic power spectrum (inherits CurieGrid).

Also included is a `mapping` module for gridding scattered data points, and converting between coordinate reference systems (CRS).

Below is a simple workflow to calculate the radial power spectrum:

```python
import pycurious

# initialise CurieOptimise object with 2D magnetic anomaly
grid = pycurious.CurieOptimise(mag_anomaly, xmin, xmax, ymin, ymax)

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
