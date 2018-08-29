# PyCurious

Magnetic data is one of the most common geophysics datasets available on the surface of the Earth. Curie depth is the depth at which rocks lose their magnetism. The most prevalent magnetic mineral is magnetite, which has a Curie point of 580C, thus the Curie depth is often interpreted as the 580C isotherm.

Current methods to derive Curie depth first compute the (fast) Fourier transform over a square window of a magnetic anomaly that has been reduced to the pole. The depth and thickness of magnetic sources is estimated from the slope of the radial power spectrum. *PyCurious* implements two common approaches:

- [Tanaka *et al.* 1999](http://linkinghub.elsevier.com/retrieve/pii/S0040195199000724)
- [Bouligand *et al.* 2009](http://doi.wiley.com/10.1029/2009JB006494)

Both of these methods to compute Curie depth are covered in the form of Jupyter notebooks. Copy these into your working directory by running,

```python
import pycurious
pycurious.install_documentation()
```

## Usage

PyCurious consists of 3 classes that inherit from each other:

1. **CurieGrid**: base class that computes radial power spectrum, centroids for processing, decomposition of subgrids.
2. **CurieOptimise**: optimisation module for fitting the synthetic power spectrum (inherits CurieGrid).
3. **CurieParallel**: parallel implementation based on the decomposition of subgrids (inherits CurieOptimise).

Also included is a `mapping` module for gridding scattered data points, and converting between coordinate reference systems (CRS).

Below is a simple workflow to calculate the radial power spectrum:

```python
import pycurious

# initialise CurieOptimise object with 2D magnetic anomaly
grid = pycurious.CurieOptimise(mag_anomaly, xmin, xmax, ymin, ymax)

# extract a square window of the magnetic anomaly
subgrid = grid.subgrid(window_size, x, y)

# compute the radial power spectrum
k, Phi, sigma_Phi = grid.radial_power_spectrum(subgrid)
```

## Dependencies

- Python 2.7 and above
- Numpy 1.9 and above
- Scipy 0.14 and above
- Cython

Optional dependencies for mapping module:

- Matplotlib
- [pyproj](https://github.com/jswhit/pyproj)
- [Cartopy](http://scitools.org.uk/cartopy/docs/latest/index.html)

Optional dependencies for parallel module:

- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [petsc4py](http://www.mcs.anl.gov/petsc/petsc4py-current/docs/usrman/install.html)

## Installation

To install:

```
python setup.py install --user
```

This will compile all C and Fortran sources and install them to the user directory (omit the `--user` flag to install to the system directory). Remember to delete the build folder if you are upgrading this package.

Alternatively, install using pip:

```
pip install pycurious --user
```

