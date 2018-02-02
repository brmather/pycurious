# PyCurious

Magnetic data is one of the most common geophysics datasets available on the surface of the Earth. Curie depth is the depth at which rocks lose their magnetism. This is most often interpreted as the 580C isotherm, which is the Curie point of magnetite - the most prevalent magnetic mineral.

All methods to derive Curie depth first compute the (fast) Fourier transform over a square window of a magnetic anomaly that has been reduced to the pole. Initial methods plotted the radial spectrum, obtained from FFT, against the wavenumber (rad/km) and used the gradient between two points to estimate the Curie depth ([Tanaka *et al.* 1999](http://linkinghub.elsevier.com/retrieve/pii/S0040195199000724)). The drawback here being that the choice of points is subjective and potentially meaningless if the window size is too small to capture long wavelength signals. Alternatively, [Bouligand *et al.* (2009)](http://doi.wiley.com/10.1029/2009JB006494) formulated an analytical expression for the radial power spectum using four parameters:

- *beta* - a fractal parameter
- *zt* - the top of magnetic sources
- *dz* - the thickness of the magnetic layer
- *C* - a field constant

The Curie depth is determined by minimising the misfit between the analytical power spectrum and the power spectrum computed from FFT.

This Python package implements the latter method for computing Curie depth. It was heavily inspired from the [pycpd](https://github.com/groupeLIAMG/pycpd) package which implements a user inferface. We have simplified the API and extended it for parallel computation. Significant effort has been made towards repurposing Curie depth determination within a Bayesian inverse framework. PyCurious implements an objective function that accepts *a priori* information, and quantifies the uncertainty of Curie depth from a sensitivity analysis.

## Usage

PyCurious consists of 3 classes that inherit from each other:

1. **CurieGrid**: base class that computes radial power spectrum, centroids for processing, decomposition of subgrids.
2. **CurieOptimise**: optimisation module for fitting the synthetic power spectrum (inherits CurieGrid).
3. **CurieParallel**: parallel implementation based on the decomposition of subgrids (inherits CurieOptimise).

Also included is several functions for mapping, gridding scattered data points, and converting between coordinate reference systems (CRS).

Two optimisation passes is generally required of a study area: in the first pass all parameters have no prior and are completely unconstrainted; in the second pass we obtain priors by taking the mean and standard deviation of the parameters after the first pass to obtain more precise Curie depth estimates. *beta* and *C* vary at long wavelengths, thus the objective function penalises these parameters more due to their low standard deviation.

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

`python setup.py install --user`

This will compile all C and Fortran sources and install them to the user directory (omit the `--user` flag to install to the system directory).

Remember to delete the build folder if you are upgrading this package.
