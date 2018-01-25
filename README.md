# PyCurious

Magnetic data is one of the most common geophysics datasets available on the surface of the Earth. The Curie depth is most often interpreted to be the Curie point of magnetite, because it is the most magnetic mineral, thus Curie depth offers a very desirable isotherm in the lower crust. This is useful for many applications that require constraints on lithospheric geotherms.

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

Remember to delete the `build` directory if you are upgrading this package.
