---
title: 'PyCurious: A Python module for computing the Curie depth from the magnetic anomaly.'
tags:
  - Python
  - Curie depth
  - magnetism
  - magnetic anomaly
  - Bayesian inference
authors:
  - name: Ben Mather
    orcid: 0000-0003-3566-1557
    affiliation: "1,2" # (Multiple affiliations must be quoted)
  - name: Robert Delhaye
    affiliation: "2"
affiliations:
 - name: School of Geoscience, The University of Sydney
   index: 1
 - name: School of Cosmic Physics, Dublin Institute for Advanced Studies
   index: 2
date: 30 April 2019
bibliography: paper.bib
---

# Summary

Magnetic data is one of the most prevelant geophysical datasets available on the surface of the Earth. Curie depth is the depth at which rocks lose their magnetism. The most prevalent magnetic mineral is magnetite, which has a Curie point of 580°C, thus the Curie depth is often interpreted as the 580°C isotherm. Current methods to derive Curie depth first compute the (fast) Fourier transform over a square window of a magnetic anomaly that has been reduced to the pole. The depth and thickness of magnetic sources is estimated from the slope of the radial power spectrum.

Our Python package, `pycurious`, ingests maps of the Earth's magnetic anomaly and distributes the computation of Curie depth across multiple CPUs. `pycurious` implements the [@Tanaka1999] and [@Bouligand2009] methods for computing the thickness of a buried magnetic source. The former selects portions of the radial power spectrum in the low and high frequency domain to compute the depth of magneic sources, while the latter fits an analytical solution to the entire power spectrum. We cast the [@Bouligand2009] method within a Bayesian framwork to estimate the uncertainty of Curie depth calculations. Common computational workflows and geospatial manipulation of magnetic data are covered in the Jupyter notebooks bundled with this package. The `mapping` module includes a set of functions that help to wrangle maps of the magnetic anomaly into a useable form for `pycurious`. Such an approach is commonly encountered for transforming global compilations of the magnetic anomaly, e.g. EMAG2 [@Meyer2017], from latitudinal/longitudinal coordinates to a local projection in eastings/northings.

<!-- `pycurious` is an object-oriented Python package that accepts grids of the Earth's magnetic anomaly and computes the Curie depth across a specified window size. The computation is distributed across all available processors to improve efficiency over large study areas.  -->

`pycurious` includes the following functionality:

- Importing and exporting GeoTiff files.
- Converting between geospatial coordinate reference systems (CRS).
- Decomposition of magnetic grids into square windows over which to compute Curie depth.
- Parallel distribution of computation across computing resources.
- Bayesian inversion framework and sensitivity analysis.


## Documentation

`pycurious` is bundled with a linked collection of jupyter notebooks that can act as a user guide and an introduction to the package. The notebooks are split into matching sets for frequentist and Bayesian estimates of Curie depth. The notebooks cover:

- Plotting the radial and azimuthal power spectrum.
- Computing Curie depth from a synthetic magnetic anomaly.
- Exploring parameter sensitivity.
- Posing the inverse problem and objective function.
- Mapping Curie depth using the EMAG2 magnetic anomaly dataset [@Meyer2017].

All documentation can be accessed from within the module via a Python function that installs the notebooks at a filesystem location specified by the user at run time. The API documentation is kept up-to-date on [GitHub pages](https://brmather.github.io/pycurious/).


## Installation, Dependencies and Usage

`pycurious` requires `numpy` and `Cython` to compile C routines that are included with the distribution. The documentation is supplied in the form of Jupyter notebooks (the Jupyter environment is a dependency) which also have optional dependencies for the `mapping` module including `matplotlib`, `pyproj`, and `cartopy`. `pycurious` and all Python dependencies can be installed through the pypi.org `pip` package manager, however, several of the dependencies for `cartopy` may cause problems for inexperienced users. We therefore provide a Docker image built with all required and optional dependencies and a deployment of the documentation / examples on [mybinder.org](https://mybinder.org/v2/gh/brmather/pycurious/binder?filepath=Notebooks%2F0-StartHere.ipynb).


# Acknowledgements

This work was made possible by the G.O.THERM.3D project, supported by an Irish Research Council Research for Policy & Society grant (RfPS/2016/50) co-funded by Geological Survey Ireland and by Sustainable Energy Authority Of Ireland. Development of `pycurious` was inspired from [`pycpd`](https://github.com/groupeLIAMG/pycpd) and [`fatiando a terra`](https://github.com/fatiando/fatiando).


# References
