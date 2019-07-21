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
    orcid: 0000-0003-2128-4295
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

Multiple geophysical methods have been proposed to resolve the thermal structure of the Earth's lithosphere with varying degrees of precision.
Geotherms may be constructed from heat flow or xenolith data, but the spatial coverage of these are often limited.
Seismic velocity has proven effective to infer upper-mantle temperature, but its application relies on building a compositional model suitable for the geological context and estimating attenuation from grainsize and water content.
In contrast, magnetic data are among the most widespread geophysical datasets available on the surface of the Earth and can offer useful insight into its thermal structure. 

The magnetic anomaly is the observed magnetic field less the contribution from the core and external fields.
Geologic features enhance or depress the local magnetic field; this response is primarily controlled by the concentration of magnetite in the Earth's crust.
Magnetite is the most prevalent magnetic mineral, in terms of both its magnetic susceptibility and concentration in the crust, and has a Curie point of 580°C.
The depth at which rocks lose their permanent magnetic properties is referred to as the Curie depth.
Due to the prevalence of magnetite in the crust, the Curie depth is often interpreted as the 580°C isotherm.
Current methods to derive the Curie depth first compute the (fast) Fourier transform over a square window of a magnetic anomaly that has been reduced to the pole.
The depth and thickness of magnetic sources is estimated from the slope of the radial power spectrum.

Our Python package, `pycurious`, ingests a map of the Earth's magnetic anomaly and distributes the computation of Curie depth across multiple CPUs.
`pycurious` implements the @Tanaka:1999 and @Bouligand:2009 methods for computing the thickness of a buried magnetic source.
The former selects portions of the radial power spectrum in the low and high frequency domain to compute the depth of magnetic sources, while the latter fits an analytical solution to the entire power spectrum.
We cast the @Bouligand:2009 method within a Bayesian framework to estimate the uncertainty of Curie depth calculations [@Mather:2019].
Common computational workflows and geospatial manipulation of magnetic data are covered in the Jupyter notebooks bundled with this package.
The `mapping` module includes a set of functions that help to wrangle maps of the magnetic anomaly into a useful form for `pycurious`.
Such an approach is commonly encountered for transforming global compilations of the magnetic anomaly, e.g. EMAG2 [@Meyer:2017], from latitudinal/longitudinal coordinates to a local projection in eastings/northings.

![Radial power spectra (right) computed from different sized windows of a synthetic magnetic anomaly (left) using `pycurious`.](figure.png)

`pycurious` includes the following functionality:

- Importing and exporting GeoTiff files.
- Converting between geospatial coordinate reference systems (CRS).
- Decomposition of magnetic grids into square windows over which to compute Curie depth.
- Parallel distribution of computation across computing resources.
- Bayesian inversion framework and sensitivity analysis.


## Documentation and examples

`pycurious` is bundled with a linked collection of Jupyter notebooks that can act as a user guide and an introduction to the package.
The notebooks are split into matching sets for frequentist and Bayesian estimates of Curie depth.
The notebooks cover:

- Plotting the radial and azimuthal power spectrum.
- Computing Curie depth from a synthetic magnetic anomaly.
- Exploring parameter sensitivity.
- Posing the inverse problem and objective function.
- Mapping Curie depth using the EMAG2 magnetic anomaly dataset [@Meyer:2017].

All documentation can be accessed from within the module via a Python function that installs the notebooks at a filesystem location specified by the user at run time.
The API documentation is kept up-to-date on [GitHub pages](https://brmather.github.io/pycurious/).
`pycurious` and all Python dependencies can be installed through the pypi.org `pip` package manager, however, several of the dependencies for `cartopy` may cause problems for inexperienced users.
We therefore provide a Docker image built with all required and optional dependencies and a deployment of the documentation / examples on [mybinder.org](https://mybinder.org/v2/gh/brmather/pycurious/binder?filepath=Notebooks%2F0-StartHere.ipynb).


# Acknowledgements

This work was made possible by the G.O.THERM.3D project, supported by an Irish Research Council Research for Policy & Society grant (RfPS/2016/50) co-funded by Geological Survey Ireland and by Sustainable Energy Authority Of Ireland.
Development of `pycurious` was inspired from [`pycpd`](https://github.com/groupeLIAMG/pycpd) and [`fatiando a terra`](https://github.com/fatiando/fatiando).


# References
