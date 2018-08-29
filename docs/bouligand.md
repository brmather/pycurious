# Bouligand method

All methods to derive Curie depth first compute the (fast) Fourier transform over a square window of a magnetic anomaly that has been reduced to the pole. Initial methods plotted the radial spectrum, obtained from FFT, against the wavenumber (rad/km) and used the gradient between two points to estimate the Curie depth ([Tanaka *et al.* 1999](http://linkinghub.elsevier.com/retrieve/pii/S0040195199000724)). The drawback here being that the choice of points is subjective and potentially meaningless if the window size is too small to capture long wavelength signals. Alternatively, [Bouligand *et al.* (2009)](http://doi.wiley.com/10.1029/2009JB006494) formulated an analytical expression for the radial power spectum using four parameters:

- *beta* - a fractal parameter
- *zt* - the top of magnetic sources
- *dz* - the thickness of the magnetic layer
- *C* - a field constant

The Curie depth is determined by minimising the misfit between the analytical power spectrum and the power spectrum computed from FFT.

This module was heavily inspired from [pycpd](https://github.com/groupeLIAMG/pycpd). We have simplified the API and repurposed Curie depth determinations within a Bayesian inverse framework. PyCurious implements an objective function that accepts *a priori* information, and quantifies the uncertainty of Curie depth.
