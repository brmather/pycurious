import pytest
import pycurious
import numpy as np
from time import time

# global variables
max_window = 100e3

# square grid 100km x 100km
xmin = 0.0
xmax = 100e3
ymin = 0.0
ymax = 100e3

nx = 100
ny = 100

# random noise
mag_grid = np.random.randn(ny,nx)


def time_routine(routine, *args, **kwargs):
    t = time()
    routine(*args, **kwargs)
    str_fmt = "- {:30} completed in  {:1.6f} seconds"
    print(str_fmt.format(str(routine.__name__), time()-t))


##
## Test PyCurious routines
##

def test_CurieGrid_routines():
    cpd = pycurious.CurieGrid(mag_grid, xmin, xmax, ymin, ymax)
    xc = 50e3
    yc = 50e3

    time_routine(cpd.__init__, mag_grid, xmin, xmax, ymin, ymax)
    time_routine(cpd.subgrid, max_window, xc, yc)
    time_routine(cpd.create_centroid_list, 0.1*max_window)
    time_routine(cpd.remove_trend_linear, cpd.data)
    time_routine(cpd.radial_spectrum, cpd.data)
    time_routine(cpd.radial_spectrum_log, cpd.data)
    time_routine(cpd.azimuthal_spectrum, cpd.data)
    time_routine(cpd.reduce_to_pole, cpd.data, 30., 30.)
    time_routine(cpd.upward_continuation, cpd.data, 10e3)


def test_CurieOptimise_routines():
    cpd = pycurious.CurieOptimise(mag_grid, xmin, xmax, ymin, ymax)
    xc_list, yc_list = cpd.create_centroid_list(0.5*max_window, spacingX=5e3, spacingY=5e3)
    xc = xc_list[0]
    yc = yc_list[0]

    time_routine(cpd.add_prior, beta=(3.0, 0.5))
    time_routine(cpd.optimise, 0.5*max_window, xc, yc)
    time_routine(cpd.optimise_routine, 0.5*max_window, xc_list, yc_list)
    time_routine(cpd.metropolis_hastings, 0.5*max_window, xc, yc, 100, 10)
    time_routine(cpd.sensitivity, 0.5*max_window, xc, yc, 100)


if __name__ == "__main__":
    test_CurieGrid_routines()
    test_CurieOptimise_routines()