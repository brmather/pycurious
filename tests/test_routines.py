import pytest
import pycurious
import numpy as np
from time import time

from conftest import load_magnetic_anomaly


def time_routine(routine, *args, **kwargs):
    t = time()
    routine(*args, **kwargs)
    str_fmt = "- {:30} completed in  {:1.6f} seconds"
    print(str_fmt.format(str(routine.__name__), time() - t))
    return True


##
## Test PyCurious routines
##


def test_CurieGrid_routines(load_magnetic_anomaly):
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]
    max_window = load_magnetic_anomaly["max_window"]

    cpd = pycurious.CurieGrid(d, xmin, xmax, ymin, ymax)

    assert time_routine(cpd.__init__, d, xmin, xmax, ymin, ymax)
    assert time_routine(cpd.subgrid, max_window, xc, yc)
    assert time_routine(cpd.create_centroid_list, 0.1 * max_window)
    assert time_routine(cpd.remove_trend_linear, cpd.data)
    assert time_routine(cpd.radial_spectrum, cpd.data)
    assert time_routine(cpd.azimuthal_spectrum, cpd.data)
    assert time_routine(cpd.reduce_to_pole, cpd.data, 30.0, 30.0)
    assert time_routine(cpd.upward_continuation, cpd.data, 10e3)


def test_CurieOptimise_routines(load_magnetic_anomaly):
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]
    max_window = load_magnetic_anomaly["max_window"]

    cpd = pycurious.CurieOptimise(d, xmin, xmax, ymin, ymax)
    xc_list, yc_list = cpd.create_centroid_list(
        0.5 * max_window, spacingX=5e3, spacingY=5e3
    )
    xc = xc_list[0]
    yc = yc_list[0]

    assert time_routine(cpd.add_prior, beta=(3.0, 0.5))
    assert time_routine(cpd.optimise, 0.5 * max_window, xc, yc)
    assert time_routine(cpd.optimise_routine, 0.5 * max_window, xc_list, yc_list)
    assert time_routine(cpd.metropolis_hastings, 0.5 * max_window, xc, yc, 100, 10)
    assert time_routine(cpd.sensitivity, 0.5 * max_window, xc, yc, 100)
