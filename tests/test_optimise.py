import pytest
import pycurious
import numpy as np
import numpy.testing as npt
from scipy.optimize import minimize

from conftest import load_magnetic_anomaly


def test_optimisation(load_magnetic_anomaly):
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]
    max_window = load_magnetic_anomaly["max_window"]

    grid = pycurious.CurieOptimise(d, xmin, xmax, ymin, ymax)
    beta, zt, dz, C = grid.optimise(max_window, xc, yc, taper=np.hanning)

    x_opt = np.array([beta, zt, dz])

    # hard-coded parameters used to generate the magnetic anomaly
    zt0 = 0.305
    dz0 = 10.0 + zt0
    beta0 = 3.0

    x0 = np.array([beta0, zt0, dz0])

    # compare if they are close or not
    # some parameters should be more similar than others
    tol = np.array([0.3, 0.1, 2.0])

    parameters = ["beta", "zt", "dz"]
    err_msg = "FAILED! {} = {:.4f} is not within an acceptable tolerance of {}"

    for i in range(x0.size):
        npt.assert_allclose(
            x_opt[i],
            x0[i],
            atol=tol[i],
            err_msg=err_msg.format(parameters[i], x_opt[i], tol[i]),
        )


def test_priors(load_magnetic_anomaly):
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]
    max_window = load_magnetic_anomaly["max_window"]

    grid = pycurious.CurieOptimise(d, xmin, xmax, ymin, ymax)
    beta0, zt0, dz0, C0 = grid.optimise(max_window, xc, yc)

    grid.add_prior(beta=(1.0, 0.1))
    beta1, zt1, dz1, C1 = grid.optimise(max_window, xc, yc)

    assert abs(beta1 - 1.0) < abs(
        beta0 - 1.0
    ), "FAILED! Optimisation with priors failed"


def test_valid_numbers(load_magnetic_anomaly):
    d = load_magnetic_anomaly["mag_data"]
    xc = load_magnetic_anomaly["xc"]
    yc = load_magnetic_anomaly["yc"]
    xmin, xmax, ymin, ymax = load_magnetic_anomaly["extent"]
    max_window = load_magnetic_anomaly["max_window"]

    # create phoney power spectrum
    S = np.array(
        [
            22.16409774,
            19.95258494,
            18.27873722,
            17.10575637,
            16.53959747,
            16.31539575,
            15.69619005,
            15.29953307,
            14.83475976,
            14.54031396,
            14.33361716,
            13.81764026,
            13.5176055,
            13.27386563,
            13.03493328,
            12.88581369,
            12.61998358,
            12.48616749,
            12.11261083,
            12.13079154,
            11.85440661,
            11.79244826,
            11.66823202,
            11.40231744,
            11.32521296,
            11.13634007,
            11.10650999,
            10.94822598,
            10.78032794,
            10.66593304,
            10.55815845,
            10.56805594,
            10.33514462,
            10.22026537,
            10.22945756,
            10.09275259,
            10.11562101,
            9.85061009,
            9.87165772,
            9.85976847,
            9.73954992,
            9.72021054,
            9.52959744,
            9.59582531,
            9.50927273,
            9.44691364,
            9.39293966,
            9.33097387,
            9.33191784,
        ]
    )

    k = np.array(
        [
            0.09237068,
            0.15443902,
            0.21486191,
            0.282031,
            0.33452175,
            0.4020083,
            0.46381582,
            0.5290359,
            0.59611689,
            0.65875047,
            0.71442694,
            0.77929408,
            0.84320843,
            0.90939865,
            0.96863721,
            1.03155349,
            1.09130906,
            1.15699532,
            1.22338685,
            1.28109263,
            1.34606572,
            1.40861721,
            1.46920824,
            1.53421046,
            1.59675833,
            1.66109235,
            1.7233503,
            1.78492941,
            1.84835837,
            1.90933735,
            1.973275,
            2.03650008,
            2.0983359,
            2.16386249,
            2.22416603,
            2.2876692,
            2.34841931,
            2.41225232,
            2.47774604,
            2.5379226,
            2.59980401,
            2.66346692,
            2.72735801,
            2.79146076,
            2.85220715,
            2.91443307,
            2.97643404,
            3.04008782,
            3.10369815,
        ]
    )

    sigma_S = np.ones_like(S)

    grid = pycurious.CurieOptimise(d, xmin, xmax, ymin, ymax)

    beta0 = 3.0
    zt0 = 1.0
    dz0 = 40.0
    C0 = 5.0
    x0 = np.array([beta0, zt0, dz0, C0])

    lower_bound = np.zeros_like(x0)
    upper_bound = [None] * len(lower_bound)

    # xi = func([5, 0., 45., 6.], S, k)
    xi = grid.min_func(x0, k, S, sigma_S)

    res = minimize(
        grid.min_func,
        x0,
        args=(k, S, sigma_S),
        method="TNC",
        bounds=list(zip(lower_bound, upper_bound)),
    )
    print("beta={:.2f}, zt={:.2f}, dz={:.2f}, C={:.2f}".format(*res.x))

    parameters = ["beta", "zt", "dz", "C"]
    err_msg = "FAILED! {} = {} is not a finite number"

    for i in range(res.x.size):
        assert np.isfinite(res.x[i]), err_msg.format(parameters[i], res.x[i])
