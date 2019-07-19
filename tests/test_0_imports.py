import pytest

## ===================


def test_numpy_import():
    import numpy

    return


def test_scipy_import():
    import scipy

    print("\t\t You have scipy version {}".format(scipy.__version__))


def test_cython_import():
    import Cython

    return


def test_pycurious_modules():
    import pycurious
    from pycurious import documentation
    from pycurious import CurieGrid
    from pycurious import CurieOptimise
    from pycurious import mapping
    from pycurious import download


# def test_jupyter_available():
#     from subprocess import check_output
#     try:
#         result = str(check_output(['which', 'jupyter']))[2:-3]
#     except:
#         assert False, "jupyter notebook system is not installed"

# def test_documentation_dependencies():
#     import matplotlib
#     import cartopy
#     import pyproj


if __name__ == "__main__":
    test_numpy_import()
    test_scipy_import()
    test_cython_import()
    test_pycurious_modules()
