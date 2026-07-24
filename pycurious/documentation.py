# Copyright 2018-2019 Ben Mather, Robert Delhaye
#
# This file is part of PyCurious.
#
# PyCurious is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# PyCurious is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PyCurious.  If not, see <http://www.gnu.org/licenses/>.

"""
Use the `pycurious.documentation.install_documentation` function to copy all
Jupyter Notebooks and example data to a local directory.

"""

import importlib.resources as _resources
import shutil as _shutil
import os


def install_documentation(path="./PyCurious-Examples"):
    """
    Install the examples for PyCurious in the given location.

    WARNING: If the path exists, the files will be written into the path
    and will overwrite any existing files with which they collide. The default
    path ("./PyCurious-Examples") is chosen to make collision less likely/problematic

    The documentation for PyCurious is in the form of jupyter notebooks.

    Some dependencies exist for the notebooks to be useful:

       - `matplotlib`: for some diagrams
       - `cartopy`: for mapping and visualisation
       - `pyepsg`: for converting between map projections

    PyCurious dependencies may be explicitly imported in the notebooks including:

       - `numpy`
       - `scipy`

    """

    Notebooks_Path = _find_examples()

    _shutil.copytree(Notebooks_Path, path, symlinks=True, dirs_exist_ok=True)

    return


def _find_examples():
    """
    Locate the bundled Examples directory.

    In an installed wheel the notebooks sit inside the package. In a source
    checkout (including `pip install -e .`) they live at the repository root,
    one level above the package, so fall back to that.
    """

    package_dir = _resources.files("pycurious")

    candidates = [
        os.path.join(str(package_dir), "Examples"),
        os.path.join(os.path.dirname(str(package_dir)), "Examples"),
    ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate the PyCurious Examples directory. Looked in:\n  "
        + "\n  ".join(candidates)
        + "\nIf you installed PyCurious from PyPI, the notebooks may not have "
        "been bundled; fetch them from "
        "https://github.com/brmather/pycurious instead."
    )
