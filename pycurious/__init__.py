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
PyCurious estimates the **Curie point depth** -- the depth at which rock loses
its magnetisation -- from the radially averaged spectrum of a magnetic anomaly.

Two methods share one grid and spectrum layer, and both return uncertainties:

- ``CurieOptimiseBouligand`` fits the four-parameter analytic spectrum of
  Bouligand *et al.* (2009) by optimisation.
- ``CurieOptimiseTanaka`` implements the centroid method of Tanaka *et al.*
  (1999), fitting two straight lines to separate wavenumber bands.

Full documentation -- a getting started guide, tutorials, the theory behind each
method, and the API reference -- lives at https://brmather.github.io/pycurious/.
"""

# -*- coding: utf-8 -*-
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("pycurious")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "2.0"

from .documentation import install_documentation
from .grid import CurieGrid, bouligand2009, tanaka1999, maus1995, ComputeTanaka
from .optimise_bouligand import CurieOptimiseBouligand
from .optimise_tanaka import CurieOptimiseTanaka
from .synthetic import fractal_anomaly
from . import mapping
from . import download
