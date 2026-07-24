#!/usr/bin/env python3
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
Stage the tutorial notebooks into the Sphinx source tree.

myst-nb can only build notebooks that live inside the documentation source
directory, so before ``sphinx-build`` the canonical example notebooks are copied
from ``Examples/`` into ``docs/tutorials/{bouligand,tanaka}/``. The bundled test
fixture is copied to ``docs/data/`` as well, because the Tanaka notebooks load it
through the relative path ``../../data/test_mag_data.txt`` and myst-nb executes
each notebook from its own directory.

The copied notebooks and data are gitignored -- this script regenerates them on
every build. Only the eleven canonical notebooks are staged (the same list as
``MANIFEST.in``); the ``Ex5`` notebooks are staged but not executed (see
``nb_execution_excludepatterns`` in ``conf.py``).
"""

import shutil
from pathlib import Path

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
EXAMPLES = REPO / "Examples"

# (source relative to Examples/Notebooks) -> (destination folder under tutorials)
NOTEBOOKS = {
    "Bouligand": [
        "Ex1-Plot-power-spectrum.ipynb",
        "Ex2-Compute-Curie-depth.ipynb",
        "Ex3-Posing-the-inverse-problem.ipynb",
        "Ex4-Spatial-variation-of-Curie-depth.ipynb",
        "Ex5-Mapping-Curie-depth-EMAG2.ipynb",
    ],
    "Tanaka": [
        "Ex1-Plot-amplitude-spectrum.ipynb",
        "Ex2-Compute-Curie-depth.ipynb",
        "Ex3-Parameter-exploration.ipynb",
        "Ex4-Spatial-variation-of-Curie-depth.ipynb",
        "Ex5-Mapping-Curie-depth-EMAG2.ipynb",
    ],
}

# Small data fixture the executed Tanaka notebooks read via ../../data/...
DATA_FILES = ["test_mag_data.txt"]


def main():
    for method, notebooks in NOTEBOOKS.items():
        dest = DOCS / "tutorials" / method.lower()
        dest.mkdir(parents=True, exist_ok=True)
        for name in notebooks:
            src = EXAMPLES / "Notebooks" / method / name
            if not src.exists():
                raise FileNotFoundError(f"missing tutorial notebook: {src}")
            shutil.copy2(src, dest / name)
            print(f"staged {src.relative_to(REPO)} -> {(dest / name).relative_to(REPO)}")

    data_dest = DOCS / "data"
    data_dest.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        src = EXAMPLES / "data" / name
        if not src.exists():
            raise FileNotFoundError(f"missing data fixture: {src}")
        shutil.copy2(src, data_dest / name)
        print(f"staged {src.relative_to(REPO)} -> {(data_dest / name).relative_to(REPO)}")


if __name__ == "__main__":
    main()
