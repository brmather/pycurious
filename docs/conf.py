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

"""Sphinx configuration for the PyCurious documentation."""

from importlib.metadata import version as _version

# -- Project information -----------------------------------------------------

project = "PyCurious"
copyright = "2018-2019, Ben Mather, Robert Delhaye"
author = "Ben Mather, Robert Delhaye"

# The single source of truth for the version is pyproject.toml; read it back
# from the installed metadata rather than duplicating it here.
try:
    release = _version("pycurious")
except Exception:  # not installed (e.g. a bare docs checkout)
    release = "2.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",  # Markdown pages + executable notebooks (pulls in myst-parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The function docstrings are written in pdoc-flavoured Markdown (single-backtick
# code spans, the odd blockquote and bullet list). Rendering single backticks as
# inline literals matches that intent instead of treating them as RST
# interpreted text.
default_role = "literal"

# Warnings we accept rather than fix. The tutorial notebooks are shipped
# verbatim and carry internal anchor links and H1->H3 jumps that MyST flags; and
# the notebooks are executed from their own directory (so they can read the
# bundled data via relative paths), which myst-nb notes. None affect the output.
suppress_warnings = [
    "myst.xref_missing",
    "myst.header",
    "mystnb.local_cwd",
]

# -- autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# The docstrings use Google-style section headers (Args:/Returns:/Notes:) with
# NumPy-style `name : type` bodies, so enable both napoleon dialects.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
# The docstrings carry NumPy-style `name : type` bodies under Google-style
# `Returns:` headers, which napoleon would otherwise emit as a bogus
# `:rtype:` field (the return *name* mistaken for its type). Fold the return
# details into the description instead of a separate "Return type" line.
napoleon_use_rtype = False

# -- MyST / myst-nb ----------------------------------------------------------

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# Execute the self-contained tutorials at build time. The two Ex5 notebooks
# each pull ~600 MB of external data (EMAG2 v3 + Li et al.), so they are
# excluded from execution and render as code with a note pointing at the data.
nb_execution_mode = "cache"
nb_execution_excludepatterns = ["**/Ex5-*.ipynb"]
nb_execution_timeout = 300
nb_execution_raise_on_error = True

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "PyCurious"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/pycurious-logo.png"
html_favicon = "_static/pycurious-logo.png"

html_theme_options = {
    "source_repository": "https://github.com/brmather/pycurious/",
    "source_branch": "master",
    "source_directory": "docs/",
}
