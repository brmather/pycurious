## Project metadata now lives in pyproject.toml.
##
## This file is kept only as a compatibility shim for tooling that still shells
## out to setup.py. Do not add metadata here -- it will conflict with the
## [project] table in pyproject.toml.
##
## To install locally:      python -m pip install -e .
## To build a release:      python -m build
## To upload to PyPI:       python -m twine upload dist/*

from setuptools import setup

if __name__ == "__main__":
    setup()
