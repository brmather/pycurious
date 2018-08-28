
import pkg_resources as _pkg_resources
from distutils import dir_util as _dir_util
import os


def install_documentation(path="./PyCurious-Examples"):
    """
    Install the examples for PyCurious in the given location.

    WARNING: If the path exists, the files will be written into the path
    and will overwrite any existing files with which they collide. The default
    path ("./PyCurious-Examples") is chosen to make collision less likely/problematic

    The documentation for PyCurious is in the form of jupyter notebooks.

    Some dependencies exist for the notebooks to be useful:

       - matplotlib: for some diagrams
       - cartopy: for mapping and visualisation
       - pyepsg: for converting between map projections

    PyCurious dependencies may be explicitly imported in the notebooks including:

       - numpy
       - scipy

    """

    Notebooks_Path = _pkg_resources.resource_filename("pycurious", os.path.join("Examples"))

    ct = _dir_util.copy_tree(Notebooks_Path, path, preserve_mode=1, preserve_times=1, preserve_symlinks=1, update=0, verbose=1, dry_run=0)

    return
