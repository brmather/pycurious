## To install locally: python setup.py build && python setup.py install
## (If there are problems with installation of the documentation, it may be that
##  the egg file is out of sync and will need to be manually deleted - see error message
##  for details of the corrupted zip file. )
##
## To push a version through to pip.
##  - Make sure it installs correctly locally as above
##  - Update the version information in this file
##  - python setup.py sdist upload -r pypitest  # for the test version
##  - python setup.py sdist upload -r pypi      # for the real version
##
## (see http://peterdowns.com/posts/first-time-with-pypi.html)


import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize
from os import path
import io

## in development set version to none and ...
PYPI_VERSION = "1.0.3"

# Return the git revision as a string (from numpy)
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "--short", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


if PYPI_VERSION is None:
    PYPI_VERSION = git_version()

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md")) as f:
    long_description = f.read()


ext = Extension(
    name="pycurious.radon",
    sources=["src/radon.pyx", "src/cradon.c"],
    include_dirs=[np.get_include(), "src"],
    library_dirs=["pycurious"],
    extra_compile_args=["-std=c99"],
)

if __name__ == "__main__":
    setup(
        name="pycurious",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Ben Mather",
        author_email="brmather1@gmail.com",
        url="https://github.com/brmather/pycurious",
        version=PYPI_VERSION,
        description="Python tool for computing the Curie depth from magnetic data",
        ext_modules=cythonize([ext]),
        install_requires=["numpy", "scipy>=0.15.0", "Cython>=0.25.2"],
        python_requires=">=2.7, >=3.5",
        setup_requires=["pytest-runner", "webdav"],
        tests_require=["pytest", "webdav"],
        packages=["pycurious"],
        package_data={
            "pycurious": [
                "Examples/data/test_mag_data.txt",
                "Examples/*.ipynb",
                "Examples/Notebooks/Tanaka/*.ipynb",
                "Examples/Notebooks/Bouligand/*.ipynb",
                "Examples/Scripts/*.py",
            ]
        },
        classifiers=[
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
    )
