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

from setuptools import dist, setup, find_packages
from os import path
import io
import os
import subprocess
import platform

links_args = []
 
if "Windows" in platform.system():
    link_args = ["-static"]


## in development set version to none and ...
PYPI_VERSION = "2.0"

# Return the git revision as a string (from numpy)
def git_version():
    
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', '--short', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


if PYPI_VERSION is None:
    PYPI_VERSION = git_version()

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding='utf-8') as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(name = 'pycurious',
          author            = "Ben Mather",
          author_email      = "brmather1@gmail.com",
          url               = "https://github.com/brmather/pycurious",
          version           = PYPI_VERSION,
          description       = "Python tool for computing the Curie depth from magnetic data",
          long_description  = long_description,
          long_description_content_type='text/markdown',
          install_requires  = ['numpy>=1.16.0',
                               'scipy>=1.0.0',
                               ],
          packages          = ['pycurious'],
          package_data      = {'pycurious': ['Examples/*ipynb', # Worked Examples is not currently used
                                             'Examples/data/*'] },
          include_package_data = True,
          classifiers       = ['Programming Language :: Python :: 2',
                               'Programming Language :: Python :: 2.6',
                               'Programming Language :: Python :: 2.7',
                               'Programming Language :: Python :: 3',
                               'Programming Language :: Python :: 3.3',
                               'Programming Language :: Python :: 3.4',
                               'Programming Language :: Python :: 3.5',
                               'Programming Language :: Python :: 3.6',
                               'Programming Language :: Python :: 3.7',
                               'Programming Language :: Python :: 3.8',
                               'Programming Language :: Python :: 3.9',
                               'Programming Language :: Python :: 3.10'
                               ]
          )
