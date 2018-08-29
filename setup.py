
# python3 setup.py build_ext --inplace

import numpy as np
from setuptools import setup
from setuptools import Extension
# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Build import cythonize
from os import path


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()


ext = Extension(name               = 'pycurious.radon',
                sources            = ['src/radon.pyx','src/cradon.c'],
                include_dirs       = [np.get_include(),'src'],
                library_dirs       = ['pycurious'],
                extra_compile_args = ['-std=c99'])

if __name__ == "__main__":
    setup(name              = 'pycurious',
          long_description  = long_description,
          long_description_content_type = 'text/markdown',
          author            = "Ben Mather",
          author_email      = "brmather1@gmail.com",
          url               = "https://github.com/brmather/pycurious",
          version           = "0.1",
          description       = "Python tool for computing the Curie depth from magnetic data",
          ext_modules       = cythonize([ext]),
          packages          = ['pycurious',
                               'pycurious.grid',
                               'pycurious.optimise',
                               'pycurious.mapping',
                               'pycurious.parallel'],
          package_data      = {'pycurious': ['Examples/data/*',
                                             'Examples/Notebooks/Tanaka/*.ipynb',
                                             'Examples/Notebooks/Bouligand/*.ipynb',
                                             'Examples/Scripts/*.py']},
          classifiers       = ['Programming Language :: Python :: 2',
                               'Programming Language :: Python :: 2.7',
                               'Programming Language :: Python :: 3',
                               'Programming Language :: Python :: 3.3',
                               'Programming Language :: Python :: 3.4',
                               'Programming Language :: Python :: 3.5',
                               'Programming Language :: Python :: 3.6',
                               'Programming Language :: Python :: 3.7']
          )
