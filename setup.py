
# python3 setup.py build_ext --inplace

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext = Extension(name               = 'pycurious.radon',
                sources            = ['pycurious/radon.pyx','pycurious/cradon.c'],
                include_dirs       = [np.get_include(),'pycurious'],
                library_dirs       = ['pycurious'],
                extra_compile_args = ['-std=c99'])

if __name__ == "__main__":
    setup(name              = 'pycurious',
          author            = "Ben Mather",
          author_email      = "brmather1@gmail.com",
          url               = "https://git.dias.ie/itherc/pycurious",
          version           = "0.1",
          description       = "Python tool for computing the Curie depth from magnetic data",
          ext_modules       = cythonize([ext]),
          packages          = ['pycurious'],
          classifiers       = ['Programming Language :: Python :: 2',
                               'Programming Language :: Python :: 2.6',
                               'Programming Language :: Python :: 2.7',
                               'Programming Language :: Python :: 3',
                               'Programming Language :: Python :: 3.3',
                               'Programming Language :: Python :: 3.4',
                               'Programming Language :: Python :: 3.5',
                               'Programming Language :: Python :: 3.6']
          )

# setup(
#     name = 'radon',
#     ext_modules = cythonize(extensions),
# )
