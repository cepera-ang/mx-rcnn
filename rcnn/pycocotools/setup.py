from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        '_mask',
        sources=['maskApi.c', '_mask.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=[ '-std=c99', '/DLL', '/link'],
    )
]

setup(name='pycocotools',
      ext_modules=cythonize(ext_modules)
)
