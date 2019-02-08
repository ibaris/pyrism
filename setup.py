# -*- coding: UTF-8 -*-

"""
This file is part of pyrism.
(c) 2017- Ismail Baris
For COPYING and LICENSE details, please refer to the LICENSE file
"""

import numpy

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False

else:
    use_cython = True

from setuptools import find_packages

cmdclass = {}
ext_modules = []

with open('requirements.txt') as f:
    required = f.read().splitlines()

if use_cython:
    print ('******** Compiling with CYTHON accomplished ******')

    ext_modules += [
        Extension("pyrism.core.rscat",
                  ["pyrism/core/rscat.pyx"], include_dirs=['.']),

        Extension("pyrism.core.rphs",
                  ["pyrism/core/rphs.pyx"], include_dirs=['.']),

        Extension("pyrism.core.fauxil",
                  ["pyrism/core/fauxil.pyx"], include_dirs=['.']),

        Extension("pyrism.core.iemauxil",
                  ["pyrism/core/iemauxil.pyx"], include_dirs=['.']),

        Extension("pyrism.core.tma",
                  ["pyrism/core/tma.pyx"], include_dirs=['.'])
    ]

    cmdclass.update({'build_ext': build_ext})

else:
    print ('******** CYTHON Not Found. Use distributed .c files *******')

    ext_modules += [
        Extension("pyrism.core.rscat",
                  ["pyrism/core/rscat.c"], include_dirs=['.']),

        Extension("pyrism.core.rphs",
                  ["pyrism/core/rphs.c"], include_dirs=['.']),

        Extension("pyrism.core.fauxil",
                  ["pyrism/core/fauxil.c"], include_dirs=['.']),

        Extension("pyrism.core.iemauxil",
                  ["pyrism/core/iemauxil.c"], include_dirs=['.']),

        Extension("pyrism.core.tma",
                  ["pyrism/core/tma.c"], include_dirs=['.'])
    ]


def get_packages():
    find_packages(exclude=['docs', 'tests']),
    return find_packages()


def get_version():
    version = dict()

    with open("pyrism/version.py") as fp:
        exec (fp.read(), version)

    return version['__version__']


setup(name='pyrism',

      version=get_version(),

      description='Python bindings for Remote Sensing Models',

      packages=get_packages(),

      cmdclass=cmdclass,

      include_dirs=[numpy.get_include()],
      ext_modules=ext_modules,

      author="Ismail Baris",
      maintainer='Ismail Baris',

      # ~ license='APACHE 2',

      url='https://github.com/ibaris/pyrism',

      long_description='A collection of optical and radar models to simulate surfaces and volumes.',
      # install_requires=install_requires,

      keywords=["radar", "remote-sensing", "optics", "integration",
                "microwave", "estimation", "physics", "radiative transfer"],

      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Atmospheric Science',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 2.7',
          'Operating System :: Microsoft',

      ],
      # package_data={"": ["*.txt"]},
      include_package_data=True,
      install_requires=required,
      setup_requires=[
          'pytest-runner',
      ],
      tests_require=[
          'pytest',
      ],
      )

# Build the f2py fortran extension
# --------------------------------
from numpy.distutils.core import Extension
from numpy.distutils.core import setup

# if use_cython:
#     print ('******** Compiling with FORTRAN accomplished ******')
#
#
#     flib = Extension(name='pyrism.fortran_tm.fotm'
#                           '',
#                      sources=['pyrism/fortran_tm/fotm.pyf',
#                               'pyrism/fortran_tm/ampld.lp.f',
#                               'pyrism/fortran_tm/lpd.f'],
#                      )
#
# else:
#     print ('******** FORTRAN Not Found. Use distributed .o files *******')
#
#     flib = Extension(name='pyrism.fortran_tm.tmatrix'
#                           '',
#                      sources=['pyrism/fortran_tm/tmatrixmodule.c',
#                               'pyrism/fortran_tm/tmatrix.o'],
#                      )
#
#
flib = Extension(name='pyrism.fortran_tm.fotm',
                 sources=['pyrism/fortran_tm/fotm.pyf',
                          'pyrism/fortran_tm/ampld.lp.f',
                          'pyrism/fortran_tm/lpd.f'],
                 )

setup(name='pyrism_ftm',

      version=get_version(),

      description='Python bindings for Remote Sensing Models and T-Matrix version',
      ext_modules=[flib])

print ('******** Installation completed ********')
