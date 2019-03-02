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
import os

cmdclass = {}
ext_modules = []

with open('requirements.txt') as f:
    required = f.read().splitlines()


def auto_ext_module(path, extension='.pyx'):
    if os.path.isdir(path):
        files = os.listdir(path)

        ext_modules = list()

        for item in files:
            if item.endswith(extension):
                path_to_item = os.path.join(path, item)
                module_call = path_to_item.replace('/', '.')
                module = Extension(module_call, [path_to_item], include_dirs=['.'])

                ext_modules.append(module)

        return ext_modules

    else:
        raise ValueError("Path {0} not found".format(path))


if use_cython:
    print ('******** Compiling with CYTHON accomplished ******')

    # ext_modules = auto_ext_module("pyrism/cython_tm")
    ext_modules += [
        # T-MATRIX----------------------------------------------------------------------------------------------------
        Extension("pyrism.cython_tm.core",
                  ["pyrism/cython_tm/core.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_tm.sz_matrix",
                  ["pyrism/cython_tm/sz_matrix.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_tm.pdf",
                  ["pyrism/cython_tm/pdf.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_tm.auxil",
                  ["pyrism/cython_tm/auxil.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_tm.xsec",
                  ["pyrism/cython_tm/xsec.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_tm.wrapper",
                  ["pyrism/cython_tm/wrapper.pyx"], include_dirs=['.']),

        # I2EM -------------------------------------------------------------------------------------------------------
        Extension("pyrism.cython_iem.auxil",
                  ["pyrism/cython_iem/auxil.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_iem.bicoef",
                  ["pyrism/cython_iem/bicoef.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_iem.ems",
                  ["pyrism/cython_iem/ems.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_iem.fresnel",
                  ["pyrism/cython_iem/fresnel.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_iem.fxxyxx",
                  ["pyrism/cython_iem/fxxyxx.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_iem.i2em",
                  ["pyrism/cython_iem/i2em.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_iem.ipp",
                  ["pyrism/cython_iem/ipp.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_iem.rspectrum",
                  ["pyrism/cython_iem/rspectrum.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_iem.sigma",
                  ["pyrism/cython_iem/sigma.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_iem.transition",
                  ["pyrism/cython_iem/transition.pyx"], include_dirs=['.']),

        Extension("pyrism.cython_iem.wrapper",
                  ["pyrism/cython_iem/wrapper.pyx"], include_dirs=['.']),

        # SOIL SECTION -----------------------------------------------------------------------------------------------

        Extension("pyrism.core.rphs",
                  ["pyrism/core/rphs.pyx"], include_dirs=['.']),

        Extension("pyrism.core.fauxil",
                  ["pyrism/core/fauxil.pyx"], include_dirs=['.']),

    ]

    cmdclass.update({'build_ext': build_ext})

else:
    print ('******** CYTHON Not Found. Use distributed .c files *******')

    ext_modules += [
        # T-MATRIX ---------------------------------------------------------------------------------------------------
        Extension("pyrism.cython_tm.core",
                  ["pyrism/cython_tm/core.c"], include_dirs=['.']),

        Extension("pyrism.cython_tm.sz_matrix",
                  ["pyrism/cython_tm/sz_matrix.c"], include_dirs=['.']),

        Extension("pyrism.cython_tm.pdf",
                  ["pyrism/cython_tm/pdf.c"], include_dirs=['.']),

        Extension("pyrism.cython_tm.auxil",
                  ["pyrism/cython_tm/auxil.c"], include_dirs=['.']),

        Extension("pyrism.cython_tm.xsec",
                  ["pyrism/cython_tm/xsec.c"], include_dirs=['.']),

        Extension("pyrism.cython_tm.wrapper",
                  ["pyrism/cython_tm/wrapper.c"], include_dirs=['.']),

        # I2EM -------------------------------------------------------------------------------------------------------
        Extension("pyrism.cython_iem.auxil",
                  ["pyrism/cython_iem/auxil.c"], include_dirs=['.']),

        Extension("pyrism.cython_iem.bicoef",
                  ["pyrism/cython_iem/bicoef.c"], include_dirs=['.']),

        Extension("pyrism.cython_iem.ems",
                  ["pyrism/cython_iem/ems.c"], include_dirs=['.']),

        Extension("pyrism.cython_iem.fresnel",
                  ["pyrism/cython_iem/fresnel.c"], include_dirs=['.']),

        Extension("pyrism.cython_iem.fxxyxx",
                  ["pyrism/cython_iem/fxxyxx.c"], include_dirs=['.']),

        Extension("pyrism.cython_iem.i2em",
                  ["pyrism/cython_iem/i2em.c"], include_dirs=['.']),

        Extension("pyrism.cython_iem.ipp",
                  ["pyrism/cython_iem/ipp.c"], include_dirs=['.']),

        Extension("pyrism.cython_iem.rspectrum",
                  ["pyrism/cython_iem/rspectrum.c"], include_dirs=['.']),

        Extension("pyrism.cython_iem.sigma",
                  ["pyrism/cython_iem/sigma.c"], include_dirs=['.']),

        Extension("pyrism.cython_iem.transition",
                  ["pyrism/cython_iem/transition.c"], include_dirs=['.']),

        Extension("pyrism.cython_iem.wrapper",
                  ["pyrism/cython_iem/wrapper.c"], include_dirs=['.']),


        Extension("pyrism.core.rphs",
                  ["pyrism/core/rphs.c"], include_dirs=['.']),

        Extension("pyrism.core.fauxil",
                  ["pyrism/core/fauxil.c"], include_dirs=['.'])
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
