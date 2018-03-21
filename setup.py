# -*- coding: UTF-8 -*-

"""
This file is part of pyrism.
(c) 2017- Ismail Baris
For COPYING and LICENSE details, please refer to the LICENSE file
"""

from setuptools import find_packages
from setuptools import setup


def get_packages():
    find_packages(exclude=['docs', 'tests']),
    return find_packages()


setup(name='pyrism',

      version='0.0.1.1',

      description='Python bindings for Remote Sensing Models',

      packages=get_packages(),
      package_dir={'core': 'core', 'models': 'models'},

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
      package_data={"": ["*.txt"]},
      include_package_data=True,
      install_requires=['numpy', 'scipy'],
      setup_requires=[
          'pytest-runner',
      ],
      tests_require=[
          'pytest',
      ],
      )
