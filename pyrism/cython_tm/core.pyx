# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 28.02.2019 by Ismail Baris
"""
from __future__ import division

import numpy as np
cimport numpy as np
from pyrism.fortran_tm.fotm import calctmat
import sys

cdef :
    float ddelt = 1e-3
    int ndgs = 2

# NOT Vectorized -------------------------------------------------------------------------------------------------------
cdef int NMAX(double radius, int radius_type, double wavelength, double eps_real, double eps_imag, double axis_ratio,
              int shape):
    """
    Calculate NMAX.
    
    Parameters
    ----------
    radius : double
        Equivalent particle radius in same unit as wavelength.
    radius_type : int, {0, 1, 2}
        Specification of particle radius:
            * 0: radius is the equivalent volume radius (default).
            * 1: radius is the equivalent area radius.
            * 2: radius is the maximum radius.
    wavelength : double
        Wavelength in same unit as radius.
    eps_real, eps_imag : double
        Real and imaginary part of the dielectric constant.
    axis_ratio : double
        The horizontal-to-rotational axis ratio.
    shape : int, {-1, -2}
        Shape of the particle:
            * -1 : spheroid,
            * -2 : cylinders.
    
    Returns
    -------
    nmax : int
    
    """
    return calctmat(radius, radius_type, wavelength, eps_real, eps_imag, axis_ratio, shape, ddelt, ndgs)

cdef int[:] NMAX_VEC(double[:] radius, int radius_type, double[:] wavelength, double[:] eps_real, double[:] eps_imag,
                     double[:] axis_ratio, int shape, int verbose):
    """
    Calculate NMAX in a vectorized function. 
    
    Parameters
    ----------
    radius : double[:]
        Equivalent particle radius in same unit as wavelength.
    radius_type : int, {0, 1, 2}
        Specification of particle radius:
            * 0: radius is the equivalent volume radius (default).
            * 1: radius is the equivalent area radius.
            * 2: radius is the maximum radius.
    wavelength : double[:]
        Wavelength in same unit as radius.
    eps_real, eps_imag : double[:]
        Real and imaginary part of the dielectric constant.
    axis_ratio : double[:]
        The horizontal-to-rotational axis ratio.
    shape : int, {-1, -2}
        Shape of the particle:
            * -1 : spheroid,
            * -2 : cylinders.
    
    Returns
    -------
    nmax : MemoryView
    
    """

    cdef:
        Py_ssize_t xmax = radius.shape[0]
        Py_ssize_t x

    result = np.zeros_like(radius, dtype=np.intc)
    cdef int[:] result_view = result

    for x in range(xmax):
        if verbose == 1:
            sys.stdout.write("\r" + "Computing NMAX: {0} of {1}".format(str(x+1), str(xmax)))
            sys.stdout.flush()

        result_view[x] = calctmat(radius[x], radius_type, wavelength[x], eps_real[x], eps_imag[x], axis_ratio[x],
                                  shape, ddelt, ndgs)
    if verbose == 1:
        sys.stdout.write("\n")

    return result
