# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 08.02.2019 by Ismail Baris
"""
from __future__ import division

import numpy as np
cimport numpy as np
from scipy.integrate import quad

cdef float PI = 3.14159265359
# ----------------------------------------------------------------------------------------------------------------------
# Orientation Function
# ----------------------------------------------------------------------------------------------------------------------
cdef gaussian(float std, float mean):
    """Gaussian probability distribution function (PDF) for orientation averaging.

    Parameters
    ----------
    std : float
        The standard deviation in degrees of the Gaussian PDF
    mean : float
        The mean in degrees of the Gaussian PDF.  This should be a number in the interval [0, 180)

    Returns
    -------
    pdf(x): callable
        A function that returns the value of the spherical Jacobian-normalized Gaussian PDF with the given STD at x
        (degrees). It is normalized for the interval [0, 180].
    """
    cdef float norm_const = 1.0

    def pdf(x):
        return norm_const * np.exp(-0.5 * ((x - mean) / std) ** 2) * \
               np.sin(PI / 180.0 * x)

    cdef float norm_dev = quad(pdf, 0.0, 180.0)[0]

    # ensure that the integral over the distribution equals 1
    norm_const /= norm_dev

    return pdf

cdef uniform():
    """Uniform probability distribution function (PDF) for orientation averaging.

    Returns
    -------
    pdf(x): callable
        A function that returns the value of the spherical Jacobian-normalized uniform PDF. It is normalized for
        the interval [0, 180].
    """
    cdef float norm_const = 1.0

    def pdf(x):
        return norm_const * np.sin(PI / 180.0 * x)

    cdef float norm_dev = quad(pdf, 0.0, 180.0)[0]
    # ensure that the integral over the distribution equals 1
    norm_const /= norm_dev
    return pdf
