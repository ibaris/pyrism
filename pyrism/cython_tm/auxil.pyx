# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 28.02.2019 by Ismail Baris
"""
from __future__ import division
import numpy as np
cimport numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Other Auxiliary Functions
# ----------------------------------------------------------------------------------------------------------------------
cdef double[:] equal_volume_from_maximum(double[:] radius, double[:] axis_ratio, int shape):
    """
    Convert maximum radius into volume equivalent radius
    
    Parameters
    ----------
    radius : double[:]
        Equivalent particle radius in same unit as wavelength.
    axis_ratio : double[:]
        The horizontal-to-rotational axis ratio.
    shape : int, {-1, -2}
        Shape of the particle:
            * -1 : spheroid,
            * -2 : cylinders.

    Returns
    -------
    Rv : MemoryView, double[:]
        Volume equivalent radius.
    """
    cdef float r_eq
    cdef:
        Py_ssize_t xmax = radius.shape[0]
        Py_ssize_t x

    result = np.zeros_like(radius, dtype=np.double)
    cdef double[:] result_view = result

    for x in range(xmax):
        if shape == -1:
            if axis_ratio[x] > 1.0:  # oblate
                result_view[x] = radius[x] / axis_ratio[x] ** (1.0 / 3.0)
            else:  # prolate
                result_view[x] = radius[x] / axis_ratio[x] ** (2.0 / 3.0)
        elif shape == -2:
            if axis_ratio[x] > 1.0:  # oblate
                result_view[x] = radius[x] * (0.75 / axis_ratio[x]) ** (1.0 / 3.0)
            else:  # prolate
                result_view[x] = radius[x] * (0.75 / axis_ratio[x]) ** (2.0 / 3.0)
        else:
            raise ValueError("Parameter shape must be -1 or -2.")

        return result
