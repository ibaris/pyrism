# cython: cdivision=True
from __future__ import division

cimport numpy as np
import numpy as np
# from libc.math cimport cos, sin, sqrt, exp, abs
import cmath
from scipy.integrate import dblquad
import cython
from scipy.special import erf
from libc.math cimport sin, cos, pow, sqrt, exp

DTYPE = np.float

ctypedef np.float_t DTYPE_t

# ----------------------------------------------------------------------------------------------------------------------
# Statistical Functions
# ----------------------------------------------------------------------------------------------------------------------
cdef int factorial(int x):
    # Basic example of a cython function, which defines
    # python-like operations and control flow on defined c types

    cdef Py_ssize_t m = x
    cdef Py_ssize_t i

    if x <= 1:
        return 1
    else:
        for i in range(1, x):
            m = m * i
        return m

# ----------------------------------------------------------------------------------------------------------------------
# Basic Math Operation
# ----------------------------------------------------------------------------------------------------------------------
cdef double[:] mul(double[:] x, double y):
    cdef:
        Py_ssize_t xmax = x.shape[0]
        Py_ssize_t i

    result = np.zeros_like(x, dtype=np.double)
    cdef double[:] result_view = result

    for i in range(xmax):
            result_view[i] = x[i] * y

    return result

cdef double[:] add(double[:] x, double y):
    cdef:
        Py_ssize_t xmax = x.shape[0]
        Py_ssize_t i

    result = np.zeros_like(x, dtype=np.double)
    cdef double[:] result_view = result

    for i in range(xmax):
            result_view[i] = x[i] + y

    return result

cdef double[:] pow(double[:] x, int n):
    cdef:
        Py_ssize_t xmax = x.shape[0]
        Py_ssize_t i

    result = np.zeros_like(x, dtype=np.double)
    cdef double[:] result_view = result

    for i in range(xmax):
            result_view[i] = pow(x[i], n)

    return result
