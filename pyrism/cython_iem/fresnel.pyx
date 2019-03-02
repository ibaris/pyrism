# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.01.2019 by Ismail Baris
"""
from __future__ import division
import numpy as np
cimport numpy as np

import cmath
from libc.math cimport cos, sin

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef:
    double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164

# ----------------------------------------------------------------------------------------------------------------------
# Computation of Rx0 and Rxi
# ----------------------------------------------------------------------------------------------------------------------
cdef tuple compute_Rx0(double complex[:] eps):
    cdef:
        Py_ssize_t xmax = eps.shape[0]
        Py_ssize_t i
        double complex[:] Rv0_view, Rh0_view

    Rv0 = np.zeros_like(eps, dtype=np.complex)
    Rh0 = np.zeros_like(eps, dtype=np.complex)
    Rv0_view = Rv0
    Rh0_view = Rh0

    for i in range(xmax):
        Rv0_view[i] = (cmath.sqrt(eps[i]) - 1) / (cmath.sqrt(eps[i]) + 1)
        Rh0_view[i] = -Rv0_view[i]

    return Rv0, Rh0

cdef tuple compute_Rxi(double[:] iza, double complex[:] eps, double complex[:] rt):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double complex[:] Rvi_view, Rhi_view
        double mui

    Rvi = np.zeros_like(iza, dtype=np.complex)
    Rhi = np.zeros_like(iza, dtype=np.complex)
    Rvi_view = Rvi
    Rhi_view = Rhi

    for i in range(xmax):
        mui = cos(iza[i])
        Rvi_view[i] = (eps[i] * mui - rt[i]) / (eps[i] * mui + rt[i])
        Rhi_view[i] = (mui - rt[i]) / (mui + rt[i])

    return Rvi, Rhi

cdef double complex[:] compute_rt(double[:] iza, double[:] epsr, double[:] epsi):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double complex[:] rt_view
        double complex eps
        double temp

    rt = np.zeros_like(iza, dtype=np.complex)
    rt_view = rt

    for i in range(xmax):
        eps = complex(epsr[i], epsi[i])
        rt[i] = cmath.sqrt(eps - pow(sin(iza[i]), 2))
        # rt[i] = complex(temp, epsi[i])

    return rt
