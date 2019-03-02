# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""
from __future__ import division
import numpy as np
cimport numpy as np
import cmath
from libc.math cimport sin, cos, pow, exp
from pyrism.cython_iem.fresnel cimport compute_Rx0
from pyrism.cython_iem.auxil cimport factorial
DTYPE = np.float

ctypedef np.float_t DTYPE_t

cdef:
    double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164

cdef double complex[:] compute_Ft(double[:] iza, double[:] vza, double complex[:] eps):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i
        double complex[:] Rv0, Rh0, Ft_view
        double sin_iza, sin_vza, cos_iza
        double complex Rv02

    Rv0, Rh0 = compute_Rx0(eps)

    Ft = np.zeros_like(iza, dtype=np.complex)
    Ft_view = Ft

    for i in range(xmax):
        sin_iza = sin(iza[i])
        sin_vza = sin(vza[i])
        cos_iza = cos(iza[i])
        Rv02 = Rv0[i] * Rv0[i]

        Ft_view[i] = 8 * Rv02 * sin_vza * (cos_iza + cmath.sqrt(eps[i] - pow(sin_iza, 2))) / (cos_iza * cmath.sqrt(eps[i] - pow(sin_iza, 2)))

    return Ft

cdef double[:] compute_Tf(double[:] iza, double[:] k, double[:] sigma, double complex[:] Rv0, double complex[:] Ft,
                          double[:, :] Wn, int[:] Ts):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t x, i, index
        double a0, a1, b1, temp, St, St0
        double[:] Tf_view

    Tf = np.zeros_like(iza, dtype=np.double)
    Tf_view = Tf

    a1, b1 = 0.0, 0.0

    for i in range(xmax):
        for x in range(1, Ts[i] + 1):
            index = x - 1
            cos_iza = cos(iza[i])

            a0 = pow(k[i] * sigma[i] * cos_iza, 2*x) / factorial(x)
            a1 += a0 * Wn[i, index]

            temp = abs(Ft[i] / 2 + pow(2, x+1) * Rv0[i] / cos_iza * exp(- pow(k[i] * sigma[i] * cos_iza, 2)))
            b1 += a0 * pow(temp, 2) * Wn[i, index]

        St = 0.25 * pow(abs(Ft[i]), 2) * a1 / b1
        St0 = 1 / pow(abs(1 + 8 * Rv0[i] / (cos_iza * Ft[i])), 2)

        Tf_view[i] = 1-St/St0

    return Tf
