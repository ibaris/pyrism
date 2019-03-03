# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.01.2019 by Ismail Baris
"""
from __future__ import division
import numpy as np
cimport numpy as np
import cmath
import scipy
from scipy.special import erf
from libc.math cimport sin, cos, pow, sqrt, exp


DTYPE = np.float

ctypedef np.float_t DTYPE_t

cdef:
    double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164


# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Function
# ----------------------------------------------------------------------------------------------------------------------
cdef int factorial(int x):
    # Basic example of a cython function, which defines
    # python-like operations and control flow on defined c types

    cdef int m = x
    cdef int i

    if x <= 1:
        return 1
    else:
        for i in range(1, x):
            m = m * i
        return m

cdef tuple compute_ABCC(double Zy, double Zx, double iza, double complex eps):
    cdef:
        double A, CC, cos_iza, sin_iza, pd
        double complex B

    cos_iza = cos(iza)
    sin_iza = sin(iza)

    A = cos_iza + Zx * sin_iza
    B = (1 + pow(Zx, 2) + pow(Zy, 2)) * eps
    CC = pow(sin_iza, 2) - 2 * Zx * sin_iza * cos_iza + pow(Zx, 2) * pow(cos_iza, 2) + pow(Zy, 2)

    return A, B, CC

# Callable Integration Functions -----------------------------------------------------------------------------------
cdef double complex RaV_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    cdef:
        double A, CC, pd
        double complex B, Rv, Rav

    A, B, CC = compute_ABCC(Zy, Zx, iza, eps)
    pd = exp(-pow(Zx, 2) / (2 * pow(sigx, 2)) - pow(Zy, 2) / (2 * pow(sigy, 2)))

    Rv = (eps * A - cmath.sqrt(B - CC)) / (eps * A + cmath.sqrt(B - CC))
    Rav = Rv * pd

    return Rav#.real

cdef double complex RaH_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    cdef:
        double A, CC, pd
        double complex B, Rh, Rah

    A, B, CC = compute_ABCC(Zy, Zx, iza, eps)
    pd = exp(-pow(Zx, 2) / (2 * pow(sigx, 2)) - pow(Zy, 2) / (2 * pow(sigy, 2)))

    Rh = (A - cmath.sqrt(B - CC)) / (A + cmath.sqrt(B - CC))
    RaH = Rh * pd

    return RaH#.real

cdef real_RaV_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    return scipy.real(RaV_integration_ifunc(Zy, Zx, iza, sigx, sigy, eps))

cdef imag_RaV_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    return scipy.imag(RaV_integration_ifunc(Zy, Zx, iza, sigx, sigy, eps))

cdef real_RaH_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    return scipy.real(RaH_integration_ifunc(Zy, Zx, iza, sigx, sigy, eps))

cdef imag_RaH_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps):
    return scipy.imag(RaH_integration_ifunc(Zy, Zx, iza, sigx, sigy, eps))

cdef double[:] compute_ShdwS(double[:] iza, double[:] vza, double[:] raa, double[:] rss):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i

        double ct, cts, rslp, ctorslp, ctsorslp, shadf, shadfs

        double[:] ShdwS_view

    ShdwS = np.zeros((xmax), dtype=np.double)
    ShdwS_view = ShdwS

    for i in range(xmax):
        if iza[i] == vza[i] and np.allclose(raa[i], PI):
            ct = cos(iza[i]) / sin(iza[i])
            cts = cos(vza[i]) / sin(vza[i])
            rslp = rss[i]
            ctorslp = ct / sqrt(2) / rslp
            ctsorslp = cts / sqrt(2) / rslp
            shadf = 0.5 * (exp(-ctorslp ** 2) / sqrt(PI) / ctorslp - erf(ctorslp))
            shadfs = 0.5 * (exp(-ctsorslp ** 2) / sqrt(PI) / ctsorslp - erf(ctsorslp))

            ShdwS_view[i] = 1 / (1 + shadf + shadfs)
        else:
            ShdwS_view[i] = 1.0

    return ShdwS

cdef double[:] compute_ShdwS_X(double[:] iza, double[:] rss):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i

        double ct, cts, rslp, ctorslp, ctsorslp, shadf, shadfs

        double[:] ShdwS_view

    ShdwS = np.zeros((xmax), dtype=np.double)
    ShdwS_view = ShdwS

    for i in range(xmax):
        ct = cos(iza[i]) / sin(iza[i])
        rslp = rss[i]
        ctorslp = ct / sqrt(2) / rslp
        shadf = 0.5 * (exp(-ctorslp ** 2) / sqrt(PI) / ctorslp - erf(ctorslp))

        ShdwS_view[i] = 1 / (1 + shadf)

    return ShdwS
