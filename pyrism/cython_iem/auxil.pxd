# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.01.2019 by Ismail Baris
"""

cdef int factorial(int x)
cdef tuple compute_ABCC(double Zy, double Zx, double iza, double complex eps)
cdef double complex RaV_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps)
cdef double complex RaH_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps)
cdef real_RaV_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps)
cdef imag_RaV_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps)
cdef real_RaH_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps)
cdef imag_RaH_integration_ifunc(double Zy, double Zx, double iza, double sigx, double sigy, double complex eps)
cdef double[:] compute_ShdwS(double[:] iza, double[:] vza, double[:] raa, double[:] rss)
cdef double[:] compute_ShdwS_X(double[:] iza, double[:] rss)
