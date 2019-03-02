# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""

cdef double complex[:] compute_Ft(double[:] iza, double[:] vza, double complex[:] eps)
cdef double[:] compute_Tf(double[:] iza, double[:] k, double[:] sigma, double complex[:] Rv0, double complex[:] Ft,
                          double[:, :] Wn, int[:] Ts)
