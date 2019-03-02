# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""

cdef tuple Rax_integration(double[:] iza, double[:] sigma, double[:] corrlength, double complex[:] eps)
cdef tuple compute_Rxt(double[:] iza, double[:] vza, double[:] raa, double[:] sigma, double[:] corrlength,
                       double complex[:] eps, double[:] Tf)
cdef tuple compute_fxx(double complex[:] Rvt, double complex[:] Rht, double[:] iza, double[:] vza, double[:] raa)
