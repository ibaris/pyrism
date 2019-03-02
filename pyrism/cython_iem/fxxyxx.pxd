# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""

cdef tuple compute_Cm1(int ud, double complex[:] Rvi, double complex[:] Rhi, double complex[:] eps, double[:]
                       k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                       double[:] phi)
cdef tuple compute_Cm2(int ud, double complex[:] Rvi, double complex[:] Rhi, double complex[:] eps, double[:]
                       k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                       double[:] phi)
cdef tuple compute_Fxxyxx(double complex[:] Rvi, double complex[:] Rhi, double complex[:] eps, double[:]
                          k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                          double[:] phi)
