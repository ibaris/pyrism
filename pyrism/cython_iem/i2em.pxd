# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""

cdef tuple compute_i2em(double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                        double[:] phi, double complex[:] eps, double[:] corrlength, double[:] sigma,
                        int[:] n, corrfunc)
cdef tuple compute_ixx(double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                        double[:] phi, double complex[:] eps, double[:] corrlength, double[:] sigma,
                        int[:] n, corrfunc)
