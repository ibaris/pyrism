# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 28.02.2019 by Ismail Baris
"""

cdef int NMAX(double radius, int radius_type, double wavelength, double eps_real, double eps_imag, double axis_ratio,
              int shape)

cdef int[:] NMAX_VEC(double[:] radius, int radius_type, double[:] wavelength, double[:] eps_real, double[:] eps_imag,
                     double[:] axis_ratio, int shape, int verbose)
