# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 28.02.2019 by Ismail Baris
"""

cdef tuple SZ_S_VEC(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] vzaDeg, double[:] iaaDeg,
                    double[:] vaaDeg, double[:] alphaDeg, double[:] betaDeg)

cdef tuple SZ_AF_VEC(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] vzaDeg, double[:] iaaDeg,
                     double[:] vaaDeg, int n_alpha, int n_beta, or_pdf)

cdef tuple SZ_S(int nmax, double wavelength, double izaDeg, double vzaDeg, double iaaDeg,
                double vaaDeg, double alphaDeg, double betaDeg)

cdef tuple SZ_AF(int nmax, double wavelength, double izaDeg, double vzaDeg, double iaaDeg, double vaaDeg, int n_alpha,
                 int n_beta, or_pdf)
