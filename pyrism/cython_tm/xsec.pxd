# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 28.09.2019 by Ismail Baris
"""

cdef double complex[:,:,:] M(double complex[:,:,:] S, double[:] wavelength)

cdef double[:,:,:] XE(double complex[:,:,:] S, double[:] wavelength)

cdef double[:,:] XS_S(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg,
                          double[:] alphaDeg, double[:] betaDeg, int Nx, int Ny, int verbose, int asy)

cdef double[:,:] XS_AF(int[:] nmax, double[:] wavelength, double[:] izaDeg, double[:] iaaDeg, int Nx, int Ny,
                       int n_alpha, int n_beta, or_pdf, int verbose, int asy)

cdef double[:,:] XSEC_QSI(double[:,:,:] Z)
