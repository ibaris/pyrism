# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 28.09.2019 by Ismail Baris
"""

cdef double complex[:,:,:] M(double complex[:,:,:] S, double[:] wavelength, double[:] N)
cdef double[:,:,:] KE(double complex[:,:,:] S, double[:] wavelength, double[:] N)
