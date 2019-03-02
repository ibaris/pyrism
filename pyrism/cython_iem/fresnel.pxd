# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""

cdef tuple compute_Rx0(double complex[:] eps)
cdef tuple compute_Rxi(double[:] iza, double complex[:] eps, double complex[:] rt)
cdef double complex[:] compute_rt(double[:] iza, double[:] epsr, double[:] epsi)
