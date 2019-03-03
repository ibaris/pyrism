# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""

cdef double[:] compute_wvnb(double[:] iza, double[:] vza, double[:] raa, double[:] phi, double[:] k)
cdef int[:] compute_TS(double[:] iza, double[:] vza, double[:] sigma, double[:] k)
cdef tuple compute_Wn_rss(corrfunc, double[:] iza, double[:] vza, double[:] raa, double[:] phi, double[:] k,
                          double[:] sigma, double[:] corrlength, int[:] n)
cdef int[:] compute_TS_X(double[:] iza, double[:] sigma, double[:] k)
