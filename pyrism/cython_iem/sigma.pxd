# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""

cdef tuple compute_sigma_nought(int[:] Ts, double[:, :] Wn, double complex[:, :] Ivv, double complex[:, :] Ihh,
                                double[:] ShdwS, double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] sigma)
