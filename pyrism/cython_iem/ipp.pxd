# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""

cdef double complex compute_Ixx(double qi, double qs, double kz_iza, double kz_vza, double sigma,
                       double complex fvv, double complex fhh, Py_ssize_t i, double complex Fxxupi,
                       double complex Fxxups, double complex Fxxdni, double complex Fxxdns)
cdef tuple compute_IPP(double[:] iza, double[:] vza, double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] sigma,
                       double complex[:] fvv, double complex[:] fhh, int[:] Ts, double complex[:] Fvvupi,
                       double complex[:] Fhhupi, double complex[:] Fvvups, double complex[:] Fhhups,
                       double complex[:] Fvvdni, double complex[:] Fhhdni, double complex[:] Fvvdns,
                       double complex[:] Fhhdns)
