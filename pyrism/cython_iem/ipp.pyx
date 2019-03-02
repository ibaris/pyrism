# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""
from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport cos, pow, exp

# ----------------------------------------------------------------------------------------------------------------------
# Computation of IPP
# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Functions ----------------------------------------------------------------------------------------------
cdef double complex compute_Ixx(double qi, double qs, double kz_iza, double kz_vza, double sigma,
                       double complex fvv, double complex fhh, Py_ssize_t i, double complex Fxxupi,
                       double complex Fxxups, double complex Fxxdni, double complex Fxxdns):

    cdef:
        double complex Ixx
        double qi2, qs2, dkz, sigma2

    dkz = (kz_vza - kz_iza)
    qi2 = pow(qi, 2)
    qs2 = pow(qs, 2)
    sigma2 = pow(sigma, 2)

    Ixx = pow((kz_iza + kz_vza), i) * fvv * exp(-sigma2 * kz_iza * kz_vza) + 0.25 * \
          (Fxxupi * pow((kz_vza - qi), (i - 1)) * exp(-sigma2 * (qi2 - qi * dkz)) +
           Fxxdni * pow((kz_vza + qi), (i - 1)) * exp(-sigma2 * (qi2 + qi * dkz)) +
           Fxxups * pow((kz_iza + qs), (i - 1)) * exp(-sigma2 * (qs2 - qs * dkz)) +
           Fxxdns * pow((kz_iza - qs), (i - 1)) * exp(-sigma2 * (qs2 + qs * dkz)))

    return Ixx

# Computation of IPP -----------------------------------------------------------------------------------------------
cdef tuple compute_IPP(double[:] iza, double[:] vza, double[:] k, double[:] kz_iza, double[:] kz_vza, double[:] sigma,
                       double complex[:] fvv, double complex[:] fhh, int[:] Ts, double complex[:] Fvvupi,
                       double complex[:] Fhhupi, double complex[:] Fvvups, double complex[:] Fhhups,
                       double complex[:] Fvvdni, double complex[:] Fhhdni, double complex[:] Fvvdns,
                       double complex[:] Fhhdns):
    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t tmax = max(Ts.base)
        Py_ssize_t i, index

        # < Temporaly Variable Definitions > ------------
        double Gqi
        double qi, qs

        double sin_iza, sin_vza, sin_raa, sin_phi, ki, k2
        double cos_iza, cos_vza, cos_raa, cos_phi

        # < Definitions of C Parameter > ------------
        double c11, c12, c21, c31, c41, c42, c51
        double complex c22, c32, c52, Rvii, Rhii

        # < Definition of Fxx > ------------
        double complex[:, :] Ivv_view, Ihh_view

    Ivv = np.zeros((xmax, tmax), dtype=np.complex)
    Ihh = np.zeros((xmax, tmax), dtype=np.complex)

    Ivv_view = Ivv
    Ihh_view = Ihh

    for index in range(xmax):
        kz_izai = kz_iza[index]
        kz_vzai = kz_vza[index]

        fvvi = fvv[index]
        fhhi = fhh[index]

        sigmai = sigma[index]

        Fvvdnii = Fvvdni[index]
        Fvvupsi = Fvvups[index]
        Fvvdnsi = Fvvdns[index]
        Fvvupii = Fvvupi[index]

        Fhhdnii = Fhhdni[index]
        Fhhupsi = Fhhups[index]
        Fhhdnsi = Fhhdns[index]
        Fhhupii = Fhhupi[index]

        qi = k[index] * cos(iza[index])
        qs = k[index] * cos(vza[index])

        for i in range(1, Ts[index] + 1):
            Ivv_view[index, i-1] = compute_Ixx(qi, qs, kz_izai, kz_vzai, sigmai, fvvi, fhhi, i, Fvvupii, Fvvupsi, Fvvdnii,
                                             Fvvdnsi)

            Ihh_view[index, i-1] = compute_Ixx(qi, qs, kz_izai, kz_vzai, sigmai, fhhi, fhhi, i, Fhhupii, Fhhupsi, Fhhdnii,
                                              Fhhdnsi)

    return Ivv, Ihh
