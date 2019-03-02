# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Created on 02.03.2019 by Ismail Baris
"""
from __future__ import division
import numpy as np
cimport numpy as np
import cmath
from libc.math cimport sin, cos, pow

# ----------------------------------------------------------------------------------------------------------------------
# Comutation of Fxxuxxx
# ----------------------------------------------------------------------------------------------------------------------
# Compute Fxx with Method One --------------------------------------------------------------------------------------
cdef tuple compute_Cm1(int ud, double complex[:] Rvi, double complex[:] Rhi, double complex[:] eps, double[:]
                       k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                       double[:] phi):

    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i

        # < Temporaly Variable Definitions > ------------
        double Gqi
        double complex Gqti, qt

        double sin_iza, sin_vza, sin_raa, sin_phi, ki, k2
        double cos_iza, cos_vza, cos_raa, cos_phi

        # < Definitions of C Parameter > ------------
        double c11, c12, c21, c31, c41, c42, c51
        double complex c22, c32, c52, Rvii, Rhii

        # < Definition of Fxx > ------------
        double complex[:] Fvv_view, Fhh_view


    Fvv = np.zeros_like(iza, dtype=np.complex)
    Fhh = np.zeros_like(Fvv, dtype=np.complex)

    Fvv_view = Fvv
    Fhh_view = Fhh

    for i in range(xmax):
        Rvii = Rvi[i]
        Rhii = Rhi[i]

        sin_iza = sin(iza[i])
        sin_vza = sin(vza[i])
        sin_raa = sin(raa[i])
        sin_phi = sin(phi[i])

        cos_iza = cos(iza[i])
        cos_vza = cos(vza[i])
        cos_raa = cos(raa[i])
        cos_phi = cos(phi[i])

        ki = k[i]
        k2 = pow(ki, 2)

        Gqi = ud * kz_iza[i]
        Gqti = ud * ki * cmath.sqrt(eps[i] - pow(sin_iza, 2))

        # < Matrix Elements > ------------
        c11 = ki * cos_raa * (kz_vza[i] - Gqi)
        c12 = c11

        c21 = cos_iza * (cos_raa *
                                 (k2 * sin_iza * cos_phi * (sin_vza * cos_raa - sin_iza * cos_phi) +
                                  Gqi * (ki * cos_vza - Gqi)) +
                                 k2 * cos_phi * sin_iza * sin_vza * pow(sin_raa, 2))

        c22 = cos_iza * (cos_raa *
                                 (k2 * sin_iza * cos_phi * (sin_vza * cos_raa - sin_iza * cos_phi) +
                                  Gqti * (ki * cos_vza - Gqi)) + k2 * cos_phi * sin_iza * sin_vza * pow(sin_raa, 2))

        c31 = ki * sin_iza * (sin_iza * cos_phi * cos_raa * (ki * cos_vza - Gqi) -
                                      Gqi * (cos_raa * (sin_vza * cos_raa - sin_iza * cos_phi) +
                                             sin_vza * pow(sin_raa, 2)))

        c32 = ki * sin_iza * (sin_iza * cos_phi * cos_raa * (ki * cos_vza - Gqi) -
                                      Gqti * (cos_raa * (sin_vza * cos_raa - sin_iza * cos_phi) -
                                              sin_vza * pow(sin_raa, 2)))

        c41 = ki * cos_iza * (cos_raa * cos_vza * (ki * cos_vza - Gqi) +
                                      ki * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi))
        c42 = c41

        c51 = Gqi * (cos_raa * cos_vza * (Gqi - ki * cos_vza) -
                             ki * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi))
        c52 = Gqti * (cos_raa * cos_vza * (Gqi - ki * cos_vza) -
                              ki * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi))


        qt = ki * cmath.sqrt(eps[i] - pow(sin_iza, 2))

        Fvv_view[i] = (1 + Rvii) * (-(1 - Rvii) * c11 / kz_iza[i] + (1 + Rvii) * c12 / qt) + \
                      (1 - Rvii) * ((1 - Rvii) * c21 / kz_iza[i] - (1 + Rvii) * c22 / qt) + \
                      (1 + Rvii) * ((1 - Rvii) * c31 / kz_iza[i] - (1 + Rvii) * c32 / eps[i] / qt) + \
                      (1 - Rvii) * ((1 + Rvii) * c41 / kz_iza[i] - eps[i] * (1 - Rvii) * c42 / qt) + \
                      (1 + Rvii) * ((1 + Rvii) * c51 / kz_iza[i] - (1 - Rvii) * c52 / qt)

        Fhh_view[i] = (1 + Rhii) * ((1 - Rhii) * c11 / kz_iza[i] - eps[i] * (1 + Rhii) * c12 / qt) - \
                      (1 - Rhii) * ((1 - Rhii) * c21 / kz_iza[i] - (1 + Rhii) * c22 / qt) - \
                      (1 + Rhii) * ((1 - Rhii) * c31 / kz_iza[i] - (1 + Rhii) * c32 / qt) - \
                      (1 - Rhii) * ((1 + Rhii) * c41 / kz_iza[i] - (1 - Rhii) * c42 / qt) - \
                      (1 + Rhii) * ((1 + Rhii) * c51 / kz_iza[i] - (1 - Rhii) * c52 / qt)

    return Fvv, Fhh

# Compute Fxx with Method Two --------------------------------------------------------------------------------------
cdef tuple compute_Cm2(int ud, double complex[:] Rvi, double complex[:] Rhi, double complex[:] eps, double[:]
                       k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                       double[:] phi):

    cdef:
        Py_ssize_t xmax = iza.shape[0]
        Py_ssize_t i

        # < Temporaly Variable Definitions > ------------
        double Gqi
        double complex Gqti, qt

        double sin_iza, sin_vza, sin_raa, sin_phi, ki, k2
        double cos_iza, cos_vza, cos_raa, cos_phi

        # < Definitions of C Parameter > ------------
        double c11, c12, c21, c31, c41, c42, c51
        double complex c22, c32, c52, Rvii, Rhii

        # < Definition of Fxx > ------------
        double complex[:] Fvv_view, Fhh_view


    Fvv = np.zeros_like(iza, dtype=np.complex)
    Fhh = np.zeros_like(Fvv, dtype=np.complex)

    Fvv_view = Fvv
    Fhh_view = Fhh

    for i in range(xmax):
        Rvii = Rvi[i]
        Rhii = Rhi[i]

        sin_iza = sin(iza[i])
        sin_vza = sin(vza[i])
        sin_raa = sin(raa[i])
        sin_phi = sin(phi[i])

        cos_iza = cos(iza[i])
        cos_vza = cos(vza[i])
        cos_raa = cos(raa[i])
        cos_phi = cos(phi[i])

        ki = k[i]
        k2 = pow(ki, 2)

        Gqs = ud * kz_vza[i]
        Gqts = ud * ki * cmath.sqrt(eps[i] - pow(sin_vza, 2))

        # < Matrix Elements > ------------
        c11 = ki * cos_raa * (kz_iza[i] + Gqs)
        c21 = Gqs * (cos_raa * (cos_iza * (ki * cos_iza + Gqs) - ki * sin_iza * (
                    sin_vza * cos_raa - sin_iza * cos_phi)) - ki * sin_iza * sin_vza * sin_raa ** 2)
        c31 = ki * sin_vza * (ki * cos_iza * (sin_vza * cos_raa - sin_iza * cos_phi) + sin_iza * (kz_iza[i] + Gqs))
        c41 = ki * cos_vza * (cos_raa * (cos_iza * (kz_iza[i] + Gqs) - ki * sin_iza * (
                    sin_vza * cos_raa - sin_iza * cos_phi)) - ki * sin_iza * sin_vza * sin_raa ** 2)
        c51 = -cos_vza * (
                    ki ** 2 * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi) + Gqs * cos_raa * (kz_iza[i] + Gqs))

        c12 = ki * cos_raa * (kz_iza[i] + Gqs)
        c22 = Gqts * (cos_raa * (cos_iza * (kz_iza[i] + Gqs) - ki * sin_iza * (
                    sin_vza * cos_raa - sin_iza * cos_phi)) - ki * sin_iza * sin_vza * sin_raa ** 2)
        c32 = ki * sin_vza * (ki * cos_iza * (sin_vza * cos_raa - sin_iza * cos_phi) + sin_iza * (kz_iza[i] + Gqs))
        c42 = ki * cos_vza * (cos_raa * (cos_iza * (kz_iza[i] + Gqs) - ki * sin_iza * (
                    sin_vza * cos_raa - sin_iza * cos_phi)) - ki * sin_iza * sin_vza * sin_raa ** 2)
        c52 = -cos_vza * (
                    ki ** 2 * sin_vza * (sin_vza * cos_raa - sin_iza * cos_phi) + Gqts * cos_raa * (kz_iza[i] + Gqs))


        qt = ki * cmath.sqrt(eps[i] - pow(sin_iza, 2))

        Fvv_view[i] = (1 + Rvii) * (-(1 - Rvii) * c11 / kz_iza[i] + (1 + Rvii) * c12 / qt) + \
                      (1 - Rvii) * ((1 - Rvii) * c21 / kz_iza[i] - (1 + Rvii) * c22 / qt) + \
                      (1 + Rvii) * ((1 - Rvii) * c31 / kz_iza[i] - (1 + Rvii) * c32 / eps[i] / qt) + \
                      (1 - Rvii) * ((1 + Rvii) * c41 / kz_iza[i] - eps[i] * (1 - Rvii) * c42 / qt) + \
                      (1 + Rvii) * ((1 + Rvii) * c51 / kz_iza[i] - (1 - Rvii) * c52 / qt)

        Fhh_view[i] = (1 + Rhii) * ((1 - Rhii) * c11 / kz_iza[i] - eps[i] * (1 + Rhii) * c12 / qt) - \
                      (1 - Rhii) * ((1 - Rhii) * c21 / kz_iza[i] - (1 + Rhii) * c22 / qt) - \
                      (1 + Rhii) * ((1 - Rhii) * c31 / kz_iza[i] - (1 + Rhii) * c32 / qt) - \
                      (1 - Rhii) * ((1 + Rhii) * c41 / kz_iza[i] - (1 - Rhii) * c42 / qt) - \
                      (1 + Rhii) * ((1 + Rhii) * c51 / kz_iza[i] - (1 - Rhii) * c52 / qt)

    return Fvv, Fhh


cdef tuple compute_Fxxyxx(double complex[:] Rvi, double complex[:] Rhi, double complex[:] eps, double[:]
                          k, double[:] kz_iza, double[:] kz_vza, double[:] iza, double[:] vza, double[:] raa,
                          double[:] phi):

    cdef:
        double complex[:] Fvvupi, Fhhupi, Fvvups, Fhhups, Fvvdni, Fhhdni, Fvvdns, Fhhdns

    Fvvupi, Fhhupi = compute_Cm1(ud=1, Rvi=Rvi, Rhi=Rhi, eps=eps, k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza,
                                 raa=raa, phi=phi)
    Fvvups, Fhhups = compute_Cm2(ud=1, Rvi=Rvi, Rhi=Rhi, eps=eps, k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza,
                                 raa=raa, phi=phi)

    Fvvdni, Fhhdni = compute_Cm1(ud=-1, Rvi=Rvi, Rhi=Rhi, eps=eps, k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza,
                                 raa=raa, phi=phi)
    Fvvdns, Fhhdns = compute_Cm2(ud=-1, Rvi=Rvi, Rhi=Rhi, eps=eps, k=k, kz_iza=kz_iza, kz_vza=kz_vza, iza=iza, vza=vza,
                                 raa=raa, phi=phi)

    return Fvvupi, Fhhupi, Fvvups, Fhhups, Fvvdni, Fhhdni, Fvvdns, Fhhdns
